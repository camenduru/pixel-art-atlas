"""Viewer for the Pixel Art Atlas dataset.

Two interchangeable backends feed the same UI, and the header can switch
between them at runtime:

  remote  Hugging Face Hub over HTTPS
          https://huggingface.co/datasets/Angais/pixel-art-atlas-1.4m
          Sprites, captions and subjects are read straight out of the remote
          Parquet files with HTTP range requests (only the row groups you look
          at are downloaded), so no copy of the dataset is kept on disk.

  local   Parquet files in ./data (or --data DIR), read off the local disk.

Both stream the caption/subject index used for search in the background.

Usage:
    python viewer.py                 # starts on the Hub, http://127.0.0.1:8765
    python viewer.py --open
    python viewer.py --source local  # start on the local copy instead
    python viewer.py --data D:/atlas # local Parquet folder
"""

import argparse
import io
import json
import os
import struct
import threading
import time
import webbrowser
from bisect import bisect_right
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
REPO = "Angais/pixel-art-atlas-1.4m"
CAP = 20000
KINDS = (("remote", "HF Hub"), ("local", "local"))


def retry(fn, tries=4, delay=1.0):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:  # network hiccups are common with HF CDN
            last = exc
            time.sleep(delay * (i + 1))
    raise last


# --------------------------------------------------------------------------- #
# parquet backend shared by the Hub and local sources
# --------------------------------------------------------------------------- #
class Parquet:
    """Row-group addressed parquet reader; subclasses supply lists and streams."""

    kind = "parquet"
    label = "parquet"

    def _list(self):
        raise NotImplementedError

    def _stream(self, name):
        raise NotImplementedError

    def __init__(self, repo=""):
        self.repo = repo
        self.names = self._list()
        if not self.names:
            raise RuntimeError("no .parquet files found for the %s source" % self.label)

        self.handles = [None] * len(self.names)
        self.locks = [threading.Lock() for _ in self.names]
        counts = [0] * len(self.names)

        with ThreadPoolExecutor(max_workers=6) as pool:
            for k, n in zip(range(len(self.names)), pool.map(self._open, range(len(self.names)))):
                counts[k] = n

        self.starts = []
        offset = 0
        for n in counts:
            self.starts.append(offset)
            offset += n
        self.total = offset
        self.train_count = 0
        for k, name in enumerate(self.names):
            if "validation" not in name:
                self.train_count += counts[k]
        self.val_count = self.total - self.train_count

        self._cache = OrderedDict()
        self._cache_lock = threading.Lock()
        self._max_groups = 48
        self._md_cache = OrderedDict()
        self._md_lock = threading.Lock()

        self.captions = None
        self.subjects = None
        self.subject_list = []
        self.index_ready = False
        self._idx_done = 0
        self._idx_lock = threading.Lock()
        threading.Thread(target=self._load_index, daemon=True).start()

    # -- handles ---------------------------------------------------------- #
    def _open(self, k):
        def go():
            f = self._stream(self.names[k])
            pf = pq.ParquetFile(f)
            sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)]
            return f, pf, sizes

        f, pf, sizes = retry(go)
        gstarts = [0]
        for s in sizes[:-1]:
            gstarts.append(gstarts[-1] + s)
        self.handles[k] = (f, pf, sizes, gstarts)
        return int(pf.metadata.num_rows)

    def _locate(self, i):
        fi = bisect_right(self.starts, i) - 1
        ri = i - self.starts[fi]
        gstarts = self.handles[fi][3]
        gi = bisect_right(gstarts, ri) - 1
        return fi, gi, ri - gstarts[gi]

    def _group(self, fi, gi):
        key = (fi, gi)
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit

        f, pf, sizes, gstarts = self.handles[fi]
        with self.locks[fi]:
            tbl = retry(lambda: pf.read_row_group(gi, columns=["image", "caption", "subject"]))
        col = tbl.column("image").combine_chunks()
        try:
            pngs = col.field("bytes").to_pylist()
        except Exception:
            pngs = [x["bytes"] if isinstance(x, dict) else x for x in col.to_pylist()]
        entry = (pngs, tbl.column("caption").to_pylist(), tbl.column("subject").to_pylist())

        with self._cache_lock:
            self._cache[key] = entry
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_groups:
                self._cache.popitem(last=False)
        with self._md_lock:
            self._md_cache[key] = (entry[1], entry[2])
            self._md_cache.move_to_end(key)
            while len(self._md_cache) > 256:
                self._md_cache.popitem(last=False)
        return entry

    def _meta_group(self, fi, gi):
        key = (fi, gi)
        with self._md_lock:
            hit = self._md_cache.get(key)
            if hit is not None:
                self._md_cache.move_to_end(key)
                return hit
        with self.locks[fi]:
            tbl = retry(lambda: self.handles[fi][1].read_row_group(gi, columns=["caption", "subject"]))
        entry = (tbl.column("caption").to_pylist(), tbl.column("subject").to_pylist())
        with self._md_lock:
            self._md_cache[key] = entry
            self._md_cache.move_to_end(key)
            while len(self._md_cache) > 256:
                self._md_cache.popitem(last=False)
        return entry

    # -- index ------------------------------------------------------------ #
    def _index_progress(self):
        return min(1.0, self._idx_done / float(self.total)) if self.total else 0.0

    def _load_index(self):
        jobs = list(range(len(self.names)))
        parts = {}

        def work(k):
            def go():
                with self._stream(self.names[k]) as f:
                    pf = pq.ParquetFile(f)
                    return pf.read(columns=["caption", "subject"])

            tbl = retry(go)
            parts[k] = (tbl.column("caption").to_pylist(), tbl.column("subject").to_pylist())
            with self._idx_lock:
                self._idx_done += tbl.num_rows

        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(work, jobs))
        except Exception:
            return  # keep browsing working even if the index fails

        caps, subs = [], []
        for k in jobs:
            c, s = parts.pop(k)
            caps.extend(c)
            subs.extend(s)
        self.captions = caps
        self.subjects = subs
        self.subject_list = sorted(set(subs))
        self.index_ready = True

    # -- api -------------------------------------------------------------- #
    def _range(self, split):
        if split == "train":
            return 0, self.train_count
        if split == "validation":
            return self.train_count, self.total
        return 0, self.total

    def meta(self):
        return {
            "total": self.total,
            "train": self.train_count,
            "validation": self.val_count,
            "subjects": self.subject_list,
            "label": self.label,
            "repo": self.repo,
            "indexing": None if self.index_ready else round(self._index_progress(), 3),
        }

    def ids_for_range(self, start, count, split):
        lo, hi = self._range(split)
        start = lo + start
        return np.arange(start, min(start + count, hi), dtype=np.uint32)

    def sprite_ids(self, ids):
        wanted = [int(x) for x in ids]
        out = np.empty((len(wanted), 16, 16, 4), dtype=np.uint8)
        groups = {}
        for k, i in enumerate(wanted):
            fi, gi, ri = self._locate(i)
            groups.setdefault((fi, gi), []).append((k, ri))
        for (fi, gi), items in groups.items():
            pngs = self._group(fi, gi)[0]
            for k, ri in items:
                img = Image.open(io.BytesIO(pngs[ri]))
                if img.mode != "RGBA" or img.size != (16, 16):
                    img = img.convert("RGBA")
                out[k] = np.asarray(img, dtype=np.uint8)
        return out

    def item(self, i):
        split = "validation" if i >= self.train_count else "train"
        if self.index_ready:
            return {"id": i, "caption": self.captions[i], "subject": self.subjects[i], "split": split}
        fi, gi, ri = self._locate(i)
        caps, subs = self._meta_group(fi, gi)
        return {"id": i, "caption": caps[ri], "subject": subs[ri], "split": split}

    def png(self, i, scale):
        fi, gi, ri = self._locate(i)
        data = self._group(fi, gi)[0][ri]
        if scale <= 1:
            return data
        img = Image.open(io.BytesIO(data)).convert("RGBA")
        img = img.resize((16 * scale, 16 * scale), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()

    def query(self, q, subject, split, limit):
        if not self.index_ready:
            return {"indexing": round(self._index_progress(), 3)}
        lo, hi = self._range(split)
        caps, subs = self.captions, self.subjects
        hits = []
        for i in range(lo, hi):
            if subject and subs[i] != subject:
                continue
            if q and q not in caps[i].lower():
                continue
            hits.append(i)
            if len(hits) >= limit:
                break
        return {"mode": "list", "ids": hits, "total": len(hits), "truncated": len(hits) >= limit}


# --------------------------------------------------------------------------- #
# remote backend: Hugging Face Hub parquet over HTTP range requests
# --------------------------------------------------------------------------- #
class Remote(Parquet):
    kind = "remote"
    label = "HF Hub"

    def __init__(self, repo=REPO):
        from huggingface_hub import HfFileSystem

        self.repo = repo
        self.fs = HfFileSystem()
        super().__init__(repo)

    def _list(self):
        from huggingface_hub import HfApi

        names = retry(lambda: HfApi().list_repo_files(self.repo, repo_type="dataset"))
        return sorted(n for n in names if n.endswith(".parquet"))

    def _stream(self, name):
        return self.fs.open("datasets/%s/%s" % (self.repo, name), "rb")


# --------------------------------------------------------------------------- #
# local backend: parquet files in the dataset folder next to this script
# --------------------------------------------------------------------------- #
class Local(Parquet):
    kind = "local"
    label = "local"

    def __init__(self, directory=DATA):
        self.directory = directory
        super().__init__("")

    def _list(self):
        try:
            names = os.listdir(self.directory)
        except OSError as exc:
            raise RuntimeError("cannot read local dataset in %s (%s)" % (self.directory, exc))
        return sorted(n for n in names if n.endswith(".parquet"))

    def _stream(self, name):
        return open(os.path.join(self.directory, name), "rb")


# --------------------------------------------------------------------------- #
# http layer
# --------------------------------------------------------------------------- #
SOURCES = {}
SOURCES_LOCK = threading.Lock()
ACTIVE = ["remote"]


def get_source(kind):
    """Built once per kind; switching back to a source reuses its open handles."""
    with SOURCES_LOCK:
        src = SOURCES.get(kind)
    if src is not None:
        return src
    if kind == "local":
        src = Local(DATA)
    elif kind == "remote":
        src = Remote(REPO)
    else:
        raise RuntimeError("unknown source: %s" % kind)
    with SOURCES_LOCK:
        SOURCES.setdefault(kind, src)
        return SOURCES[kind]


def current():
    return get_source(ACTIVE[0])


def meta_payload():
    m = current().meta()
    m["source"] = ACTIVE[0]
    m["sources"] = [{"kind": k, "label": l} for k, l in KINDS]
    return m


def batch_bytes(ids):
    rgba = current().sprite_ids(ids)
    head = struct.pack("<I", len(ids)) + np.asarray(ids, dtype="<u4").tobytes()
    return head + rgba.tobytes()


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pixel Art Atlas viewer</title>
<style>
  :root {
    --bg: #12141a;
    --bg2: #171a21;
    --line: #262b36;
    --fg: #e7eaf0;
    --dim: #9aa3b2;
    --accent: #ffcc4d;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    background: var(--bg);
    color: var(--fg);
    font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  header {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    padding: 10px 14px;
    background: var(--bg2);
    border-bottom: 1px solid var(--line);
    z-index: 5;
  }
  header h1 {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: .04em;
    margin: 0 10px 0 0;
    white-space: nowrap;
  }
  header h1 span { color: var(--accent); }
  .field { display: flex; align-items: center; gap: 6px; }
  .field label { color: var(--dim); white-space: nowrap; }
  input[type=search], select, button {
    background: #0e1015;
    color: var(--fg);
    border: 1px solid var(--line);
    border-radius: 5px;
    padding: 5px 8px;
    font: inherit;
    outline: none;
  }
  input[type=search] { width: 240px; }
  input[type=search]:focus, select:focus { border-color: #3d475c; }
  button { cursor: pointer; }
  button:hover { border-color: #4a5670; background: #141821; }
  input[type=range] { width: 110px; accent-color: var(--accent); }
  #status { color: var(--dim); margin-left: auto; white-space: nowrap; }
  #viewport { flex: 1; overflow-y: auto; overflow-x: hidden; position: relative; }
  #spacer { position: relative; width: 100%; }
  #layer { position: absolute; inset: 0; }
  .tile {
    position: absolute;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 3px;
    background-color: #1b1f28;
    background-image:
      linear-gradient(45deg, #21262f 25%, transparent 25%, transparent 75%, #21262f 75%),
      linear-gradient(45deg, #21262f 25%, transparent 25%, transparent 75%, #21262f 75%);
    background-size: 10px 10px;
    background-position: 0 0, 5px 5px;
    outline: 1px solid #0b0d11;
    outline-offset: -1px;
    cursor: pointer;
    contain: strict;
  }
  .tile:hover { outline: 2px solid var(--accent); outline-offset: -2px; z-index: 2; }
  .tile canvas {
    width: 100%; height: 100%;
    image-rendering: pixelated;
    image-rendering: crisp-edges;
    display: block;
  }
  #modal {
    position: fixed; inset: 0; background: rgba(6, 8, 12, .82);
    display: flex; align-items: center; justify-content: center;
    padding: 24px; z-index: 20; backdrop-filter: blur(3px);
  }
  #modal.hidden { display: none; }
  .card {
    background: var(--bg2); border: 1px solid var(--line); border-radius: 10px;
    max-width: 720px; width: 100%; display: flex; gap: 20px; padding: 20px;
    box-shadow: 0 24px 60px rgba(0,0,0,.55);
  }
  .preview {
    width: 320px; height: 320px; flex: none;
    border-radius: 6px; display: flex; align-items: center; justify-content: center;
    background-color: #1b1f28;
    background-image:
      linear-gradient(45deg, #21262f 25%, transparent 25%, transparent 75%, #21262f 75%),
      linear-gradient(45deg, #21262f 25%, transparent 25%, transparent 75%, #21262f 75%);
    background-size: 20px 20px; background-position: 0 0, 10px 10px;
  }
  .preview canvas { width: 288px; height: 288px; image-rendering: pixelated; }
  .meta { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 10px; }
  .meta .caption { font-size: 17px; line-height: 1.35; }
  .meta .k { color: var(--dim); }
  .meta .row { color: #cfd6e2; }
  .actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: auto; }
  .actions a, .actions button { text-decoration: none; }
  .actions a {
    background: #0e1015; color: var(--fg); border: 1px solid var(--line);
    border-radius: 5px; padding: 5px 8px;
  }
  .actions a:hover { border-color: #4a5670; }
  #close { position: absolute; top: 18px; right: 24px; }
  .badge {
    display: inline-block; padding: 1px 6px; border-radius: 999px;
    font-size: 11px; border: 1px solid var(--line); color: var(--dim);
  }
</style>
</head>
<body>
<header>
  <h1>PIXEL ART <span>ATLAS</span></h1>
  <div class="field">
    <label for="source">source</label>
    <select id="source" title="local disk or Hugging Face Hub"></select>
  </div>
  <div class="field">
    <input id="q" type="search" placeholder="search captions..." autocomplete="off" spellcheck="false">
  </div>
  <div class="field">
    <label for="subject">subject</label>
    <select id="subject"><option value="">any</option></select>
  </div>
  <div class="field">
    <label for="split">split</label>
    <select id="split">
      <option value="all">all</option>
      <option value="train">train</option>
      <option value="validation">validation</option>
    </select>
  </div>
  <div class="field">
    <label for="size">size</label>
    <input id="size" type="range" min="24" max="160" step="8" value="64">
  </div>
  <button id="random">random</button>
  <div id="status">connecting...</div>
</header>

<div id="viewport"><div id="spacer"><div id="layer"></div></div></div>

<div id="modal" class="hidden">
  <button id="close">esc</button>
  <div class="card" id="card">
    <div class="preview"><canvas id="big" width="16" height="16"></canvas></div>
    <div class="meta">
      <div class="caption" id="m-caption">...</div>
      <div class="row"><span class="k">subject</span> <span id="m-subject">...</span></div>
      <div class="row"><span class="k">row</span> <span id="m-row">...</span> <span class="badge" id="m-split">...</span></div>
      <div class="row"><span class="k">size</span> 16 x 16 RGBA</div>
      <div class="actions">
        <button id="m-copy">copy caption</button>
        <a id="m-png" href="#" target="_blank" rel="noopener">open PNG</a>
        <a id="m-download" href="#" download>download 8x PNG</a>
      </div>
    </div>
  </div>
</div>

<script>
(function () {
  "use strict";

  var el = function (id) { return document.getElementById(id); };
  var viewport = el("viewport"), spacer = el("spacer"), layer = el("layer");

  var meta = { total: 0, subjects: [], label: "", indexing: null, source: "" };
  var source = { kind: "range", total: 0, split: "all" };
  var cell = 64, gap = 6, pad = 12;
  var cols = 1, rows = 0;
  var tiles = new Map();       // position -> element
  var posToId = new Map();     // position -> global id
  var idToImage = new Map();   // global id -> ImageData
  var pendingChunks = new Set();
  var pendingIds = new Set();
  var visible = { first: 0, last: -1 };
  var current = null;
  var hoverTimer = null;
  var renderQueued = false;
  var querySeq = 0;
  var MAX_CACHE = 48000;

  function fmt(n) { return n.toLocaleString("en-US"); }
  function setStatus(text) { el("status").textContent = text; }
  // keeps the browser HTTP cache from mixing up ids between sources
  function srcParam() { return "&src=" + encodeURIComponent(meta.source || "unknown"); }

  // ---------- networking ----------
  function parseBatch(buf) {
    var dv = new DataView(buf);
    var n = dv.getUint32(0, true);
    var ids = new Uint32Array(buf, 4, n);
    var dataStart = 4 + n * 4;
    var out = [];
    for (var i = 0; i < n; i++) {
      out.push({ id: ids[i], bytes: new Uint8ClampedArray(buf, dataStart + i * 1024, 1024) });
    }
    return out;
  }

  function storeBatch(items) {
    for (var i = 0; i < items.length; i++) {
      var it = items[i];
      if (!idToImage.has(it.id)) idToImage.set(it.id, new ImageData(it.bytes, 16, 16));
    }
    trim();
  }

  function trim() {
    if (idToImage.size <= MAX_CACHE) return;
    idToImage.clear();
    posToId.clear();
    pendingChunks.clear();
    pendingIds.clear();
    tiles.forEach(function (t) { t.dataset.drawn = ""; });
  }

  function fetchRange(start, count) {
    var key = source.split + ":" + start + ":" + count;
    if (pendingChunks.has(key)) return;
    pendingChunks.add(key);
    var url = "/api/sprites?start=" + start + "&count=" + count + "&split=" + source.split + srcParam();
    fetch(url).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.arrayBuffer();
    }).then(function (buf) {
      var items = parseBatch(buf);
      storeBatch(items);
      for (var i = 0; i < items.length; i++) posToId.set(start + i, items[i].id);
      pendingChunks.delete(key);
    }).catch(function () {
      pendingChunks.delete(key);
    }).then(scheduleDraw);
  }

  function fetchIds(ids) {
    var need = ids.filter(function (id) { return !idToImage.has(id) && !pendingIds.has(id); });
    if (!need.length) return;
    for (var i = 0; i < need.length; i += 96) {
      (function (chunk) {
        chunk.forEach(function (id) { pendingIds.add(id); });
        fetch("/api/sprites?ids=" + chunk.join(",") + srcParam()).then(function (r) {
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.arrayBuffer();
        }).then(function (buf) {
          storeBatch(parseBatch(buf));
        }).catch(function () {
          chunk.forEach(function (id) { pendingIds.delete(id); });
        }).then(scheduleDraw);
      })(need.slice(i, i + 96));
    }
  }

  function queryApi(opts) {
    var params = new URLSearchParams();
    if (opts.q) params.set("q", opts.q);
    if (opts.subject) params.set("subject", opts.subject);
    if (opts.split) params.set("split", opts.split);
    params.set("limit", String(opts.limit || 20000));
    return fetch("/api/query?" + params.toString()).then(function (r) { return r.json(); });
  }

  // ---------- layout ----------
  function layout() {
    var available = viewport.clientWidth - pad * 2;
    cols = Math.max(1, Math.floor((available + gap) / (cell + gap)));
    rows = Math.max(1, Math.ceil(source.total / cols));
    spacer.style.height = (rows * (cell + gap) - gap + pad * 2) + "px";
    tiles.forEach(function (t) { layer.removeChild(t); });
    tiles.clear();
    visible.first = 0; visible.last = -1;
    render();
  }

  function at(pos) {
    var row = Math.floor(pos / cols), col = pos % cols;
    var used = Math.min(cols, source.total - row * cols);
    var left = pad + Math.floor((viewport.clientWidth - pad * 2 - (used * (cell + gap) - gap)) / 2);
    return { x: left + col * (cell + gap), y: pad + row * (cell + gap) };
  }

  function ensureData(p0, p1) {
    if (source.kind === "range") {
      var missing = false;
      for (var q = p0; q < p1; q++) {
        if (!posToId.has(q)) { missing = true; break; }
      }
      if (missing) {
        var cs = Math.floor(p0 / 256) * 256;
        var ce = Math.min(source.total, Math.ceil(p1 / 256) * 256);
        for (var s = cs; s < ce; s += 256) {
          var n = Math.min(256, source.total - s);
          var k = source.split + ":" + s + ":" + n;
          if (!pendingChunks.has(k)) fetchRange(s, n);
        }
      }
      return;
    }
    var ids = [];
    for (var r = p0; r < p1; r++) {
      var id = source.ids[r];
      if (id !== undefined) ids.push(id);
    }
    fetchIds(ids);
  }

  function render() {
    if (renderQueued) return;
    renderQueued = true;
    setTimeout(function () {
      renderQueued = false;
      var top = viewport.scrollTop;
      var rowH = cell + gap;
      var first = Math.max(0, Math.floor((top - pad) / rowH) - 1);
      var last = Math.min(rows - 1, Math.ceil((top + viewport.clientHeight - pad) / rowH) + 1);
      if (first === visible.first && last === visible.last) { drawVisible(); return; }
      visible.first = first; visible.last = last;
      var p0 = first * cols, p1 = Math.min(source.total, (last + 1) * cols);
      ensureData(p0, p1);

      var keep = new Set();
      for (var p = p0; p < p1; p++) {
        keep.add(p);
        var t = tiles.get(p);
        if (!t) {
          t = document.createElement("div");
          t.className = "tile";
          t.style.width = t.style.height = cell + "px";
          var c = document.createElement("canvas");
          c.width = 16; c.height = 16;
          t.appendChild(c);
          t.addEventListener("click", (function (pos) {
            return function () { openDetail(pos); };
          })(p));
          t.addEventListener("mouseenter", (function (pos) {
            return function () { peek(pos); };
          })(p));
          tiles.set(p, t);
          layer.appendChild(t);
        }
        var xy = at(p);
        t.style.left = xy.x + "px";
        t.style.top = xy.y + "px";
        draw(p, t);
      }
      tiles.forEach(function (t, pos) {
        if (!keep.has(pos)) { layer.removeChild(t); tiles.delete(pos); }
      });
      updateStatus();
    }, 0);
  }

  function drawVisible() {
    var p0 = visible.first * cols;
    var p1 = Math.min(source.total, (visible.last + 1) * cols);
    for (var p = p0; p < p1; p++) {
      var t = tiles.get(p);
      if (t) draw(p, t);
    }
  }

  function scheduleDraw() { setTimeout(drawVisible, 0); }

  function draw(pos, tile) {
    var id = source.kind === "range" ? posToId.get(pos) : source.ids[pos];
    if (id === undefined) return;
    var canvas = tile.firstChild;
    if (tile.dataset.drawn === String(id)) return;
    var img = idToImage.get(id);
    if (!img) { tile.dataset.drawn = ""; canvas.style.visibility = "hidden"; return; }
    canvas.getContext("2d").putImageData(img, 0, 0);
    canvas.style.visibility = "visible";
    tile.dataset.drawn = String(id);
    tile.dataset.id = String(id);
    tile.title = "";
  }

  function updateStatus() {
    var label = source.kind === "range" ? "sprites" : "matches";
    var prefix = source.truncated ? "at least " : "";
    var note = meta.indexing != null ? " · indexing captions " + Math.round(meta.indexing * 100) + "%" : "";
    setStatus(prefix + fmt(source.total) + " " + label + " · " + (meta.label || "local") + note + " · 16x16 RGBA");
  }

  // ---------- interaction ----------
  var peekCache = new Map();
  function peek(pos) {
    clearTimeout(hoverTimer);
    hoverTimer = setTimeout(function () {
      var id = source.kind === "range" ? posToId.get(pos) : source.ids[pos];
      if (id === undefined) return;
      var tile = tiles.get(pos);
      if (!tile) return;
      var apply = function (info) { if (tile.parentNode) tile.title = info.caption; };
      if (peekCache.has(id)) return apply(peekCache.get(id));
      fetch("/api/item?id=" + id).then(function (r) { return r.json(); }).then(function (info) {
        peekCache.set(id, info);
        apply(info);
      }).catch(function () {});
    }, 240);
  }

  function openDetail(pos) {
    var id = source.kind === "range" ? posToId.get(pos) : source.ids[pos];
    if (id === undefined) return;
    current = { id: id, pos: pos };
    el("modal").classList.remove("hidden");
    var img = idToImage.get(id);
    if (img) el("big").getContext("2d").putImageData(img, 0, 0);
    el("m-caption").textContent = "...";
    el("m-subject").textContent = "...";
    el("m-split").textContent = "...";
    el("m-row").textContent = fmt(id);
    el("m-png").href = "/api/png?id=" + id + "&scale=1" + srcParam();
    el("m-download").href = "/api/png?id=" + id + "&scale=8" + srcParam();
    el("m-download").setAttribute("download", String(id).padStart(7, "0") + ".png");
    fetch("/api/item?id=" + id).then(function (r) { return r.json(); }).then(function (info) {
      if (!current || current.id !== id) return;
      el("m-caption").textContent = info.caption;
      el("m-subject").textContent = info.subject;
      el("m-split").textContent = info.split;
      el("m-row").textContent = fmt(id);
    }).catch(function () {});
  }

  function closeDetail() {
    el("modal").classList.add("hidden");
    current = null;
  }

  function step(delta) {
    if (!current) return;
    var pos = current.pos + delta;
    if (pos < 0 || pos >= source.total) return;
    var tile = tiles.get(pos);
    if (tile) tile.scrollIntoView({ block: "nearest" });
    openDetail(pos);
  }

  function applySource(next) {
    source = next;
    posToId.clear();
    tiles.forEach(function (t) { layer.removeChild(t); });
    tiles.clear();
    viewport.scrollTop = 0;
    layout();
  }

  function resetCaches() {
    idToImage.clear();
    posToId.clear();
    peekCache.clear();
    pendingChunks.clear();
    pendingIds.clear();
    subjectKey = "";
  }

  function switchSource() {
    var sel = el("source");
    var kind = sel.value;
    if (!kind || kind === meta.source) return;
    var label = sel.options[sel.selectedIndex].textContent;
    sel.disabled = true;
    setStatus("loading " + label + " source...");
    var seq = ++querySeq;
    fetch("/api/source?kind=" + encodeURIComponent(kind), { method: "POST" })
      .then(function (r) {
        if (r.ok) return r.json();
        return r.text().then(function (t) { throw new Error(t || ("HTTP " + r.status)); });
      })
      .then(function (m) {
        if (seq !== querySeq) return;
        resetCaches();
        el("q").value = "";
        el("subject").value = "";
        el("split").value = "all";
        applyMeta(m);
        applySource({ kind: "range", total: m.total, split: "all", truncated: false });
        if (m.indexing != null) pollIndex();
      })
      .catch(function (e) {
        if (seq !== querySeq) return;
        sel.value = meta.source;
        setStatus("cannot switch source: " + e.message);
      })
      .then(function () { sel.disabled = false; });
  }

  // ---------- wiring ----------
  var qTimer = null;
  el("q").addEventListener("input", function () {
    clearTimeout(qTimer);
    qTimer = setTimeout(runQuery, 260);
  });
  el("source").addEventListener("change", switchSource);
  el("subject").addEventListener("change", runQuery);
  el("split").addEventListener("change", runQuery);
  el("size").addEventListener("input", function () {
    cell = parseInt(this.value, 10);
    layout();
  });
  el("random").addEventListener("click", function () {
    var pos = Math.floor(Math.random() * source.total);
    viewport.scrollTop = Math.floor(pos / cols) * (cell + gap);
    render();
    setTimeout(function () { openDetail(pos); }, 60);
  });
  el("close").addEventListener("click", closeDetail);
  el("modal").addEventListener("click", function (e) {
    if (e.target === el("modal")) closeDetail();
  });
  el("card").addEventListener("click", function (e) { e.stopPropagation(); });
  el("m-copy").addEventListener("click", function () {
    if (navigator.clipboard) navigator.clipboard.writeText(el("m-caption").textContent);
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") closeDetail();
    if (el("modal").classList.contains("hidden")) return;
    if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
    if (e.key === "ArrowRight") { e.preventDefault(); step(1); }
  });
  viewport.addEventListener("scroll", render, { passive: true });
  window.addEventListener("resize", function () { layout(); });

  var subjectKey = "";

  function fillSubjects(list) {
    if (!list) return;
    var key = list.length + "|" + (list[0] || "") + "|" + (list[list.length - 1] || "");
    if (key === subjectKey) return;
    subjectKey = key;
    var sel = el("subject");
    var keep = sel.value;
    while (sel.options.length > 1) sel.remove(1);
    var frag = document.createDocumentFragment();
    for (var i = 0; i < list.length; i++) {
      var o = document.createElement("option");
      o.value = o.textContent = list[i];
      frag.appendChild(o);
    }
    sel.appendChild(frag);
    sel.value = keep;
    if (sel.selectedIndex < 0) sel.value = "";
  }

  function fillSources(m) {
    var sel = el("source");
    if (!m.sources || sel.options.length) return;
    for (var i = 0; i < m.sources.length; i++) {
      var o = document.createElement("option");
      o.value = m.sources[i].kind;
      o.textContent = m.sources[i].label;
      sel.appendChild(o);
    }
  }

  function applyMeta(m) {
    meta = m;
    fillSources(m);
    fillSubjects(m.subjects);
    if (m.source) el("source").value = m.source;
    updateStatus();
  }

  function pollIndex() {
    setTimeout(function () { loadMeta(false); }, 1500);
  }

  function loadMeta(first) {
    fetch("/api/meta").then(function (r) { return r.json(); }).then(function (m) {
      applyMeta(m);
      if (first) {
        source = { kind: "range", total: m.total, split: "all", truncated: false };
        layout();
      }
      if (m.indexing != null) pollIndex();
    }).catch(function (e) {
      setStatus("cannot reach the viewer server: " + e.message);
    });
  }

  function runQuery() {
    var q = el("q").value.trim();
    var subject = el("subject").value;
    var split = el("split").value;
    var seq = ++querySeq;
    if (!q && !subject) {
      var total = meta.total;
      if (split === "train") total = meta.train;
      if (split === "validation") total = meta.validation;
      applySource({ kind: "range", total: total, split: split, truncated: false });
      return;
    }
    askQuery(seq, q, subject, split);
  }

  function askQuery(seq, q, subject, split) {
    setStatus("searching...");
    queryApi({ q: q, subject: subject, split: split, limit: 20000 }).then(function (res) {
      if (seq !== querySeq) return;
      if (res.indexing != null) {
        setStatus("building caption index for " + (meta.label || "this source") + "... " + Math.round(res.indexing * 100) + "%");
        setTimeout(function () { askQuery(seq, q, subject, split); }, 1500);
        return;
      }
      applySource({
        kind: "list", ids: res.ids, total: res.ids.length,
        truncated: res.total > res.ids.length, split: split
      });
    }).catch(function () {
      if (seq === querySeq) setStatus("search failed");
    });
  }

  loadMeta(true);
})();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True

    def _send(self, status, ctype, body, cache="no-store"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, "application/json; charset=utf-8", json.dumps(obj).encode("utf-8"))

    def do_GET(self):
        u = urlparse(self.path)
        qs = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", INDEX_HTML.encode("utf-8"))
                return

            if u.path == "/api/meta":
                self._json(meta_payload())
                return

            if u.path == "/api/sprites":
                if "ids" in qs:
                    ids = np.array([int(x) for x in qs["ids"].split(",") if x.strip()], dtype=np.uint32)[:1024]
                else:
                    start = max(0, int(qs.get("start", 0)))
                    count = min(max(1, int(qs.get("count", 128))), 1024)
                    ids = current().ids_for_range(start, count, qs.get("split", "all"))
                self._send(200, "application/octet-stream", batch_bytes(ids),
                           "public, max-age=31536000, immutable")
                return

            if u.path == "/api/query":
                q = qs.get("q", "").strip().lower()
                subject = qs.get("subject", "").strip()
                split = qs.get("split", "all")
                limit = min(max(1, int(qs.get("limit", 500))), 20000)
                if not q and not subject:
                    lo, hi = current()._range(split)
                    self._json({"mode": "range", "total": hi - lo})
                    return
                self._json(current().query(q, subject, split, limit))
                return

            if u.path == "/api/item":
                i = int(qs.get("id", 0))
                if not (0 <= i < current().total):
                    self._send(404, "text/plain", b"out of range")
                    return
                self._json(current().item(i))
                return

            if u.path == "/api/png":
                i = int(qs.get("id", 0))
                scale = min(max(1, int(qs.get("scale", 1))), 32)
                if not (0 <= i < current().total):
                    self._send(404, "text/plain", b"out of range")
                    return
                body = current().png(i, scale)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Disposition", 'inline; filename="%07d.png"' % i)
                self.send_header("Cache-Control", "public, max-age=31536000, immutable")
                self.end_headers()
                self.wfile.write(body)
                return

            self._send(404, "text/plain", b"not found")
        except Exception as exc:
            self._send(500, "text/plain", str(exc).encode("utf-8"))

    def do_POST(self):
        u = urlparse(self.path)
        qs = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/api/source":
                kind = qs.get("kind", "")
                if kind not in dict(KINDS):
                    self._send(400, "text/plain", b"unknown source")
                    return
                if kind != ACTIVE[0]:
                    get_source(kind)  # may raise; reported to the client as 500
                    ACTIVE[0] = kind
                    print("source switched to %s" % kind, flush=True)
                self._json(meta_payload())
                return
            self._send(404, "text/plain", b"not found")
        except Exception as exc:
            self._send(500, "text/plain", str(exc).encode("utf-8"))


def main():
    global DATA, REPO
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open the viewer in a browser")
    ap.add_argument("--source", choices=[k for k, _ in KINDS], default="remote",
                    help="source to start on: remote = Hugging Face Hub (default), local = --data folder")
    ap.add_argument("--repo", default=REPO, help="dataset repo on the Hugging Face Hub")
    ap.add_argument("--data", default=DATA, help="folder with local .parquet files")
    args = ap.parse_args()

    DATA = os.path.abspath(args.data)
    REPO = args.repo
    ACTIVE[0] = args.source
    if args.source == "local":
        print("loading local dataset from %s" % DATA, flush=True)
    else:
        print("connecting to https://huggingface.co/datasets/%s" % REPO, flush=True)
    SOURCE = current()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = "http://%s:%d/" % (args.host, args.port)
    print("Pixel Art Atlas viewer: %s" % url, flush=True)
    print("%s sprites (%s train / %s validation) - %s"
          % (f"{SOURCE.total:,}", f"{SOURCE.train_count:,}", f"{SOURCE.val_count:,}", SOURCE.label), flush=True)
    print("local dataset: %s" % DATA, flush=True)
    if args.open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
