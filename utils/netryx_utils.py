"""
netryx_utils.py

All non-GUI logic for Netryx Astra: configuration, device/model setup,
encoder abstraction, compact index build/load/search, panorama
download & stitching, geo utilities, and equirectangular projection math.

Nothing in this file imports tkinter or any GUI toolkit. main.py is
responsible for all UI and should import this module as:

    from utils import netryx_utils as nx

and reference mutable state as `nx.COMPACT_INDEX_DIR`, `nx.ACTIVE_ENCODER`,
etc. (rather than `from utils.netryx_utils import COMPACT_INDEX_DIR`),
since several functions here reassign these globals at runtime with the
`global` keyword and a star/name import would only capture a stale copy.
"""

import os
# [MPS FIX] Enable CPU fallback for operators not implemented on MPS (like aten::kthvalue used by DISK)
# MUST be set before importing torch
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import re
import math
import time
import json
import uuid
import glob
import gc
import queue
import random
import asyncio
import itertools
import threading
import concurrent.futures
from collections import defaultdict

import numpy as np
import torch
import aiohttp
from PIL import Image, ImageDraw

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import kornia.feature as KF
except ImportError:
    KF = None

from utils.megaloc_utils import (
    get_megaloc_model, extract_megaloc_descriptor,
    megaloc_similarity, batch_extract_megaloc,
    fit_pca, apply_pca, save_pca, load_pca,
    MEGALOC_RAW_DIM, MEGALOC_PCA_DIM
)

import torch._dynamo
torch._dynamo.config.suppress_errors = True

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS / PERFORMANCE TUNING
# ═══════════════════════════════════════════════════════════════════

# PCA matching dimensions
INDEX_TARGET_DIM = 1024

MAX_PANOID_WORKERS = 32
MAX_HEADING_WORKERS = 4
MAX_DOWNLOAD_WORKERS = 128
MAX_MATCH_WORKERS = 16
EARLY_EXIT_INLIER_THRESHOLD = 300
MEGALOC_BATCH_SIZE = 64  # descriptors are identical at any batch size; larger = better GPU utilization
CROP_QUEUE_SIZE = 1024

# Each panoid API call already searches a radius around the query point
# (the "2d50" param in _panoids_url below -- 50 meters) and returns every
# pano Google finds inside it, not just the one nearest the exact
# coordinate. Grid points closer together than this radius search
# overlapping circles and mostly rediscover the same panos, which is pure
# wasted API calls.
# PANOID_SEARCH_RADIUS_M must match the "2d{N}" value in _panoids_url below
# -- if you change one, change the other.
PANOID_SEARCH_RADIUS_M = 50
# Grid spacing = radius * this factor. <1.0 leaves deliberate overlap so
# thin strips of coverage (e.g. a road running between two grid points)
# don't get missed; 1.0 is the "just barely touching" tiling. Don't push
# above ~1.0 or genuine gaps start opening up between search circles.
GRID_SPACING_OVERLAP_FACTOR = 0.85

IMGX = 4
IMGY = 2

# ═══════════════════════════════════════════════════════════════════
# DEVICE AND MODEL SETUP
# ═══════════════════════════════════════════════════════════════════

device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

extractor_lock = threading.Lock()

try:
    from utils.mast3r_utils import get_mast3r_model, get_mast3r_matches
    MAST3R_AVAILABLE = True
except ImportError:
    MAST3R_AVAILABLE = False
    print("[MASt3R] mast3r_utils.py not found. MASt3R matching disabled.")

try:
    from utils.netryx_hub import NetryxHub, create_bundle, extract_bundle
    HUB_AVAILABLE = True
except ImportError:
    HUB_AVAILABLE = False
    print("[HUB] netryx_hub.py not found. Community sharing disabled.")

mast3r_model_instance = None
mast3r_lock = threading.Lock()


def get_lazy_mast3r():
    global mast3r_model_instance
    with mast3r_lock:
        if mast3r_model_instance is None:
            mast3r_model_instance = get_mast3r_model()
    return mast3r_model_instance


# ═══════════════════════════════════════════════════════════════════
# DATA DIRECTORIES
# ═══════════════════════════════════════════════════════════════════

# where we save all the data and stuff
# check if EXPANSION disk exists, otherwise use local folder
_potential_dir = "/Volumes/Expansion/netryx"
if os.path.exists(_potential_dir):
    DATA_DIR = _potential_dir
else:
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "netryx_data")

MEGALOC_PARTS_DIR = os.path.join(DATA_DIR, "megaloc_parts")
EMB_CSV = os.path.join(DATA_DIR, "embeddings_index.csv")
INDEXES_DIR = os.path.join(DATA_DIR, "indexes")
# COMPACT_INDEX_DIR and friends now point at whatever index load_index()/
# build_compact_index() last activated, under INDEXES_DIR/{uuid}/. There is
# no more single default "index" folder -- an index must be built or loaded
# before these paths are meaningful.
COMPACT_INDEX_DIR = None
COMPACT_DESCS_PATH = None
COMPACT_META_PATH = None
COMPACT_INFO_PATH = None
COMPACT_PCA_PATH = None


def has_active_index():
    """True if an index has been built or loaded this session. Guard any
    COMPACT_* path access with this -- they are None until load_index()
    or build_compact_index() runs."""
    return COMPACT_INDEX_DIR is not None


# ── Encoder abstraction (MegaLoc | MixVPR) ──────────────────────────────────
# The retrieval encoder is switchable. Each encoder has its OWN parts + index
# dir (descriptors from different models are not comparable), so switching never
# corrupts or mixes indexes. set_encoder() reassigns the path globals above so
# the rest of the code keeps working unchanged.
ACTIVE_ENCODER = "megaloc"
ENCODER_USES_PCA = True          # MegaLoc: 8448-dim -> PCA. MixVPR: already compact.
_BATCH_ENCODE = batch_extract_megaloc

os.makedirs(INDEXES_DIR, exist_ok=True)


def set_encoder(name):
    """Switch the active retrieval encoder for the NEXT indexing run.

    This only controls where raw part files get written (each encoder has
    its own parts dir, since descriptors from different models aren't
    comparable) and which encoder build_compact_index() will use. It does
    NOT point at an index anymore -- indexes are UUID-keyed under
    INDEXES_DIR and are selected with load_index(), not set_encoder().
    Any COMPACT_* globals get cleared here so stale paths from a previously
    loaded index can't silently leak into a new build.
    """
    global ACTIVE_ENCODER, ENCODER_USES_PCA, _BATCH_ENCODE
    global MEGALOC_PARTS_DIR, COMPACT_INDEX_DIR, COMPACT_DESCS_PATH
    global COMPACT_META_PATH, COMPACT_INFO_PATH, COMPACT_PCA_PATH, _compact_cache
    name = (name or "megaloc").lower()

    # No-op if this encoder is already active AND an index is currently
    # loaded. Without this, any redundant call (re-clicking the same radio
    # button, a caller re-asserting the encoder defensively, etc.) silently
    # unloads a perfectly good index for no reason -- the caller wanted to
    # confirm the encoder, not wipe the loaded index.
    if name == ACTIVE_ENCODER and has_active_index():
        return

    if name == "mixvpr":
        import utils.mixvpr_utils as _MX
        _MX.get_mixvpr_model  # ensure module import succeeds early
        MEGALOC_PARTS_DIR = os.path.join(DATA_DIR, "mixvpr_parts")
        ENCODER_USES_PCA = False
        _BATCH_ENCODE = _MX.batch_extract_mixvpr
        ACTIVE_ENCODER = "mixvpr"
    else:
        MEGALOC_PARTS_DIR = os.path.join(DATA_DIR, "megaloc_parts")
        ENCODER_USES_PCA = True
        _BATCH_ENCODE = batch_extract_megaloc
        ACTIVE_ENCODER = "megaloc"
    os.makedirs(MEGALOC_PARTS_DIR, exist_ok=True)
    os.makedirs(INDEXES_DIR, exist_ok=True)
    # Clear any previously loaded index's paths -- caller must build a new
    # index or load_index() an existing one before searching/building again.
    COMPACT_INDEX_DIR = None
    COMPACT_DESCS_PATH = None
    COMPACT_META_PATH = None
    COMPACT_INFO_PATH = None
    COMPACT_PCA_PATH = None
    _compact_cache = None
    print(f"[ENCODER] Active encoder: {ACTIVE_ENCODER} (parts: {MEGALOC_PARTS_DIR})")


def encode_query(pil_img):
    """Encode a query image to a search-ready descriptor for the active encoder."""
    if ACTIVE_ENCODER == "mixvpr":
        import utils.mixvpr_utils as _MX
        return _MX.extract_mixvpr_descriptor(pil_img)
    return extract_megaloc_descriptor(pil_img, apply_pca_reduction=True)


def batch_encode(pil_images, batch_size=None):
    """Batch-encode crops for indexing with the active encoder."""
    if batch_size is None:
        return _BATCH_ENCODE(pil_images)
    return _BATCH_ENCODE(pil_images, batch_size=batch_size)


# Create dirs on startup. COMPACT_INDEX_DIR is intentionally excluded --
# it's None until an index is built or loaded via load_index().
for d in [DATA_DIR, MEGALOC_PARTS_DIR, INDEXES_DIR]:
    os.makedirs(d, exist_ok=True)

_mps_cleanup_counter = 0
_mps_cleanup_lock = threading.Lock()


def aggressive_mps_cleanup(force=False):
    global _mps_cleanup_counter
    with _mps_cleanup_lock:
        _mps_cleanup_counter += 1
        should_clean = force or (_mps_cleanup_counter % 100 == 0)
    if not should_clean:
        return
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
    if force or (_mps_cleanup_counter % 50 == 0):
        import subprocess
        try:
            subprocess.run(
                ['find', '/private/var/folders', '-name', 'mpsgraph-*', '-type', 'f', '-mmin', '+1', '-delete'],
                capture_output=True, timeout=10
            )
        except Exception:
            pass


def pil_to_tensor(im):
    return torch.from_numpy(np.array(im.convert('RGB'))).float().permute(2, 0, 1).unsqueeze(0).div(255.0).to(device)


def tensor_to_pil(t):
    t = t.squeeze(0).cpu().clamp(0, 1).mul(255).add_(0.5).to(torch.uint8).permute(1, 2, 0).numpy()
    if t.shape[2] == 1:
        t = t.squeeze(2)
    return Image.fromarray(t)


def scan_indexes():
    indexes = []
    indexes_dir = os.path.join(DATA_DIR, "indexes")

    if not os.path.exists(indexes_dir):
        return indexes

    for index_id in os.listdir(indexes_dir):
        index_path = os.path.join(indexes_dir, index_id)
        manifest_path = os.path.join(index_path, "manifest.json")

        if not os.path.isfile(manifest_path):
            continue

        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            manifest["path"] = index_path
            manifest["index_id"] = index_id
            if "coverage_center" not in manifest:
                manifest["coverage_center"] = {
                    "lat": manifest.get("center_lat"),
                    "lon": manifest.get("center_lon"),
                }

            indexes.append(manifest)

        except Exception as e:
            print(f"[INDEX] Failed loading {index_id}: {e}")

    return indexes


def load_index(index_id):
    global COMPACT_INDEX_DIR
    global COMPACT_DESCS_PATH
    global COMPACT_META_PATH
    global COMPACT_INFO_PATH
    global COMPACT_PCA_PATH
    global ACTIVE_ENCODER
    global ENCODER_USES_PCA
    global _compact_cache

    index_path = os.path.join(INDEXES_DIR, index_id)

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index not found: {index_id}")

    manifest_path = os.path.join(index_path, "manifest.json")

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"manifest.json missing for index: {index_id}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Normalize coverage fields -- Hub-built bundles use flat center_lat/
    # center_lon/radius_km, while write_index_manifest() (used for indexes
    # built locally in this app) nests them under coverage_center. Callers
    # should always read manifest["coverage_center"]["lat"/"lon"] and
    # manifest["radius_km"] after this.
    if "coverage_center" not in manifest:
        manifest["coverage_center"] = {
            "lat": manifest.get("center_lat"),
            "lon": manifest.get("center_lon"),
        }

    encoder = manifest.get("descriptor_model", "MegaLoc").lower()

    # Don't blindly trust descriptor_model -- verify against what's actually
    # on disk. A manifest can end up self-contradictory (e.g. built while
    # ACTIVE_ENCODER said one thing but the real extraction ran as another),
    # and trusting the label alone means load_index() looks for a filename
    # that was never written, silently behaving as if no index exists.
    mixvpr_path = os.path.join(index_path, "mixvpr_descriptors.npy")
    megaloc_path = os.path.join(index_path, "megaloc_descriptors.npy")
    mixvpr_exists = os.path.isfile(mixvpr_path)
    megaloc_exists = os.path.isfile(megaloc_path)

    if encoder == "mixvpr" and not mixvpr_exists and megaloc_exists:
        print(f"[INDEX] WARNING: manifest for {index_id} claims 'mixvpr' but "
              f"only megaloc_descriptors.npy exists on disk. Using megaloc "
              f"instead (the manifest is stale/wrong, not the data).")
        encoder = "megaloc"
    elif encoder != "mixvpr" and not megaloc_exists and mixvpr_exists:
        print(f"[INDEX] WARNING: manifest for {index_id} claims '{encoder}' but "
              f"only mixvpr_descriptors.npy exists on disk. Using mixvpr "
              f"instead (the manifest is stale/wrong, not the data).")
        encoder = "mixvpr"

    if encoder == "mixvpr":
        desc_name = "mixvpr_descriptors.npy"
        ENCODER_USES_PCA = False
    else:
        desc_name = "megaloc_descriptors.npy"
        ENCODER_USES_PCA = True

    desc_path_check = os.path.join(index_path, desc_name)
    if not os.path.isfile(desc_path_check):
        raise FileNotFoundError(
            f"Index {index_id}: expected descriptor file not found at "
            f"{desc_path_check}, and no alternate-encoder descriptor file "
            f"was found either. This index looks incomplete or corrupted."
        )

    ACTIVE_ENCODER = encoder
    COMPACT_INDEX_DIR = index_path
    COMPACT_DESCS_PATH = os.path.join(index_path, desc_name)
    COMPACT_META_PATH = os.path.join(index_path, "metadata.npz")
    COMPACT_INFO_PATH = os.path.join(index_path, "index_info.txt")
    COMPACT_PCA_PATH = os.path.join(index_path, "megaloc_pca.pkl")

    _compact_cache = None

    print(
        f"[INDEX] Loaded {manifest.get('name', index_id)} "
        f"({encoder})"
    )

    return manifest


def write_index_manifest(index_dir, index_id, *, name=None, encoder="megaloc",
                          descriptor_dim=None, num_entries=None,
                          center_lat=None, center_lon=None, radius_km=None,
                          format_version=1):
    """Write manifest.json for an index dir. Called right after an index's
    data files are saved so scan_indexes()/load_index() can discover it."""
    manifest = {
        "index_id": index_id,
        "name": name or index_id,
        "descriptor_model": encoder,
        "descriptor_dim": descriptor_dim,
        "num_entries": num_entries,
        "coverage_center": {"lat": center_lat, "lon": center_lon},
        "radius_km": radius_km,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "format_version": format_version,
    }
    manifest_path = os.path.join(index_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def draw_matches(img1, img2, kp1, kp2, matches=None, color=(0, 255, 0)):
    w1, h1 = img1.size
    w2, h2 = img2.size
    new_h = max(h1, h2)
    result = Image.new("RGB", (w1 + w2, new_h), (255, 255, 255))
    result.paste(img1, (0, 0))
    result.paste(img2, (w1, 0))
    draw = ImageDraw.Draw(result)
    if matches is None:
        return result
    if isinstance(matches, np.ndarray) and matches.ndim == 2 and matches.shape[1] == 2:
        for i in range(len(matches)):
            idx0, idx1 = matches[i]
            p1, p2 = kp1[idx0], kp2[idx1]
            draw.line(((p1[0], p1[1]), (p2[0] + w1, p2[1])), fill=color, width=1)
    elif isinstance(matches, np.ndarray) and matches.ndim == 1:
        for idx, m in enumerate(matches):
            if m > -1:
                x1, y1 = kp1[idx]
                x2, y2 = kp2[m]
                draw.line(((x1, y1), (x2 + w1, y2)), fill=color, width=1)
    return result


# ═══════════════════════════════════════════════════════════════════
# PANORAMA DOWNLOAD & STITCHING
# ═══════════════════════════════════════════════════════════════════

def _panoids_url(lat, lon):
    url = "https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{0:}!4d{1:}!2d50!3m10!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2&callback=_xdc_._v2mub5"
    return url.format(lat, lon)


def panoids_from_response(text):
    matches = re.findall(r'"([A-Za-z0-9_-]{22})"', text)
    out = []
    for panoid in matches:
        latlon = re.findall(r'"' + panoid + r'".+?\[null,null,(-?\d+\.\d+),(-?\d+\.\d+)', text)
        if latlon:
            lat, lon = map(float, latlon[0])
        else:
            lat, lon = None, None
        out.append({"panoid": panoid, "lat": lat, "lon": lon})
    filtered = []
    seen = set()
    for p in out:
        if p['panoid'] not in seen:
            seen.add(p['panoid'])
            filtered.append(p)
    return filtered


def tiles_info(panoid):
    # cbk0.google.com now returns 403 for all requests; use the current tile endpoint
    image_url = "https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid={0:}&x={1:}&y={2:}&zoom=2&nbt=1&fover=2"
    coord = list(itertools.product(range(IMGX), range(IMGY)))
    tiles = [(x, y, "%s_%dx%d.jpg" % (panoid, x, y), image_url.format(panoid, x, y)) for x, y in coord]
    return tiles

_tile_429_count = 0
_tile_200_count = 0
_tile_other_count = 0
_tile_exception_count = 0
_tile_debug_lock = threading.Lock()


async def download_tile_aiohttp(session, x, y, fname, url, max_attempts=5, semaphore=None):
    global _tile_429_count, _tile_200_count, _tile_other_count, _tile_exception_count
    attempt = 0
    rate_limit_retries = 0
    while attempt < max_attempts:
        try:
            async with semaphore:
                async with session.get(url.replace("http://", "https://"), timeout=aiohttp.ClientTimeout(total=20, sock_connect=10)) as response:
                    if response.status == 200:
                        data = await response.read()
                        with _tile_debug_lock:
                            _tile_200_count += 1
                        return x, y, data
                    elif response.status == 429:
                        with _tile_debug_lock:
                            _tile_429_count += 1
                        if rate_limit_retries < 5:
                            rate_limit_retries += 1
                            await asyncio.sleep(min(2 ** rate_limit_retries, 10))
                            continue
                        return x, y, None
                    else:
                        with _tile_debug_lock:
                            _tile_other_count += 1
                        attempt += 1
                        await asyncio.sleep(0.5)
        except Exception as e:
            with _tile_debug_lock:
                _tile_exception_count += 1
            attempt += 1
            await asyncio.sleep(0.5)
    return x, y, None


def reset_tile_debug_counters():
    global _tile_429_count, _tile_200_count, _tile_other_count, _tile_exception_count
    with _tile_debug_lock:
        _tile_429_count = 0
        _tile_200_count = 0
        _tile_other_count = 0
        _tile_exception_count = 0


def get_tile_debug_counters():
    with _tile_debug_lock:
        return {
            "200": _tile_200_count,
            "429": _tile_429_count,
            "other": _tile_other_count,
            "exception": _tile_exception_count,
        }


def download_tiles(tiles, status_callback=None, max_workers=64):
    total = len(tiles)
    results = {}

    async def main():
        connector = aiohttp.TCPConnector(limit=max_workers)
        # Google endpoints 403 without browser-like headers
        _headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
            "Referer": "https://www.google.com/maps/",
        }
        async with aiohttp.ClientSession(connector=connector, headers=_headers) as session:
            tasks = []
            for i, (x, y, fname, url) in enumerate(tiles):
                tasks.append(download_tile_aiohttp(session, x, y, fname, url))
            for idx, coro in enumerate(asyncio.as_completed(tasks), 1):
                x, y, data = await coro
                if data:
                    results[(x, y)] = data
                if status_callback:
                    status_callback(idx, total)

    asyncio.run(main())
    return results


async def _download_tiles_multi(panoid_tile_lists, max_workers=120, status_callback=None):
    """Download tiles for MANY panoramas concurrently in ONE shared event loop
    and ONE shared connection pool.

    IMPORTANT: a semaphore gates task concurrency, not just the
    TCPConnector's connection limit. Previously all coroutines for every
    tile were created and started at once via asyncio.gather(), so with
    (e.g.) 4024 tiles and max_workers=32, ~3992 of them sat waiting for a
    free connection slot while their individual request timeout was
    already ticking -- causing them to fail in a synchronized mass wave
    once the queue backed up past the timeout window, instead of failing
    gradually or not at all. The semaphore ensures only max_workers
    coroutines are ever actively inside a request (queued-for-connection
    time included) at once, so the rest simply haven't started their
    timeout clock yet.

    panoid_tile_lists: dict {panoid_id: [(x, y, fname, url), ...]}
    Returns: dict {panoid_id: {(x, y): bytes}}
    """
    reset_tile_debug_counters()
    results = {pid: {} for pid in panoid_tile_lists}
    connector = aiohttp.TCPConnector(limit=max_workers)
    semaphore = asyncio.Semaphore(max_workers)
    _headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
        "Referer": "https://www.google.com/maps/",
    }
    total_tiles = sum(len(t) for t in panoid_tile_lists.values())
    done = 0
    t_start = time.time()
    last_log_t = [t_start]
    last_log_done = [0]

    print(f"[TILE-DEBUG] Starting download of {total_tiles} tiles across "
          f"{len(panoid_tile_lists)} panoids, max_workers={max_workers}")

    async with aiohttp.ClientSession(connector=connector, headers=_headers) as session:
        async def fetch_one(pid, x, y, fname, url):
            nonlocal done
            xr, yr, data = await download_tile_aiohttp(session, x, y, fname, url, semaphore=semaphore)
            if data:
                results[pid][(xr, yr)] = data
            done += 1

            now = time.time()
            if now - last_log_t[0] >= 2.0 or (done - last_log_done[0]) >= 1000:
                elapsed = now - t_start
                interval_done = done - last_log_done[0]
                interval_t = now - last_log_t[0]
                rate = interval_done / interval_t if interval_t > 0 else 0
                counters = get_tile_debug_counters()
                print(f"[TILE-DEBUG] {done}/{total_tiles} ({elapsed:.1f}s elapsed) | "
                      f"rate={rate:.1f} tiles/s | "
                      f"200={counters['200']} 429={counters['429']} "
                      f"other={counters['other']} exc={counters['exception']}")
                last_log_t[0] = now
                last_log_done[0] = done

            if status_callback:
                status_callback(done, total_tiles)

        tasks = [
            fetch_one(pid, x, y, fname, url)
            for pid, tiles in panoid_tile_lists.items()
            for (x, y, fname, url) in tiles
        ]
        await asyncio.gather(*tasks)

    final_counters = get_tile_debug_counters()
    print(f"[TILE-DEBUG] FINISHED: {done}/{total_tiles} in {time.time()-t_start:.1f}s | "
          f"200={final_counters['200']} 429={final_counters['429']} "
          f"other={final_counters['other']} exc={final_counters['exception']} | "
          f"successful_panoids={sum(1 for v in results.values() if v)}/{len(results)}")

    return results

def download_tiles_for_panoids(panoid_ids, max_workers=120, status_callback=None):
    reset_tile_debug_counters()
    panoid_tile_lists = {pid: tiles_info(pid) for pid in panoid_ids}
    return asyncio.run(_download_tiles_multi(panoid_tile_lists, max_workers=max_workers,
                                              status_callback=status_callback))


def stitch_tiles(tiles_data):
    tile_w, tile_h = 512, 512
    import io
    pano_np = np.zeros((IMGY * tile_h, IMGX * tile_w, 3), dtype=np.uint8)
    for (x, y), data in tiles_data.items():
        try:
            tile = Image.open(io.BytesIO(data))
            tile_np = np.array(tile)
            th, tw, _ = tile_np.shape
            pano_np[y*tile_h:y*tile_h+th, x*tile_w:x*tile_w+tw] = tile_np
            tile.close()
        except Exception:
            continue
    return Image.fromarray(pano_np)


# ═══════════════════════════════════════════════════════════════════
# GEO UTILITIES
# ═══════════════════════════════════════════════════════════════════

def haversine(p1, p2):
    R = 6371
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def haversine_vec(center_lat, center_lon, lats, lons):
    """Vectorized haversine: distance in km from one (lat, lon) point to
    every point in the lats/lons numpy arrays. Same formula as haversine()
    above, just array-based instead of scalar, for computing something like
    the max distance from a center to thousands of indexed points without
    a Python-level loop."""
    R = 6371
    lat1 = math.radians(center_lat)
    lon1 = math.radians(center_lon)
    lat2 = np.radians(lats)
    lon2 = np.radians(lons)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + math.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def grid_points(center, radius, spacing_m):
    """Build a grid of coordinates to probe for Street View panoramas.

    spacing_m: target distance in meters between adjacent grid points
    (this is what the GUI's "Grid Resolution" field actually documents
    itself as -- previously the code silently treated this number as a
    point-count instead, so a 300 they typed meaning "300m apart" became
    a 301x301 grid instead).

    Each panoid API call already searches PANOID_SEARCH_RADIUS_M around its
    query point, so spacing tighter than that just re-searches overlapping
    areas and rediscovers the same panos -- pure wasted API calls. spacing
    is floored here at PANOID_SEARCH_RADIUS_M * GRID_SPACING_OVERLAP_FACTOR
    for that reason; going below it buys duplicate coverage, not real
    coverage.
    """
    lat, lon = center
    min_spacing_m = PANOID_SEARCH_RADIUS_M * GRID_SPACING_OVERLAP_FACTOR
    if spacing_m < min_spacing_m:
        print(f"[GRID] Requested spacing {spacing_m}m is tighter than "
              f"{PANOID_SEARCH_RADIUS_M}m search radius can usefully cover "
              f"-- flooring to {min_spacing_m:.0f}m (tighter would just "
              f"re-search overlapping circles).")
        spacing_m = min_spacing_m

    top_left = (lat - radius / 70, lon + radius / 70)
    bottom_right = (lat + radius / 70, lon - radius / 70)
    lat_diff = top_left[0] - bottom_right[0]
    lon_diff = top_left[1] - bottom_right[1]

    # Convert the requested meter spacing into a point count across the
    # bounding box (diameter = 2 * radius, in km -> meters).
    diameter_m = radius * 2 * 1000
    resolution = max(1, round(diameter_m / spacing_m))

    test_points = list(itertools.product(range(resolution + 1), range(resolution + 1)))
    test_points = [
        (bottom_right[0] + x * lat_diff / resolution, bottom_right[1] + y * lon_diff / resolution)
        for (x, y) in test_points
    ]
    test_points = [p for p in test_points if haversine(p, center) <= radius]
    return test_points


def get_panoids(points, status_callback=None, max_workers=64):
    async def fetch_one(session, idx, lat, lon, max_attempts=3):
        # 3 attempts + a short timeout: repeated failures are almost always
        # genuinely empty spots (water/no coverage), so extra retries and a long
        # timeout just add a slow tail with no extra panos found.
        url = _panoids_url(lat, lon)
        attempt = 0
        rate_limit_retries = 0  # 429s get their own budget so real points aren't dropped
        while attempt < max_attempts:
            try:
                async with session.get(url, timeout=15) as resp:
                    status = resp.status
                    text = await resp.text()
                    if status == 429:
                        # Transient throttling — back off and retry without
                        # consuming the fast-fail budget (up to a cap).
                        if rate_limit_retries < 5:
                            rate_limit_retries += 1
                            await asyncio.sleep(min(2 ** rate_limit_retries, 10))
                            continue
                        return []
                    elif status != 200:
                        attempt += 1
                        continue
                    pans = panoids_from_response(text)
                    if not pans:
                        return []
                    return pans
            except asyncio.TimeoutError:
                attempt += 1
                await asyncio.sleep(0.5)
            except Exception:
                attempt += 1
                await asyncio.sleep(0.5)
        return []

    async def main():
        connector = aiohttp.TCPConnector(limit=max_workers)
        # Google endpoints 403 without browser-like headers
        _headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
            "Referer": "https://www.google.com/maps/",
        }
        async with aiohttp.ClientSession(connector=connector, headers=_headers) as session:
            tasks = []
            for idx, (lat, lon) in enumerate(points):
                task = asyncio.create_task(fetch_one(session, idx, lat, lon))
                tasks.append(task)
            results = []
            for idx, task in enumerate(asyncio.as_completed(tasks), 1):
                pans = await task
                results.extend(pans)
                if status_callback:
                    status_callback(idx, len(points))
            return results

    panoids_raw = asyncio.run(main())
    already = set()
    filtered = []
    for pan in panoids_raw:
        if pan['panoid'] not in already:
            already.add(pan['panoid'])
            filtered.append(pan)
    print(f"[SUMMARY] Fetched {len(points)} grid points, found {len(filtered)} unique panoids.")
    return filtered


def generate_circle_points(center_lat, center_lon, radius_km, num_points=36):
    points = []
    R = 6371.0
    lat_rad = math.radians(center_lat)
    lon_rad = math.radians(center_lon)
    angular_dist = radius_km / R
    for i in range(num_points):
        bearing = math.radians(i * (360 / num_points))
        new_lat = math.asin(math.sin(lat_rad) * math.cos(angular_dist) +
                            math.cos(lat_rad) * math.sin(angular_dist) * math.cos(bearing))
        new_lon = lon_rad + math.atan2(math.sin(bearing) * math.sin(angular_dist) * math.cos(lat_rad),
                                       math.cos(angular_dist) - math.sin(lat_rad) * math.sin(new_lat))
        points.append((math.degrees(new_lat), math.degrees(new_lon)))
    return points


# ═══════════════════════════════════════════════════════════════════
# EQUIRECTANGULAR PROJECTION
# ═══════════════════════════════════════════════════════════════════

def get_projection_base_dirs(fov_deg, out_hw):
    fov = math.radians(fov_deg)
    out_h, out_w = out_hw
    cx, cy = out_w / 2.0, out_h / 2.0
    fx = fy = (out_w / 2.0) / math.tan(fov / 2.0)
    xx, yy = torch.meshgrid(
        torch.arange(out_w, device=device, dtype=torch.float32),
        torch.arange(out_h, device=device, dtype=torch.float32),
        indexing='xy'
    )
    x = (xx - cx) / fx
    y = (yy - cy) / fy
    z = torch.ones_like(x)
    dirs = torch.stack([x, -y, z], dim=-1)
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
    return dirs.reshape(-1, 3).T


def equirectangular_to_rectilinear_torch(pano_tensor, fov_deg=90, out_hw=(400, 400), yaw_deg=0, pitch_deg=0, base_dirs=None):
    _, _, h, w = pano_tensor.shape
    out_h, out_w = out_hw
    if isinstance(yaw_deg, (float, int)):
        yaws = torch.tensor([yaw_deg], device=device, dtype=torch.float32)
    elif isinstance(yaw_deg, list):
        yaws = torch.tensor(yaw_deg, device=device, dtype=torch.float32)
    else:
        yaws = yaw_deg.to(device).float()
    B = len(yaws)
    yaws_rad = torch.deg2rad(yaws)
    cos_vals = torch.cos(yaws_rad)
    sin_vals = torch.sin(yaws_rad)
    zeros = torch.zeros_like(cos_vals)
    ones = torch.ones_like(cos_vals)
    row1 = torch.stack([cos_vals, zeros, sin_vals], dim=1)
    row2 = torch.stack([zeros, ones, zeros], dim=1)
    row3 = torch.stack([-sin_vals, zeros, cos_vals], dim=1)
    R = torch.stack([row1, row2, row3], dim=1)
    if base_dirs is None:
        base_dirs = get_projection_base_dirs(fov_deg, out_hw)
    dirs = torch.matmul(R, base_dirs.unsqueeze(0))
    dirs = dirs.permute(0, 2, 1)
    x = dirs[:, :, 0]
    y = dirs[:, :, 1]
    z = dirs[:, :, 2]
    lon = torch.atan2(x, z)
    lat = torch.asin(y.clamp(-1+1e-7, 1-1e-7))
    grid_x = lon / math.pi
    grid_y = -lat / (math.pi / 2.0)
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(B, out_h, out_w, 2)
    pano_batch = pano_tensor.expand(B, -1, -1, -1)
    out = torch.nn.functional.grid_sample(pano_batch, grid, mode='bilinear', align_corners=True)
    return out


def equirectangular_to_rectilinear(pano_img, fov_deg=90, out_hw=(400, 400), yaw_deg=0, pitch_deg=0):
    pano_tensor = pil_to_tensor(pano_img)
    out_tensor = equirectangular_to_rectilinear_torch(pano_tensor, fov_deg, out_hw, yaw_deg, pitch_deg)
    return tensor_to_pil(out_tensor)


# ═══════════════════════════════════════════════════════════════════
# COMPACT INDEX — BUILD, LOAD, SEARCH
# (Merged from compact_index.py)
# ═══════════════════════════════════════════════════════════════════

_compact_cache = None


def parse_emb_path(emb_path):
    """Extract panoid and heading from path like '/path/to/PANOID_HEADING.npz'."""
    filename = os.path.basename(emb_path)
    name = filename.replace('.npz', '')
    parts = name.rsplit('_', 1)
    if len(parts) == 2:
        try:
            return parts[0], int(parts[1])
        except ValueError:
            pass
    return None, None


def build_compact_index():
    """Build compact index from part files + CSV coordinates.

    Auto-applies PCA if descriptors are high-dimensional (e.g., 8448 from MegaLoc).
    """
    global COMPACT_INDEX_DIR
    global COMPACT_DESCS_PATH
    global COMPACT_META_PATH
    global COMPACT_INFO_PATH
    global COMPACT_PCA_PATH
    global _compact_cache

    # Capture the encoder NOW, at the start of the build, and use this
    # captured value everywhere below -- including the final manifest write.
    # ACTIVE_ENCODER is a live global; if anything calls set_encoder() while
    # this build is running (a build can take a while), reading
    # ACTIVE_ENCODER again at the end would write a manifest describing a
    # DIFFERENT encoder than the one whose part files actually got indexed.
    # That exact mismatch previously produced an index whose manifest said
    # "mixvpr" while the descriptor file on disk was really MegaLoc's.
    build_encoder = ACTIVE_ENCODER

    index_id = str(uuid.uuid4())
    index_dir = os.path.join(INDEXES_DIR, index_id)
    os.makedirs(index_dir, exist_ok=True)

    megaloc_pattern = os.path.join(MEGALOC_PARTS_DIR, "megaloc_part_*.npz")
    part_files = sorted(glob.glob(megaloc_pattern))
    part_files = sorted(set(part_files))

    if not part_files:
        print(f"[INDEX] ERROR: No part files found")
        return False

    print(f"[INDEX] Found {len(part_files)} part files")

    # ── Pass 1: Count total entries and detect descriptor dimension ──
    total = 0
    raw_dim = None
    for pf in part_files:
        data = np.load(pf, allow_pickle=True)
        total += len(data['paths'])
        if raw_dim is None:
            raw_dim = data['descriptors'].shape[1]
        del data

    print(f"[INDEX] Total entries: {total}, raw descriptor dim: {raw_dim}")

    # ── Decide if PCA is needed ──
    needs_pca = raw_dim > INDEX_TARGET_DIM
    final_dim = INDEX_TARGET_DIM if needs_pca else raw_dim

    if needs_pca:
        print(f"[INDEX] Will apply PCA: {raw_dim} -> {final_dim}")

        # Fit PCA on a subsample (avoids 63GB RAM spike for 2M×8448)
        MAX_PCA_SAMPLES = 100_000
        COMPACT_PCA_PATH = os.path.join(index_dir, "megaloc_pca.pkl")
        pca_path = COMPACT_PCA_PATH

        # Collect subsample for PCA fitting
        print(f"[INDEX] Collecting subsample for PCA fitting (max {MAX_PCA_SAMPLES})...")
        pca_samples = []
        pca_count = 0
        for pf in part_files:
            if pca_count >= MAX_PCA_SAMPLES:
                break
            data = np.load(pf, allow_pickle=True)
            descs = data['descriptors']
            remaining = MAX_PCA_SAMPLES - pca_count
            pca_samples.append(descs[:remaining])
            pca_count += len(descs[:remaining])
            del data

        pca_matrix = np.vstack(pca_samples)
        del pca_samples
        print(f"[INDEX] Fitting PCA on {pca_matrix.shape[0]} samples...")

        from sklearn.decomposition import PCA
        pca = PCA(n_components=final_dim, whiten=True)
        pca.fit(pca_matrix)
        explained = pca.explained_variance_ratio_.sum()
        print(f"[INDEX] PCA fitted. Explained variance: {explained*100:.1f}%")
        del pca_matrix

        # Save PCA model for query-time use
        import pickle
        with open(pca_path, 'wb') as f:
            pickle.dump(pca, f)
        print(f"[INDEX] Saved PCA model to {pca_path}")

        # Also load it into megaloc_utils global
        try:
            from utils.megaloc_utils import load_pca as _load_pca
            _load_pca(pca_path)
        except Exception:
            pass
    else:
        pca = None
        print(f"[INDEX] Descriptors already {raw_dim}-dim, no PCA needed")

    # ── Pass 2: Load descriptors and metadata ──
    print(f"[INDEX] Loading and merging {len(part_files)} files...")
    all_descs = np.zeros((total, final_dim), dtype=np.float32)
    all_paths = []
    all_embedded_lats = []
    all_embedded_lons = []

    idx = 0
    t0 = time.time()
    for i, pf in enumerate(part_files):
        data = np.load(pf, allow_pickle=True)
        n = len(data['paths'])

        descs = data['descriptors']

        # Apply PCA if needed
        if needs_pca and descs.shape[1] > final_dim:
            descs = pca.transform(descs).astype(np.float32)
            # L2-normalize after PCA
            norms = np.linalg.norm(descs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            descs = descs / norms
        elif descs.shape[1] != final_dim:
            print(f"[INDEX] WARNING: Skipping {pf} — dim {descs.shape[1]} != expected {final_dim}")
            del data
            continue

        all_descs[idx:idx+n] = descs
        all_paths.extend(data['paths'].tolist())

        if 'lats' in data and 'lons' in data:
            all_embedded_lats.extend(data['lats'].tolist())
            all_embedded_lons.extend(data['lons'].tolist())
        else:
            all_embedded_lats.extend([0.0] * n)
            all_embedded_lons.extend([0.0] * n)

        idx += n
        del data, descs
        if (i+1) % 100 == 0:
            print(f"  Loaded {i+1}/{len(part_files)} ({idx} entries) [{time.time()-t0:.0f}s]")

    # Trim if we skipped any files
    if idx < total:
        all_descs = all_descs[:idx]
        print(f"[INDEX] Trimmed to {idx} entries (skipped some incompatible files)")
        total = idx

    print(f"[INDEX] Loaded all {idx} entries in {time.time()-t0:.1f}s")

    # ── Load lat/lon from CSV ──
    print(f"[INDEX] Loading coordinates from {EMB_CSV}...")
    csv_locations = {}
    csv_full_locations = {}
    if os.path.exists(EMB_CSV):
        with open(EMB_CSV, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    try:
                        lat, lon = float(parts[1]), float(parts[2])
                        csv_full_locations[parts[0]] = (lat, lon)
                        csv_locations[os.path.basename(parts[0])] = (lat, lon)
                    except ValueError:
                        pass
    print(f"[INDEX] CSV has {len(csv_locations)} location entries")

    # ── Match paths to coordinates ──
    lats = np.zeros(idx, dtype=np.float32)
    lons = np.zeros(idx, dtype=np.float32)
    headings = np.zeros(idx, dtype=np.int16)
    panoids = []
    valid_mask = np.zeros(idx, dtype=bool)
    matched = 0

    for i, path in enumerate(all_paths):
        filename = os.path.basename(path)
        name = filename.replace('.npz', '')
        parts_split = name.rsplit('_', 1)
        panoid = parts_split[0] if len(parts_split) == 2 else None
        try:
            heading = int(parts_split[1]) if len(parts_split) == 2 else 0
        except ValueError:
            heading = 0

        panoids.append(panoid or "")
        headings[i] = heading

        emb_lat = all_embedded_lats[i]
        emb_lon = all_embedded_lons[i]

        if emb_lat != 0 or emb_lon != 0:
            lats[i], lons[i] = emb_lat, emb_lon
            valid_mask[i] = True
            matched += 1
        else:
            loc = csv_full_locations.get(path) or csv_locations.get(filename)
            if loc:
                lats[i], lons[i] = loc
                valid_mask[i] = True
                matched += 1
        if (i + 1) % 200000 == 0:
            print(f"  Matching {i+1}/{idx}... ({matched} matched)")

    print(f"[INDEX] Matched {matched}/{idx} paths to coordinates")
    # Keep all embeddings. Coordinates are useful but not required.
    # Search can still use descriptors.
    valid_idx = np.arange(idx)

    missing = idx - matched
    if missing:
        print(f"[INDEX] WARNING: {missing} entries missing coordinates, keeping them anyway")

    # ── Filter and normalize ──
    # NOTE: fancy-index copy (all_descs[valid_idx].copy()) briefly doubles
    # peak RAM (~2x index size) and gets OOM-killed on multi-GB indexes.
    print("[INDEX] Filtering valid descriptors...")
    n_valid = len(valid_idx)
    if n_valid == idx:
        descs_valid = all_descs  # nothing filtered — no copy needed
    else:
        # valid_idx is sorted ascending, so row j's source index is >= j and
        # forward compaction never overwrites a row before it is read
        for j, src in enumerate(valid_idx):
            if j != src:
                all_descs[j] = all_descs[src]
        descs_valid = all_descs[:n_valid]

    print("[INDEX] Normalizing in-place (chunked)...")
    NORM_CHUNK = 200_000
    for s in range(0, n_valid, NORM_CHUNK):
        chunk = descs_valid[s:s + NORM_CHUNK]
        norms = np.sqrt(np.einsum('ij,ij->i', chunk, chunk))[:, None]
        norms[norms == 0] = 1
        chunk /= norms

    COMPACT_INDEX_DIR = index_dir
    COMPACT_DESCS_PATH = os.path.join(COMPACT_INDEX_DIR, "megaloc_descriptors.npy")
    COMPACT_META_PATH = os.path.join(COMPACT_INDEX_DIR, "metadata.npz")
    COMPACT_INFO_PATH = os.path.join(COMPACT_INDEX_DIR, "index_info.txt")
    np.save(COMPACT_DESCS_PATH, descs_valid)
    del descs_valid, all_descs

    print("[INDEX] Saving metadata...")
    np.savez_compressed(COMPACT_META_PATH,
        lats=lats[valid_idx], lons=lons[valid_idx], headings=headings[valid_idx],
        panoids=np.array([panoids[i] for i in valid_idx], dtype=object),
        paths=np.array([all_paths[i] for i in valid_idx], dtype=object)
    )

    COMPACT_INFO_PATH = os.path.join(COMPACT_INDEX_DIR, "index_info.txt")
    size_d = os.path.getsize(COMPACT_DESCS_PATH) / 1024 / 1024
    size_m = os.path.getsize(COMPACT_META_PATH) / 1024 / 1024
    with open(COMPACT_INFO_PATH, 'w') as f:
        f.write(f"Compact Index Info\n")
        f.write(f"Built: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Entries: {len(valid_idx)}\n")
        f.write(f"Descriptor dim: {final_dim}\n")
        f.write(f"Raw dim (pre-PCA): {raw_dim}\n")
        f.write(f"Total: {size_d + size_m:.1f} MB\n")

    # Write manifest.json LAST, once every data file for this index exists.
    # scan_indexes()/load_index() treat manifest.json's presence as the
    # signal that an index is complete and safe to use -- writing it first
    # (or not at all) would make a half-built index look ready, or make a
    # fully-built one invisible.
    try:
        idx_lats = lats[valid_idx]
        idx_lons = lons[valid_idx]
        if len(idx_lats):
            # Don't use a raw mean here -- for coastal/scattered coverage
            # (e.g. points along a curving shoreline, or split across a
            # fjord/strait), the arithmetic average of lat/lon can land in
            # open water or otherwise nowhere any real indexed point exists,
            # even though every actual point is on land. Instead, snap to
            # whichever real indexed point is closest to that average, so
            # "coverage center" is always a genuine indexed location.
            mean_lat = float(np.mean(idx_lats))
            mean_lon = float(np.mean(idx_lons))
            dists_sq = (idx_lats - mean_lat) ** 2 + (idx_lons - mean_lon) ** 2
            nearest_idx = int(np.argmin(dists_sq))
            center_lat = float(idx_lats[nearest_idx])
            center_lon = float(idx_lons[nearest_idx])

            # radius_km was previously never computed here at all -- every
            # manifest this function wrote had radius_km stuck at null, so
            # the GUI's "sync radius from the loaded index" logic silently
            # did nothing and left radius_var at whatever stale value it
            # already had (e.g. a leftover 0.09km from an earlier search),
            # producing a near-zero search radius around an otherwise
            # correct center. Compute it as the true max distance from the
            # chosen center to any indexed point, with small headroom.
            _dists_km = haversine_vec(center_lat, center_lon, idx_lats, idx_lons)
            radius_km = float(np.max(_dists_km)) * 1.05 if len(_dists_km) else None
        else:
            center_lat = center_lon = radius_km = None
    except Exception:
        center_lat = center_lon = radius_km = None

    write_index_manifest(
        index_dir,
        index_id,
        name=index_id,
        encoder=build_encoder,
        descriptor_dim=final_dim,
        num_entries=len(valid_idx),
        center_lat=center_lat,
        center_lon=center_lon,
        radius_km=radius_km,
    )

    print(f"\n[INDEX] ✅ Saved compact index:")
    print(f"  ID: {index_id}")
    print(f"  Descriptors: {COMPACT_DESCS_PATH} ({size_d:.1f} MB)")
    print(f"  Metadata: {COMPACT_META_PATH} ({size_m:.1f} MB)")
    print(f"  Descriptor dim: {final_dim} (from raw {raw_dim})")
    print(f"  Total: {size_d + size_m:.1f} MB")

    _compact_cache = None  # Force reload
    return True


def load_compact_index():
    """Load compact index into memory. Returns (descriptors, metadata_dict)."""
    global _compact_cache
    if _compact_cache is not None:
        return _compact_cache
    if not has_active_index() or not os.path.exists(COMPACT_DESCS_PATH) or not os.path.exists(COMPACT_META_PATH):
        print("[INDEX] ERROR: No active index. Build or load one first.")
        return None, None
    print("[INDEX] Loading compact index (memory-mapped)...")
    t0 = time.time()
    # Use mmap_mode='r' to keep the 7.4GB descriptors on disk and stream them into RAM
    descs = np.load(COMPACT_DESCS_PATH, mmap_mode='r')
    meta = np.load(COMPACT_META_PATH, allow_pickle=True)
    metadata = {
        'lats': meta['lats'].copy(), 'lons': meta['lons'].copy(),
        'headings': meta['headings'].copy(),
        'panoids': meta['panoids'], 'paths': meta['paths'],
    }
    del meta
    elapsed = time.time() - t0
    print(f"[INDEX] Loaded {len(descs)} entries ({descs.shape[1]}-dim) in {elapsed:.1f}s [mmap]")
    _compact_cache = (descs, metadata)
    return descs, metadata


def search_compact_index(query_desc, center, radius_km, top_k=100):
    """Search: radius filter → chunked dot-product → panoid dedup → top-K."""
    descs, metadata = load_compact_index()
    if descs is None:
        return []
    t0 = time.time()
    lat1 = np.radians(center[0])
    lon1 = np.radians(center[1])
    lat2 = np.radians(metadata['lats'])
    lon2 = np.radians(metadata['lons'])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    distances = 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    radius_mask = distances <= radius_km
    radius_indices = np.where(radius_mask)[0]
    n_in_radius = len(radius_indices)
    print(f"[INDEX] Radius filter: {n_in_radius}/{len(descs)} in {radius_km}km ({time.time()-t0:.2f}s)")
    if n_in_radius == 0:
        return []

    t1 = time.time()
    query_norm = query_desc / (np.linalg.norm(query_desc) + 1e-8)
    query_norm = query_norm.astype(np.float32)

    # Guard against a stale/mismatched query descriptor -- e.g. the encoder
    # radio button was switched after an index was already loaded, so
    # ACTIVE_ENCODER (and therefore the query descriptor's dimension) no
    # longer matches the dimension of the currently loaded index's
    # descriptors. This used to crash deep inside the matmul with a cryptic
    # numpy ValueError; fail clearly here instead, and don't require a
    # rebuild -- the index on disk is fine, only the in-memory encoder
    # selection is out of sync with what's loaded.
    if query_norm.shape[-1] != descs.shape[-1]:
        print(f"[INDEX] ERROR: Query descriptor is {query_norm.shape[-1]}-dim "
              f"but the loaded index's descriptors are {descs.shape[-1]}-dim. "
              f"The active encoder ('{ACTIVE_ENCODER}') doesn't match the "
              f"encoder this index was built with. Re-select this index "
              f"from 'Select Index...' to resync, then search again.")
        return []

    # Chunked dot product — caps RAM at ~200MB per chunk
    CHUNK_SIZE = 100_000
    top_scores = np.full(top_k * 2, -np.inf, dtype=np.float32)  # keep 2x for panoid dedup
    top_indices = np.zeros(top_k * 2, dtype=np.int64)

    for chunk_start in range(0, n_in_radius, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, n_in_radius)
        chunk_idx = radius_indices[chunk_start:chunk_end]
        chunk_descs = np.array(descs[chunk_idx], dtype=np.float32)
        chunk_sims = chunk_descs @ query_norm
        del chunk_descs

        combined_scores = np.concatenate([top_scores, chunk_sims])
        combined_indices = np.concatenate([top_indices, chunk_idx])
        k = min(top_k * 2, len(combined_scores))
        best_k = np.argsort(combined_scores)[::-1][:k]
        top_scores = combined_scores[best_k]
        top_indices = combined_indices[best_k]

    # Panoid dedup: keep best heading per panoid, then top_k unique panoids
    seen_panoids = {}
    for gi, score in zip(top_indices, top_scores):
        if score == -np.inf:
            break
        pid = str(metadata['panoids'][gi])
        if pid not in seen_panoids or score > seen_panoids[pid]['score']:
            seen_panoids[pid] = {
                'panoid': pid,
                'heading': int(metadata['headings'][gi]),
                'lat': float(metadata['lats'][gi]),
                'lon': float(metadata['lons'][gi]),
                'score': float(score),
                'path': str(metadata['paths'][gi]),
            }

    results = sorted(seen_panoids.values(), key=lambda x: x['score'], reverse=True)[:top_k]
    print(f"[INDEX] Search: top-{len(results)} unique panoids in {time.time()-t1:.2f}s (best: {results[0]['score']:.3f})")
    return results


class ProgressTracker:
    def __init__(self, total_items, estimate_storage=False, embeddings_per_item=4, avg_bytes_per_embedding=2560):
        self.total = total_items
        self.start_time = time.time()
        self.processed = 0
        self.estimate_storage = estimate_storage
        self.embeddings_per_item = embeddings_per_item
        self.avg_bytes_per_embedding = avg_bytes_per_embedding

    def update(self, current_count):
        self.processed = current_count

    def get_status(self):
        elapsed = time.time() - self.start_time
        if elapsed > 0.5 and self.processed > 0:
            speed = self.processed / elapsed
            remaining = self.total - self.processed
            # calc eta string
            eta_seconds = remaining / speed if speed > 0 else 0
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds)) if eta_seconds > 3600 else time.strftime("%M:%S", time.gmtime(eta_seconds))
            speed_fmt = f"{speed:.2f}"
        else:
            eta_str = "calculating..."
            speed_fmt = "--"
        percent = int((self.processed / self.total) * 100) if self.total > 0 else 0
        storage_str = ""
        if self.estimate_storage:
            total_bytes = self.total * self.embeddings_per_item * self.avg_bytes_per_embedding
            if total_bytes < 1024 * 1024:
                storage_str = f" | Storage: {total_bytes / 1024:.1f} KB"
            elif total_bytes < 1024 * 1024 * 1024:
                storage_str = f" | Storage: {total_bytes / (1024 * 1024):.1f} MB"
            else:
                storage_str = f" | Storage: {total_bytes / (1024 * 1024 * 1024):.2f} GB"
        return f"{self.processed}/{self.total} ({percent}%) | {speed_fmt} it/s | ETA: {eta_str}{storage_str}"