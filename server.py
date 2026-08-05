"""
server.py

FastAPI backend for Netryx Astra v2's web UI. This is the HTTP-facing
equivalent of test_supoer.py's Tkinter GUI: no indexing/matching/geo logic lives
here either -- it all comes from utils/netryx_utils.py (imported as `nx`),
exactly like main.py does.

Long-running work (index creation, search) runs in a background thread per
request and reports progress into an in-memory per-job queue. The frontend
polls GET /api/job/{job_id} to drain that queue, the same way main.py's
poll_match_queue() drains self.match_queue every 100ms.

CHANGES vs. the previous version (see inline "# FIX" comments):
  1. _do_create_embeddings now explicitly logs how many panoids were
     skipped (already embedded) vs. how many actually need processing,
     right after that split is computed -- previously the progress bar
     silently pre-loaded the "skipped" count into the tracker, so
     "Stitching & projecting: 8654/8654" could complete almost instantly
     while only a handful of panoids were ever really stitched/encoded,
     with no message telling you that's what happened.
  2. total_extracted is now incremented AFTER a batch is successfully
     appended to buf_descs/buf_paths/buf_lats/buf_lons, not before.
     Previously it was incremented at the top of the try block, so a
     batch_encode() failure still counted those crops as "extracted"
     in the final "N new entries added" message even though nothing
     was written to the part file for them.
  3. Batch failures are now reported into the job's status queue (not
     just printed server-side), so they're visible from the frontend.
  4. The final "Done!" message now also reports the index's total
     cumulative entry count (across ALL part files ever written into
     MEGALOC_PARTS_DIR), not just this run's new-entries delta, since
     those are two different numbers and only the delta was shown
     before.
"""

import os
import io
import gc
import time
import uuid
import queue
import shutil
import tempfile
import threading
import traceback
import concurrent.futures
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from utils import netryx_utils as nx

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="web"), name="static")


@app.get("/")
def home():
    return FileResponse("web/index.html")


@app.get("/api")
def health():
    return {"status": "online"}


# ═══════════════════════════════════════════════════════════════════
# JOB STORE
# Every long-running op (create index / search) gets a job_id. Its
# background thread pushes {"type": ..., ...} dicts into JOBS[id]["queue"].
# The frontend polls /api/job/{id}, which drains whatever has accumulated
# since the last poll (same drain-on-poll pattern as main.py's
# poll_match_queue()).
# ═══════════════════════════════════════════════════════════════════

JOBS = {}
JOBS_LOCK = threading.Lock()


def new_job():
    jid = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[jid] = {
            "queue": queue.Queue(),
            "cancel": threading.Event(),
            "status": "running",   # running | done | error | cancelled
            "result": None,
        }
    return jid


def job_put(jid, **msg):
    job = JOBS.get(jid)
    if job:
        job["queue"].put(msg)


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "job not found"}, status_code=404)
    messages = []
    while True:
        try:
            messages.append(job["queue"].get_nowait())
        except queue.Empty:
            break
    return {
        "status": job["status"],
        "messages": messages,
        "result": job["result"],
    }


@app.post("/api/job/{job_id}/cancel")
def job_cancel(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "job not found"}, status_code=404)
    job["cancel"].set()
    return {"ok": True}


# ═══════════════════════════════════════════════════════════════════
# INDEX MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

def _index_state():
    active_name = None
    active_id = None
    if nx.has_active_index():
        active_id = os.path.basename(nx.COMPACT_INDEX_DIR)
        for m in nx.scan_indexes():
            if m.get("index_id") == active_id:
                active_name = m.get("name", active_id)
                break
    return {
        "active_index_id": active_id,
        "active_index_name": active_name,
        "active_encoder": nx.ACTIVE_ENCODER,
        "has_active_index": nx.has_active_index(),
    }


@app.get("/api/state")
def get_state():
    """Everything the frontend needs to reflect current backend state on load
    or after any action: active index, active encoder, lat/lon/radius if an
    index is loaded."""
    state = _index_state()
    if nx.has_active_index():
        for m in nx.scan_indexes():
            if m.get("index_id") == state["active_index_id"]:
                cov = m.get("coverage_center", {})
                state["coverage_center"] = cov
                state["radius_km"] = m.get("radius_km")
                break
    return state


@app.get("/api/indexes")
def list_indexes():
    return {"indexes": nx.scan_indexes(), **_index_state()}


@app.post("/api/indexes/select")
def select_index(index_id: str = Form(...)):
    try:
        manifest = nx.load_index(index_id)
        nx._compact_cache = None
        return {"ok": True, "manifest": manifest, **_index_state()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.delete("/api/indexes/{index_id}")
def delete_index(index_id: str):
    index_path = os.path.join(nx.INDEXES_DIR, index_id)
    if not os.path.isdir(index_path):
        return JSONResponse({"error": "Index not found."}, status_code=404)
    try:
        # If this is the currently active index, clear the COMPACT_* globals
        # first so nothing keeps pointing at a directory we're about to
        # delete out from under it.
        if nx.has_active_index() and os.path.basename(nx.COMPACT_INDEX_DIR) == index_id:
            nx.COMPACT_INDEX_DIR = None
            nx.COMPACT_DESCS_PATH = None
            nx.COMPACT_META_PATH = None
            nx.COMPACT_INFO_PATH = None
            nx.COMPACT_PCA_PATH = None
            nx._compact_cache = None
        shutil.rmtree(index_path)
        return {"ok": True, **_index_state()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


def _cleanup_temp_file(path):
    try:
        os.remove(path)
    except Exception:
        pass


@app.get("/api/indexes/{index_id}/export")
def export_index(index_id: str, background_tasks: BackgroundTasks):
    if not nx.HUB_AVAILABLE:
        return JSONResponse({"error": "Bundle export unavailable (netryx_hub.py not found)."}, status_code=400)

    index_path = os.path.join(nx.INDEXES_DIR, index_id)
    if not os.path.isdir(index_path):
        return JSONResponse({"error": "Index not found."}, status_code=404)

    manifest_path = os.path.join(index_path, "manifest.json")
    if not os.path.isfile(manifest_path):
        return JSONResponse({"error": "Index has no manifest -- looks incomplete."}, status_code=400)

    import json as _json
    with open(manifest_path) as f:
        manifest = _json.load(f)

    tmp_dir = tempfile.mkdtemp(prefix="netryx_export_")
    out_name = f"{manifest.get('name', index_id)}.netryx"
    out_path = os.path.join(tmp_dir, out_name)

    try:
        cov = manifest.get("coverage_center", {})
        nx.create_bundle(
            index_dir=index_path,
            output_path=out_path,
            name=manifest.get("name", index_id),
            description="Exported via Netryx Astra web UI",
            center_lat=cov.get("lat"),
            center_lon=cov.get("lon"),
            radius_km=manifest.get("radius_km"),
        )
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return JSONResponse({"error": f"Export failed: {e}"}, status_code=500)

    background_tasks.add_task(shutil.rmtree, tmp_dir, ignore_errors=True)
    return FileResponse(out_path, filename=out_name, media_type="application/octet-stream",
                         background=background_tasks)


@app.post("/api/indexes/import")
async def import_index(file: UploadFile = File(...)):
    if not nx.HUB_AVAILABLE:
        return JSONResponse({"error": "Bundle import unavailable (netryx_hub.py not found)."}, status_code=400)

    tmp_dir = tempfile.mkdtemp(prefix="netryx_import_")
    tmp_path = os.path.join(tmp_dir, file.filename or "upload.netryx")
    try:
        with open(tmp_path, "wb") as f:
            f.write(await file.read())

        import zipfile, json as _json
        with zipfile.ZipFile(tmp_path, "r") as zf:
            peek_manifest = _json.loads(zf.read("manifest.json"))
        bundle_id = peek_manifest.get("index_id")
        if bundle_id:
            local_ids = {m.get("index_id") for m in nx.scan_indexes()}
            if bundle_id in local_ids:
                return {"ok": False, "already_imported": True,
                        "name": peek_manifest.get("name", bundle_id)}

        manifest = nx.extract_bundle(tmp_path, nx.DATA_DIR)
        nx.load_index(manifest["index_id"])
        nx._compact_cache = None
        return {"ok": True, "manifest": manifest, **_index_state()}
    except Exception as e:
        return JSONResponse({"error": f"Import failed: {e}"}, status_code=500)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/api/encoder")
def set_encoder(name: str = Form(...)):
    try:
        nx.set_encoder(name)
        return {"ok": True, **_index_state()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/api/coverage")
def coverage():
    if not nx.has_active_index() or not os.path.exists(nx.COMPACT_META_PATH):
        return {"points": []}
    try:
        meta = np.load(nx.COMPACT_META_PATH, allow_pickle=True)
        lats, lons = meta["lats"], meta["lons"]
        pts = {(round(float(a), 6), round(float(b), 6)) for a, b in zip(lats, lons)}
        del meta
        return {"points": [{"lat": p[0], "lon": p[1]} for p in pts]}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ═══════════════════════════════════════════════════════════════════
# CREATE INDEX  (mirrors main.py's StreetViewMatcherGUI._create_embeddings)
# ═══════════════════════════════════════════════════════════════════

def _count_total_part_entries():
    """Cumulative entry count across ALL part files currently in
    MEGALOC_PARTS_DIR (not just this run's delta). Used for the final
    status message so 'N new entries added' and 'total in parts dir'
    are never confused with each other again."""
    total = 0
    try:
        for pf in nx.glob.glob(os.path.join(nx.MEGALOC_PARTS_DIR, "megaloc_part_*.npz")):
            data = np.load(pf, allow_pickle=True)
            total += len(data["paths"])
            del data
    except Exception:
        pass
    return total


def _do_create_embeddings(jid, center, radius, res, crop_fov, crop_size, crop_step):
    def status(text):
        job_put(jid, type="status", text=text)

    status("Getting grid points...")
    points = nx.grid_points(center, radius, res)
    status(f"Generated {len(points)} grid points. Downloading scan nodes...")

    panoids = nx.get_panoids(
        points,
        status_callback=lambda i, t: status(f"Scan node fetch {i}/{t}..."),
        max_workers=nx.MAX_PANOID_WORKERS,
    )
    status(f"Found {len(panoids)} scan nodes. Extracting features...")

    headings_all = sorted(set(((h // crop_step) * crop_step) % 360 for h in range(0, 360, crop_step)))
    embeddings_per_panoid = len(headings_all)

    os.makedirs(nx.MEGALOC_PARTS_DIR, exist_ok=True)

    existing_files = set()
    try:
        for ep in nx.glob.glob(os.path.join(nx.MEGALOC_PARTS_DIR, "megaloc_part_*.npz")):
            data = np.load(ep, allow_pickle=True)
            for p in data["paths"]:
                existing_files.add(os.path.basename(str(p)))
            del data
        if existing_files:
            status(f"Loaded {len(existing_files)} existing entries from part files. Starting...")
    except Exception as e:
        status(f"Warning: Could not load existing parts: {e}")

    crop_queue = queue.Queue(maxsize=nx.CROP_QUEUE_SIZE)
    tracker = nx.ProgressTracker(len(panoids), estimate_storage=True,
                                  embeddings_per_item=embeddings_per_panoid, avg_bytes_per_embedding=2560)
    total_extracted = 0
    failed_batches = 0

    def batch_extractor():
        nonlocal total_extracted, failed_batches
        target_batch_size = nx.MEGALOC_BATCH_SIZE
        batch_buffer = []
        buf_descs, buf_paths, buf_lats, buf_lons = [], [], [], []

        def save_chunk():
            if not buf_descs:
                return
            try:
                ts = int(time.time() * 1000)
                part_filename = os.path.join(nx.MEGALOC_PARTS_DIR, f"megaloc_part_{ts}.npz")
                all_descs = np.vstack(buf_descs)
                np.savez(part_filename, descriptors=all_descs,
                          paths=np.array(buf_paths, dtype=object),
                          lats=np.array(buf_lats, dtype=np.float32),
                          lons=np.array(buf_lons, dtype=np.float32))
                status(f"Saved index chunk: {len(buf_paths)} items")
                buf_descs.clear(); buf_paths.clear(); buf_lats.clear(); buf_lons.clear()
            except Exception as e:
                # FIX: this used to silently drop an entire chunk (up to
                # 5000 items) with only a server-side print(). Any items
                # already counted into total_extracted for this chunk were
                # never actually persisted to disk, so the final "N new
                # entries added" message lied. Report it to the job queue
                # AND roll back total_extracted for the items just lost.
                lost = len(buf_paths)
                total_extracted -= lost
                status(f"ERROR saving chunk ({lost} items lost): {e}")
                print(f"Error saving chunk: {e}")
                buf_descs.clear(); buf_paths.clear(); buf_lats.clear(); buf_lons.clear()

        def process_batch(buffer):
            nonlocal total_extracted, failed_batches
            crops = [b[0] for b in buffer]
            meta = [b[1] for b in buffer]
            try:
                crops_pil = [nx.tensor_to_pil(c) for c in crops]
                cos_descs = nx.batch_encode(crops_pil, batch_size=len(crops))
                buf_descs.append(cos_descs)
                buf_paths.extend([m["path"] for m in meta])
                buf_lats.extend([m["lat"] for m in meta])
                buf_lons.extend([m["lon"] for m in meta])
                # FIX: only count as "extracted" AFTER the batch is
                # actually appended to the save buffer, not before
                # encoding was even attempted.
                total_extracted += len(meta)
                if len(buf_paths) >= 5000:
                    save_chunk()
            except Exception as e:
                # FIX: previously print()-only, and total_extracted was
                # already incremented before this try ran, so a failed
                # batch was still counted in the final summary while
                # producing zero saved entries. Now nothing is counted
                # for a failed batch, and the failure is visible in the
                # job's status feed instead of only the server console.
                failed_batches += 1
                status(f"ERROR: batch of {len(meta)} crops failed to encode ({e}); skipped, not counted")
                print(f"Batch processing error: {e}")

        while True:
            item = crop_queue.get()
            if item == "DONE":
                if batch_buffer:
                    process_batch(batch_buffer)
                save_chunk()
                crop_queue.task_done()
                break
            batch_buffer.append(item)
            if len(batch_buffer) >= target_batch_size:
                process_batch(batch_buffer)
                batch_buffer = []
            crop_queue.task_done()

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    extractor_thread = threading.Thread(target=batch_extractor)
    extractor_thread.start()

    base_dirs = nx.get_projection_base_dirs(crop_fov, (crop_size, crop_size))

    panoids_needing_download = []
    missing_yaws_by_id = {}
    for p in panoids:
        pid = p["panoid"]
        missing = [y for y in headings_all if f"{pid}_{y}.npz" not in existing_files]
        if missing:
            panoids_needing_download.append(p)
            missing_yaws_by_id[pid] = missing
    skipped = len(panoids) - len(panoids_needing_download)

    # FIX: this is the key visibility fix. Previously the progress tracker
    # was silently pre-loaded with `skipped` and the loop only ever ran
    # over panoids_needing_download, so "Stitching & projecting: 8654/8654"
    # could report 100% almost instantly with no indication that most of
    # those 8654 were never touched this run because they were already
    # embedded from earlier coverage. Make that explicit now.
    status(
        f"{skipped}/{len(panoids)} scan nodes already embedded (skipping); "
        f"{len(panoids_needing_download)} need stitching/projection/encoding this run."
    )
    if len(panoids_needing_download) == 0:
        status("Nothing new to process -- all scan nodes in this area are already indexed. "
               "Rebuilding compact index from existing part files only.")

    all_tiles_data = {}
    if panoids_needing_download:
        all_tiles_data = nx.download_tiles_for_panoids(
            [p["panoid"] for p in panoids_needing_download],
            max_workers=nx.MAX_DOWNLOAD_WORKERS,
            status_callback=lambda d, t: status(f"Downloading tiles: {d}/{t}..."),
        )

    stitch_failures = 0

    def process_one_panoid(panoid):
        nonlocal stitch_failures
        pid = panoid["panoid"]
        missing_yaws = missing_yaws_by_id.get(pid)
        if not missing_yaws:
            return True
        tiles_data = all_tiles_data.get(pid)
        if not tiles_data:
            stitch_failures += 1
            return False
        try:
            pano_img = nx.stitch_tiles(tiles_data)
        except Exception:
            stitch_failures += 1
            return False
        maxw = 2048
        if pano_img.size[0] > maxw:
            pano_img = pano_img.resize((maxw, int(pano_img.size[1] * (maxw / pano_img.size[0]))), Image.BILINEAR)
        pano_t = nx.pil_to_tensor(pano_img)
        crops_batch = nx.equirectangular_to_rectilinear_torch(
            pano_t, fov_deg=crop_fov, out_hw=(crop_size, crop_size),
            yaw_deg=missing_yaws, pitch_deg=0, base_dirs=base_dirs)
        for i, yaw in enumerate(missing_yaws):
            crop_t = crops_batch[i].unsqueeze(0)
            meta = {"path": f"{pid}_{yaw}.npz", "lat": panoid["lat"], "lon": panoid["lon"], "yaw": yaw}
            crop_queue.put((crop_t, meta))
        pano_img.close()
        del pano_t
        return True

    tracker.update(skipped)
    with concurrent.futures.ThreadPoolExecutor(max_workers=nx.MAX_PANOID_WORKERS) as executor:
        for idx, _ in enumerate(executor.map(process_one_panoid, panoids_needing_download), skipped + 1):
            tracker.update(idx)
            status(f"Stitching & projecting: {tracker.get_status()}")

    crop_queue.put("DONE")
    extractor_thread.join()

    if stitch_failures:
        status(f"Warning: {stitch_failures} panoids failed to stitch/download tiles and were skipped.")
    if failed_batches:
        status(f"Warning: {failed_batches} encode batch(es) failed and were not saved.")

    status(f"All embeddings saved ({total_extracted} new this run). Building index (fits PCA)...")
    nx.build_compact_index()

    # FIX: report both numbers so "new entries added" is never mistaken
    # for "total size of the index" again.
    total_in_parts = _count_total_part_entries()
    status(
        f"Done! Index ready. {total_extracted} new entries added this run "
        f"({skipped} scan nodes were already covered and skipped). "
        f"Total entries across all part files: {total_in_parts}."
    )
    nx._compact_cache = None


@app.post("/api/create_index")
def create_index(lat: float = Form(...), lon: float = Form(...), radius: float = Form(...),
                  resolution: int = Form(300), fov: int = Form(90),
                  crop_size: int = Form(256), crop_step: int = Form(90)):
    jid = new_job()

    def run():
        try:
            _do_create_embeddings(jid, (lat, lon), radius, resolution, fov, crop_size, crop_step)
            JOBS[jid]["status"] = "done"
        except Exception as e:
            traceback.print_exc()
            job_put(jid, type="status", text=f"Error: {e}")
            JOBS[jid]["status"] = "error"

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": jid}


# ═══════════════════════════════════════════════════════════════════
# SEARCH  (mirrors main.py's _run_search / _handle_match_done)
# ═══════════════════════════════════════════════════════════════════

def _do_run_search(jid, query_img_resize, center, radius, threshold, crop_fov, crop_size, crop_step, cancel_event):
    def status(text):
        job_put(jid, type="status", text=text)

    if nx.ENCODER_USES_PCA:
        pca_path = nx.COMPACT_PCA_PATH or (
            os.path.join(nx.COMPACT_INDEX_DIR, "megaloc_pca.pkl") if nx.COMPACT_INDEX_DIR else None)
        if pca_path and os.path.exists(pca_path):
            from utils.megaloc_utils import load_pca, _pca_model
            if _pca_model is None:
                load_pca(pca_path)

    status(f"Extracting query descriptor ({nx.ACTIVE_ENCODER}, multi-scale)...")
    desc_original = nx.encode_query(query_img_resize)

    w, h = query_img_resize.size
    mx, my = int(w * 0.1), int(h * 0.1)
    cropped = query_img_resize.crop((mx, my, w - mx, h - my)).resize((crop_size, crop_size), Image.BILINEAR)
    desc_zoom = nx.encode_query(cropped)
    cropped.close()

    query_desc = 0.65 * desc_original + 0.35 * desc_zoom
    query_desc = query_desc / (np.linalg.norm(query_desc) + 1e-8)

    flipped = query_img_resize.transpose(Image.FLIP_LEFT_RIGHT)
    desc_flipped = nx.encode_query(flipped)
    desc_flipped_zoom = nx.encode_query(
        flipped.crop((mx, my, w - mx, h - my)).resize((crop_size, crop_size), Image.BILINEAR))
    desc_flipped = 0.65 * desc_flipped + 0.35 * desc_flipped_zoom
    desc_flipped = desc_flipped / (np.linalg.norm(desc_flipped) + 1e-8)

    status("Searching index (original + flipped)...")
    K_MEGALOC = 1000

    loaded_descs, _ = nx.load_compact_index()
    if loaded_descs is not None and query_desc.shape[-1] != loaded_descs.shape[-1]:
        status(f"Encoder mismatch: query is {query_desc.shape[-1]}-dim but "
               f"the loaded index is {loaded_descs.shape[-1]}-dim. Re-select the index to resync.")
        return {"inliers": 0, "panoid": None, "lat": None, "lon": None, "all_top_clusters": []}

    results_original = nx.search_compact_index(query_desc=query_desc, center=center, radius_km=radius, top_k=100)
    results_flipped = nx.search_compact_index(query_desc=desc_flipped, center=center, radius_km=radius, top_k=100)

    seen = {}
    for r in results_original + results_flipped:
        k = r["panoid"]
        if k not in seen or r["score"] > seen[k]["score"]:
            seen[k] = r
    compact_results = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:K_MEGALOC]

    if not compact_results:
        status("No candidates found in radius.")
        return {"inliers": 0, "panoid": None, "lat": None, "lon": None, "all_top_clusters": []}

    MAST3R_STAGE2_TOP_N = 100
    candidates = compact_results[:MAST3R_STAGE2_TOP_N]
    status(f"Stage 2: Running MASt3R on top {len(candidates)} candidates...")

    all_matches = []
    best = {"inliers": 0, "panoid": None, "heading": None, "lat": None, "lon": None}

    try:
        mast3r = nx.get_lazy_mast3r()
        if mast3r is not None:
            PREFETCH_LOOKAHEAD = 4
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=PREFETCH_LOOKAHEAD)
            futures = {}

            def fetch_tiles(cand):
                cpid = cand.get("panoid")
                if not cpid:
                    return None
                try:
                    return nx.download_tiles(nx.tiles_info(cpid), max_workers=16)
                except Exception:
                    return None

            for j in range(min(PREFETCH_LOOKAHEAD, len(candidates))):
                futures[j] = pool.submit(fetch_tiles, candidates[j])

            for i, match in enumerate(candidates):
                if cancel_event.is_set():
                    status("Search cancelled.")
                    break
                status(f"MASt3R Match: {i+1}/{len(candidates)}")
                job_put(jid, type="progress", value=i, max=len(candidates))
                pid, hdg = match.get("panoid"), match.get("heading")
                if not pid or hdg is None:
                    continue

                nxt = i + PREFETCH_LOOKAHEAD
                if nxt < len(candidates) and nxt not in futures:
                    futures[nxt] = pool.submit(fetch_tiles, candidates[nxt])

                pano_img = None
                try:
                    fut = futures.pop(i, None)
                    td = fut.result() if fut is not None else nx.download_tiles(nx.tiles_info(pid), max_workers=16)
                    if td:
                        pano_img = nx.stitch_tiles(td)
                        maxw = 2048
                        if pano_img.size[0] > maxw:
                            pano_img = pano_img.resize((maxw, int(pano_img.size[1] * (maxw / pano_img.size[0]))), Image.BILINEAR)
                except Exception:
                    continue

                if pano_img:
                    pano_t = nx.pil_to_tensor(pano_img)
                    base_dirs = nx.get_projection_base_dirs(crop_fov, (crop_size, crop_size))
                    crop_t = nx.equirectangular_to_rectilinear_torch(
                        pano_t, fov_deg=crop_fov, out_hw=(crop_size, crop_size),
                        yaw_deg=[hdg], pitch_deg=0, base_dirs=base_dirs)[0].unsqueeze(0)
                    crop_pil = nx.tensor_to_pil(crop_t)
                    m0, m1, _ = nx.get_mast3r_matches(query_img_resize, crop_pil, mast3r)
                    score = len(m0)

                    match_res = {"inliers": score, "panoid": pid, "heading": hdg,
                                 "lat": match.get("lat"), "lon": match.get("lon")}

                    if score > 50:
                        all_matches.append(match_res)
                        job_put(jid, type="scan_blip", lat=match.get("lat"), lon=match.get("lon"), inliers=score)

                    if score > best["inliers"]:
                        best = match_res

                    del pano_t, crop_t
                    pano_img.close()
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()

                    if best["inliers"] >= 450:
                        status(f"Ultra-strong match! {best['inliers']} points — stopping early")
                        break

            pool.shutdown(wait=False, cancel_futures=True)

            if len(all_matches) >= 3:
                CELL_SIZE = 0.00045
                cells = defaultdict(list)
                for m in all_matches:
                    cell = (round(m["lat"] / CELL_SIZE), round(m["lon"] / CELL_SIZE))
                    cells[cell].append(m)
                scored = []
                for ck, cm in cells.items():
                    neigh = []
                    for dlat in (-1, 0, 1):
                        for dlon in (-1, 0, 1):
                            neigh.extend(cells.get((ck[0] + dlat, ck[1] + dlon), []))
                    cell_score = sum(nx.math.sqrt(m["inliers"]) for m in neigh)
                    cluster_best = max(neigh, key=lambda m: m["inliers"])
                    scored.append({"score": cell_score, "match": cluster_best})
                scored.sort(key=lambda x: x["score"], reverse=True)

                top = []
                seen_pids = set()
                for sc in scored:
                    r = sc["match"]
                    if r["panoid"] not in seen_pids:
                        top.append(r)
                        seen_pids.add(r["panoid"])
                    if len(top) >= 10:
                        break
                if top:
                    best = dict(top[0])
                    best["all_top_clusters"] = top
            elif best["inliers"] > 0:
                best["all_top_clusters"] = [best]

    except Exception as e:
        print(f"Stage 2 MASt3R error: {e}")
        traceback.print_exc()

    if best["inliers"] > 0:
        best["inliers"] = 200 + best["inliers"] // 10
        if "all_top_clusters" in best:
            for c in best["all_top_clusters"]:
                c["inliers"] = 200 + c["inliers"] // 10
    return best


@app.post("/api/search")
async def search(image: UploadFile = File(...), lat: float = Form(...), lon: float = Form(...),
                  radius: float = Form(...), threshold: int = Form(50), fov: int = Form(90),
                  crop_size: int = Form(256), crop_step: int = Form(90)):
    if not nx.has_active_index():
        return JSONResponse({"error": "No active index selected."}, status_code=400)

    img_bytes = await image.read()
    try:
        query_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return JSONResponse({"error": f"Could not read image: {e}"}, status_code=400)
    query_img_resize = query_img.resize((crop_size, crop_size), Image.BILINEAR)

    jid = new_job()
    job = JOBS[jid]

    def run():
        try:
            best = _do_run_search(jid, query_img_resize, (lat, lon), radius, threshold,
                                   fov, crop_size, crop_step, job["cancel"])
            if job["cancel"].is_set():
                job["status"] = "cancelled"
                job_put(jid, type="status", text="Search cancelled.")
                return
            if best["inliers"] > threshold and best.get("panoid") is not None:
                job_put(jid, type="status",
                        text=f"Best match: {best['inliers']} inliers at heading {best.get('heading')}°")
            else:
                job_put(jid, type="status", text="No match found.")
            job["result"] = best
            job["status"] = "done"
        except Exception as e:
            traceback.print_exc()
            job_put(jid, type="status", text=f"Search error: {e}")
            job["status"] = "error"

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": jid}


if __name__ == "__main__":
    for d in [nx.DATA_DIR, nx.MEGALOC_PARTS_DIR, nx.INDEXES_DIR]:
        os.makedirs(d, exist_ok=True)
    available = nx.scan_indexes()
    print(f"[INDEX] Found {len(available)} index(es): "
          f"{[m.get('name', m.get('index_id')) for m in available]}")
    if available:
        nx.load_index(available[-1]["index_id"])

    uvicorn.run("server:app", host="127.0.0.1", port=2026, reload=False)