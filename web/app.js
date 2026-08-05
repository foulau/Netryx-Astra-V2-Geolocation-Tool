// ═══════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════

const latInput = document.getElementById("lat");
const lonInput = document.getElementById("lon");
const radiusInput = document.getElementById("radius");
const resolutionInput = document.getElementById("resolution");
const thresholdInput = document.getElementById("threshold");

const runBtn = document.getElementById("runBtn");
const statusBar = document.getElementById("statusBar");
const progressFill = document.getElementById("progressFill");
const resultsList = document.getElementById("resultsList");
const indexSelect = document.getElementById("indexSelect");
const importBtn = document.getElementById("importBtn");
const exportBtn = document.getElementById("exportBtn");
const deleteBtn = document.getElementById("deleteBtn");
const importInput = document.getElementById("importInput");
const uploadBox = document.getElementById("uploadBox");
const uploadBoxText = document.getElementById("uploadBoxText");
const uploadBoxPreview = document.getElementById("uploadBoxPreview");
const imageInput = document.getElementById("imageInput");

let modeVar = "search";        // "search" | "create"
let encoderVar = "megaloc";    // "megaloc" | "mixvpr"
let selectedFile = null;
let currentJobId = null;
let pollTimer = null;
let coverageLayer = null;
let resultMarkers = [];
let isRunning = false;

function setStatus(text) {
    statusBar.textContent = text;
}

function setProgress(value, max) {
    const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
    progressFill.style.width = pct + "%";
}

// ═══════════════════════════════════════════════════════════════════
// MODE / ENCODER SELECTORS
// ═══════════════════════════════════════════════════════════════════

document.querySelectorAll('.selector[data-group="mode"]').forEach(el => {
    el.addEventListener("click", () => {
        document.querySelectorAll('.selector[data-group="mode"]').forEach(s => {
            s.classList.remove("active");
            s.querySelector(".selector-dot").classList.remove("active");
        });
        el.classList.add("active");
        el.querySelector(".selector-dot").classList.add("active");
        modeVar = el.dataset.value;
        if (!isRunning) {
            runBtn.textContent = modeVar === "search" ? "▶ Run Search" : "▶ Create Index";
        }
        uploadBox.style.display = modeVar === "search" ? "flex" : "none";
    });
});

document.querySelectorAll('.selector[data-group="encoder"]').forEach(el => {
    el.addEventListener("click", async () => {
        const value = el.dataset.value;
        if (value === encoderVar) return;
        document.querySelectorAll('.selector[data-group="encoder"]').forEach(s => {
            s.classList.remove("active");
            s.querySelector(".selector-dot").classList.remove("active");
        });
        el.classList.add("active");
        el.querySelector(".selector-dot").classList.add("active");
        encoderVar = value;

        try {
            const fd = new FormData();
            fd.append("name", value);
            const res = await fetch("/api/encoder", { method: "POST", body: fd });
            const data = await res.json();
            if (data.error) {
                setStatus(`Encoder '${value}' unavailable: ${data.error}`);
            } else {
                setStatus(`Encoder: ${value.toUpperCase()} — ${data.has_active_index ? "index ready." : "no index selected."}`);
                await refreshIndexList();
            }
        } catch (e) {
            setStatus(`Encoder switch failed: ${e}`);
        }
    });
});

// ═══════════════════════════════════════════════════════════════════
// IMAGE UPLOAD
// ═══════════════════════════════════════════════════════════════════

uploadBox.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (!file) return;
    selectedFile = file;

    const reader = new FileReader();
    reader.onload = e => {
        uploadBoxPreview.src = e.target.result;
        uploadBoxPreview.style.display = "block";
    };
    reader.readAsDataURL(file);

    uploadBoxText.textContent = file.name;
    uploadBox.classList.add("has-image");
});

// ═══════════════════════════════════════════════════════════════════
// INDEX SELECTOR
// ═══════════════════════════════════════════════════════════════════

async function refreshIndexList() {
    try {
        const res = await fetch("/api/indexes");
        const data = await res.json();
        indexSelect.innerHTML = "";
        if (!data.indexes || data.indexes.length === 0) {
            const opt = document.createElement("option");
            opt.value = "";
            opt.textContent = "No indexes found";
            indexSelect.appendChild(opt);
            return;
        }
        data.indexes.forEach(m => {
            const opt = document.createElement("option");
            opt.value = m.index_id;
            const entries = m.num_entries ?? "?";
            const enc = m.descriptor_model ?? "?";
            opt.textContent = `${m.name || m.index_id} [${enc}, ${entries} entries]`;
            if (m.index_id === data.active_index_id) opt.selected = true;
            indexSelect.appendChild(opt);
        });
        if (data.active_index_name) {
            setStatus(`Active index: ${data.active_index_name}`);
        }
    } catch (e) {
        setStatus(`Failed to load indexes: ${e}`);
    }
}

indexSelect.addEventListener("change", async () => {
    const id = indexSelect.value;
    if (!id) return;
    setStatus("Loading index...");
    try {
        const fd = new FormData();
        fd.append("index_id", id);
        const res = await fetch("/api/indexes/select", { method: "POST", body: fd });
        const data = await res.json();
        if (data.error) {
            setStatus(`Failed to load index: ${data.error}`);
            return;
        }
        setStatus(`Index loaded: ${data.manifest.name || id}`);
        const cov = data.manifest.coverage_center || {};
        if (cov.lat != null && cov.lon != null) {
            latInput.value = cov.lat;
            lonInput.value = cov.lon;
            moveCenter(L.latLng(cov.lat, cov.lon));
            map.setView([cov.lat, cov.lon], 13);
        }
        if (data.manifest.radius_km != null) {
            radiusInput.value = data.manifest.radius_km.toFixed(2);
            circle.setRadius(data.manifest.radius_km * 1000);
            updateHandle();
        }
        await loadCoverage();
    } catch (e) {
        setStatus(`Failed to load index: ${e}`);
    }
});

importBtn.addEventListener("click", () => importInput.click());

importInput.addEventListener("change", async () => {
    const file = importInput.files[0];
    if (!file) return;
    setStatus(`Importing ${file.name}...`);
    try {
        const fd = new FormData();
        fd.append("file", file);
        const res = await fetch("/api/indexes/import", { method: "POST", body: fd });
        const data = await res.json();
        if (data.error) {
            setStatus(`Import failed: ${data.error}`);
        } else if (data.already_imported) {
            setStatus(`'${data.name}' is already imported.`);
        } else {
            setStatus(`✅ Imported: ${data.manifest.name || "index"}`);
            await refreshIndexList();
            await loadCoverage();
        }
    } catch (e) {
        setStatus(`Import failed: ${e}`);
    } finally {
        importInput.value = "";
    }
});

exportBtn.addEventListener("click", async () => {
    const id = indexSelect.value;
    if (!id) {
        setStatus("Select an index first.");
        return;
    }
    setStatus("Exporting index...");
    try {
        const res = await fetch(`/api/indexes/${id}/export`);
        if (!res.ok) {
            const data = await res.json().catch(() => ({}));
            setStatus(`Export failed: ${data.error || res.statusText}`);
            return;
        }
        const blob = await res.blob();
        const disposition = res.headers.get("Content-Disposition") || "";
        const match = disposition.match(/filename="?([^"]+)"?/);
        const filename = match ? match[1] : `${id}.netryx`;
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
        setStatus(`✅ Exported: ${filename}`);
    } catch (e) {
        setStatus(`Export failed: ${e}`);
    }
});

deleteBtn.addEventListener("click", async () => {
    const id = indexSelect.value;
    if (!id) {
        setStatus("Select an index first.");
        return;
    }
    const label = indexSelect.options[indexSelect.selectedIndex].textContent;
    if (!confirm(`Delete "${label}"? This permanently removes it from disk.`)) return;

    setStatus("Deleting index...");
    try {
        const res = await fetch(`/api/indexes/${id}`, { method: "DELETE" });
        const data = await res.json();
        if (data.error) {
            setStatus(`Delete failed: ${data.error}`);
            return;
        }
        setStatus("Index deleted.");
        await refreshIndexList();
        await loadCoverage();
    } catch (e) {
        setStatus(`Delete failed: ${e}`);
    }
});

// ═══════════════════════════════════════════════════════════════════
// RUN / CANCEL
// ═══════════════════════════════════════════════════════════════════

runBtn.addEventListener("click", async () => {
    if (isRunning) {
        if (currentJobId) {
            await fetch(`/api/job/${currentJobId}/cancel`, { method: "POST" });
            setStatus("Cancelling...");
        }
        return;
    }
    if (modeVar === "create") {
        await startCreateIndex();
    } else {
        await startSearch();
    }
});

function setRunning(running) {
    isRunning = running;
    runBtn.classList.toggle("running", running);
    if (running) {
        runBtn.textContent = "■ Stop";
    } else {
        runBtn.textContent = modeVar === "search" ? "▶ Run Search" : "▶ Create Index";
    }
}

async function startCreateIndex() {
    clearResultMarkers();
    const fd = new FormData();
    fd.append("lat", latInput.value);
    fd.append("lon", lonInput.value);
    fd.append("radius", radiusInput.value);
    fd.append("resolution", resolutionInput.value);

    setStatus("Creating embeddings in background...");
    setRunning(true);

    try {
        const res = await fetch("/api/create_index", { method: "POST", body: fd });
        const data = await res.json();
        if (data.error) {
            setStatus(`Error: ${data.error}`);
            resetButtons();
            return;
        }
        currentJobId = data.job_id;
        pollJob();
    } catch (e) {
        setStatus(`Error: ${e}`);
        resetButtons();
    }
}

async function startSearch() {
    if (!selectedFile) {
        setStatus("⚠ No image selected. Click the upload box first.");
        return;
    }

    clearResultMarkers();
    const fd = new FormData();
    fd.append("image", selectedFile);
    fd.append("lat", latInput.value);
    fd.append("lon", lonInput.value);
    fd.append("radius", radiusInput.value);
    fd.append("threshold", thresholdInput.value);

    setStatus("Starting search...");
    setRunning(true);

    try {
        const res = await fetch("/api/search", { method: "POST", body: fd });
        const data = await res.json();
        if (data.error) {
            setStatus(`⚠ ${data.error}`);
            resetButtons();
            return;
        }
        currentJobId = data.job_id;
        pollJob();
    } catch (e) {
        setStatus(`Error: ${e}`);
        resetButtons();
    }
}

function resetButtons() {
    setRunning(false);
    currentJobId = null;
    if (pollTimer) clearTimeout(pollTimer);
}

async function pollJob() {
    if (!currentJobId) return;
    try {
        const res = await fetch(`/api/job/${currentJobId}`);
        const data = await res.json();

        for (const msg of data.messages) {
            if (msg.type === "status") setStatus(msg.text);
            else if (msg.type === "progress") setProgress(msg.value, msg.max);
            else if (msg.type === "scan_blip") handleScanBlip(msg.lat, msg.lon, msg.inliers);
        }

        if (data.status === "done" || data.status === "cancelled") {
            if (data.result) handleSearchResult(data.result);
            resetButtons();
            return;
        }
        if (data.status === "error") {
            resetButtons();
            return;
        }
        pollTimer = setTimeout(pollJob, 700);
    } catch (e) {
        setStatus(`Polling error: ${e}`);
        resetButtons();
    }
}

// ═══════════════════════════════════════════════════════════════════
// RESULT HANDLING
// ═══════════════════════════════════════════════════════════════════

function buildGmapsPopup(lat, lon, subtitle) {
    const gmapsUrl = `https://www.google.com/maps/search/?api=1&query=${lat},${lon}`;
    const el = document.createElement("div");
    el.className = "gmaps-popup";
    el.innerHTML =
        `📍 ${lat.toFixed(6)}, ${lon.toFixed(6)}<br>${subtitle}` +
        `<div class="gmaps-hint">Click to open in Google Maps ↗</div>`;
    el.addEventListener("click", () => window.open(gmapsUrl, "_blank"));
    return el;
}

function clearResultMarkers() {
    resultMarkers.forEach(m => map.removeLayer(m));
    resultMarkers = [];
    resultsList.innerHTML = '<span class="muted">No results yet</span>';
}

function handleScanBlip(lat, lon, inliers) {
    if (lat == null || lon == null) return;
    const color = inliers > 50 ? "#00ff9d" : inliers > 20 ? "#ffd700" : "#ff4444";
    const blip = L.circleMarker([lat, lon], {
        radius: 6, color, fillColor: color, fillOpacity: 0.8, weight: 2
    }).addTo(map);
    setTimeout(() => map.removeLayer(blip), 1200);
}

function handleSearchResult(best) {
    clearResultMarkers();
    const thresh = Number(thresholdInput.value);

    if (!best || best.inliers <= thresh || best.panoid == null) {
        setStatus("No match found.");
        return;
    }

    map.setView([best.lat, best.lon], 16);
    const marker = L.marker([best.lat, best.lon]).addTo(map)
        .bindPopup(buildGmapsPopup(best.lat, best.lon, `${best.inliers} inliers | ${best.heading}°`))
        .openPopup();
    resultMarkers.push(marker);

    setStatus(`Best match: ${best.inliers} inliers at heading ${best.heading}°`);

    const clusters = (best.all_top_clusters && best.all_top_clusters.length) ? best.all_top_clusters : [best];
    resultsList.innerHTML = "";
    clusters.slice(0, 10).forEach((r, i) => {
        const row = document.createElement("div");
        row.className = "result-row";
        row.innerHTML = `<span>#${i + 1}</span><span>${r.inliers}</span><span>${r.lat.toFixed(6)}, ${r.lon.toFixed(6)}</span>`;
        row.addEventListener("click", () => {
            map.setView([r.lat, r.lon], 18);
            const m = L.marker([r.lat, r.lon]).addTo(map)
                .bindPopup(buildGmapsPopup(r.lat, r.lon, `${r.inliers} inliers`))
                .openPopup();
            resultMarkers.push(m);
        });
        resultsList.appendChild(row);
    });
}

// ═══════════════════════════════════════════════════════════════════
// COVERAGE (loaded once on startup / index switch)
// ═══════════════════════════════════════════════════════════════════

async function loadCoverage() {
    try {
        const res = await fetch("/api/coverage");
        const data = await res.json();
        if (coverageLayer) {
            map.removeLayer(coverageLayer);
            coverageLayer = null;
        }
        if (!data.points || data.points.length === 0) return;
        coverageLayer = L.layerGroup(
            data.points.map(p => L.circleMarker([p.lat, p.lon], {
                radius: 2, color: "#3b82f6", fillColor: "#3b82f6", fillOpacity: 0.6, weight: 0
            }))
        ).addTo(map);
    } catch (e) {
        console.error("Coverage load failed", e);
    }
}

// ═══════════════════════════════════════════════════════════════════
// MAP (radius-drag circle, unchanged from original)
// ═══════════════════════════════════════════════════════════════════

const map = L.map("map", { zoomControl: true }).setView(
    [Number(latInput.value), Number(lonInput.value)], 13
);

L.tileLayer("https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png", { maxZoom: 19 }).addTo(map);

let center = map.getCenter();
let radius = Number(radiusInput.value) * 1000;
let angle = 0;

const marker = L.marker(center, { draggable: true }).addTo(map);
const circle = L.circle(center, {
    radius, color: "#facc15", weight: 2, fillColor: "#facc15", fillOpacity: 0.15
}).addTo(map);

const handle = L.DomUtil.create("div", "radius-hitbox");
const knob = L.DomUtil.create("div", "radius-knob", handle);
map.getPanes().overlayPane.appendChild(handle);

function calculateHandlePosition() {
    const c = circle.getLatLng();
    const r = circle.getRadius();
    const earth = 6378137;
    const latOffset = (Math.sin(angle) * r / earth) * 180 / Math.PI;
    const lngOffset = (Math.cos(angle) * r / (earth * Math.cos(c.lat * Math.PI / 180))) * 180 / Math.PI;
    const handleLatLng = L.latLng(c.lat + latOffset, c.lng + lngOffset);
    return map.latLngToLayerPoint(handleLatLng);
}

function updateHandle() {
    const pos = calculateHandlePosition();
    handle.style.left = pos.x + "px";
    handle.style.top = pos.y + "px";
}

function moveCenter(position) {
    marker.setLatLng(position);
    circle.setLatLng(position);
    latInput.value = position.lat.toFixed(6);
    lonInput.value = position.lng.toFixed(6);
    updateHandle();
}

marker.on("drag", e => moveCenter(e.latlng));
map.on("click", e => moveCenter(e.latlng));

let resizing = false;

handle.addEventListener("mousedown", e => {
    resizing = true;
    L.DomEvent.stopPropagation(e);
    L.DomEvent.preventDefault(e);
});

map.on("mousemove", e => {
    if (!resizing) return;
    const c = circle.getLatLng();
    const distance = c.distanceTo(e.latlng);
    if (distance < 100) return;
    circle.setRadius(distance);
    angle = Math.atan2(e.latlng.lat - c.lat, e.latlng.lng - c.lng);
    radiusInput.value = (distance / 1000).toFixed(2);
    updateHandle();
});

map.on("mouseup", () => { resizing = false; });

radiusInput.addEventListener("input", () => {
    const meters = Number(radiusInput.value) * 1000;
    circle.setRadius(meters);
    updateHandle();
});

map.on("zoom move", updateHandle);
updateHandle();

// ═══════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════

(async function init() {
    try {
        const res = await fetch("/api/state");
        const data = await res.json();
        encoderVar = data.active_encoder || "megaloc";
        document.querySelectorAll('.selector[data-group="encoder"]').forEach(el => {
            const active = el.dataset.value === encoderVar;
            el.classList.toggle("active", active);
            el.querySelector(".selector-dot").classList.toggle("active", active);
        });
        if (data.coverage_center && data.coverage_center.lat != null) {
            latInput.value = data.coverage_center.lat;
            lonInput.value = data.coverage_center.lon;
            moveCenter(L.latLng(data.coverage_center.lat, data.coverage_center.lon));
            map.setView([data.coverage_center.lat, data.coverage_center.lon], 13);
        }
        // Radius intentionally NOT synced from the loaded index's manifest here --
        // an index's built coverage radius can be huge (it's the max distance to
        // any indexed point), which isn't a sensible default search radius. Keep
        // the 10km default; the radius only changes when the user explicitly
        // picks an index from the dropdown or drags the map circle.
        if (data.active_index_name) {
            setStatus(`Active index: ${data.active_index_name}`);
        } else {
            setStatus("No index selected. Choose one from the Index dropdown, or switch to Create Index mode.");
        }
    } catch (e) {
        setStatus("Could not reach API — is the server running?");
    }
    await refreshIndexList();
    await loadCoverage();
})();