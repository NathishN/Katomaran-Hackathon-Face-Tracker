/* =========================================================
   FaceTracker AI — Frontend Logic
   ========================================================= */

// ---- State ----
let currentView = "dashboard";
let currentEventFilter = "all";
let allEvents = [];
let uploadedSource = null;
let sse = null;

// ---- On load ----
document.addEventListener("DOMContentLoaded", () => {
  navigateTo("dashboard");
  fetchStats();
  connectSSE();
  setInterval(fetchStats, 5000);

  // Drag & drop
  const dz = document.getElementById("drop-zone");
  dz.addEventListener("dragover", e => { e.preventDefault(); dz.classList.add("dragover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", e => {
    e.preventDefault(); dz.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  });
});

// =========================================================
// NAVIGATION
// =========================================================
function navigateTo(view) {
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
  document.getElementById("view-" + view).classList.add("active");
  const navEl = document.getElementById("nav-" + view);
  if (navEl) navEl.classList.add("active");
  currentView = view;
  document.getElementById("page-title").textContent =
    { dashboard: "Dashboard", visitors: "Visitors", events: "Events Log", settings: "Settings" }[view] || view;

  if (view === "visitors") loadVisitors();
  if (view === "events")   loadEvents();
  if (view === "settings") loadConfig();

  // Mobile: close sidebar
  document.getElementById("sidebar").classList.remove("open");
}

document.querySelectorAll(".nav-item").forEach(item => {
  item.addEventListener("click", e => { e.preventDefault(); navigateTo(item.dataset.view); });
});
document.getElementById("hamburger").addEventListener("click", () => {
  document.getElementById("sidebar").classList.toggle("open");
});

// =========================================================
// SSE — Real-time events
// =========================================================
function connectSSE() {
  if (sse) { sse.close(); }
  sse = new EventSource("/api/stream");
  sse.onmessage = e => {
    const d = JSON.parse(e.data);
    if (d.type === "event") handleLiveEvent(d);
    if (d.type === "stats") updateStats(d);
    if (d.type === "stopped") {
      setRunning(false);
      fetchStats();
    }
    if (d.type === "error") {
      setRunning(false);
      showToast("Pipeline error: " + d.message, "error");
    }
  };
  sse.onerror = () => setTimeout(connectSSE, 3000);
}

function handleLiveEvent(d) {
  // Prepend to live feed
  const feed = document.getElementById("event-feed");
  const empty = feed.querySelector(".event-empty");
  if (empty) empty.remove();

  const item = document.createElement("div");
  item.className = "event-item";
  const thumbHTML = d.image
    ? `<img class="event-thumb" src="${d.image}" alt="crop" onclick="openLightbox('${d.image}')" />`
    : "";
  const timeStr = d.ts ? d.ts.split("T")[1].substring(0, 8) : "";
  item.innerHTML = `
    <span class="event-type-badge ${d.event_type}">${d.event_type}</span>
    <span class="event-face">${d.face}</span>
    <span class="event-time">${timeStr}</span>
    ${thumbHTML}
  `;
  feed.prepend(item);
  // Keep max 60 items
  while (feed.children.length > 60) feed.removeChild(feed.lastChild);

  // Flash stat card
  const cardId = d.event_type === "entry" ? "card-entries" : "card-exits";
  flashCard(cardId);
}

// =========================================================
// STATS
// =========================================================
function fetchStats() {
  fetch("/api/stats")
    .then(r => r.json())
    .then(d => {
      updateStats(d);
      if (d.running !== undefined) setRunning(d.running);
    })
    .catch(() => {});
}

function updateStats(d) {
  animateCount("stat-unique",  d.unique);
  animateCount("stat-entries", d.entry  !== undefined ? d.entry : (d.entries ?? 0));
  animateCount("stat-exits",   d.exit   !== undefined ? d.exit  : (d.exits ?? 0));
  animateCount("stat-active",  d.active ?? 0);
}

function animateCount(id, target) {
  const el = document.getElementById(id);
  if (!el) return;
  const current = parseInt(el.textContent) || 0;
  if (current === target) return;
  const diff = target - current;
  const steps = Math.min(Math.abs(diff), 20);
  let step = 0;
  const tick = setInterval(() => {
    step++;
    el.textContent = Math.round(current + diff * (step / steps));
    if (step >= steps) clearInterval(tick);
  }, 20);
}

function flashCard(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.style.boxShadow = id.includes("entries") ? "0 0 20px rgba(16,185,129,0.4)" : "0 0 20px rgba(239,68,68,0.4)";
  setTimeout(() => { el.style.boxShadow = ""; }, 600);
}

function setRunning(running) {
  const startBtn  = document.getElementById("btn-start");
  const stopBtn   = document.getElementById("btn-stop");
  const liveBadge = document.getElementById("live-badge");
  const pulseDot  = document.getElementById("pulse-dot");
  const statusTxt = document.getElementById("status-text");
  const videoBadge = document.getElementById("video-badge");
  const overlay   = document.getElementById("video-overlay");

  if (running) {
    startBtn.style.display  = "none";
    stopBtn.style.display   = "inline-flex";
    liveBadge.style.display = "inline-flex";
    pulseDot.classList.add("live");
    statusTxt.textContent = "Live";
    videoBadge.textContent = "ONLINE";
    videoBadge.classList.add("online");
    overlay.classList.add("hidden");
  } else {
    startBtn.style.display  = "inline-flex";
    stopBtn.style.display   = "none";
    liveBadge.style.display = "none";
    pulseDot.classList.remove("live");
    statusTxt.textContent = "Idle";
    videoBadge.textContent = "OFFLINE";
    videoBadge.classList.remove("online");
    overlay.classList.remove("hidden");
  }
}

// =========================================================
// PIPELINE CONTROL
// =========================================================
function openStartModal() {
  document.getElementById("modal-overlay").classList.add("open");
  document.getElementById("start-modal").classList.add("open");
}
function closeStartModal() {
  document.getElementById("modal-overlay").classList.remove("open");
  document.getElementById("start-modal").classList.remove("open");
}

function switchModalTab(btn, pane) {
  document.querySelectorAll(".modal-tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".modal-pane").forEach(p => p.classList.remove("active"));
  btn.classList.add("active");
  document.getElementById("pane-" + pane).classList.add("active");
}

async function launchPipeline() {
  const activeTab = document.querySelector(".modal-tab.active").textContent.trim().toLowerCase();
  let source = "";

  if (activeTab.includes("upload")) {
    if (!uploadedSource) {
      alert("Please upload a video file first.");
      return;
    }
    source = uploadedSource;
  } else if (activeTab.includes("rtsp") || activeTab.includes("url")) {
    source = document.getElementById("rtsp-url").value.trim();
    if (!source) { alert("Enter an RTSP/URL."); return; }
  } else if (activeTab.includes("webcam")) {
    source = document.getElementById("webcam-idx").value;
  }

  const fresh = document.getElementById("opt-fresh").checked;
  const launchBtn = document.getElementById("launch-btn");
  launchBtn.disabled = true; launchBtn.textContent = "Starting…";

  try {
    const res = await fetch("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source, fresh })
    });
    const data = await res.json();
    if (data.ok) {
      closeStartModal();
      setRunning(true);
    } else {
      alert("Error: " + (data.error || "Unknown error"));
    }
  } catch (e) {
    alert("Network error: " + e.message);
  } finally {
    launchBtn.disabled = false; launchBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg> Launch';
  }
}

async function stopPipeline() {
  await fetch("/api/stop", { method: "POST" });
  setRunning(false);
}

// =========================================================
// UPLOAD
// =========================================================
function handleFileSelect(input) {
  const file = input.files[0];
  if (file) uploadFile(file);
}

async function uploadFile(file) {
  const progress  = document.getElementById("upload-progress");
  const bar       = document.getElementById("progress-bar");
  const text      = document.getElementById("progress-text");
  const status    = document.getElementById("upload-status");

  progress.style.display = "block";
  status.textContent = ""; status.className = "upload-status";

  const formData = new FormData();
  formData.append("file", file);

  try {
    // Simulate progress with XHR for real progress events
    await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/upload");
      xhr.upload.onprogress = e => {
        if (e.lengthComputable) {
          const pct = Math.round(e.loaded / e.total * 100);
          bar.style.width = pct + "%";
          text.textContent = `Uploading… ${pct}%`;
        }
      };
      xhr.onload = () => {
        if (xhr.status === 200) {
          const d = JSON.parse(xhr.responseText);
          if (d.ok) {
            uploadedSource = d.source;
            status.textContent = `✓ Uploaded: ${file.name}`;
            status.className = "upload-status success";
            bar.style.width = "100%";
            text.textContent = "Upload complete";
            resolve();
          } else { reject(new Error(d.error)); }
        } else { reject(new Error("Upload failed")); }
      };
      xhr.onerror = () => reject(new Error("Network error"));
      xhr.send(formData);
    });
  } catch (e) {
    status.textContent = "✗ " + e.message;
    status.className = "upload-status error";
  }
}

// =========================================================
// VISITORS
// =========================================================
async function loadVisitors() {
  const grid = document.getElementById("visitor-grid");
  grid.innerHTML = '<div class="empty-state"><p>Loading…</p></div>';
  try {
    const data = await fetch("/api/visitors").then(r => r.json());
    if (!data.length) {
      grid.innerHTML = `<div class="empty-state">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
        <p>No visitors yet</p><small>Start the tracker on a video to detect faces</small></div>`;
      return;
    }
    grid.innerHTML = data.map(v => {
      const initials = (v.label || "?").replace("FACE-","").replace(/^0+/,"") || "?";
      const dateStr  = v.registered_at ? v.registered_at.split("T")[0] : "";
      const imgHTML  = v.image
        ? `<img class="visitor-avatar" src="${v.image}" alt="${v.label}" onclick="openLightbox('${v.image}')" />`
        : `<div class="visitor-avatar-placeholder">${initials}</div>`;
      return `<div class="visitor-card">
        ${imgHTML}
        <div class="visitor-label">${v.label || "Unknown"}</div>
        <div class="visitor-date">${dateStr}</div>
        <span class="visitor-visits">${v.visit_count} visit${v.visit_count !== 1 ? "s" : ""}</span>
      </div>`;
    }).join("");
  } catch (e) {
    grid.innerHTML = `<div class="empty-state"><p>Error loading visitors</p></div>`;
  }
}

// =========================================================
// EVENTS TABLE
// =========================================================
async function loadEvents() {
  try {
    allEvents = await fetch("/api/events?limit=200").then(r => r.json());
    renderEvents();
  } catch (e) {
    document.getElementById("events-tbody").innerHTML =
      '<tr><td colspan="7" style="text-align:center;color:var(--text-muted);padding:24px">Error loading events</td></tr>';
  }
}

function filterEvents(btn, filter) {
  document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  currentEventFilter = filter;
  renderEvents();
}

function renderEvents() {
  const filtered = currentEventFilter === "all"
    ? allEvents
    : allEvents.filter(e => e.event_type === currentEventFilter);

  const tbody = document.getElementById("events-tbody");
  if (!filtered.length) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--text-muted);padding:32px">No events found</td></tr>';
    return;
  }
  tbody.innerHTML = filtered.map(e => {
    const typeClass = e.event_type === "entry" ? "entry" : "exit";
    const timeStr   = e.timestamp ? e.timestamp.replace("T", " ").substring(0, 19) : "";
    const vt = e.video_time_sec != null ? e.video_time_sec.toFixed(2) + "s" : "—";
    const imgHTML = e.image
      ? `<img src="${e.image}" style="width:36px;height:36px;border-radius:6px;object-fit:cover;cursor:pointer;border:1px solid var(--glass-border)" onclick="openLightbox('${e.image}')" alt="crop" />`
      : "—";
    return `<tr>
      <td class="mono" style="color:var(--purple-light)">${e.label || "—"}</td>
      <td><span class="event-type-badge ${typeClass}">${e.event_type}</span></td>
      <td style="color:var(--text-secondary);font-size:0.75rem">${timeStr}</td>
      <td class="mono">${e.frame_idx ?? "—"}</td>
      <td class="mono">${vt}</td>
      <td class="mono">${e.track_id ?? "—"}</td>
      <td>${imgHTML}</td>
    </tr>`;
  }).join("");
}

// =========================================================
// CONFIG
// =========================================================
async function loadConfig() {
  try {
    const cfg = await fetch("/api/config").then(r => r.json());
    const d = cfg.detector    || {};
    const r = cfg.recognition || {};
    const t = cfg.tracking    || {};
    setVal("cfg-confidence", d.confidence  ?? 0.5);
    setVal("cfg-min-face",   d.min_face_size_px ?? 32);
    setVal("cfg-skip",       cfg.detection_skip_frames ?? 2);
    setVal("cfg-sim",        r.similarity_threshold ?? 0.4);
    setVal("cfg-sharp",      r.min_sharpness ?? 10.0);
    setVal("cfg-attempts",   r.max_attempts_per_track ?? 8);
    setVal("cfg-iou",        t.iou_threshold ?? 0.25);
    setVal("cfg-maxage",     t.max_age_frames ?? 45);
    setVal("cfg-minhits",    t.min_hits ?? 2);
    updateRangeLabel(document.getElementById("cfg-confidence"), "lbl-confidence");
    updateRangeLabel(document.getElementById("cfg-skip"), "lbl-skip");
    updateRangeLabel(document.getElementById("cfg-sim"), "lbl-sim");
    updateRangeLabel(document.getElementById("cfg-iou"), "lbl-iou");
  } catch (e) {}
}

async function saveConfig() {
  try {
    const current = await fetch("/api/config").then(r => r.json());
    const merged = {
      ...current,
      detection_skip_frames: parseInt(getVal("cfg-skip")),
      detector: {
        ...(current.detector || {}),
        confidence: parseFloat(getVal("cfg-confidence")),
        min_face_size_px: parseInt(getVal("cfg-min-face")),
      },
      recognition: {
        ...(current.recognition || {}),
        similarity_threshold: parseFloat(getVal("cfg-sim")),
        min_sharpness: parseFloat(getVal("cfg-sharp")),
        max_attempts_per_track: parseInt(getVal("cfg-attempts")),
      },
      tracking: {
        ...(current.tracking || {}),
        iou_threshold: parseFloat(getVal("cfg-iou")),
        max_age_frames: parseInt(getVal("cfg-maxage")),
        min_hits: parseInt(getVal("cfg-minhits")),
      }
    };
    const res = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(merged)
    });
    const d = await res.json();
    showToast(d.ok ? "✓ Config saved successfully" : "✗ Save failed: " + d.error, d.ok ? "success" : "error");
  } catch (e) {
    showToast("✗ Error: " + e.message, "error");
  }
}

function setVal(id, val) {
  const el = document.getElementById(id);
  if (el) el.value = val;
}
function getVal(id) {
  const el = document.getElementById(id);
  return el ? el.value : "";
}
function updateRangeLabel(input, labelId) {
  const lbl = document.getElementById(labelId);
  if (lbl) lbl.textContent = parseFloat(input.value).toFixed(input.step < 1 ? 2 : 0);
}
function showToast(msg, type) {
  const t = document.getElementById("config-toast");
  t.textContent = msg;
  t.style.color = type === "success" ? "var(--green)" : "var(--red)";
  setTimeout(() => { t.textContent = ""; }, 4000);
}

// =========================================================
// LIGHTBOX
// =========================================================
function openLightbox(src) {
  document.getElementById("lightbox-img").src = src;
  document.getElementById("lightbox").classList.add("open");
}
function closeLightbox() {
  document.getElementById("lightbox").classList.remove("open");
}

// =========================================================
// UTILS
// =========================================================
function clearEvents() {
  document.getElementById("event-feed").innerHTML = `<div class="event-empty">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2"><path d="M22 2 11 13"/><path d="M22 2 15 22 11 13 2 9l20-7z"/></svg>
    <p>Events will appear here in real-time</p></div>`;
}
