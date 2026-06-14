const els = {
  backendUrl: document.getElementById("backendUrl"),
  checkHealthBtn: document.getElementById("checkHealthBtn"),
  healthPill: document.getElementById("healthPill"),
  transcribeForm: document.getElementById("transcribeForm"),
  audioFile: document.getElementById("audioFile"),
  userId: document.getElementById("userId"),
  enableCorrection: document.getElementById("enableCorrection"),
  enablePersonalization: document.getElementById("enablePersonalization"),
  batchBtn: document.getElementById("batchBtn"),
  userStatsBtn: document.getElementById("userStatsBtn"),
  demoScriptBtn: document.getElementById("demoScriptBtn"),
  confidenceValue: document.getElementById("confidenceValue"),
  latencyValue: document.getElementById("latencyValue"),
  backendValue: document.getElementById("backendValue"),
  modeValue: document.getElementById("modeValue"),
  originalText: document.getElementById("originalText"),
  correctedText: document.getElementById("correctedText"),
  audioWrap: document.getElementById("audioWrap"),
  correctedAudio: document.getElementById("correctedAudio"),
  timingBars: document.getElementById("timingBars"),
  runtimeJson: document.getElementById("runtimeJson"),
  consoleLog: document.getElementById("consoleLog"),
  judgeScore: document.getElementById("judgeScore"),
  scoreRadar: document.getElementById("scoreRadar"),
};

const stageOrder = [
  "preprocess_ms",
  "feature_extraction_ms",
  "asr_ms",
  "correction_ms",
  "personalization_ms",
  "tts_ms",
  "total_ms",
];

const appState = {
  lastTranscription: null,
  lastUserStats: null,
};

function getBaseUrl() {
  return (els.backendUrl.value || "http://localhost:8000").replace(/\/$/, "");
}

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

function log(message) {
  const now = new Date().toLocaleTimeString();
  const current = els.consoleLog.textContent;
  els.consoleLog.textContent = `[${now}] ${message}\n${current}`.trim();
}

function setHealth(ok, text) {
  els.healthPill.classList.remove("neutral", "ok", "bad");
  els.healthPill.classList.add(ok ? "ok" : "bad");
  els.healthPill.textContent = text;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function normalizeScore(value, min, max) {
  if (max === min) {
    return 0;
  }
  return clamp((value - min) / (max - min), 0, 1);
}

function getRadarMetrics() {
  const t = appState.lastTranscription;
  const stats = appState.lastUserStats;
  const confidence = Number(t?.confidence || 0);
  const latencyMs = Number(t?.stage_timings?.total_ms || 0);
  const correctionMode = t?.correction_metadata?.strategy?.mode || "-";
  const backend = t?.runtime?.asr?.gpu_backend || "cpu";
  const correctionCount = Number(stats?.total_corrections || 0);

  return [
    { label: "Accuracy", value: normalizeScore(confidence, 0.5, 1.0) },
    { label: "Speed", value: 1 - normalizeScore(latencyMs, 800, 8000) },
    { label: "Personalization", value: normalizeScore(correctionCount, 0, 30) },
    {
      label: "Adaptivity",
      value: correctionMode === "high_recovery" || correctionMode === "dysarthria_sensitive" ? 1 : 0.6,
    },
    { label: "Infra", value: backend === "rocm" ? 1 : backend === "cuda" ? 0.85 : 0.45 },
  ];
}

function drawRadar(metrics) {
  const canvas = els.scoreRadar;
  if (!canvas) {
    return;
  }

  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const cx = w / 2;
  const cy = h / 2 + 8;
  const maxRadius = Math.min(w, h) * 0.34;
  const levels = 4;
  const count = metrics.length;

  ctx.strokeStyle = "rgba(180, 200, 230, 0.22)";
  for (let level = 1; level <= levels; level += 1) {
    const r = (maxRadius * level) / levels;
    ctx.beginPath();
    for (let i = 0; i < count; i += 1) {
      const angle = -Math.PI / 2 + (i * 2 * Math.PI) / count;
      const x = cx + r * Math.cos(angle);
      const y = cy + r * Math.sin(angle);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(255,255,255,0.16)";
  ctx.fillStyle = "rgba(156, 178, 216, 0.85)";
  ctx.font = "12px IBM Plex Mono";
  metrics.forEach((metric, i) => {
    const angle = -Math.PI / 2 + (i * 2 * Math.PI) / count;
    const x = cx + maxRadius * Math.cos(angle);
    const y = cy + maxRadius * Math.sin(angle);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(x, y);
    ctx.stroke();
    const lx = cx + (maxRadius + 18) * Math.cos(angle);
    const ly = cy + (maxRadius + 18) * Math.sin(angle);
    ctx.fillText(metric.label, lx - 24, ly);
  });

  ctx.beginPath();
  metrics.forEach((metric, i) => {
    const angle = -Math.PI / 2 + (i * 2 * Math.PI) / count;
    const r = maxRadius * metric.value;
    const x = cx + r * Math.cos(angle);
    const y = cy + r * Math.sin(angle);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.fillStyle = "rgba(255, 138, 61, 0.32)";
  ctx.strokeStyle = "rgba(255, 209, 102, 0.94)";
  ctx.lineWidth = 2;
  ctx.fill();
  ctx.stroke();
}

function updateScorecard() {
  const metrics = getRadarMetrics();
  const avg = metrics.reduce((sum, metric) => sum + metric.value, 0) / metrics.length;
  els.judgeScore.textContent = String(Math.round(avg * 100));
  drawRadar(metrics);
}

function updateTimingBars(stageTimings = {}) {
  const total = Number(stageTimings.total_ms || 1);
  els.timingBars.innerHTML = "";

  stageOrder.forEach((key) => {
    if (!(key in stageTimings)) {
      return;
    }

    const value = Number(stageTimings[key] || 0);
    const percent = Math.max(2, Math.round((value / total) * 100));

    const row = document.createElement("div");
    row.className = "timing-row";
    row.innerHTML = `
      <span>${key.replace("_ms", "")}</span>
      <div class="timing-track"><div class="timing-fill" style="width:${percent}%"></div></div>
      <span>${value.toFixed(1)} ms</span>
    `;

    els.timingBars.appendChild(row);
  });
}

async function checkHealth() {
  const url = `${getBaseUrl()}/health`;
  try {
    log(`Checking backend health: ${url}`);
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`Health check failed (${res.status})`);
    }
    const data = await res.json();
    const backend = data?.runtime?.asr?.gpu_backend || "unknown";
    setHealth(true, `Healthy • backend=${backend}`);
    els.backendValue.textContent = backend;
    els.runtimeJson.textContent = pretty(data);
    log("Health check success.");
  } catch (err) {
    setHealth(false, `Backend unavailable • ${err.message}`);
    log(`Health check error: ${err.message}`);
  }
}

async function postTranscribeFromSelectedFile() {
  const file = els.audioFile.files[0];
  if (!file) {
    throw new Error("Select an audio file first.");
  }

  const params = new URLSearchParams({
    user_id: els.userId.value || "default",
    enable_correction: String(els.enableCorrection.checked),
    enable_personalization: String(els.enablePersonalization.checked),
  });

  const formData = new FormData();
  formData.append("file", file);

  const url = `${getBaseUrl()}/transcribe?${params.toString()}`;
  const res = await fetch(url, { method: "POST", body: formData });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Transcribe failed (${res.status}): ${text}`);
  }

  return res.json();
}

function renderTranscribeResult(data) {
  appState.lastTranscription = data;

  els.originalText.textContent = data.original_text || "";
  els.correctedText.textContent = data.corrected_text || "";
  els.confidenceValue.textContent = `${((data.confidence || 0) * 100).toFixed(1)}%`;
  els.latencyValue.textContent = `${(data.stage_timings?.total_ms || 0).toFixed(1)} ms`;
  els.backendValue.textContent = data.runtime?.asr?.gpu_backend || "-";
  els.modeValue.textContent = data.correction_metadata?.strategy?.mode || "-";

  if (data.corrected_audio_url) {
    els.audioWrap.classList.remove("hidden");
    els.correctedAudio.src = `${getBaseUrl()}${data.corrected_audio_url}`;
  } else {
    els.audioWrap.classList.add("hidden");
    els.correctedAudio.removeAttribute("src");
  }

  updateTimingBars(data.stage_timings || {});
  els.runtimeJson.textContent = pretty({
    runtime: data.runtime,
    features: data.features,
    correction_metadata: data.correction_metadata,
    stage_timings: data.stage_timings,
  });
  updateScorecard();
}

async function runTranscribe(event) {
  event.preventDefault();

  const btn = document.getElementById("transcribeBtn");
  btn.disabled = true;
  btn.textContent = "Processing...";

  try {
    const file = els.audioFile.files[0];
    if (!file) {
      throw new Error("Select an audio file first.");
    }
    log(`Running /transcribe for ${file.name}`);
    const data = await postTranscribeFromSelectedFile();
    renderTranscribeResult(data);

    log("Transcription completed.");
  } catch (err) {
    log(`Transcribe error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Demo Inference";
  }
}

async function runBatch() {
  const url = `${getBaseUrl()}/batch-process`;
  try {
    log("Running batch process...");
    const res = await fetch(url, { method: "POST" });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Batch failed (${res.status}): ${text}`);
    }
    const data = await res.json();
    els.runtimeJson.textContent = pretty(data);
    log(`Batch done: total=${data.total_files}, success=${data.successful}, failed=${data.failed}`);
  } catch (err) {
    log(`Batch error: ${err.message}`);
  }
}

async function fetchUserStats() {
  const userId = els.userId.value || "default";
  const url = `${getBaseUrl()}/user/${encodeURIComponent(userId)}/stats`;
  try {
    log(`Fetching user stats for ${userId}`);
    const res = await fetch(url);
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Stats failed (${res.status}): ${text}`);
    }
    const data = await res.json();
    appState.lastUserStats = data;
    els.runtimeJson.textContent = pretty({ user_stats: data });
    log(`User stats loaded: corrections=${data.total_corrections}`);
    updateScorecard();
  } catch (err) {
    log(`User stats error: ${err.message}`);
  }
}

async function runDemoScriptMode() {
  const btn = els.demoScriptBtn;
  btn.disabled = true;
  btn.textContent = "Running Script...";

  try {
    log("Demo Script Mode: Step 1/4 health check");
    await checkHealth();

    log("Demo Script Mode: Step 2/4 transcription");
    const data = await postTranscribeFromSelectedFile();
    renderTranscribeResult(data);

    log("Demo Script Mode: Step 3/4 batch process");
    await runBatch();

    log("Demo Script Mode: Step 4/4 user stats");
    await fetchUserStats();

    log("Demo Script Mode completed successfully.");
  } catch (err) {
    log(`Demo Script Mode error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Demo Script Mode";
  }
}

els.checkHealthBtn.addEventListener("click", checkHealth);
els.transcribeForm.addEventListener("submit", runTranscribe);
els.batchBtn.addEventListener("click", runBatch);
els.userStatsBtn.addEventListener("click", fetchUserStats);
els.demoScriptBtn.addEventListener("click", runDemoScriptMode);

checkHealth();
updateScorecard();
