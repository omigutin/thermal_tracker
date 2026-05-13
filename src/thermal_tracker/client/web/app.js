const TARGET_WIDTH = 512;
const TARGET_HEIGHT = 640;
const VIDEO_SUFFIXES = [".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv", ".m4v", ".webm"];
const STATE_KEY = "thermal_tracker_web_client_state_v1";

const filePicker = document.getElementById("filePicker");
const folderPicker = document.getElementById("folderPicker");
const playlistElement = document.getElementById("playlist");
const playlistStatus = document.getElementById("playlistStatus");
const presetSelect = document.getElementById("presetSelect");
const modelPathField = document.getElementById("modelPathField");
const modelPathInput = document.getElementById("modelPathInput");
const chooseModelButton = document.getElementById("chooseModel");
const frameIntervalInput = document.getElementById("frameInterval");
const sequentialCheckbox = document.getElementById("sequentialPlayback");
const recordCheckbox = document.getElementById("recordOutput");
const recordDownload = document.getElementById("recordDownload");
const displayCanvas = document.getElementById("displayCanvas");
const emptyOverlay = document.getElementById("emptyOverlay");
const videoStage = document.querySelector(".video-stage");
const sourceVideo = document.getElementById("sourceVideo");
const previousVideoButton = document.getElementById("previousVideo");
const stepBackButton = document.getElementById("stepBack");
const playPauseButton = document.getElementById("playPause");
const stepForwardButton = document.getElementById("stepForward");
const stopVideoButton = document.getElementById("stopVideo");
const nextVideoButton = document.getElementById("nextVideo");
const timeline = document.getElementById("timeline");
const timeLabel = document.getElementById("timeLabel");
const sendStatus = document.getElementById("sendStatus");
const summary = document.getElementById("summary");
const metrics = document.getElementById("metrics");
const modelDialog = document.getElementById("modelDialog");
const dialogModelPath = document.getElementById("dialogModelPath");
const modelFilePicker = document.getElementById("modelFilePicker");
const cancelModelDialog = document.getElementById("cancelModelDialog");
const applyModelDialog = document.getElementById("applyModelDialog");

const displayContext = displayCanvas.getContext("2d");
const uploadCanvas = document.createElement("canvas");
uploadCanvas.width = TARGET_WIDTH;
uploadCanvas.height = TARGET_HEIGHT;
const uploadContext = uploadCanvas.getContext("2d", { willReadFrequently: true });

let playlist = [];
let selectedIndex = -1;
let objectUrl = "";
let sendTimer = 0;
let metricsTimer = 0;
let frameTimer = 0;
let timelineTimer = 0;
let frameId = 0;
let sentFrames = 0;
let isSending = false;
let lastMetrics = null;
let lastDisplayedFrame = false;
let presetScenarios = new Map();
let presetMetadata = new Map();
let modelPathsByPreset = {};
let lastConfiguredSignature = "";
let lastUploadContentRect = { x: 0, y: 0, width: TARGET_WIDTH, height: TARGET_HEIGHT };
let displayedContentRect = lastUploadContentRect;
let serverRecordingActive = false;
let recordingArmed = false;
let playbackActive = false;

restoreState();
loadPresets();
startMetricsLoops();
refreshControls();

function restoreState() {
  try {
    const state = JSON.parse(localStorage.getItem(STATE_KEY) || "{}");
    if (typeof state.frameIntervalMs === "number") {
      frameIntervalInput.value = String(Math.max(5, Math.min(1000, Math.round(state.frameIntervalMs))));
    }
    sequentialCheckbox.checked = Boolean(state.sequentialPlayback);
    if (state.modelPathsByPreset && typeof state.modelPathsByPreset === "object") {
      modelPathsByPreset = state.modelPathsByPreset;
    }
  } catch {
    // Локальное состояние не критично.
  }
}

function saveState() {
  const state = {
    presetName: presetSelect.value,
    frameIntervalMs: currentFrameIntervalMs(),
    sequentialPlayback: sequentialCheckbox.checked,
    modelPathsByPreset,
  };
  localStorage.setItem(STATE_KEY, JSON.stringify(state));
}

async function loadPresets() {
  try {
    const response = await fetch("/api/presets", { cache: "no-store" });
    const data = await response.json();
    const presets = Array.isArray(data.presets) ? data.presets : [];
    renderPresets(presets);
  } catch {
    renderPresets([
      { name: "opencv_general", title: "OpenCV General", description: "Базовый OpenCV-пресет.", has_neural: false },
      { name: "opencv_small_target", title: "Small Target", description: "Для маленьких целей.", has_neural: false },
      { name: "opencv_clutter", title: "OpenCV Clutter", description: "Для сложного фона.", has_neural: false },
      {
        name: "yolo_general",
        title: "YOLO General",
        description: "Нейросетевой ручной режим.",
        has_neural: true,
        model_path: "models/model.pt",
      },
      {
        name: "yolo_auto",
        title: "YOLO Auto",
        description: "Нейросетевой авто-режим.",
        has_neural: true,
        model_path: "models/model.pt",
      },
    ]);
  }
}

function renderPresets(presets) {
  const savedPreset = readSavedPreset();
  presetScenarios = new Map();
  presetMetadata = new Map();
  presetSelect.innerHTML = "";
  for (const preset of presets) {
    presetScenarios.set(preset.name, preset.scenario || scenarioForPresetName(preset.name));
    presetMetadata.set(preset.name, preset);
    if (preset.has_neural && preset.model_path && !modelPathsByPreset[preset.name]) {
      modelPathsByPreset[preset.name] = preset.model_path;
    }
    const option = document.createElement("option");
    option.value = preset.name;
    option.textContent = preset.title || preset.name;
    option.title = preset.description || preset.tooltip || preset.name;
    presetSelect.appendChild(option);
  }
  if (savedPreset && [...presetSelect.options].some((option) => option.value === savedPreset)) {
    presetSelect.value = savedPreset;
  }
  syncModelControls();
  saveState();
}

function scenarioForPresetName(presetName) {
  if (presetName === "yolo_general") {
    return "nn_manual";
  }
  if (presetName === "yolo_auto") {
    return "nn_auto";
  }
  return "opencv_manual";
}

function readSavedPreset() {
  try {
    return JSON.parse(localStorage.getItem(STATE_KEY) || "{}").presetName || "";
  } catch {
    return "";
  }
}

function currentPresetMetadata() {
  return presetMetadata.get(presetSelect.value) || {};
}

function isCurrentPresetNeural() {
  const preset = currentPresetMetadata();
  return Boolean(preset.has_neural) || scenarioForPresetName(presetSelect.value).startsWith("nn_");
}

function currentModelPath() {
  const presetName = presetSelect.value;
  const preset = currentPresetMetadata();
  return modelPathsByPreset[presetName] || preset.model_path || "";
}

function syncModelControls() {
  const needsModel = isCurrentPresetNeural();
  modelPathField.hidden = !needsModel;
  modelPathInput.value = needsModel ? currentModelPath() : "";
}

function openModelDialog() {
  if (!isCurrentPresetNeural()) {
    return;
  }
  dialogModelPath.value = currentModelPath();
  modelDialog.hidden = false;
  window.setTimeout(() => dialogModelPath.focus(), 0);
}

function closeModelDialog() {
  modelDialog.hidden = true;
  modelFilePicker.value = "";
}

async function applyModelDialogValue() {
  const modelPath = dialogModelPath.value.trim();
  if (modelPath) {
    modelPathsByPreset[presetSelect.value] = modelPath;
  }
  syncModelControls();
  saveState();
  closeModelDialog();
  await configureRuntime({ force: true });
  await sendCurrentFrame();
}

function updateModelPathFromInput() {
  const modelPath = modelPathInput.value.trim();
  if (!modelPath) {
    return;
  }
  modelPathsByPreset[presetSelect.value] = modelPath;
  saveState();
}

function loadFiles(fileList) {
  const files = Array.from(fileList || [])
    .filter((file) => VIDEO_SUFFIXES.some((suffix) => file.name.toLowerCase().endsWith(suffix)))
    .sort((left, right) => displayName(left).localeCompare(displayName(right), "ru", { numeric: true }));

  playlist = files;
  selectedIndex = files.length > 0 ? 0 : -1;
  closeObjectUrl();
  renderPlaylist();
  refreshControls();
  if (selectedIndex >= 0) {
    loadSelectedVideo({ autoplay: false, sendFirstFrame: true });
  } else {
    setEmpty("Видео не выбраны.");
  }
}

function displayName(file) {
  return file.webkitRelativePath || file.name;
}

function renderPlaylist() {
  playlistElement.innerHTML = "";
  playlist.forEach((file, index) => {
    const item = document.createElement("li");
    item.textContent = displayName(file);
    item.title = displayName(file);
    item.className = index === selectedIndex ? "active" : "";
    item.addEventListener("click", () => {
      selectedIndex = index;
      loadSelectedVideo({ autoplay: false, sendFirstFrame: true });
      renderPlaylist();
      refreshControls();
    });
    playlistElement.appendChild(item);
  });

  if (playlist.length === 0) {
    playlistStatus.textContent = "Видео не выбраны.";
  } else {
    playlistStatus.textContent = `Файлов в плейлисте: ${playlist.length}`;
  }
}

function closeObjectUrl() {
  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
    objectUrl = "";
  }
}

async function loadSelectedVideo({ autoplay, sendFirstFrame }) {
  stopSender();
  playbackActive = false;
  if (selectedIndex < 0 || selectedIndex >= playlist.length) {
    setEmpty("Выберите видео.");
    return;
  }

  closeObjectUrl();
  const file = playlist[selectedIndex];
  objectUrl = URL.createObjectURL(file);
  sourceVideo.src = objectUrl;
  sourceVideo.load();
  setEmpty(`Загрузка: ${file.name}`);

  await waitForVideoEvent("loadedmetadata");
  // Синхронно подтягиваем content rect и canvas под новый ролик ДО того, как
  // пользователь успеет кликнуть. Закрывает race condition между переключением
  // ролика и первым image.onload в refreshFrame.
  applyContentRectToCanvas();
  timeline.max = Number.isFinite(sourceVideo.duration) ? String(sourceVideo.duration) : "0";
  timeline.value = "0";
  sourceVideo.currentTime = 0;
  await waitForSeekIfNeeded();
  setEmpty("");
  renderPlaylist();
  refreshTimeline();

  if (sendFirstFrame) {
    await configureRuntime({ force: true });
    await sendCurrentFrame();
  }
  if (autoplay) {
    await startPlayback();
  }
}

function waitForVideoEvent(eventName) {
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      sourceVideo.removeEventListener(eventName, onDone);
      sourceVideo.removeEventListener("error", onError);
    };
    const onDone = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error("Не удалось открыть видео."));
    };
    sourceVideo.addEventListener(eventName, onDone, { once: true });
    sourceVideo.addEventListener("error", onError, { once: true });
  });
}

function waitForSeekIfNeeded() {
  if (!sourceVideo.seeking) {
    return Promise.resolve();
  }
  return waitForVideoEvent("seeked");
}

function currentRuntimeSignature() {
  const presetName = presetSelect.value || "opencv_general";
  const scenario = presetScenarios.get(presetName) || scenarioForPresetName(presetName);
  const modelPath = isCurrentPresetNeural() ? currentModelPath().trim() : "";
  return JSON.stringify({ presetName, scenario, modelPath });
}

async function configureRuntime({ force = false } = {}) {
  saveState();
  const presetName = presetSelect.value || "opencv_general";
  const scenario = presetScenarios.get(presetName) || scenarioForPresetName(presetName);
  const signature = currentRuntimeSignature();
  if (!force && signature === lastConfiguredSignature) {
    return;
  }
  const payload = { preset_name: presetName, scenario };
  if (isCurrentPresetNeural()) {
    const modelPath = currentModelPath().trim();
    if (modelPath) {
      payload.model_path = modelPath;
    }
  }
  await fetch("/api/commands/configure", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  lastConfiguredSignature = signature;
}

async function startPlayback() {
  if (selectedIndex < 0) {
    return;
  }
  await configureRuntime();
  try {
    await sourceVideo.play();
  } catch (error) {
    metrics.textContent = `Браузер не разрешил воспроизведение: ${error}`;
    return;
  }
  if (recordCheckbox.checked) {
    try {
      await startRecording();
    } catch (error) {
      recordCheckbox.checked = false;
      recordingArmed = false;
      serverRecordingActive = false;
      recordDownload.hidden = false;
      recordDownload.textContent = "Ошибка записи";
      metrics.textContent = String(error);
    }
  }
  playbackActive = true;
  startSender();
  startTimelineLoop();
  refreshControls();
}

function pausePlayback() {
  playbackActive = false;
  sourceVideo.pause();
  stopSender();
  refreshControls();
}

async function stopPlayback() {
  if (selectedIndex < 0) {
    return;
  }
  playbackActive = false;
  sourceVideo.pause();
  stopSender();
  stopTimelineLoop();
  sourceVideo.currentTime = 0;
  await waitForSeekIfNeeded();
  refreshTimeline();
  refreshControls();
  await sendCurrentFrame();
  stopRecordingAndSave();
}

function startSender() {
  stopSender();
  sendCurrentFrame();
  sendTimer = window.setInterval(sendCurrentFrame, currentFrameIntervalMs());
}

function stopSender() {
  if (sendTimer) {
    window.clearInterval(sendTimer);
    sendTimer = 0;
  }
}

function currentFrameIntervalMs() {
  const value = Number(frameIntervalInput.value);
  if (!Number.isFinite(value)) {
    return 40;
  }
  return Math.max(5, Math.min(1000, Math.round(value)));
}

async function sendCurrentFrame() {
  if (isSending || selectedIndex < 0 || sourceVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return;
  }

  isSending = true;
  try {
    drawSourceVideoToUploadCanvas();
    const imageData = uploadContext.getImageData(0, 0, TARGET_WIDTH, TARGET_HEIGHT);
    const raw = rgbaToGray(imageData.data);
    frameId += 1;
    const timestampNs = BigInt(Date.now()) * 1000000n;
    const params = new URLSearchParams({
      frame_id: String(frameId),
      timestamp_ns: timestampNs.toString(),
      content_x: String(lastUploadContentRect.x),
      content_y: String(lastUploadContentRect.y),
      content_width: String(lastUploadContentRect.width),
      content_height: String(lastUploadContentRect.height),
      source_width: String(sourceVideo.videoWidth || 0),
      source_height: String(sourceVideo.videoHeight || 0),
      source_name: selectedIndex >= 0 ? displayName(playlist[selectedIndex]) : "",
      preset_name: presetSelect.value || "",
    });
    const response = await fetch(`/api/frames/raw-y8?${params.toString()}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: raw,
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    sentFrames += 1;
    sendStatus.textContent = `кадров: ${sentFrames}`;
    refreshFrame();
  } catch (error) {
    metrics.textContent = String(error);
  } finally {
    isSending = false;
  }
}

function updateLastUploadContentRect() {
  // Единственный источник истины для координатного пространства, в котором
  // живёт сервер. Считается синхронно от текущего source video, не зависит от
  // image.onload и от isSending guard в sendCurrentFrame.
  const sourceWidth = sourceVideo.videoWidth || TARGET_WIDTH;
  const sourceHeight = sourceVideo.videoHeight || TARGET_HEIGHT;
  lastUploadContentRect = containRect(sourceWidth, sourceHeight, TARGET_WIDTH, TARGET_HEIGHT);
}

function applyContentRectToCanvas() {
  // Синхронно подтягивает display state под lastUploadContentRect. Нужен,
  // чтобы клик после переключения ролика мапился по актуальным размерам,
  // а не по stale значениям от прошлого ролика или от init.
  updateLastUploadContentRect();
  displayedContentRect = { ...lastUploadContentRect };
  if (
    displayCanvas.width !== lastUploadContentRect.width ||
    displayCanvas.height !== lastUploadContentRect.height
  ) {
    displayCanvas.width = lastUploadContentRect.width;
    displayCanvas.height = lastUploadContentRect.height;
  }
  // Очищаем canvas, чтобы кадр предыдущего ролика не вводил в заблуждение
  // в окне до первого latest.jpg.
  displayContext.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
  fitDisplayCanvasToStage();
}

function drawSourceVideoToUploadCanvas() {
  const sourceWidth = sourceVideo.videoWidth || TARGET_WIDTH;
  const sourceHeight = sourceVideo.videoHeight || TARGET_HEIGHT;
  uploadContext.fillStyle = "black";
  uploadContext.fillRect(0, 0, TARGET_WIDTH, TARGET_HEIGHT);

  updateLastUploadContentRect();
  const rect = lastUploadContentRect;
  uploadContext.imageSmoothingEnabled = true;
  uploadContext.drawImage(
    sourceVideo,
    0,
    0,
    sourceWidth,
    sourceHeight,
    rect.x,
    rect.y,
    rect.width,
    rect.height,
  );
}

function containRect(sourceWidth, sourceHeight, targetWidth, targetHeight) {
  const scale = Math.min(targetWidth / sourceWidth, targetHeight / sourceHeight);
  const width = Math.max(1, Math.round(sourceWidth * scale));
  const height = Math.max(1, Math.round(sourceHeight * scale));
  return {
    x: Math.floor((targetWidth - width) / 2),
    y: Math.floor((targetHeight - height) / 2),
    width,
    height,
  };
}

function rgbaToGray(rgba) {
  const gray = new Uint8Array(TARGET_WIDTH * TARGET_HEIGHT);
  for (let src = 0, dst = 0; src < rgba.length; src += 4, dst += 1) {
    gray[dst] = Math.round(rgba[src] * 0.299 + rgba[src + 1] * 0.587 + rgba[src + 2] * 0.114);
  }
  return gray;
}

async function refreshFrame() {
  try {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => {
      displayedContentRect = clampRect(lastUploadContentRect, image.naturalWidth, image.naturalHeight);
      if (
        displayCanvas.width !== displayedContentRect.width ||
        displayCanvas.height !== displayedContentRect.height
      ) {
        displayCanvas.width = displayedContentRect.width;
        displayCanvas.height = displayedContentRect.height;
      }
      displayContext.drawImage(
        image,
        displayedContentRect.x,
        displayedContentRect.y,
        displayedContentRect.width,
        displayedContentRect.height,
        0,
        0,
        displayedContentRect.width,
        displayedContentRect.height,
      );
      fitDisplayCanvasToStage();
      lastDisplayedFrame = true;
      emptyOverlay.hidden = true;
    };
    image.onerror = () => {
      if (!lastDisplayedFrame) {
        emptyOverlay.hidden = false;
      }
    };
    image.src = `/api/frame/latest.jpg?overlay=true&t=${Date.now()}`;
  } catch {
    if (!lastDisplayedFrame) {
      emptyOverlay.hidden = false;
    }
  }
}

function clampRect(rect, width, height) {
  const x = Math.min(Math.max(0, rect.x || 0), Math.max(0, width - 1));
  const y = Math.min(Math.max(0, rect.y || 0), Math.max(0, height - 1));
  return {
    x,
    y,
    width: Math.min(Math.max(1, rect.width || width), width - x),
    height: Math.min(Math.max(1, rect.height || height), height - y),
  };
}

function fitDisplayCanvasToStage() {
  if (!videoStage || displayCanvas.width <= 0 || displayCanvas.height <= 0) {
    return;
  }
  const stageRect = videoStage.getBoundingClientRect();
  if (stageRect.width <= 0 || stageRect.height <= 0) {
    return;
  }

  const scale = Math.min(stageRect.width / displayCanvas.width, stageRect.height / displayCanvas.height);
  const cssWidth = Math.max(1, Math.floor(displayCanvas.width * scale));
  const cssHeight = Math.max(1, Math.floor(displayCanvas.height * scale));
  displayCanvas.style.aspectRatio = `${displayCanvas.width} / ${displayCanvas.height}`;
  displayCanvas.style.width = `${cssWidth}px`;
  displayCanvas.style.height = `${cssHeight}px`;
}

async function refreshMetrics() {
  try {
    const response = await fetch("/api/metrics", { cache: "no-store" });
    const data = await response.json();
    lastMetrics = data;
    renderSummary(data);
    metrics.textContent = JSON.stringify(data, null, 2);
    refreshRecordingStatus();
  } catch (error) {
    metrics.textContent = String(error);
  }
}

function startMetricsLoops() {
  frameTimer = window.setInterval(refreshFrame, 160);
  metricsTimer = window.setInterval(refreshMetrics, 500);
  refreshFrame();
  refreshMetrics();
}

function valueOrDash(value) {
  return value === undefined || value === null || value === "" ? "-" : value;
}

function numberOrDash(value, digits = 1) {
  return typeof value === "number" ? value.toFixed(digits) : "-";
}

function renderSummary(data) {
  const result = data.result || {};
  const snapshot = result.snapshot || {};
  const frameData = data.frame || {};
  const ingress = data.ingress || {};
  const lag = data.lag || {};
  const items = [
    ["frame_id", valueOrDash(frameData.frame_id)],
    ["ingress_fps", numberOrDash(ingress.recent_fps, 1)],
    ["processed_frame", valueOrDash(lag.latest_result_frame_id)],
    ["frame_id_lag", valueOrDash(lag.frame_id_lag)],
    ["state", valueOrDash(snapshot.state)],
    ["track_id", valueOrDash(snapshot.track_id)],
    ["processing_ms", numberOrDash(result.processing_ms, 2)],
    ["source_to_result_ms", numberOrDash(result.source_to_result_ms, 2)],
    ["ingress_to_runtime_ms", numberOrDash(result.ingress_to_runtime_ms, 2)],
    ["result_age_ms", numberOrDash(lag.latest_result_age_ms, 1)],
    ["frame_age_ms", numberOrDash(lag.latest_frame_age_ms, 1)],
  ];
  summary.innerHTML = items
    .map(([name, value]) => `<div class="metric"><span>${name}</span><span>${value}</span></div>`)
    .join("");
}

function refreshTimeline() {
  const duration = Number.isFinite(sourceVideo.duration) ? sourceVideo.duration : 0;
  timeline.max = String(duration);
  timeline.value = String(sourceVideo.currentTime || 0);
  timeLabel.textContent = `${formatSeconds(sourceVideo.currentTime || 0)} / ${formatSeconds(duration)}`;
}

function startTimelineLoop() {
  stopTimelineLoop();
  timelineTimer = window.setInterval(refreshTimeline, 120);
}

function stopTimelineLoop() {
  if (timelineTimer) {
    window.clearInterval(timelineTimer);
    timelineTimer = 0;
  }
}

function formatSeconds(seconds) {
  const safeSeconds = Math.max(0, Math.round(seconds || 0));
  const minutes = Math.floor(safeSeconds / 60);
  const rest = safeSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
}

function refreshControls() {
  const hasVideo = selectedIndex >= 0;
  previousVideoButton.disabled = !hasVideo || selectedIndex <= 0;
  nextVideoButton.disabled = !hasVideo || selectedIndex >= playlist.length - 1;
  stepBackButton.disabled = !hasVideo;
  stepForwardButton.disabled = !hasVideo;
  stopVideoButton.disabled = !hasVideo;
  playPauseButton.disabled = !hasVideo;
  playPauseButton.textContent = playbackActive ? "⏸" : "▶";
}

function setEmpty(message) {
  emptyOverlay.textContent = message || "";
  emptyOverlay.hidden = !message;
}

async function goToVideo(index, { autoplay = false } = {}) {
  if (index < 0 || index >= playlist.length) {
    return;
  }
  selectedIndex = index;
  await loadSelectedVideo({ autoplay, sendFirstFrame: true });
  renderPlaylist();
  refreshControls();
}

async function stepByFrame(direction) {
  if (selectedIndex < 0) {
    return;
  }
  pausePlayback();
  const stepSeconds = currentFrameIntervalMs() / 1000;
  const nextTime = Math.max(0, Math.min(sourceVideo.duration || 0, sourceVideo.currentTime + direction * stepSeconds));
  sourceVideo.currentTime = nextTime;
  await waitForSeekIfNeeded();
  refreshTimeline();
  await sendCurrentFrame();
}

async function resetTrack() {
  await fetch("/api/commands/reset", { method: "POST" });
  refreshMetrics();
}

async function handleVideoEnded() {
  playbackActive = false;
  stopSender();
  stopTimelineLoop();
  refreshTimeline();
  refreshControls();
  if (sequentialCheckbox.checked && selectedIndex + 1 < playlist.length) {
    await goToVideo(selectedIndex + 1, { autoplay: true });
    return;
  }
  stopRecordingAndSave();
}

async function toggleRecording() {
  try {
    if (recordCheckbox.checked) {
      recordingArmed = true;
      recordDownload.hidden = false;
      recordDownload.removeAttribute("href");
      if (sourceVideo.paused) {
        recordDownload.textContent = "Запись начнётся при воспроизведении";
      } else {
        await startRecording();
      }
    } else {
      await stopRecording();
    }
  } catch (error) {
    recordCheckbox.checked = false;
    recordingArmed = false;
    serverRecordingActive = false;
    recordDownload.hidden = false;
    recordDownload.textContent = "Ошибка записи";
    metrics.textContent = String(error);
  }
}

async function startRecording() {
  if (serverRecordingActive) {
    return;
  }
  if (!recordCheckbox.checked) {
    recordingArmed = false;
    return;
  }
  if (selectedIndex < 0) {
    recordCheckbox.checked = false;
    recordingArmed = false;
    return;
  }
  drawSourceVideoToUploadCanvas();
  const baseName = recordingBaseName();
  const response = await fetch("/api/recording/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      base_name: baseName,
      fps: Math.max(1, Math.round(1000 / currentFrameIntervalMs())),
      content_rect: lastUploadContentRect,
      source_name: selectedIndex >= 0 ? displayName(playlist[selectedIndex]) : "",
      preset_name: presetSelect.value || "",
    }),
  });
  if (!response.ok) {
    recordCheckbox.checked = false;
    recordingArmed = false;
    throw new Error(await response.text());
  }
  const state = await response.json();
  serverRecordingActive = true;
  recordingArmed = false;
  recordDownload.hidden = false;
  recordDownload.removeAttribute("href");
  recordDownload.textContent = `Идёт запись: ${state.video_path || baseName}`;
}

async function stopRecording() {
  recordingArmed = false;
  if (!serverRecordingActive) {
    recordDownload.hidden = true;
    return;
  }
  const response = await fetch("/api/recording/stop", { method: "POST" });
  serverRecordingActive = false;
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const state = await response.json();
  recordDownload.hidden = false;
  recordDownload.textContent = state.video_path
    ? `Сохранено: ${state.video_path} + ${state.jsonl_path}`
    : "Запись остановлена";
}

function stopRecordingAndSave() {
  if (!recordCheckbox.checked && !serverRecordingActive && !recordingArmed) {
    return;
  }
  recordCheckbox.checked = false;
  recordingArmed = false;
  stopRecording().catch((error) => {
    recordDownload.hidden = false;
    recordDownload.textContent = "Ошибка остановки записи";
    metrics.textContent = String(error);
  });
}

function recordingBaseName() {
  const sourceName = selectedIndex >= 0 ? displayName(playlist[selectedIndex]) : "web";
  const cleanSource = sourceName.split(/[\\/]/).pop().replace(/\.[^.]+$/, "");
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  return `${presetSelect.value || "preset"}_${cleanSource}_${timestamp}`;
}

async function refreshRecordingStatus() {
  if (!serverRecordingActive) {
    if (recordingArmed) {
      recordDownload.hidden = false;
      recordDownload.textContent = "Запись начнётся при воспроизведении";
    }
    return;
  }
  const response = await fetch("/api/recording/status", { cache: "no-store" });
  if (!response.ok) {
    return;
  }
  const state = await response.json();
  if (state.active) {
    recordDownload.textContent = `Идёт запись: ${state.frames || 0} кадров`;
  }
}

async function onCanvasClick(event) {
  if (selectedIndex < 0) {
    return;
  }
  // Страховка: синхронно обновляем lastUploadContentRect под текущий source
  // video перед расчётом координат. Не зависит от isSending guard в
  // sendCurrentFrame, поэтому корректные координаты получаются даже если
  // параллельно идёт другая отправка кадра.
  updateLastUploadContentRect();

  const rect = displayCanvas.getBoundingClientRect();
  const scaleX = lastUploadContentRect.width / rect.width;
  const scaleY = lastUploadContentRect.height / rect.height;
  const x = Math.round(lastUploadContentRect.x + (event.clientX - rect.left) * scaleX);
  const y = Math.round(lastUploadContentRect.y + (event.clientY - rect.top) * scaleY);

  // Временная диагностика на время проверки фикса. Уберём после подтверждения.
  console.log("[click->frame]", {
    cssX: event.clientX - rect.left,
    cssY: event.clientY - rect.top,
    cssRect: { width: rect.width, height: rect.height },
    canvas: { width: displayCanvas.width, height: displayCanvas.height },
    upload: { ...lastUploadContentRect },
    frame: { x, y },
  });

  await sendCurrentFrame();
  await fetch("/api/commands/click", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ x, y }),
  });
  await sendCurrentFrame();
  window.setTimeout(refreshFrame, 120);
  window.setTimeout(refreshMetrics, 180);
}

function bindEvents() {
  filePicker.addEventListener("change", () => loadFiles(filePicker.files));
  folderPicker.addEventListener("change", () => loadFiles(folderPicker.files));
  presetSelect.addEventListener("change", async () => {
    syncModelControls();
    saveState();
    if (isCurrentPresetNeural()) {
      openModelDialog();
      return;
    }
    await configureRuntime({ force: true });
    await sendCurrentFrame();
  });
  chooseModelButton.addEventListener("click", openModelDialog);
  modelPathInput.addEventListener("change", async () => {
    updateModelPathFromInput();
    await configureRuntime({ force: true });
    await sendCurrentFrame();
  });
  modelFilePicker.addEventListener("change", () => {
    const file = modelFilePicker.files && modelFilePicker.files[0];
    if (file) {
      dialogModelPath.value = `models/${file.name}`;
    }
  });
  cancelModelDialog.addEventListener("click", closeModelDialog);
  applyModelDialog.addEventListener("click", applyModelDialogValue);
  dialogModelPath.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      applyModelDialogValue();
    } else if (event.key === "Escape") {
      event.preventDefault();
      closeModelDialog();
    }
  });
  modelDialog.addEventListener("click", (event) => {
    if (event.target === modelDialog) {
      closeModelDialog();
    }
  });
  frameIntervalInput.addEventListener("change", () => {
    frameIntervalInput.value = String(currentFrameIntervalMs());
    saveState();
    if (!sourceVideo.paused) {
      startSender();
    }
  });
  sequentialCheckbox.addEventListener("change", saveState);
  recordCheckbox.addEventListener("change", toggleRecording);
  previousVideoButton.addEventListener("click", () => goToVideo(selectedIndex - 1, { autoplay: playbackActive }));
  nextVideoButton.addEventListener("click", () => goToVideo(selectedIndex + 1, { autoplay: playbackActive }));
  playPauseButton.addEventListener("click", () => {
    if (!playbackActive) {
      startPlayback();
    } else {
      pausePlayback();
    }
  });
  stepBackButton.addEventListener("click", () => stepByFrame(-1));
  stepForwardButton.addEventListener("click", () => stepByFrame(1));
  stopVideoButton.addEventListener("click", stopPlayback);
  timeline.addEventListener("input", async () => {
    if (selectedIndex < 0) {
      return;
    }
    pausePlayback();
    sourceVideo.currentTime = Number(timeline.value);
    await waitForSeekIfNeeded();
    refreshTimeline();
    await sendCurrentFrame();
  });
  displayCanvas.addEventListener("click", onCanvasClick);
  window.addEventListener("resize", fitDisplayCanvasToStage);
  if ("ResizeObserver" in window && videoStage) {
    const resizeObserver = new ResizeObserver(fitDisplayCanvasToStage);
    resizeObserver.observe(videoStage);
  }
  sourceVideo.addEventListener("ended", handleVideoEnded);
  sourceVideo.addEventListener("pause", () => {
    playbackActive = false;
    stopSender();
    stopTimelineLoop();
    refreshControls();
  });
  sourceVideo.addEventListener("play", () => {
    playbackActive = true;
    startTimelineLoop();
    refreshControls();
  });
  window.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) {
      return;
    }
    if (event.code === "Space") {
      event.preventDefault();
      if (!playbackActive) {
        startPlayback();
      } else {
        pausePlayback();
      }
    } else if (event.key.toLowerCase() === "n") {
      stepByFrame(1);
    } else if (event.key.toLowerCase() === "r") {
      resetTrack();
    }
  });
}

bindEvents();
