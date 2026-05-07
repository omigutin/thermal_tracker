const frame = document.getElementById("frame");
const empty = document.getElementById("empty");
const metrics = document.getElementById("metrics");
const summary = document.getElementById("summary");

let hasFrame = false;

function valueOrDash(value) {
  return value === undefined || value === null ? "-" : value;
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

function refreshFrame() {
  if (!hasFrame) {
    return;
  }
  frame.src = `/api/frame/latest.jpg?overlay=true&t=${Date.now()}`;
}

async function refreshMetrics() {
  try {
    const response = await fetch("/api/metrics", { cache: "no-store" });
    const data = await response.json();
    hasFrame = Boolean(data.frame);
    frame.hidden = !hasFrame;
    empty.hidden = hasFrame;
    renderSummary(data);
    metrics.textContent = JSON.stringify(data, null, 2);
    if (hasFrame) {
      refreshFrame();
    }
  } catch (error) {
    metrics.textContent = String(error);
  }
}

frame.addEventListener("click", async (event) => {
  const rect = frame.getBoundingClientRect();
  const scaleX = frame.naturalWidth / rect.width;
  const scaleY = frame.naturalHeight / rect.height;
  const x = Math.round((event.clientX - rect.left) * scaleX);
  const y = Math.round((event.clientY - rect.top) * scaleY);
  await fetch("/api/commands/click", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ x, y }),
  });
  refreshMetrics();
});

document.getElementById("reset").addEventListener("click", async () => {
  await fetch("/api/commands/reset", { method: "POST" });
  refreshMetrics();
});

setInterval(refreshFrame, 120);
setInterval(refreshMetrics, 500);
refreshFrame();
refreshMetrics();
