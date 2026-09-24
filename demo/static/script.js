// Road Damage Detector -- upload, pick a model (or compare all four), see
// the result. No framework.

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const chooseBtn = document.getElementById("chooseBtn");
const modelSelect = document.getElementById("modelSelect");
const confidenceInput = document.getElementById("confidence");
const confidenceValue = document.getElementById("confidenceValue");
const analyzeBtn = document.getElementById("analyzeBtn");
const compareBtn = document.getElementById("compareBtn");
const errorText = document.getElementById("errorText");

const resultCard = document.getElementById("resultCard");
const resultLoading = document.getElementById("resultLoading");
const resultLoadingText = document.getElementById("resultLoadingText");
const resultContent = document.getElementById("resultContent");
const resultImage = document.getElementById("resultImage");
const resultSummary = document.getElementById("resultSummary");
const detectionsBody = document.getElementById("detectionsBody");
const compareGrid = document.getElementById("compareGrid");

let selectedFile = null;

confidenceInput.addEventListener("input", () => {
  confidenceValue.textContent = parseFloat(confidenceInput.value).toFixed(2);
});

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

async function loadModels() {
  try {
    const res = await fetch("/api/models");
    const models = await res.json();
    modelSelect.innerHTML = models
      .map((m) => `<option value="${m.id}">${escapeHtml(m.label)}</option>`)
      .join("");
    const yolo26 = models.find((m) => m.id === "yolo26n");
    if (yolo26) modelSelect.value = "yolo26n";
  } catch (err) {
    showError("Could not load model list -- is the server running?");
  }
}
loadModels();

function setFile(file) {
  if (!file || !file.type.match(/^image\/(jpeg|png)$/)) {
    showError("Please choose a JPG or PNG image.");
    return;
  }
  selectedFile = file;
  hideError();
  analyzeBtn.disabled = false;
  compareBtn.disabled = false;

  const reader = new FileReader();
  reader.onload = (e) => {
    dropzone.innerHTML = `
      <img class="dropzone-preview" src="${e.target.result}" alt="Selected image preview">
      <p class="dropzone-text">${escapeHtml(file.name)}</p>
      <button type="button" class="btn-choose" id="chooseBtn">Choose a different image</button>
    `;
    // The button was replaced by innerHTML above, so its click listener
    // needs re-wiring onto the new element.
    document.getElementById("chooseBtn").addEventListener("click", (e) => {
      e.stopPropagation();
      fileInput.click();
    });
  };
  reader.readAsDataURL(file);
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
chooseBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

function showError(msg) { errorText.textContent = msg; errorText.hidden = false; }
function hideError() { errorText.hidden = true; }

function buildForm() {
  const form = new FormData();
  form.append("image", selectedFile);
  form.append("confidence", confidenceInput.value);
  form.append("model", modelSelect.value);
  return form;
}

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  hideError();
  resultCard.hidden = false;
  resultContent.hidden = true;
  compareGrid.hidden = true;
  resultLoadingText.textContent = "Running detection…";
  resultLoading.hidden = false;
  analyzeBtn.disabled = true;
  compareBtn.disabled = true;

  try {
    const res = await fetch("/api/predict", { method: "POST", body: buildForm() });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
    renderResult(data);
    resultLoading.hidden = true;
    resultContent.hidden = false;
  } catch (err) {
    resultCard.hidden = true;
    showError(err.message || "Something went wrong -- try again.");
  } finally {
    analyzeBtn.disabled = false;
    compareBtn.disabled = false;
  }
});

compareBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  hideError();
  resultCard.hidden = false;
  resultContent.hidden = true;
  compareGrid.hidden = true;
  resultLoadingText.textContent = "Running all four models — this takes longer…";
  resultLoading.hidden = false;
  analyzeBtn.disabled = true;
  compareBtn.disabled = true;

  try {
    const res = await fetch("/api/compare", { method: "POST", body: buildForm() });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
    renderCompare(data);
    resultLoading.hidden = true;
    compareGrid.hidden = false;
  } catch (err) {
    resultCard.hidden = true;
    showError(err.message || "Something went wrong -- try again.");
  } finally {
    analyzeBtn.disabled = false;
    compareBtn.disabled = false;
  }
});

function renderResult(data) {
  resultImage.src = data.image;

  const countParts = Object.entries(data.counts).map(([cls, n]) => `${n}× ${cls}`);
  const countStr = countParts.length ? countParts.join(", ") : "no damage detected";
  resultSummary.innerHTML =
    `<strong>${data.detections.length} detection(s)</strong> — ${escapeHtml(countStr)}. ` +
    `Inference time: ${data.latency_ms} ms.`;

  detectionsBody.innerHTML = "";
  for (const det of data.detections) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${escapeHtml(det.class)}</td><td>${det.confidence.toFixed(3)}</td>`;
    detectionsBody.appendChild(tr);
  }
}

function renderCompare(data) {
  compareGrid.innerHTML = "";
  for (const modelId of Object.keys(data)) {
    const r = data[modelId];
    const card = document.createElement("div");
    card.className = "compare-card";

    if (r.error) {
      card.innerHTML = `<h3>${escapeHtml(r.label)}</h3><p class="compare-error">${escapeHtml(r.error)}</p>`;
      compareGrid.appendChild(card);
      continue;
    }

    const countParts = Object.entries(r.counts).map(([cls, n]) => `${n}× ${cls}`);
    const countStr = countParts.length ? countParts.join(", ") : "no damage detected";

    let html = `<h3>${escapeHtml(r.label)}</h3>`;
    html += `<img src="${r.image}" alt="${escapeHtml(r.label)} detections">`;
    html += `<p class="compare-summary"><strong>${r.detections.length} detection(s)</strong> — ${escapeHtml(countStr)}. ${r.latency_ms} ms.</p>`;

    if (r.heatmap) {
      html += `<p class="heatmap-label">EigenCAM attention</p>`;
      html += `<img src="${r.heatmap}" alt="${escapeHtml(r.label)} EigenCAM heatmap">`;
      html += `<p class="heatmap-legend">low <span class="legend-gradient"></span> high</p>`;
    }
    if (r.sentences && r.sentences.length) {
      html += `<ul class="explain-list">${r.sentences.map((s) => `<li>${escapeHtml(s)}</li>`).join("")}</ul>`;
    }

    card.innerHTML = html;
    compareGrid.appendChild(card);
  }
}
