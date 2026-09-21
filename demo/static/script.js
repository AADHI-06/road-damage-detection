// Road Damage Detector -- simple single-card UI. No framework, no
// offline-metrics section -- just upload, analyze, show the result.

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const chooseBtn = document.getElementById("chooseBtn");
const confidenceInput = document.getElementById("confidence");
const confidenceValue = document.getElementById("confidenceValue");
const analyzeBtn = document.getElementById("analyzeBtn");
const errorText = document.getElementById("errorText");

const resultCard = document.getElementById("resultCard");
const resultLoading = document.getElementById("resultLoading");
const resultContent = document.getElementById("resultContent");
const resultImage = document.getElementById("resultImage");
const resultSummary = document.getElementById("resultSummary");
const detectionsBody = document.getElementById("detectionsBody");

let selectedFile = null;

confidenceInput.addEventListener("input", () => {
  confidenceValue.textContent = parseFloat(confidenceInput.value).toFixed(2);
});

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function setFile(file) {
  if (!file || !file.type.match(/^image\/(jpeg|png)$/)) {
    showError("Please choose a JPG or PNG image.");
    return;
  }
  selectedFile = file;
  hideError();
  analyzeBtn.disabled = false;

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

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  hideError();
  resultCard.hidden = false;
  resultContent.hidden = true;
  resultLoading.hidden = false;
  analyzeBtn.disabled = true;

  const form = new FormData();
  form.append("image", selectedFile);
  form.append("confidence", confidenceInput.value);

  try {
    const res = await fetch("/api/predict", { method: "POST", body: form });
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
