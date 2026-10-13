const picker = document.querySelector("#video-file");
const dropzone = document.querySelector("#dropzone");
const fileTitle = document.querySelector("#file-title");
const fileRow = document.querySelector("#file-row");
const fileInfo = document.querySelector("#file-info");
const consent = document.querySelector("#consent");
const analyzeButton = document.querySelector("#analyze");
const errorBox = document.querySelector("#form-error");
const progress = document.querySelector("#progress");
const resultBox = document.querySelector("#result");
let selectedFile = null;

const permittedExtensions = [".mp4", ".mov", ".webm", ".avi"];
const maxFileSize = 100 * 1024 * 1024;
const extensionOf = (name) => name.slice(name.lastIndexOf(".")).toLowerCase();

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = !message;
}

function syncButton() {
  analyzeButton.disabled = !selectedFile || !consent.checked || analyzeButton.dataset.busy === "true";
}

function selectFile(file) {
  resultBox.hidden = true;
  showError("");
  if (!file) return;
  if (!permittedExtensions.includes(extensionOf(file.name))) {
    selectedFile = null;
    picker.value = "";
    fileRow.hidden = true;
    fileTitle.innerHTML = 'Drop a video here, or <u>browse files</u>';
    showError("Choose an MP4, MOV, WebM, or AVI video.");
    syncButton();
    return;
  }
  if (file.size > maxFileSize) {
    selectedFile = null;
    picker.value = "";
    fileRow.hidden = true;
    fileTitle.innerHTML = 'Drop a video here, or <u>browse files</u>';
    showError("This video is over the 100 MB limit.");
    syncButton();
    return;
  }
  selectedFile = file;
  fileTitle.textContent = "Video selected";
  fileInfo.textContent = `${file.name} · ${(file.size / (1024 * 1024)).toFixed(1)} MB`;
  fileRow.hidden = false;
  syncButton();
}

dropzone.addEventListener("click", () => picker.click());
picker.addEventListener("change", () => selectFile(picker.files?.[0]));
dropzone.addEventListener("dragover", (event) => { event.preventDefault(); dropzone.classList.add("dragging"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
dropzone.addEventListener("drop", (event) => { event.preventDefault(); dropzone.classList.remove("dragging"); selectFile(event.dataTransfer.files?.[0]); });
consent.addEventListener("change", syncButton);
document.querySelector("#remove-file").addEventListener("click", () => {
  selectedFile = null;
  picker.value = "";
  fileRow.hidden = true;
  fileTitle.innerHTML = 'Drop a video here, or <u>browse files</u>';
  resultBox.hidden = true;
  showError("");
  syncButton();
});

analyzeButton.addEventListener("click", async () => {
  if (!selectedFile || !consent.checked) return;
  analyzeButton.dataset.busy = "true";
  analyzeButton.innerHTML = '<span class="spinner"></span> Checking video…';
  progress.hidden = false;
  resultBox.hidden = true;
  showError("");
  syncButton();
  const body = new FormData();
  body.append("video", selectedFile);
  body.append("consent", "true");
  try {
    const response = await fetch("/api/video-check", { method: "POST", body });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "The analysis could not be completed.");
    showResult(result);
  } catch (error) {
    showError(error instanceof Error ? error.message : "The analysis could not be completed.");
  } finally {
    analyzeButton.dataset.busy = "false";
    analyzeButton.innerHTML = 'Analyze video <span aria-hidden="true">↗</span>';
    progress.hidden = true;
    syncButton();
  }
});

function showResult(data) {
  if (data.status !== "complete" || !data.frames_analyzed) {
    resultBox.innerHTML = '<div class="result-kicker">RESULT</div><h3>Unable to determine</h3><p>The video service did not return frame scores. Try another supported video. A missing result does not indicate that a video is authentic.</p>';
  } else {
    const frameWord = data.frames_analyzed === 1 ? "frame" : "frames";
    const peak = `${(data.highest_frame_score * 100).toFixed(1)}%`;
    const average = `${(data.average_frame_score * 100).toFixed(1)}%`;
    const times = (data.findings || []).map((finding) => `<li>At ${Number(finding.time_seconds).toFixed(1)} seconds <b>${(Number(finding.score) * 100).toFixed(1)}%</b></li>`).join("");
    resultBox.innerHTML = `<div class="result-head"><div><div class="result-kicker">SIGHTENGINE VIDEO CHECK</div><h3>Signals from ${data.frames_analyzed} sampled ${frameWord}</h3></div><span class="result-badge">✓</span></div><p class="result-copy">These are the detector scores returned for this video. They are signals from an automated system, not proof of who created or shared it.</p><div class="scores"><div><span>Highest frame score</span><b>${peak}</b></div><div><span>Average frame score</span><b>${average}</b></div></div>${times ? `<div class="flagged"><h4>Frame with the highest returned score</h4><ul>${times}</ul></div>` : ""}<p class="caveat">A low score cannot establish that a video is authentic. Editing, compression, lighting, and new generation methods can affect automated checks.</p>`;
  }
  resultBox.hidden = false;
  resultBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
