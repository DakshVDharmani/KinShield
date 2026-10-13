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
let requestId = null;
const permittedExtensions = [".mp4", ".mov", ".webm", ".mkv"];
const maxFileSize = 25_000_000;
const busy = () => analyzeButton.dataset.busy === "true";
try { sessionStorage.removeItem("kinshield-pending-scan"); } catch { /* Old provider state is no longer used. */ }
function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = !message;
}
function syncButton() {
  analyzeButton.disabled = !selectedFile || !consent.checked || busy();
  picker.disabled = busy();
  dropzone.disabled = busy();
  consent.disabled = busy();
  document.querySelector("#remove-file").disabled = busy();
}
function requestForFile(file) {
  // Keep retry identity across reloads; the server also binds it to the file bytes.
  const fingerprint = JSON.stringify([file.name, file.size, file.lastModified]);
  let id = crypto.randomUUID();
  try {
    const saved = JSON.parse(sessionStorage.getItem("kinshield-fauxlens-request") || "null");
    if (saved?.fingerprint === fingerprint && saved.expires > Date.now()) id = saved.id;
    else sessionStorage.setItem("kinshield-fauxlens-request", JSON.stringify({ fingerprint, id, expires: Date.now() + 23 * 3600000 }));
  } catch { /* In-memory identity still protects retries in this page. */ }
  return id;
}
function clearFile() {
  selectedFile = null;
  requestId = null;
  picker.value = "";
  fileRow.hidden = true;
  fileTitle.innerHTML = 'Drop a video here, or <u>browse files</u>';
  resultBox.hidden = true;
  consent.checked = false;
  syncButton();
}
function selectFile(file) {
  if (busy() || !file) return;
  clearFile();
  showError("");
  const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (!permittedExtensions.includes(extension)) return showError("Choose an MP4, MOV, WebM, or MKV video.");
  if (!file.size || file.size > maxFileSize) return showError("Choose a non-empty video up to 25 MB.");
  selectedFile = file;
  requestId = requestForFile(file);
  fileTitle.textContent = "Video selected";
  fileInfo.textContent = `${file.name} · ${(file.size / 1000000).toFixed(1)} MB`;
  fileRow.hidden = false;
  syncButton();
}
dropzone.addEventListener("click", () => picker.click());
picker.addEventListener("change", () => selectFile(picker.files?.[0]));
dropzone.addEventListener("dragover", event => { event.preventDefault(); if (!busy()) dropzone.classList.add("dragging"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
dropzone.addEventListener("drop", event => { event.preventDefault(); dropzone.classList.remove("dragging"); selectFile(event.dataTransfer.files?.[0]); });
consent.addEventListener("change", syncButton);
document.querySelector("#remove-file").addEventListener("click", () => {
  if (busy()) return;
  clearFile();
  showError("");
});
analyzeButton.addEventListener("click", async () => {
  if (!selectedFile || !consent.checked || busy()) return;
  analyzeButton.dataset.busy = "true";
  analyzeButton.innerHTML = '<span class="spinner"></span> Checking video…';
  progress.textContent = "Sending your video to FauxLens for analysis. This can take up to two minutes.";
  progress.hidden = false;
  resultBox.hidden = true;
  showError("");
  syncButton();
  const body = new FormData();
  body.append("video", selectedFile);
  body.append("consent", "true");
  body.append("request_id", requestId);
  try {
    const response = await fetch("/api/video-check", { method: "POST", body, signal: AbortSignal.timeout(150000) });
    const result = await response.json().catch(() => null);
    if (response.status === 409) {
      try { sessionStorage.removeItem("kinshield-fauxlens-request"); } catch { /* Storage can be unavailable. */ }
      requestId = requestForFile(selectedFile);
    }
    if (!response.ok || !result) throw new Error(result?.error || "The server did not return a result. Keep the same file selected and retry.");
    showResult(result);
  } catch (error) {
    showError(error.name === "TimeoutError" ? "The connection timed out. Keep this file selected and retry to reuse the same request." : error.message || "The analysis could not be completed.");
  } finally {
    analyzeButton.dataset.busy = "false";
    analyzeButton.innerHTML = 'Analyze video <span aria-hidden="true">↗</span>';
    progress.hidden = true;
    syncButton();
  }
});
function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function showResult(data) {
  resultBox.replaceChildren();
  const heading = element("div", null, "result-head");
  const title = element("div");
  title.append(element("div", "FAUXLENS VIDEO CHECK", "result-kicker"));
  const complete = data.status === "complete" && typeof data.verdict === "string";
  title.append(element("h3", complete ? data.verdict : "No clear result returned"));
  heading.append(title);
  resultBox.append(heading);
  if (complete) {
    resultBox.append(element("p", "This is FauxLens's assessment of the video. It is a signal to review alongside the original source and context."));
    if (typeof data.confidence_pct === "number" && Number.isFinite(data.confidence_pct)) {
      const scores = element("div", null, "scores");
      const score = element("div");
      score.append(element("span", "Confidence in this verdict"), element("b", `${data.confidence_pct.toFixed(1)}%`));
      scores.append(score);
      resultBox.append(scores, element("p", "Confidence is the provider's ranking of supporting evidence, not the probability that this video is fake.", "caveat"));
    }
    if (Array.isArray(data.evidence) && data.evidence.length) {
      const section = element("div", null, "detection-evidence");
      section.append(element("h4", "What informed this result"));
      const list = element("ul");
      for (const evidence of data.evidence) {
        const item = element("li");
        item.append(element("strong", evidence.label), element("p", evidence.detail || "No further detail supplied."));
        list.append(item);
      }
      section.append(list);
      resultBox.append(section);
    }
    if (data.early_result) resultBox.append(element("p", "One signal settled the provider's verdict early; not every analysis layer ran.", "caveat"));
    if (data.replayed) resultBox.append(element("p", "Showing the saved result for this request; FauxLens reports that it did not charge again.", "caveat"));
    if (data.request_id) resultBox.append(element("p", `Reference: ${data.request_id}`, "caveat"));
  }
  resultBox.append(element("p", "A clean result does not prove authenticity. Video detection can miss manipulation or flag genuine footage, especially after compression or editing. Verify the source before acting.", "caveat"));
  resultBox.hidden = false;
  resultBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
