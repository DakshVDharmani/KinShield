import express from "express";
import multer from "multer";
import dotenv from "dotenv";
import fs from "node:fs";
import path from "node:path";
import { createHmac } from "node:crypto";
import { fileURLToPath } from "node:url";

dotenv.config();
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const uploadDir = path.join(__dirname, "uploads");
fs.mkdirSync(uploadDir, { recursive: true });
const app = express();
const MAX_BYTES = 25_000_000;
const types = { ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm", ".mkv": "video/x-matroska" };
const summaries = new Set(["No AI Detected", "Likely Authentic", "Likely AI-Generated", "AI-Generated"]);
const upload = multer({
  dest: uploadDir,
  limits: { fileSize: MAX_BYTES, files: 1, fields: 3 },
  fileFilter: (_req, file, callback) => {
    const extension = path.extname(file.originalname).toLowerCase();
    const validMime = !file.mimetype || file.mimetype.startsWith("video/") || file.mimetype === "application/octet-stream";
    if (!types[extension] || !validMime) return callback(new Error("Choose an MP4, MOV, WebM, or MKV video."));
    callback(null, true);
  },
});
app.use(express.static(path.join(__dirname, "public")));
app.get("/learn", (_req, res) => res.sendFile(path.join(__dirname, "public", "learn.html")));
app.get("/api/video-check/config", (_req, res) => {
  res.set("Cache-Control", "no-store").json({ provider: "FauxLens", configured: Boolean(process.env.FAUXLENS_API_KEY), max_bytes: MAX_BYTES });
});
app.post("/api/video-check", (req, res) => {
  res.set("Cache-Control", "no-store");
  if (!process.env.FAUXLENS_API_KEY) return res.status(503).json({ error: "Video checks need a FauxLens API key. Create one at fauxlens.com/developers, add FAUXLENS_API_KEY to the server .env file, and restart the server." });
  upload.single("video")(req, res, async uploadError => {
    if (uploadError) return res.status(uploadError.code === "LIMIT_FILE_SIZE" ? 413 : 400).json({ error: uploadError.code === "LIMIT_FILE_SIZE" ? "Video must be 25 MB or smaller." : "Upload one MP4, MOV, WebM, or MKV video, up to 25 MB." });
    const tempPath = req.file?.path;
    try {
      if (!req.file || req.file.size === 0) return res.status(400).json({ error: "Choose a non-empty video to analyze." });
      if (req.body.consent !== "true") return res.status(400).json({ error: "Consent is required before sending your video to FauxLens." });
      if (!/^[a-f0-9-]{36}$/i.test(req.body.request_id || "")) return res.status(400).json({ error: "Refresh this page and select your video again." });
      const key = process.env.FAUXLENS_API_KEY;
      const video = await fs.promises.readFile(tempPath);
      const filename = path.basename(req.file.originalname);
      // The browser keeps request_id on retries. Bind it to the actual file so a
      // changed file cannot accidentally replay a different video's verdict.
      const digest = createHmac("sha256", key).update(req.body.request_id).update(filename).update(video).digest("hex");
      const idempotencyKey = `${digest.slice(0,8)}-${digest.slice(8,12)}-4${digest.slice(13,16)}-a${digest.slice(17,20)}-${digest.slice(20,32)}`;
      const body = new FormData();
      body.append("file", new Blob([video], { type: types[path.extname(filename).toLowerCase()] }), filename);
      const response = await fetch("https://api.fauxlens.com/v1/analyze", {
        method: "POST",
        headers: { Authorization: `Bearer ${key}`, "Idempotency-Key": idempotencyKey },
        body,
        signal: AbortSignal.timeout(120000),
      });
      const output = await response.json().catch(() => null);
      if (!response.ok) {
        // Do not log raw provider bodies: they can contain submitted media or credentials.
        console.error("FauxLens request failed", JSON.stringify({ status: response.status }));
        const errors = {
          400: "FauxLens rejected the upload request. Check the selected file and try again.",
          401: "FauxLens rejected the API key. Check FAUXLENS_API_KEY in the server .env file.",
          402: "Your FauxLens credits are exhausted. Free starting credits do not renew monthly. Check your account balance before scanning again.",
          403: "This FauxLens account does not have API access. Check the account and key permissions.",
          409: "FauxLens reported a request conflict. Click Analyze video again to submit a new request.",
          413: "FauxLens rejected the file size. Choose a video under 25 MB.",
          415: "FauxLens does not support this video's format or codec. Try an MP4 with H.264 video.",
          422: "FauxLens could not analyze this video. No detection result is available.",
          429: "FauxLens is rate limiting requests. Wait a minute, then retry with the same selected file.",
        };
        return res.status([400,402,409,413,415,422,429].includes(response.status) ? response.status : 502).json({ error: errors[response.status] || "FauxLens could not complete this check. Retry with the same selected file to reuse this request." });
      }
      const data = output?.data;
      if (!data || !summaries.has(data.summary)) return res.status(502).json({ error: "FauxLens returned an unrecognized result. No verdict can be shown." });
      const confidence = typeof data.confidence_score === "number" && data.confidence_score >= 0 && data.confidence_score <= 1 ? data.confidence_score : null;
      res.json({
        status: "complete",
        provider: "FauxLens",
        verdict: data.summary,
        confidence_pct: confidence === null ? null : confidence * 100,
        request_id: typeof output.request_id === "string" ? output.request_id : null,
        early_result: data.is_short_circuited === true,
        replayed: output.idempotent_replay === true,
        evidence: (Array.isArray(data.evidence_chain) ? data.evidence_chain : []).slice(0, 30).map(item => ({
          label: String(item?.label || item?.layer || "Detection signal").slice(0, 200),
          status: ["passed", "warning", "failed"].includes(item?.status) ? item.status : "unknown",
          detail: String(item?.detail || "").slice(0, 3000),
        })),
      });
    } catch (error) {
      const timeout = ["TimeoutError", "AbortError"].includes(error.name);
      console.error("Video check failed", timeout ? "provider timeout" : "provider connection error");
      res.status(timeout ? 504 : 502).json({ error: timeout
        ? "FauxLens did not return a result within two minutes. Keep this file selected and retry to reuse the same request; no result is available yet."
        : "Could not connect to FauxLens. Keep this file selected and retry to reuse the same request." });
    } finally {
      if (tempPath) await fs.promises.unlink(tempPath).catch(() => console.error("Could not remove temporary upload."));
    }
  });
});
app.use((error, _req, res, _next) => {
  console.error("Request failed", error.name);
  res.status(500).json({ error: "The request could not be completed." });
});
const port = Number(process.env.PORT || 3001);
app.listen(port, (error) => {
  if (error) {
    console.error(error.code === "EADDRINUSE"
      ? `Cannot start KinShield: port ${port} is already in use. Stop the existing server or set a different PORT in .env.`
      : `Cannot start KinShield: ${error.message}`);
    process.exitCode = 1;
    return;
  }
  console.log(`KinShield video checker listening on http://localhost:${port}`);
});
