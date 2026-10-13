import express from "express";
import multer from "multer";
import FormData from "form-data";
import fetch from "node-fetch";
import dotenv from "dotenv";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

dotenv.config();
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const uploadDir = path.join(__dirname, "uploads");
fs.mkdirSync(uploadDir, { recursive: true });

const app = express();
const MAX_BYTES = 100 * 1024 * 1024;
const allowedExtensions = new Set([".mp4", ".mov", ".webm", ".avi"]);
const upload = multer({
  dest: uploadDir,
  limits: { fileSize: MAX_BYTES, files: 1 },
  fileFilter: (_req, file, callback) => {
    const extension = path.extname(file.originalname).toLowerCase();
    if (!allowedExtensions.has(extension) || !file.mimetype.startsWith("video/")) {
      callback(new Error("Upload an MP4, MOV, WebM, or AVI video."));
      return;
    }
    callback(null, true);
  },
});

app.use(express.static(path.join(__dirname, "public")));

app.post("/api/video-check", (req, res) => {
  upload.single("video")(req, res, async (uploadError) => {
    if (uploadError) {
      const status = uploadError instanceof multer.MulterError && uploadError.code === "LIMIT_FILE_SIZE" ? 413 : 400;
      res.status(status).json({ error: status === 413 ? "Video must be 100 MB or smaller." : uploadError.message });
      return;
    }

    const tempPath = req.file?.path;
    try {
      if (!req.file) return res.status(400).json({ error: "Choose a video to analyze." });
      if (req.body.consent !== "true") return res.status(400).json({ error: "Consent is required before the video is sent for analysis." });

      const apiUser = process.env.SIGHTENGINE_API_USER;
      const apiSecret = process.env.SIGHTENGINE_API_SECRET;
      if (!apiUser || !apiSecret) return res.status(503).json({ error: "Sightengine is not configured. Add credentials to the server .env file." });

      const body = new FormData();
      body.append("media", fs.createReadStream(req.file.path), {
        filename: path.basename(req.file.originalname),
        contentType: req.file.mimetype,
      });
      body.append("models", "deepfake");
      body.append("api_user", apiUser);
      body.append("api_secret", apiSecret);

      const response = await fetch("https://api.sightengine.com/1.0/video/check-sync.json", {
        method: "POST",
        body,
        headers: body.getHeaders(),
        signal: AbortSignal.timeout(180_000),
      });
      const payload = await response.json();
      if (!response.ok || payload.status !== "success") {
        return res.status(502).json({ error: payload.error?.message || "Sightengine could not analyze this video." });
      }

      const frames = (payload.data?.frames || []).flatMap((frame) => {
        const rawScore = frame.type?.deepfake;
        const position = Number(frame.info?.position);
        if (!Number.isFinite(Number(rawScore)) || !Number.isFinite(position)) return [];
        return [{ time_seconds: position / 1000, score: Number(rawScore) }];
      });
      if (!frames.length) return res.json({ status: "inconclusive", frames_analyzed: 0, findings: [] });

      const scores = frames.map((frame) => frame.score);
      const highest = Math.max(...scores);
      const average = scores.reduce((sum, score) => sum + score, 0) / scores.length;
      return res.json({
        status: "complete",
        frames_analyzed: frames.length,
        highest_frame_score: highest,
        average_frame_score: average,
        findings: frames.filter((frame) => frame.score === highest),
        provider: "Sightengine",
      });
    } catch (error) {
      if (error.name === "TimeoutError" || error.name === "AbortError") {
        return res.status(504).json({ error: "The video check took too long. Try a shorter video." });
      }
      console.error("Video check failed:", error.message);
      return res.status(502).json({ error: "The video check could not be completed. Please try again." });
    } finally {
      if (tempPath && fs.existsSync(tempPath)) fs.unlink(tempPath, () => {});
    }
  });
});

app.use((error, _req, res, _next) => {
  console.error("Request failed:", error.message);
  res.status(500).json({ error: "The request could not be completed." });
});

const port = Number(process.env.PORT || 3001);
app.listen(port, () => console.log(`KinShield video checker listening on http://localhost:${port}`));
