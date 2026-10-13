# KinShield

KinShield is a parent-facing video deepfake checker. It sends a video to Sightengine only after the user gives consent, reports the frame scores returned by that service, and explains that a detector score is not proof of authenticity, identity, or intent.

## What it does

- Checks one MP4, MOV, WebM, or AVI video at a time (maximum 100 MB).
- Uses Sightengine's video deepfake endpoint. Credentials stay on the server.
- Shows sampled frame count, highest returned score, average score, and the timestamp(s) for the highest-scoring frame(s).
- Deletes the temporary upload after the request succeeds or fails.
- Does not keep a public analysis history or require an account.

Images, audio, misinformation, login, chat, and rPPG analysis are outside this app's scope. The original `deepfake_rppg` directory in the surrounding workspace has not been modified.

## Run locally

1. Install Node.js 18 or newer.
2. In this directory, run `npm install`.
3. Copy `.env.example` to `.env` and set `SIGHTENGINE_API_USER` and `SIGHTENGINE_API_SECRET` using your Sightengine account credentials.
4. Run `npm start` and open `http://localhost:3001`.

Do not commit `.env`. Keep the server private or deploy it behind appropriate upload limits and abuse controls before exposing it publicly.

## Privacy and limitations

The browser asks for explicit consent before upload. The server validates extension, video MIME type, consent, and the 100 MB size limit. The video is sent to Sightengine for analysis; processing therefore involves that external provider. KinShield stores the upload temporarily on the server only while the request is handled, then deletes the temporary file. Sightengine's own handling is subject to its terms and privacy policy.

Frame scores are returned as supplied by Sightengine. KinShield does not convert them into a binary "real" or "fake" verdict because the score is not proof of authenticity or intent. A low score is not proof that a video is genuine. Compression, editing, lighting, and newer generation methods can affect detector results. If no frame scores are returned, the result is inconclusive.

The supported formats are limited by the file extension and browser MIME type and may also be subject to Sightengine's current video requirements. Very long clips can time out; try a shorter clip if that happens.
