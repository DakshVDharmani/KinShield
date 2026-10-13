# KinShield

A parent-facing video checker and Learning Centre at `/learn`, with lessons, a quiz, and a downloadable completion certificate.

## Run

1. Install Node.js 22 or newer and run `npm install` in this directory.
2. Sign in at https://fauxlens.com/developers and create your own developer API key.
3. Set `FAUXLENS_API_KEY` in `.env` (see `.env.example`). Never put the key in browser code or commit it.
4. Run `npm start` and open http://localhost:3001. Restart the server after changing `.env`.

## Video pipeline

Browser upload and consent → Express validation → FauxLens → verdict, confidence, and evidence.

- Accepts MP4, MOV, WebM, and MKV, up to 25,000,000 bytes. Codec support is ultimately enforced by the provider.
- Calls `POST https://api.fauxlens.com/v1/analyze` with Bearer authentication, multipart `file`, and `Idempotency-Key`.
- Waits up to 120 seconds for the synchronous result. No fabricated progress or queued scan is shown.
- Keeps a request identity for retries in the same browser tab, including reloads, for 23 hours. The server binds the identity to the video bytes and filename. Keep the same file selected when retrying. A new tab or different file can create a new billable request.
- Returns the provider's verdict and evidence. Confidence describes support for that verdict; it is not a calibrated probability that the video is fake. No human probability is invented.
- Handles missing/invalid keys, exhausted credits, rate limits, unsupported media, invalid responses, and timeouts explicitly. No automatic retries or paid fallback.
- Deletes the temporary local video when processing ends, including failures. The API key remains on the server.

`GET /api/video-check/config` returns the provider name, whether a key is configured (never its value), and the upload limit. The old ZeroTrue endpoints and pending-scan flow are no longer used. An old `ZEROTRUE_API_KEY` environment entry is ignored.

## Free access and provider selection

Reviewed official pages on 14 October 2026:

| Provider | Relevant free access | Decision |
| --- | --- | --- |
| [FauxLens](https://fauxlens.com/developers) | Starting credits can fund video API calls; each call costs 10 credits. No recurring monthly free allowance. | Integrated. Self-service key and documented video upload/response contract. |
| [Sightengine](https://sightengine.com/pricing) | Pricing comparison excludes video processing from the free API plan. | Does not meet free video API requirement. |
| [Reality Defender](https://www.realitydefender.com/product/realapi) | Free plan lists image/audio; video is on paid plans. | Does not meet free video API requirement. |

This is a free-credit integration, not unlimited free detection. Account balance and current provider terms determine availability. Code integration and key configuration alone do not establish live detection quality or successful processing.

## Limitations and privacy

FauxLens says its video detection has been evaluated less than its image detection. Treat results as review signals, never proof of authenticity or grounds for accusations. Compression, editing, and unfamiliar generation methods can affect detection. See [methodology](https://fauxlens.com/methodology).

Uploads go to FauxLens after explicit consent. It says media is deleted after analysis and result records are retained separately; review its [privacy policy](https://fauxlens.com/privacy). KinShield stores only retry metadata in browser session storage and does not publish reports. Avoid uploading sensitive content or any intimate or exploitative imagery involving a child.

Keep this development server private; public deployment needs authentication, abuse controls, and appropriate hosting timeouts for video uploads.
