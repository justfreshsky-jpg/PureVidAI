# PureVid AI

Educator-ready AI video and teaching-content assistant.

## Environment variables

### Recommended (Google Cloud Vertex AI)
- `VERTEX_PROJECT_ID` (or `GOOGLE_CLOUD_PROJECT`)
- `VERTEX_LOCATION` (optional, default: `us-central1`)
- `VERTEX_MODEL` (optional, default: `gemini-1.5-flash`)
- Workload/service account with Vertex AI access (`aiplatform.endpoints.predict` permission).

### Optional fallback
- `GROQ_KEY` (used when Vertex AI is not configured or as fallback if enabled)

### Video generation (no extra third-party spend by default)
- `VIDEO_PROVIDER` (optional, default: `google`) → `google` or `fal`
- `VERTEX_VIDEO_MODEL` (optional, default: `veo-2.0-generate-001`) for Google video generation
- `ALLOW_FAL_FALLBACK` (optional, default: `false`) set `true` only if you want fallback to fal
- `FAL_KEY` required only when `VIDEO_PROVIDER=fal` or when fal fallback is enabled

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000`.
