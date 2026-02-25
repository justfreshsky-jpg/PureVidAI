# PureVid AI

Educator-ready AI video and teaching-content assistant.

## Environment variables

### Recommended (Google Cloud credits via Vertex AI)
- `VERTEX_PROJECT_ID` (or `GOOGLE_CLOUD_PROJECT`)
- `VERTEX_LOCATION` (optional, default: `us-central1`)
- `VERTEX_MODEL` (optional, default: `gemini-1.5-flash`)
- Workload/service account with Vertex AI access (`aiplatform.endpoints.predict` permission).

### Optional fallback
- `GROQ_KEY` (used when Vertex AI is not configured or as fallback if enabled)

### Video generation
- `FAL_KEY` for CogVideoX video rendering.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000`.
