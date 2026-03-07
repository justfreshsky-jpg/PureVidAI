# PureImage AI

AI-powered image generation web app. Type a text prompt and generate stunning images using multiple AI providers.

## Features

- **Multiple image providers** with automatic fallback chain:
  1. **fal.ai** — Primary (FLUX Schnell, FLUX Pro, SD3, Recraft v3)
  2. **Hugging Face** — Secondary (FLUX.1-schnell, SDXL)
  3. **Stability AI** — Tertiary (SDXL 1.0)
  4. **Replicate** — Quaternary (FLUX Schnell)
  5. **Pollinations.ai** — Free fallback, always available (no key needed)
- **Style presets**: Photorealistic, Artistic, Anime, Digital Art, Oil Painting, Watercolor, Sketch, Cinematic, Abstract
- **Aspect ratios**: Square (1:1), Landscape (16:9), Portrait (9:16), Wide (3:2), Tall (2:3)
- **Multiple images**: Generate 1, 2, or 4 images at once
- **Prompt enhancement**: Optional AI-powered prompt improvement (requires an LLM key)
- **Negative prompts**: Advanced control over what to exclude
- **Download buttons** on every generated image
- **Rate limiting** and **response caching** built in

## Environment Variables

### Image Providers
| Variable | Description |
|---|---|
| `FAL_KEY` | fal.ai API key (https://fal.ai) |
| `HF_KEY` | Hugging Face API key (https://huggingface.co) |
| `STABILITY_KEY` | Stability AI API key (https://stability.ai) |
| `REPLICATE_KEY` | Replicate API key (https://replicate.com) |

Pollinations.ai requires no key and is always used as the final fallback.

### Text LLM Keys (for prompt enhancement, optional)
| Variable | Provider |
|---|---|
| `GROQ_KEY` | Groq (Llama 3.3 70B) |
| `CEREBRAS_KEY` | Cerebras |
| `GEMINI_KEY` | Google Gemini |
| `COHERE_KEY` | Cohere |
| `MISTRAL_KEY` | Mistral |
| `OPENROUTER_KEY` | OpenRouter |

### App Config
| Variable | Default | Description |
|---|---|---|
| `PORT` | `8080` | HTTP port |
| `PUREIMAGE_LOG_PATH` | `/tmp/pureimage_feedback.log.jsonl` | Generation log path |

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:8080.

## Deploy on Render

1. Create a new **Web Service**
2. Set **Build Command**: `pip install -r requirements.txt`
3. Set **Start Command**: `gunicorn app:app`
4. Add environment variables for your API keys
5. Deploy — Pollinations.ai works with no keys, so images generate immediately

## Deploy on Google Cloud Run

1. Build and push the Docker image:
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT_ID/pureimage-ai
   ```
2. Deploy the service:
   ```bash
   gcloud run deploy pureimage-ai --image gcr.io/PROJECT_ID/pureimage-ai --platform managed --allow-unauthenticated
   ```
3. Add environment variables for your API keys via the Cloud Run console or `--set-env-vars`.

### Image Proxying

Generated images from external providers are served through the `/proxy_image` endpoint to avoid CORS issues. The allowed upstream hosts are defined in `app.py` in the `allowed_hosts` tuple inside `proxy_image()`. If a new image provider returns URLs from a host not in the list, add it there.

### Debugging on Cloud Run

- Visit `/debug` to check which API keys are configured and retrieve the Cloud Run trace ID.
- Check Cloud Run logs for lines containing `Unhandled route error` — each entry includes a `request_id` that is also returned in the JSON error response to the client, making it easy to correlate user-reported errors with server logs.
