import os
os.environ.setdefault('HTTPX_PROXIES', 'null')  # Fix Render/httpx proxies bug
import base64
import collections
import hashlib
import json
import logging
import re
import threading
import time
import unicodedata
import urllib.parse
import requests
from flask import Flask, request, jsonify, render_template_string, Response
from groq import Groq

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# ── CONFIG ───────────────────────────────────────────────────
GROQ_KEY        = os.environ.get("GROQ_KEY")
FAL_KEY         = os.environ.get("FAL_KEY")
HF_KEY          = os.environ.get("HF_KEY")
STABILITY_KEY   = os.environ.get("STABILITY_KEY")
REPLICATE_KEY   = os.environ.get("REPLICATE_KEY")
PUREIMAGE_LOG_PATH = os.environ.get("PUREIMAGE_LOG_PATH", "/tmp/pureimage_feedback.log.jsonl")

client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

app = Flask(__name__)
FEEDBACK_LOG: list = []
FEEDBACK_LOG_MAX = 1000
FEEDBACK_LOG_LOCK = threading.Lock()

# ── GLOBAL RATE LIMITER (20/min per IP) ──────────────────────
_RATE_LIMIT = 20
_RATE_WINDOW = 60
_rate_data: dict = {}
_rate_lock = threading.Lock()
_rate_request_count = 0


def _check_global_rate_limit():
    global _rate_request_count
    ip = (request.access_route[0] if request.access_route else request.remote_addr) or "0.0.0.0"
    now = time.time()
    with _rate_lock:
        _rate_request_count += 1
        if ip not in _rate_data:
            _rate_data[ip] = collections.deque()
        dq = _rate_data[ip]
        while dq and dq[0] < now - _RATE_WINDOW:
            dq.popleft()
        if len(dq) >= _RATE_LIMIT:
            return False
        dq.append(now)
        if _rate_request_count % 100 == 0:
            stale = [k for k, v in _rate_data.items() if not v]
            for k in stale:
                del _rate_data[k]
    return True


@app.before_request
def enforce_rate_limit():
    if request.method == "POST":
        if not _check_global_rate_limit():
            return jsonify(error="Rate limit exceeded. Please wait a minute before making another request."), 429


# ── GENERATION-SPECIFIC RATE LIMITER (5/min per IP) ──────────
_GEN_RATE_LIMIT = 5
_GEN_RATE_WINDOW = 60
_gen_rate_data: dict = {}
_gen_rate_lock = threading.Lock()


def _check_gen_rate_limit():
    ip = (request.access_route[0] if request.access_route else request.remote_addr) or "0.0.0.0"
    now = time.time()
    with _gen_rate_lock:
        if ip not in _gen_rate_data:
            _gen_rate_data[ip] = collections.deque()
        dq = _gen_rate_data[ip]
        while dq and dq[0] < now - _GEN_RATE_WINDOW:
            dq.popleft()
        if len(dq) >= _GEN_RATE_LIMIT:
            return False
        dq.append(now)
    return True


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https: blob:; "
        "connect-src 'self'"
    )
    return response


# ── HELPERS ──────────────────────────────────────────────────
def _sanitize_text(text):
    text = (text or "").replace("**", "")
    return "".join(c for c in text if unicodedata.category(c) not in ("Cc", "Cs")).strip()


def _json_body():
    return request.get_json(silent=True) or {}


_MAX_FIELD_LEN = 4000

# ── FAMILY-SAFE CONTENT FILTERING ────────────────────────────
FAMILY_SAFE_SUFFIX = ", family friendly, appropriate for all ages, wholesome"
SAFETY_NEGATIVE = (
    "nudity, sexual content, revealing clothing, violence, gore, blood, weapons, "
    "alcohol, drugs, gambling, immodest clothing, inappropriate content, nsfw, "
    "adult content, suggestive, provocative, scary, frightening"
)

_SANITIZE_RULES = [
    (r'\b(bikini|lingerie|underwear|swimsuit|naked|nude|topless|shirtless)\b', 'person in modest clothing'),
    (r'\b(blood|gore|violent|violence|murder|kill|killing|dead\s+body)\b', 'dramatic scene'),
    (r'\b(alcohol|beer|wine|whiskey|drunk)\b', 'beverage'),
    (r'\b(gambling|casino|poker)\b', 'game'),
    (r'\b(nsfw|adult|sexual|explicit|erotic|xxx)\b', ''),
]


def _sanitize_prompt(prompt: str) -> str:
    """Silently sanitize prompts to ensure family-safe content."""
    result = prompt
    for pattern, replacement in _SANITIZE_RULES:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result.strip()


def _internal_error():
    logger.exception("Unhandled route error")
    return jsonify(error="Internal server error. Please try again."), 500


# ── RESPONSE CACHE (LRU, TTL 1 hr) ───────────────────────────
_resp_cache: collections.OrderedDict = collections.OrderedDict()
_CACHE_MAX = 500
_CACHE_TTL = 3600
_resp_cache_lock = threading.Lock()


def _cache_get(key):
    with _resp_cache_lock:
        if key in _resp_cache:
            val, ts = _resp_cache[key]
            if time.time() - ts < _CACHE_TTL:
                _resp_cache.move_to_end(key)
                return val
            del _resp_cache[key]
    return None


def _cache_set(key, val):
    with _resp_cache_lock:
        if key in _resp_cache:
            _resp_cache.move_to_end(key)
        elif len(_resp_cache) >= _CACHE_MAX:
            _resp_cache.popitem(last=False)
        _resp_cache[key] = (val, time.time())


# ── LLM FALLBACK CHAIN ────────────────────────────────────────
def _groq_llm(system, user):
    if not client:
        raise ValueError("GROQ_KEY not configured")
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=1500,
        temperature=0.7,
    )
    return _sanitize_text(r.choices[0].message.content)


def _cerebras_llm(system, user):
    key = os.environ.get("CEREBRAS_KEY")
    if not key:
        raise ValueError("CEREBRAS_KEY not configured")
    resp = requests.post(
        "https://api.cerebras.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.3-70b",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 1500, "temperature": 0.7,
        },
        timeout=45,
    )
    resp.raise_for_status()
    return _sanitize_text(resp.json()["choices"][0]["message"]["content"])


def _gemini_llm(system, user):
    key = os.environ.get("GEMINI_KEY")
    if not key:
        raise ValueError("GEMINI_KEY not configured")
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}",
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"role": "user", "parts": [{"text": system + "\n\n" + user}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1500},
        },
        timeout=45,
    )
    resp.raise_for_status()
    parts = resp.json()["candidates"][0]["content"]["parts"]
    return _sanitize_text("\n".join(p.get("text", "") for p in parts if p.get("text")))


def _cohere_llm(system, user):
    key = os.environ.get("COHERE_KEY")
    if not key:
        raise ValueError("COHERE_KEY not configured")
    resp = requests.post(
        "https://api.cohere.com/v2/chat",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "command-r-plus",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 1500, "temperature": 0.7,
        },
        timeout=45,
    )
    resp.raise_for_status()
    return _sanitize_text(resp.json()["message"]["content"][0]["text"])


def _mistral_llm(system, user):
    key = os.environ.get("MISTRAL_KEY")
    if not key:
        raise ValueError("MISTRAL_KEY not configured")
    resp = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "mistral-small-latest",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 1500, "temperature": 0.7,
        },
        timeout=45,
    )
    resp.raise_for_status()
    return _sanitize_text(resp.json()["choices"][0]["message"]["content"])


def _openrouter_llm(system, user):
    key = os.environ.get("OPENROUTER_KEY")
    if not key:
        raise ValueError("OPENROUTER_KEY not configured")
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "meta-llama/llama-3.3-70b-instruct:free",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 1500, "temperature": 0.7,
        },
        timeout=45,
    )
    resp.raise_for_status()
    return _sanitize_text(resp.json()["choices"][0]["message"]["content"])


def _huggingface_llm(system, user):
    key = os.environ.get("HF_KEY")
    if not key:
        raise ValueError("HF_KEY not configured")
    resp = requests.post(
        "https://router.hugging-face.cn/models/mistralai/Mistral-7B-Instruct-v0.3/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 1500, "temperature": 0.7,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return _sanitize_text(resp.json()["choices"][0]["message"]["content"])


_LLM_PROVIDERS = [
    ("Groq", _groq_llm),
    ("Cerebras", _cerebras_llm),
    ("Gemini", _gemini_llm),
    ("Cohere", _cohere_llm),
    ("Mistral", _mistral_llm),
    ("OpenRouter", _openrouter_llm),
    ("HuggingFace", _huggingface_llm),
]


def llm(system, user):
    full_system = system + "\n\nReturn ONLY the requested output, no extra commentary."
    cache_key = hashlib.md5((system + user).encode()).hexdigest()
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    for name, fn in _LLM_PROVIDERS:
        try:
            result = fn(full_system, user)
            if result:
                _cache_set(cache_key, result)
                return result
        except Exception as exc:
            logger.warning("LLM provider %s failed: %s", name, exc)
    return None


def _has_llm_key():
    return any([
        GROQ_KEY,
        os.environ.get("CEREBRAS_KEY"),
        os.environ.get("GEMINI_KEY"),
        os.environ.get("COHERE_KEY"),
        os.environ.get("MISTRAL_KEY"),
        os.environ.get("OPENROUTER_KEY"),
        HF_KEY,
    ])


# ── STYLE PROMPT ENHANCEMENT ─────────────────────────────────
_STYLE_TEMPLATES = {
    "photorealistic": "photorealistic, ultra detailed, 8k, professional photography, sharp focus, modest and family friendly, appropriate for all ages, {prompt}",
    "artistic":       "{prompt}, digital art, trending on artstation, vibrant colors, family friendly",
    "anime":          "anime style, {prompt}, studio ghibli inspired, detailed illustration, family friendly",
    "digital-art":    "digital art, {prompt}, concept art, highly detailed, family friendly",
    "oil-painting":   "oil painting, {prompt}, classical art, rich textures, family friendly",
    "watercolor":     "watercolor painting, {prompt}, soft colors, artistic, family friendly",
    "sketch":         "pencil sketch, {prompt}, detailed linework, clean lines",
    "cinematic":      "cinematic, {prompt}, movie still, dramatic lighting, family friendly",
    "abstract":       "abstract art, {prompt}, surreal, vibrant colors, geometric",
}


def _apply_style(prompt, style):
    template = _STYLE_TEMPLATES.get((style or "").lower().replace(" ", "-"), "{prompt}")
    return template.replace("{prompt}", prompt)


# ── ASPECT RATIO → DIMENSIONS ────────────────────────────────
_ASPECT_DIMS = {
    "square":    (1024, 1024),
    "landscape": (1344, 768),
    "portrait":  (768, 1344),
    "wide":      (1216, 832),
    "tall":      (832, 1216),
}


def _get_dims(aspect_ratio):
    return _ASPECT_DIMS.get((aspect_ratio or "square").lower(), (1024, 1024))


# ── IMAGE PROVIDERS ───────────────────────────────────────────

def _generate_via_fal(prompt, negative_prompt, width, height, num_images, **kwargs):
    if not FAL_KEY:
        raise ValueError("FAL_KEY not configured")
    model = kwargs.get("fal_model", "fal-ai/flux/schnell")
    if model not in ("fal-ai/flux/schnell", "fal-ai/flux-pro",
                     "fal-ai/stable-diffusion-v3-medium", "fal-ai/recraft-v3"):
        model = "fal-ai/flux/schnell"

    payload = {
        "prompt": prompt,
        "image_size": {"width": width, "height": height},
        "num_images": num_images,
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt

    submit = requests.post(
        f"https://queue.fal.run/{model}",
        headers={"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    if submit.status_code not in (200, 201):
        raise RuntimeError(f"fal.ai submission failed ({submit.status_code}): {submit.text[:300]}")

    request_id = submit.json().get("request_id")
    if not request_id:
        raise RuntimeError("fal.ai returned no request_id")

    # Poll until COMPLETED (up to 90 seconds, every 2 seconds)
    for _ in range(45):
        time.sleep(2)
        poll = requests.get(
            f"https://queue.fal.run/{model}/requests/{request_id}",
            headers={"Authorization": f"Key {FAL_KEY}"},
            timeout=30,
        )
        if poll.status_code != 200:
            continue
        data = poll.json()
        status = data.get("status", "")
        if status == "COMPLETED":
            images = data.get("images") or []
            urls = [img.get("url") for img in images if img.get("url")]
            if urls:
                return urls
            response_url = data.get("responseUrl") or data.get("response_url")
            if response_url:
                r2 = requests.get(response_url, headers={"Authorization": f"Key {FAL_KEY}"}, timeout=30)
                if r2.status_code == 200:
                    imgs = r2.json().get("images") or []
                    return [img.get("url") for img in imgs if img.get("url")]
            raise RuntimeError("fal.ai completed but no image URLs found")
        if status in ("FAILED", "ERROR"):
            raise RuntimeError(f"fal.ai generation failed: {data.get('error', 'unknown error')}")

    raise RuntimeError("fal.ai timed out after 90 seconds")


def _generate_via_huggingface(prompt, negative_prompt, width, height, num_images, **kwargs):
    if not HF_KEY:
        raise ValueError("HF_KEY not configured")
    models_to_try = [
        "black-forest-labs/FLUX.1-schnell",
        "stabilityai/stable-diffusion-xl-base-1.0",
    ]
    urls = []
    for _ in range(min(num_images, 4)):
        image_b64 = None
        for model in models_to_try:
            try:
                resp = requests.post(
                    f"https://api-inference.huggingface.co/models/{model}",
                    headers={"Authorization": f"Bearer {HF_KEY}", "Content-Type": "application/json"},
                    json={"inputs": prompt},
                    timeout=60,
                )
                if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/"):
                    content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
                    image_b64 = f"data:{content_type};base64," + base64.b64encode(resp.content).decode()
                    break
            except Exception as exc:
                logger.warning("HuggingFace model %s failed: %s", model, exc)
                continue
        if image_b64:
            urls.append(image_b64)
        else:
            raise RuntimeError("HuggingFace returned no image data")
    if not urls:
        raise RuntimeError("HuggingFace returned no images")
    return urls


def _generate_via_stability(prompt, negative_prompt, width, height, num_images, **kwargs):
    if not STABILITY_KEY:
        raise ValueError("STABILITY_KEY not configured")
    payload = {
        "text_prompts": [{"text": prompt, "weight": 1.0}],
        "cfg_scale": 7,
        "height": height,
        "width": width,
        "steps": 30,
        "samples": min(num_images, 4),
    }
    if negative_prompt:
        payload["text_prompts"].append({"text": negative_prompt, "weight": -1.0})

    resp = requests.post(
        "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
        headers={
            "Authorization": f"Bearer {STABILITY_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=payload,
        timeout=90,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Stability AI error ({resp.status_code}): {resp.text[:300]}")
    artifacts = resp.json().get("artifacts", [])
    urls = []
    for art in artifacts:
        b64 = art.get("base64")
        if b64:
            urls.append("data:image/png;base64," + b64)
    if not urls:
        raise RuntimeError("Stability AI returned no images")
    return urls


def _generate_via_replicate(prompt, negative_prompt, width, height, num_images, **kwargs):
    if not REPLICATE_KEY:
        raise ValueError("REPLICATE_KEY not configured")
    payload = {
        "input": {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_outputs": min(num_images, 4),
        }
    }
    if negative_prompt:
        payload["input"]["negative_prompt"] = negative_prompt

    submit = requests.post(
        "https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions",
        headers={"Authorization": f"Bearer {REPLICATE_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    if submit.status_code not in (200, 201):
        raise RuntimeError(f"Replicate submission failed ({submit.status_code}): {submit.text[:300]}")

    pred_id = submit.json().get("id")
    if not pred_id:
        raise RuntimeError("Replicate returned no prediction ID")

    # Poll until succeeded (up to 90 seconds, every 3 seconds)
    for _ in range(30):
        time.sleep(3)
        poll = requests.get(
            f"https://api.replicate.com/v1/predictions/{pred_id}",
            headers={"Authorization": f"Bearer {REPLICATE_KEY}"},
            timeout=30,
        )
        if poll.status_code != 200:
            continue
        data = poll.json()
        status = data.get("status", "")
        if status == "succeeded":
            output = data.get("output") or []
            if isinstance(output, str):
                output = [output]
            urls = [u for u in output if u]
            if urls:
                return urls
            raise RuntimeError("Replicate succeeded but no output URLs")
        if status == "failed":
            raise RuntimeError(f"Replicate generation failed: {data.get('error', 'unknown error')}")

    raise RuntimeError("Replicate timed out after 90 seconds")


def _generate_via_pollinations(prompt, negative_prompt, width, height, num_images, **kwargs):
    encoded = urllib.parse.quote(prompt)
    base_url = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&nologo=true&safe=1"
    urls = []
    for i in range(min(num_images, 4)):
        seed_url = base_url + (f"&seed={i * 1000 + int(time.time()) % 10000}" if i > 0 else "")
        urls.append(seed_url)
    return urls


# ── PROVIDER DISPATCH ─────────────────────────────────────────
_IMAGE_PROVIDERS = [
    ("fal.ai",       _generate_via_fal),
    ("huggingface",  _generate_via_huggingface),
    ("stability",    _generate_via_stability),
    ("replicate",    _generate_via_replicate),
    ("pollinations", _generate_via_pollinations),
]


def _generate_images(prompt, negative_prompt, width, height, num_images):
    """Try each provider in order, silently fall back on failure."""
    errors = []
    for name, fn in _IMAGE_PROVIDERS:
        try:
            urls = fn(prompt, negative_prompt, width, height, num_images)
            if urls:
                logger.info("Images generated via %s", name)
                return urls, name
        except Exception as exc:
            logger.warning("Image provider %s failed: %s", name, exc)
            errors.append(f"{name}: {exc}")
            continue
    logger.error("All image providers failed: %s", " | ".join(errors))
    return None, None


# ── UI HTML ───────────────────────────────────────────────────
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PureImage AI</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>&#127912;</text></svg>">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --primary: #4f46e5;
    --accent: #7c3aed;
    --bg: #0f0f1a;
    --surface: #1a1a2e;
    --surface2: #16213e;
    --border: #2d2d4e;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --danger: #ef4444;
  }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }

  header {
    background: linear-gradient(135deg, var(--primary), var(--accent));
    padding: 2rem 1rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(79,70,229,.4);
  }
  header h1 { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px; }
  header p { margin-top: .4rem; opacity: .85; font-size: 1rem; }

  main { max-width: 900px; margin: 2rem auto; padding: 0 1rem 4rem; }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
  }

  label { display: block; font-size: .85rem; font-weight: 600; color: var(--muted); margin-bottom: .4rem; text-transform: uppercase; letter-spacing: .05em; }

  textarea, input[type="text"], select {
    width: 100%;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    padding: .75rem 1rem;
    font-size: 1rem;
    outline: none;
    transition: border-color .2s;
  }
  textarea:focus, input[type="text"]:focus, select:focus { border-color: var(--primary); }
  textarea { resize: vertical; min-height: 100px; }
  select option { background: var(--surface2); }

  .prompt-actions { display: flex; gap: .5rem; margin-top: .5rem; flex-wrap: wrap; }

  .controls-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
  }

  .advanced-toggle {
    background: none; border: none; color: var(--primary); cursor: pointer;
    font-size: .9rem; padding: .3rem 0; text-decoration: underline;
  }
  .advanced-section { display: none; margin-top: 1rem; }
  .advanced-section.open { display: block; }

  button {
    cursor: pointer; border: none; border-radius: 8px;
    font-size: .9rem; font-weight: 600; padding: .6rem 1.2rem;
    transition: opacity .2s, transform .1s;
  }
  button:active { transform: scale(.97); }
  button:disabled { opacity: .5; cursor: not-allowed; }

  .btn-primary {
    background: linear-gradient(135deg, var(--primary), var(--accent));
    color: #fff;
    font-size: 1.1rem;
    padding: .9rem 2rem;
    width: 100%;
    margin-top: 1.2rem;
  }
  .btn-secondary {
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
  }
  .btn-secondary:hover { border-color: var(--primary); }

  #status-bar {
    display: none; text-align: center; padding: 1rem;
    color: var(--muted); font-size: .95rem;
  }
  .spinner {
    display: inline-block; width: 20px; height: 20px;
    border: 3px solid var(--border); border-top-color: var(--primary);
    border-radius: 50%; animation: spin .8s linear infinite;
    vertical-align: middle; margin-right: .5rem;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  #error-msg {
    display: none; background: rgba(239,68,68,.1);
    border: 1px solid var(--danger); border-radius: 8px;
    color: var(--danger); padding: 1rem; margin-top: 1rem;
    font-size: .95rem;
  }

  #image-gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1rem;
    margin-top: 1.5rem;
  }
  .img-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    transition: transform .2s, box-shadow .2s;
  }
  .img-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(79,70,229,.2); }
  .img-card img { width: 100%; display: block; aspect-ratio: 1; object-fit: cover; min-height: 200px; background: var(--surface2); }
  .img-card-footer {
    padding: .75rem 1rem;
    display: flex; justify-content: space-between; align-items: center; gap: .5rem;
  }
  .provider-badge {
    font-size: .75rem; background: rgba(79,70,229,.2);
    color: var(--primary); border: 1px solid rgba(79,70,229,.3);
    border-radius: 20px; padding: .2rem .7rem; font-weight: 600;
  }
  .btn-download { background: var(--primary); color: #fff; font-size: .8rem; padding: .35rem .8rem; }
  .btn-download:hover { opacity: .85; }

  @media (max-width: 600px) {
    header h1 { font-size: 1.6rem; }
    .controls-grid { grid-template-columns: 1fr 1fr; }
    #image-gallery { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<header>
  <h1>PureImage AI</h1>
  <p>Create stunning images with AI &mdash; Family Safe &#10003;</p>
  <div style="display:flex;gap:.6rem;justify-content:center;flex-wrap:wrap;margin-top:.8rem;">
    <span style="background:rgba(255,255,255,.15);border-radius:20px;padding:.25rem .75rem;font-size:.8rem;">&#10024; AI Powered</span>
    <span style="background:rgba(255,255,255,.15);border-radius:20px;padding:.25rem .75rem;font-size:.8rem;">&#128106; Family Safe</span>
    <span style="background:rgba(255,255,255,.15);border-radius:20px;padding:.25rem .75rem;font-size:.8rem;">&#128444; Multiple Styles</span>
    <span style="background:rgba(255,255,255,.15);border-radius:20px;padding:.25rem .75rem;font-size:.8rem;">&#9889; Fast Generation</span>
  </div>
</header>

<main>
  <div class="card">
    <label for="prompt">Image Prompt</label>
    <textarea id="prompt" placeholder="Describe the image you want to create..."></textarea>
    <div class="prompt-actions">
      <button class="btn-secondary" onclick="copyPrompt(this)">&#128203; Copy Prompt</button>
      ENHANCE_BTN_PLACEHOLDER
    </div>

    <div style="margin-top:1rem;">
      <button class="advanced-toggle" onclick="toggleAdvanced()">&#9881; Advanced Options</button>
    </div>
    <div id="advanced-section" class="advanced-section">
      <label for="negative-prompt" style="margin-top:.5rem;">Negative Prompt</label>
      <input type="text" id="negative-prompt" placeholder="blurry, low quality, distorted...">
    </div>

    <div class="controls-grid">
      <div>
        <label for="style">Style</label>
        <select id="style">
          <option value="none">None</option>
          <option value="photorealistic">Photorealistic</option>
          <option value="artistic">Artistic</option>
          <option value="anime">Anime</option>
          <option value="digital-art">Digital Art</option>
          <option value="oil-painting">Oil Painting</option>
          <option value="watercolor">Watercolor</option>
          <option value="sketch">Sketch</option>
          <option value="cinematic">Cinematic</option>
          <option value="abstract">Abstract</option>
        </select>
      </div>
      <div>
        <label for="aspect-ratio">Aspect Ratio</label>
        <select id="aspect-ratio">
          <option value="square">Square (1:1)</option>
          <option value="landscape">Landscape (16:9)</option>
          <option value="portrait">Portrait (9:16)</option>
          <option value="wide">Wide (3:2)</option>
          <option value="tall">Tall (2:3)</option>
        </select>
      </div>
      <div>
        <label for="num-images">Number of Images</label>
        <select id="num-images">
          <option value="1">1</option>
          <option value="2">2</option>
          <option value="4">4</option>
        </select>
      </div>
    </div>

    <button class="btn-primary" id="gen-btn" onclick="generateImages()">Generate Images</button>
  </div>

  <div id="status-bar"><span class="spinner"></span><span id="status-text">Generating...</span></div>
  <div id="error-msg"></div>
  <div id="image-gallery"></div>
</main>

<footer style="text-align:center;padding:1.5rem 1rem;color:var(--muted);font-size:.85rem;border-top:1px solid var(--border);margin-top:2rem;">
  PureImage AI &nbsp;|&nbsp; &#10003; Family Safe AI Image Generation &nbsp;|&nbsp; &#9888;&#65039; AI-generated images may not always be perfect. Review before use.
</footer>

<script>
function toggleAdvanced() {
  document.getElementById('advanced-section').classList.toggle('open');
}

function copyPrompt(btn) {
  const p = document.getElementById('prompt').value.trim();
  if (!p) return;
  navigator.clipboard.writeText(p).then(() => {
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = orig, 1500);
  });
}

ENHANCE_JS_PLACEHOLDER

async function generateImages() {
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) { showError('Please enter a prompt.'); return; }

  const btn = document.getElementById('gen-btn');
  const statusBar = document.getElementById('status-bar');
  const statusText = document.getElementById('status-text');
  const errDiv = document.getElementById('error-msg');
  const gallery = document.getElementById('image-gallery');

  btn.disabled = true;
  statusBar.style.display = 'block';
  errDiv.style.display = 'none';
  gallery.innerHTML = '';

  statusText.textContent = '\u2728 Creating your images...';

  try {
    const resp = await fetch('/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        prompt,
        negative_prompt: document.getElementById('negative-prompt').value.trim(),
        style: document.getElementById('style').value,
        aspect_ratio: document.getElementById('aspect-ratio').value,
        num_images: parseInt(document.getElementById('num-images').value, 10),
      })
    });

    const data = await resp.json();
    if (!resp.ok || data.error) {
      showError(data.error || 'Generation failed. Please try again.');
      return;
    }

    renderImages(data.images, data.elapsed_ms);
  } catch(e) {
    showError('Network error: ' + e.message);
  } finally {
    btn.disabled = false;
    statusBar.style.display = 'none';
  }
}

function showError(msg) {
  const errDiv = document.getElementById('error-msg');
  errDiv.textContent = msg;
  errDiv.style.display = 'block';
}

function renderImages(images, elapsedMs) {
  const gallery = document.getElementById('image-gallery');
  gallery.innerHTML = '';
  if (!images || images.length === 0) {
    showError('No images returned. Please try again.');
    return;
  }
  images.forEach((img, i) => {
    const card = document.createElement('div');
    card.className = 'img-card';
    const src = img.url;

    const imgEl = document.createElement('img');
    imgEl.alt = 'Generated image ' + (i + 1);
    imgEl.style.background = 'var(--surface2)';
    imgEl.style.minHeight = '280px';
    imgEl.loading = 'lazy';

    imgEl.onerror = function() {
      const placeholder = document.createElement('div');
      placeholder.style.cssText = 'width:100%;min-height:280px;display:flex;flex-direction:column;align-items:center;justify-content:center;background:var(--surface2);color:var(--muted);font-size:.85rem;gap:.5rem;padding:1rem;text-align:center;';
      placeholder.innerHTML = '<span style="font-size:2rem">\uD83D\uDDBC\uFE0F</span><span>Image failed to load.<br>Try generating again.</span>';
      this.replaceWith(placeholder);
    };

    // Use proxy for external URLs to avoid CORS issues
    const displaySrc = src.startsWith('data:') ? src : '/proxy_image?url=' + encodeURIComponent(src);
    imgEl.src = displaySrc;

    const footer = document.createElement('div');
    footer.className = 'img-card-footer';

    const badge = document.createElement('span');
    badge.className = 'provider-badge';
    badge.textContent = '\u2728 AI Generated';

    const dlBtn = document.createElement('button');
    dlBtn.className = 'btn-download';
    dlBtn.textContent = '\u2B07 Download';
    dlBtn.dataset.src = src;
    dlBtn.dataset.idx = String(i + 1);
    dlBtn.addEventListener('click', function() {
      downloadImage(this.dataset.src, this.dataset.idx);
    });

    footer.appendChild(badge);
    footer.appendChild(dlBtn);
    card.appendChild(imgEl);
    card.appendChild(footer);
    gallery.appendChild(card);
  });

  if (elapsedMs) {
    const info = document.createElement('p');
    info.style.cssText = 'text-align:center;color:var(--muted);font-size:.85rem;margin-top:.5rem;';
    info.textContent = 'Generated in ' + (elapsedMs / 1000).toFixed(1) + 's';
    gallery.appendChild(info);
  }
}

function escHtml(str) {
  return String(str).replace(/[&<>"']/g, function(c) {
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
  });
}

async function downloadImage(src, idx) {
  try {
    if (src.startsWith('data:')) {
      const a = document.createElement('a');
      a.href = src; a.download = 'pureimage-' + idx + '.png'; a.click();
      return;
    }
    const resp = await fetch(src);
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'pureimage-' + idx + '.png'; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  } catch(e) {
    window.open(src, '_blank');
  }
}
</script>
</body>
</html>"""

_ENHANCE_BTN_HTML = '<button class="btn-secondary" onclick="enhancePrompt(this)">&#10024; Enhance Prompt</button>'

_ENHANCE_JS_CODE = """
async function enhancePrompt(btn) {
  const promptEl = document.getElementById('prompt');
  const raw = promptEl.value.trim();
  if (!raw) { showError('Enter a prompt first.'); return; }
  const orig = btn.textContent;
  btn.textContent = 'Enhancing...';
  btn.disabled = true;
  try {
    const resp = await fetch('/enhance_prompt', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt: raw})
    });
    const data = await resp.json();
    if (data.enhanced) {
      promptEl.value = data.enhanced;
    } else {
      showError(data.error || 'Enhancement failed.');
    }
  } catch(e) {
    showError('Network error: ' + e.message);
  } finally {
    btn.textContent = orig;
    btn.disabled = false;
  }
}
"""


def _render_html():
    if _has_llm_key():
        html = _HTML.replace("ENHANCE_BTN_PLACEHOLDER", _ENHANCE_BTN_HTML)
        html = html.replace("ENHANCE_JS_PLACEHOLDER", _ENHANCE_JS_CODE)
    else:
        html = _HTML.replace("ENHANCE_BTN_PLACEHOLDER", "")
        html = html.replace("ENHANCE_JS_PLACEHOLDER", "function enhancePrompt(){}")
    return html


# ── ROUTES ────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(_render_html())


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/proxy_image")
def proxy_image():
    url = request.args.get("url", "").strip()
    if not url:
        return jsonify(error="Missing url parameter"), 400
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return jsonify(error="Invalid URL"), 400
    allowed_hosts = (
        "image.pollinations.ai",
        "fal.media", "fal.run",
        "replicate.delivery",
        "pbxt.replicate.delivery",
    )
    netloc = parsed.netloc.split(":")[0]  # strip port if present
    if parsed.scheme not in ("http", "https") or not any(
        netloc == h or netloc.endswith("." + h) for h in allowed_hosts
    ):
        return jsonify(error="Disallowed image URL"), 400
    # Reconstruct URL from validated components to prevent SSRF bypass
    safe_url = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", parsed.query, ""))
    try:
        resp = requests.get(safe_url, timeout=30, stream=True)
        if resp.status_code != 200:
            return jsonify(error="Image fetch failed"), 502
        content_type = (resp.headers.get("Content-Type") or "image/jpeg").split(";")[0].strip()
        return Response(resp.content, status=200, content_type=content_type)
    except Exception as exc:
        logger.warning("proxy_image failed for %s: %s", safe_url, exc)
        return jsonify(error="Could not fetch image"), 502


@app.route("/generate", methods=["POST"])
def generate():
    t0 = time.time()
    try:
        if not _check_gen_rate_limit():
            return jsonify(error="Too many requests. Please wait a moment and try again."), 429

        d = _json_body()
        raw_prompt = (d.get("prompt") or "").strip()
        if not raw_prompt:
            return jsonify(error="Please enter a prompt."), 400
        if len(raw_prompt) > _MAX_FIELD_LEN:
            return jsonify(error="Prompt is too long."), 400

        user_negative = (d.get("negative_prompt") or "").strip()
        style = (d.get("style") or "none").strip().lower()
        aspect_ratio = (d.get("aspect_ratio") or "square").strip().lower()
        num_images = int(d.get("num_images") or 1)
        num_images = max(1, min(num_images, 4))

        # Silently sanitize prompt for family-safe content
        sanitized_prompt = _sanitize_prompt(raw_prompt)

        if style and style != "none":
            styled_prompt = _apply_style(sanitized_prompt, style)
        else:
            styled_prompt = sanitized_prompt

        # Append family-safe suffix to every prompt
        final_prompt = styled_prompt + FAMILY_SAFE_SUFFIX

        # Merge user negative prompt with safety negatives
        if user_negative:
            final_negative = user_negative + ", " + SAFETY_NEGATIVE
        else:
            final_negative = SAFETY_NEGATIVE

        width, height = _get_dims(aspect_ratio)

        # Cache key based on final prompt + settings (robust serialization to avoid collisions)
        cache_key = hashlib.sha256(
            json.dumps([final_prompt, final_negative, aspect_ratio, num_images], sort_keys=True).encode()
        ).hexdigest()
        cached = _cache_get(cache_key)
        if cached is not None:
            return jsonify(**cached)

        image_urls, provider_used = _generate_images(
            final_prompt, final_negative, width, height, num_images
        )

        if not image_urls:
            logger.error("All image providers failed")
            return jsonify(error="Image generation is temporarily unavailable. Please try again in a moment."), 502

        elapsed_ms = int((time.time() - t0) * 1000)
        # Never expose provider name to users — just return image URLs
        images = [{"url": u} for u in image_urls]

        entry = {
            "ts": time.time(),
            "prompt_hash": hashlib.sha256(sanitized_prompt.encode()).hexdigest(),
            "style": style,
            "aspect_ratio": aspect_ratio,
            "num_images": len(images),
            "provider_used": provider_used,
            "elapsed_ms": elapsed_ms,
            "success": True,
        }
        with FEEDBACK_LOG_LOCK:
            FEEDBACK_LOG.append(entry)
            if len(FEEDBACK_LOG) > FEEDBACK_LOG_MAX:
                FEEDBACK_LOG.pop(0)
        try:
            with open(PUREIMAGE_LOG_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

        result = {"images": images, "elapsed_ms": elapsed_ms}
        _cache_set(cache_key, result)
        return jsonify(**result)

    except Exception:
        return _internal_error()


@app.route("/enhance_prompt", methods=["POST"])
def enhance_prompt():
    try:
        d = _json_body()
        raw = (d.get("prompt") or "").strip()
        if not raw:
            return jsonify(error="Please enter a prompt."), 400
        if not _has_llm_key():
            return jsonify(error="No LLM key configured for prompt enhancement."), 503

        enhanced = llm(
            "You are an expert AI image prompt engineer. Expand and enhance the user's prompt to be more "
            "descriptive, vivid, and detailed for AI image generation. Return ONLY the enhanced prompt, no explanation.",
            raw,
        )
        if not enhanced:
            return jsonify(error="Enhancement failed. Please try again."), 502
        return jsonify(enhanced=enhanced)
    except Exception:
        return _internal_error()


if __name__ == "__main__":
    logger.info("PureImage AI starting...")
    logger.info("FAL_KEY: %s", "configured" if FAL_KEY else "not configured")
    logger.info("HF_KEY: %s", "configured" if HF_KEY else "not configured")
    logger.info("STABILITY_KEY: %s", "configured" if STABILITY_KEY else "not configured")
    logger.info("REPLICATE_KEY: %s", "configured" if REPLICATE_KEY else "not configured")
    logger.info("Pollinations: always available (no key needed)")
    logger.info("LLM for enhancement: %s", "available" if _has_llm_key() else "no key configured")
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
