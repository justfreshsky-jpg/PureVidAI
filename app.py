import os
os.environ['HTTPX_PROXIES'] = 'null'  # Fix Render/httpx proxies bug
import re
import logging
import time
import threading
import unicodedata
import collections
import hashlib
import requests
from flask import Flask, request, jsonify, render_template_string
from groq import Groq

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# ── CONFIG ───────────────────────────────────────────────────
GROQ_KEY = os.environ.get("GROQ_KEY")
FAL_KEY = os.environ.get("FAL_KEY")
VERTEX_PROJECT_ID = (
    os.environ.get("VERTEX_PROJECT_ID")
    or os.environ.get("GOOGLE_CLOUD_PROJECT")
    or os.environ.get("GCP_PROJECT")
    or os.environ.get("GCLOUD_PROJECT")
)
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.environ.get("VERTEX_MODEL", "gemini-1.5-flash")
VIDEO_PROVIDER = os.environ.get("VIDEO_PROVIDER", "google").strip().lower()  # google|fal
VERTEX_VIDEO_MODEL = os.environ.get("VERTEX_VIDEO_MODEL", "veo-2.0-generate-001")
ALLOW_FAL_FALLBACK = os.environ.get("ALLOW_FAL_FALLBACK", "false").lower() == "true"
FEEDBACK_LOG_PATH = os.environ.get("FEEDBACK_LOG_PATH", "/tmp/purevid_feedback.log.jsonl")
client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

app = Flask(__name__)
FEEDBACK_LOG = []
FEEDBACK_LOG_MAX = 1000
FEEDBACK_LOG_LOCK = threading.Lock()

# ── GLOBAL RATE LIMITER (all POST requests) ──────────────────
_RATE_LIMIT = 20
_RATE_WINDOW = 60
_rate_data: dict = {}
_rate_lock = threading.Lock()
_rate_request_count = 0

def _check_global_rate_limit():
    global _rate_request_count
    ip = (request.access_route[0] if request.access_route else request.remote_addr) or '0.0.0.0'
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
        # Clean up empty keys every 100 requests to avoid unbounded growth
        if _rate_request_count % 100 == 0:
            stale = [k for k, v in _rate_data.items() if not v]
            for k in stale:
                del _rate_data[k]
    return True

@app.before_request
def enforce_rate_limit():
    if request.method == 'POST':
        if not _check_global_rate_limit():
            return jsonify(error="Rate limit exceeded. Please wait a minute before making another request."), 429

# ── VIDEO-SPECIFIC RATE LIMITER (5/min per IP) ───────────────
_VIDEO_RATE_LIMIT = 5
_VIDEO_RATE_WINDOW = 60
_video_rate_data: dict = {}
_video_rate_lock = threading.Lock()

def _check_video_rate_limit():
    ip = (request.access_route[0] if request.access_route else request.remote_addr) or '0.0.0.0'
    now = time.time()
    with _video_rate_lock:
        if ip not in _video_rate_data:
            _video_rate_data[ip] = collections.deque()
        dq = _video_rate_data[ip]
        while dq and dq[0] < now - _VIDEO_RATE_WINDOW:
            dq.popleft()
        if len(dq) >= _VIDEO_RATE_LIMIT:
            return False
        dq.append(now)
    return True

@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "font-src 'self' https://cdnjs.cloudflare.com; "
        "img-src 'self' data:; "
        "media-src 'self' blob: data: https://storage.googleapis.com https://queue.fal.run; "
        "connect-src 'self'"
    )
    return response

UNSAFE = ["nudity","naked","violence","blood","kill","alcohol","drugs","gambling","weapon","gore","nsfw","sexy","adult","explicit","hate","terrorist"]
UNSAFE_PATTERN = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in UNSAFE) + r")\b", re.IGNORECASE)

def is_safe(prompt):
    return not bool(UNSAFE_PATTERN.search(prompt or ""))

# ── STATIC CONTEXT ──────────────────────────────────────────
FALLBACK = """
[VIDEO TIPS] CogVideoX works best with detailed cinematic descriptions. Include: lighting, camera angle, motion, mood.
[PROMPTS] Good structure: [Subject] + [Action] + [Setting] + [Lighting] + [Camera] + [Style]
[STYLES] Cinematic, 4K, golden hour, soft bokeh, aerial drone, slow motion, time lapse all improve results.
[SAFETY] Family-safe content: nature, children, animals, food, travel, celebrations, seasons.
[FAL.AI] CogVideoX-5b generates 49 frames (~6 sec). 16:9 best for YouTube. 9:16 for Reels/TikTok.
[IDEAS] Trending safe topics: Eid celebrations, nature scenery, family moments, cooking, travel vlogs.
[NEGATIVE] Always add: no violence, no adult content, no text overlays, no watermarks.
[ENHANCE] Add: "cinematic lighting, photorealistic, 8K, shallow depth of field, professional grade".
"""

def get_context():
    return FALLBACK

def _normalized_video_provider():
    provider = (VIDEO_PROVIDER or "google").strip().lower()
    return provider if provider in {"google", "fal"} else "google"

def _json_body():
    return request.get_json(silent=True) or {}

_MAX_FIELD_LEN = 4000

def _get_json(required_fields):
    data = request.get_json(silent=True) or {}
    missing = [field for field in required_fields if not str(data.get(field, "")).strip()]
    if missing:
        return None, jsonify(error=f"Missing required field(s): {', '.join(missing)}"), 400
    for field, val in data.items():
        if isinstance(val, str) and len(val) > _MAX_FIELD_LEN:
            return None, jsonify(error=f"Field '{field}' exceeds maximum length of {_MAX_FIELD_LEN} characters."), 400
    return data, None, None

def _internal_error():
    logger.exception("Unhandled route error")
    return jsonify(error="Internal server error. Please try again."), 500

# ── LLM ─────────────────────────────────────────────────────
def _sanitize_text(text):
    text = (text or "").replace('**', '')
    return ''.join(c for c in text
                   if unicodedata.category(c) not in ('Cc', 'Cs')).strip()

# Response cache (OrderedDict for O(1) LRU eviction)
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


def _vertex_llm(full_system, user):
    token, project_id = _get_google_auth_context()
    if not project_id:
        raise RuntimeError("Google Cloud project is not configured. Set VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT.")

    endpoint = (
        f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1/projects/"
        f"{project_id}/locations/{VERTEX_LOCATION}/publishers/google/models/"
        f"{VERTEX_MODEL}:generateContent"
    )
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": f"System:\n{full_system}\n\nUser:\n{user}"}]}
        ],
        "generationConfig": {"temperature": 0.6, "maxOutputTokens": 2000}
    }
    res = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=40,
    )
    if res.status_code != 200:
        raise RuntimeError(f"Vertex AI error: {res.text[:400]}")

    data = res.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Vertex AI returned no candidates.")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "\n".join(part.get("text", "") for part in parts if part.get("text"))
    return _sanitize_text(text)


def _groq_llm(full_system, user):
    if not client:
        raise RuntimeError("Groq client is not configured. Provide GROQ_KEY.")

    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": full_system}, {"role": "user", "content": user}],
        max_tokens=2000,
        temperature=0.6,
    )
    return _sanitize_text(r.choices[0].message.content)


def _cerebras_llm(full_system, user):
    key = os.environ.get("CEREBRAS_KEY")
    if not key:
        raise ValueError("CEREBRAS_KEY is not set.")
    res = requests.post(
        "https://api.cerebras.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.3-70b",
            "messages": [{"role": "system", "content": full_system}, {"role": "user", "content": user}],
            "max_tokens": 900,
            "temperature": 0.6,
        },
        timeout=45,
    )
    if res.status_code != 200:
        raise RuntimeError(f"Cerebras error: {res.text[:400]}")
    return _sanitize_text(res.json()["choices"][0]["message"]["content"])


def _gemini_llm(full_system, user):
    key = os.environ.get("GEMINI_KEY")
    if not key:
        raise ValueError("GEMINI_KEY is not set.")
    res = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}",
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"role": "user", "parts": [{"text": f"System:\n{full_system}\n\nUser:\n{user}"}]}],
            "generationConfig": {"temperature": 0.6, "maxOutputTokens": 900},
        },
        timeout=45,
    )
    if res.status_code != 200:
        raise RuntimeError(f"Gemini error: {res.text[:400]}")
    data = res.json()
    parts = data["candidates"][0]["content"]["parts"]
    return _sanitize_text("\n".join(p.get("text", "") for p in parts if p.get("text")))


def _cohere_llm(full_system, user):
    key = os.environ.get("COHERE_KEY")
    if not key:
        raise ValueError("COHERE_KEY is not set.")
    res = requests.post(
        "https://api.cohere.com/v2/chat",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "command-r-plus",
            "messages": [
                {"role": "system", "content": full_system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 900,
            "temperature": 0.6,
        },
        timeout=45,
    )
    if res.status_code != 200:
        raise RuntimeError(f"Cohere error: {res.text[:400]}")
    return _sanitize_text(res.json()["message"]["content"][0]["text"])


def _mistral_llm(full_system, user):
    key = os.environ.get("MISTRAL_KEY")
    if not key:
        raise ValueError("MISTRAL_KEY is not set.")
    res = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "mistral-small-latest",
            "messages": [{"role": "system", "content": full_system}, {"role": "user", "content": user}],
            "max_tokens": 900,
            "temperature": 0.6,
        },
        timeout=45,
    )
    if res.status_code != 200:
        raise RuntimeError(f"Mistral error: {res.text[:400]}")
    return _sanitize_text(res.json()["choices"][0]["message"]["content"])


def _openrouter_llm(full_system, user):
    key = os.environ.get("OPENROUTER_KEY")
    if not key:
        raise ValueError("OPENROUTER_KEY is not set.")
    res = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "meta-llama/llama-3.3-70b-instruct:free",
            "messages": [{"role": "system", "content": full_system}, {"role": "user", "content": user}],
            "max_tokens": 900,
            "temperature": 0.6,
        },
        timeout=45,
    )
    if res.status_code != 200:
        raise RuntimeError(f"OpenRouter error: {res.text[:400]}")
    return _sanitize_text(res.json()["choices"][0]["message"]["content"])


def _huggingface_llm(full_system, user):
    key = os.environ.get("HF_KEY")
    if not key:
        raise ValueError("HF_KEY is not set.")
    res = requests.post(
        "https://router.hugging-face.cn/models/mistralai/Mistral-7B-Instruct-v0.3/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "messages": [{"role": "system", "content": full_system}, {"role": "user", "content": user}],
            "max_tokens": 900,
            "temperature": 0.6,
        },
        timeout=60,
    )
    if res.status_code != 200:
        raise RuntimeError(f"HuggingFace error: {res.text[:400]}")
    return _sanitize_text(res.json()["choices"][0]["message"]["content"])


_PROVIDERS = [
    ("Groq", _groq_llm),
    ("Cerebras", _cerebras_llm),
    ("Gemini", _gemini_llm),
    ("Cohere", _cohere_llm),
    ("Mistral", _mistral_llm),
    ("OpenRouter", _openrouter_llm),
    ("HuggingFace", _huggingface_llm),
]


def _is_vertex_config_error(exc):
    msg = str(exc or "").lower()
    config_markers = [
        "project is not configured",
        "default credentials",
        "could not automatically determine credentials",
        "google-auth package is required",
        "your default credentials were not found",
    ]
    return any(marker in msg for marker in config_markers)


def llm(system, user):
    assistant_prompt = """
    🌍 GENERAL CREATIVE ASSISTANT
    ✅ Produce clear outputs that general users can use quickly.
    ✅ Include practical options, clear structure, and useful suggestions.
    ✅ Keep tone practical, family-safe, and globally usable.
    ✅ Avoid violence, explicit, hateful, or unsafe content.
    """

    full_system = system + "\n\n" + assistant_prompt + "\n\nReference data:\n" + get_context()

    cache_key = hashlib.md5((system + user).encode()).hexdigest()
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    for name, fn in _PROVIDERS:
        try:
            result = fn(full_system, user)
            _cache_set(cache_key, result)
            return result
        except Exception as exc:
            logger.warning("LLM provider %s failed: %s", name, exc)

    return "⚠️ All AI providers are currently busy. Please try again in a few minutes."

# ── HTML ─────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<title>🎥 PureVid AI</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="AI video and prompt generator for general users. Family-safe and easy to use.">
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E%F0%9F%93%B9%3C/text%3E%3C/svg%3E">
<style>
:root{--bd:#1a2e4a;--bm:#2563eb;--bl:#60a5fa;--pale:#f0f4ff;--border:#bfdbfe;--w:#fff;--gray:#666;--r:12px}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:var(--pale);color:#222}
.header{background:linear-gradient(135deg,var(--bd),var(--bm));color:#fff;padding:32px 20px;text-align:center}
.header h1{font-size:2.2em;margin-bottom:8px;letter-spacing:1px}
.header p{font-size:1em;opacity:.92;max-width:580px;margin:0 auto}
.badges{display:flex;flex-wrap:wrap;justify-content:center;gap:8px;margin-top:14px}
.badge{background:rgba(255,255,255,.2);border:1px solid rgba(255,255,255,.4);border-radius:20px;padding:5px 14px;font-size:.82em}
.container{max-width:960px;margin:28px auto;padding:0 16px}
.tabs{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:24px}
@media(min-width:700px){.tabs{grid-template-columns:repeat(6,1fr)}}
@media(max-width:380px){.tabs{grid-template-columns:repeat(2,1fr)}}
.tabs button{background:var(--bm);color:#fff;border:none;padding:12px 6px;border-radius:var(--r);cursor:pointer;font-size:11px;font-weight:700;transition:all .2s;display:flex;flex-direction:column;align-items:center;gap:4px;width:100%}
.tabs button:hover{background:var(--bd);transform:translateY(-2px)}
.tabs button.active{background:var(--bd);border-bottom:3px solid var(--bl)}
.tab-icon{font-size:1.4em}
.tab{display:none}.tab.active{display:block;animation:fadeIn .35s}
@keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.card{background:var(--w);padding:28px;border-radius:16px;box-shadow:0 4px 20px rgba(0,0,0,.09);margin-bottom:4px}
.card h2{color:var(--bd);margin-bottom:8px;font-size:1.35em;display:flex;align-items:center;gap:8px}
.hint{color:var(--gray);font-size:.87em;margin-bottom:18px;background:#eff6ff;padding:12px 14px;border-radius:0 10px 10px 0;border-left:4px solid var(--bl)}
.form-row{display:grid;grid-template-columns:1fr;gap:14px;margin-bottom:8px}
@media(min-width:500px){.form-row.two{grid-template-columns:1fr 1fr}}
.field{display:flex;flex-direction:column;gap:6px;margin-top:10px}
label{font-weight:700;color:var(--bd);font-size:.9em}
input,select,textarea{width:100%;padding:11px 13px;border:1.5px solid #ddd;border-radius:var(--r);font-size:14px;background:#fafafa;transition:border .2s}
input:focus,select:focus,textarea:focus{border-color:var(--bl);outline:none;background:#fff}
textarea{resize:vertical;min-height:90px}
.btn{background:linear-gradient(135deg,var(--bm),var(--bd));color:#fff;border:none;padding:14px;width:100%;border-radius:var(--r);font-size:15px;cursor:pointer;margin:14px 0 8px;font-weight:bold;transition:all .2s;box-shadow:0 3px 8px rgba(0,0,0,.15)}
.btn:hover{transform:translateY(-2px);box-shadow:0 6px 18px rgba(0,0,0,.2)}
.btn:disabled{opacity:.6;cursor:not-allowed;transform:none}
.btn.green{background:linear-gradient(135deg,#22c55e,#16a34a)}
.btn.green:hover{box-shadow:0 6px 18px rgba(34,197,94,.35)}
.output-wrap{position:relative;margin-top:8px}
.output{background:#f0f4ff;border:1.5px solid var(--border);border-radius:var(--r);padding:18px;min-height:60px;white-space:pre-wrap;font-size:14px;line-height:1.75}
.copy-btn{position:absolute;top:8px;right:8px;background:var(--bm);color:#fff;border:none;border-radius:6px;padding:5px 12px;font-size:12px;cursor:pointer;opacity:0;transition:opacity .2s}
.output-wrap:hover .copy-btn{opacity:1}
.video-box{margin-top:16px;text-align:center;display:none}
.video-box video{max-width:100%;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,.2)}
.download-btn{display:inline-block;margin-top:10px;background:#16a34a;color:#fff;padding:10px 24px;border-radius:8px;text-decoration:none;font-weight:bold;transition:all .2s}
.download-btn:hover{background:#15803d;transform:translateY(-1px)}
.progress{background:#e0e7ff;border-radius:8px;height:8px;margin:12px 0;overflow:hidden;display:none}
.progress-bar{height:100%;background:var(--bm);width:0%;transition:width .5s;border-radius:8px}
.spinner{display:inline-block;width:16px;height:16px;border:3px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .8s linear infinite;vertical-align:middle;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
hr{border:none;border-top:1px solid #e8eaf0;margin:18px 0}
.tip-box{background:#eff6ff;border-left:3px solid var(--bl);padding:12px 14px;border-radius:0 10px 10px 0;font-size:13px;color:#1e40af;margin-top:12px}
.footer{text-align:center;padding:28px 16px;color:var(--gray);font-size:13px;line-height:2;background:var(--w);border-radius:16px;margin-top:20px}
.topnav{display:flex;justify-content:center;gap:10px;flex-wrap:wrap;margin:14px 0 4px}
.nav-pill{color:#fff;text-decoration:none;font-weight:700;font-size:12px;padding:6px 10px;border:1px solid rgba(255,255,255,.35);border-radius:999px;background:rgba(255,255,255,.12);cursor:pointer}
.nav-pill:hover{background:rgba(255,255,255,.22)}
.quickstart{background:#ffffffd9;border:1px solid #dbeafe;border-radius:14px;padding:14px;margin:0 auto 18px;max-width:960px}
.quickstart h3{color:var(--bd);font-size:16px;margin-bottom:8px}
.step-list{display:grid;grid-template-columns:1fr;gap:8px;font-size:13px;color:#334155}
@media(min-width:800px){.step-list{grid-template-columns:repeat(3,1fr)}}
.step{background:#f8fbff;border:1px solid #dbeafe;border-radius:10px;padding:10px}
.examples{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
.ex{border:1px solid var(--border);background:#fff;color:#1d4ed8;border-radius:999px;padding:6px 10px;font-size:12px;cursor:pointer}
.ex:hover{background:#eff6ff}
</style>
</head>
<body>
<div class="header">
  <h1 style="cursor:pointer" onclick="window.location.reload()">🎥 PureVid AI</h1>
  <p><b>Beautiful, family-safe AI videos and creative content for everyone</b></p>
  <div class="topnav">
    <button class="nav-pill" type="button" onclick="switchTab('generate')">Start</button><button class="nav-pill" type="button" onclick="switchTab('prompt')">Prompts</button><button class="nav-pill" type="button" onclick="switchTab('story')">Story</button><button class="nav-pill" type="button" onclick="switchTab('safety')">Safety</button><button class="nav-pill" type="button" onclick="switchTab('enhance')">Enhance</button><button class="nav-pill" type="button" onclick="switchTab('ideas')">Ideas</button><button class="nav-pill" type="button" onclick="switchTab('followup')">Ask More</button><button class="nav-pill" type="button" onclick="switchTab('feedback')">Feedback</button>
  </div>
  <div class="badges">
    <span class="badge">✅ Classroom Safe</span>
    <span class="badge">🔒 No Data Stored</span>
    <span class="badge">🌍 General Users</span>
    <span class="badge">⚡ Powered by CogVideoX</span>
  </div>
</div>

<div class="container">
  <div class="quickstart">
    <h3>✨ Quick start for everyone</h3>
    <div class="step-list">
      <div class="step"><b>1) Describe your idea</b><br/>What should happen in the video and the style you want.</div>
      <div class="step"><b>2) Generate and review</b><br/>Wait for the video, then check safety and clarity.</div>
      <div class="step"><b>3) Download and share</b><br/>Use on social media, presentations, or personal projects.</div>
    </div>
  </div>
  <div class="tabs">
    <button id="tab-generate" class="active" onclick="show('generate',this)"><span class="tab-icon">🎬</span>Generate</button>
    <button id="tab-prompt" onclick="show('prompt',this)"><span class="tab-icon">✨</span>Prompts</button>
    <button id="tab-story" onclick="show('story',this)"><span class="tab-icon">📖</span>Story</button>
    <button id="tab-safety" onclick="show('safety',this)"><span class="tab-icon">🛡️</span>Safety</button>
    <button id="tab-enhance" onclick="show('enhance',this)"><span class="tab-icon">⚡</span>Enhance</button>
    <button id="tab-ideas" onclick="show('ideas',this)"><span class="tab-icon">💡</span>Ideas</button>
    <button id="tab-followup" onclick="show('followup',this)"><span class="tab-icon">💬</span>Ask More</button>
    <button id="tab-feedback" onclick="show('feedback',this)"><span class="tab-icon">📝</span>Feedback</button>
  </div>

  <!-- GENERATE -->
  <div id="generate" class="tab active"><div class="card">
    <h2>🎬 Generate a Video</h2>
    <p class="hint">Describe your idea (subject, scene, mood). PureVid AI creates a family-safe video with CogVideoX. This may take up to 10 minutes.</p>
    <hr>
    <div class="field">
      <label>What do you want in your video?</label>
      <textarea id="vp" rows="4" placeholder="e.g. A peaceful mountain sunrise with cinematic drone movement, warm colors, soft clouds..."></textarea>
      <div class="examples">
        <button class="ex" type="button" onclick="setExample('Serene forest river at sunrise, cinematic lighting, slow camera pan, photorealistic')">Nature Example</button>
        <button class="ex" type="button" onclick="setExample('A traveler walking through an old city market at golden hour, cinematic and detailed')">Travel Example</button>
        <button class="ex" type="button" onclick="setExample('A cozy morning coffee scene by a window with rain outside, warm mood, shallow depth of field')">Lifestyle Example</button>
      </div>
    </div>
    <div class="field">
      <label>Aspect Ratio</label>
      <select id="va">
        <option value="16:9">16:9 (YouTube / Wide)</option>
        <option value="9:16">9:16 (Reels / TikTok)</option>
        <option value="1:1">1:1 (Square)</option>
      </select>
    </div>
    <div class="tip-box">💡 Add words like <b>cinematic, golden light, slow motion, peaceful, nature</b> for better results. Unsafe words are auto-blocked.</div>
    <button class="btn green" id="vbtn" onclick="generateVideo()">🎬 Generate Video</button>
    <div class="progress" id="prog"><div class="progress-bar" id="progbar"></div></div>
    <div id="vstatus" aria-live="polite" style="text-align:center;color:var(--gray);font-size:13px;margin-top:8px"></div>
    <div class="video-box" id="vbox">
      <video id="vplayer" controls autoplay loop></video><br>
      <a id="vdownload" class="download-btn" download="purevid.mp4">⬇️ Download Video</a>
    </div>
    <div class="output-wrap" style="margin-top:10px">
      <div id="vo" class="output" aria-live="polite" style="min-height:30px"></div>
    </div>
  </div></div>

  <!-- PROMPTS -->
  <div id="prompt" class="tab"><div class="card">
    <h2>✨ Safe Prompt Generator</h2>
    <p class="hint">Generate a detailed, family-safe AI video prompt from a simple idea.</p>
    <hr>
    <div class="form-row two">
      <div class="field"><label>Your Idea</label><input id="p1" placeholder="e.g. children playing in a park"></div>
      <div class="field">
        <label>Style</label>
        <select id="p2"><option>Cinematic</option><option>Animated</option><option>Nature Documentary</option><option>Warm & Cozy</option><option>Inspirational</option></select>
      </div>
    </div>
    <div class="form-row two">
      <div class="field">
        <label>Mood</label>
        <select id="p3"><option>Happy & Joyful</option><option>Peaceful & Calm</option><option>Inspiring</option><option>Educational</option></select>
      </div>
      <div class="field">
        <label>Duration</label>
        <select id="p4"><option>5 seconds</option><option>10 seconds</option><option>30 seconds</option></select>
      </div>
    </div>
    <button class="btn" id="pb" onclick="call('/gen_prompt',{idea:g('p1'),style:g('p2'),mood:g('p3'),duration:g('p4')},'po','pb','✨ Generate Prompt')">✨ Generate Prompt</button>
    <div class="output-wrap"><div id="po" class="output">Your prompt will appear here...</div><button class="copy-btn" onclick="cp('po')">📋 Copy</button></div>
  </div></div>

  <!-- STORY -->
  <div id="story" class="tab"><div class="card">
    <h2>📖 Story → Video Prompts</h2>
    <p class="hint">Turn any story into scene-by-scene AI video prompts.</p>
    <hr>
    <div class="field"><label>Your Story</label><textarea id="s1" rows="5" placeholder="e.g. A child plants a seed and watches it grow into a beautiful tree..."></textarea></div>
    <div class="form-row two">
      <div class="field">
        <label>Scenes</label>
        <select id="s2"><option>3</option><option selected>5</option><option>8</option></select>
      </div>
      <div class="field">
        <label>Style</label>
        <select id="s3"><option>Cinematic</option><option>Animated</option><option>Storybook</option><option>Documentary</option></select>
      </div>
    </div>
    <button class="btn" id="sb" onclick="call('/story_to_video',{story:g('s1'),scenes:g('s2'),style:g('s3')},'so','sb','📖 Generate Scene Prompts')">📖 Generate Scene Prompts</button>
    <div class="output-wrap"><div id="so" class="output">Scene prompts will appear here...</div><button class="copy-btn" onclick="cp('so')">📋 Copy</button></div>
  </div></div>

  <!-- SAFETY -->
  <div id="safety" class="tab"><div class="card">
    <h2>🛡️ Content Safety Checker</h2>
    <p class="hint">Check if your prompt is family-safe before generating.</p>
    <hr>
    <div class="field"><label>Paste Your Prompt</label><textarea id="sc1" rows="4" placeholder="Paste any AI video prompt..."></textarea></div>
    <div class="field">
      <label>Audience</label>
      <select id="sc2"><option>General (All Ages)</option><option>Children (Under 12)</option><option>Family</option><option>Islamic Guidelines</option></select>
    </div>
    <button class="btn" id="scb" onclick="call('/safety_check',{prompt:g('sc1'),audience:g('sc2')},'sco','scb','🛡️ Check Safety')">🛡️ Check Safety</button>
    <div class="output-wrap"><div id="sco" class="output">Safety report will appear here...</div><button class="copy-btn" onclick="cp('sco')">📋 Copy</button></div>
  </div></div>

  <!-- ENHANCE -->
  <div id="enhance" class="tab"><div class="card">
    <h2>⚡ Prompt Enhancer</h2>
    <p class="hint">Turn a basic idea into a cinematic, detailed AI prompt.</p>
    <hr>
    <div class="field"><label>Basic Prompt</label><input id="e1" placeholder="e.g. sunset beach"></div>
    <div class="form-row two">
      <div class="field">
        <label>Camera</label>
        <select id="e2"><option>Cinematic Wide Shot</option><option>Close Up</option><option>Drone Aerial</option><option>Time Lapse</option></select>
      </div>
      <div class="field">
        <label>Lighting</label>
        <select id="e3"><option>Golden Hour</option><option>Soft Natural Light</option><option>Warm Indoor</option><option>Sunrise</option></select>
      </div>
    </div>
    <button class="btn" id="eb" onclick="call('/enhance_prompt',{prompt:g('e1'),camera:g('e2'),lighting:g('e3')},'eo','eb','⚡ Enhance Prompt')">⚡ Enhance Prompt</button>
    <div class="output-wrap"><div id="eo" class="output">Enhanced prompt will appear here...</div><button class="copy-btn" onclick="cp('eo')">📋 Copy</button></div>
  </div></div>

  <!-- IDEAS -->
  <div id="ideas" class="tab"><div class="card">
    <h2>💡 Content Ideas</h2>
    <p class="hint">Get 10 creative, family-safe AI video ideas.</p>
    <hr>
    <div class="form-row two">
      <div class="field"><label>Theme</label><input id="i1" placeholder="e.g. Eid, family picnic, nature"></div>
      <div class="field">
        <label>Platform</label>
        <select id="i2"><option>YouTube</option><option>Instagram Reels</option><option>TikTok</option><option>WhatsApp Status</option></select>
      </div>
    </div>
    <div class="field">
      <label>Audience</label>
      <select id="i3"><option>Children</option><option selected>Family</option><option>Muslim Community</option><option>General Public</option></select>
    </div>
    <button class="btn" id="ib" onclick="call('/gen_ideas',{theme:g('i1'),platform:g('i2'),audience:g('i3')},'io','ib','💡 Generate 10 Video Ideas')">💡 Generate 10 Video Ideas</button>
    <div class="output-wrap"><div id="io" class="output">Ideas will appear here...</div><button class="copy-btn" onclick="cp('io')">📋 Copy</button></div>
  </div></div>

  <!-- FOLLOW-UP -->
  <div id="followup" class="tab"><div class="card">
    <h2>💬 Ask Follow-up Questions</h2>
    <p class="hint">Want to learn more from a result? Paste it below and ask a follow-up question.</p>
    <hr>
    <div class="field">
      <label>Previous AI Response</label>
      <textarea id="fu1" rows="5" placeholder="Paste the AI output you want to explore further..."></textarea>
    </div>
    <div class="field">
      <label>Your Follow-up Question</label>
      <input id="fu2" placeholder="e.g. Can you explain this in simpler steps?">
    </div>
    <button class="btn" id="fub" onclick="call('/follow_up',{context:g('fu1'),question:g('fu2')},'fuo','fub','💬 Ask Follow-up')">💬 Ask Follow-up</button>
    <div class="output-wrap"><div id="fuo" class="output">Follow-up answer will appear here...</div><button class="copy-btn" onclick="cp('fuo')">📋 Copy</button></div>
  </div></div>

  <!-- FEEDBACK -->
  <div id="feedback" class="tab"><div class="card">
    <h2>📝 Feedback</h2>
    <p class="hint">Tell us what works and what should improve. Your feedback helps make PureVid better.</p>
    <hr>
    <div class="field"><label>Your Name (optional)</label><input id="fb1" placeholder="Your name"></div>
    <div class="field"><label>Email (optional)</label><input id="fb2" placeholder="you@example.com"></div>
    <div class="field"><label>Feedback</label><textarea id="fb3" rows="5" placeholder="Share bugs, ideas, or suggestions..."></textarea></div>
    <button class="btn" id="fbb" onclick="submitFeedback()">📝 Submit Feedback</button>
    <div class="output-wrap"><div id="fbo" class="output">Feedback status will appear here...</div></div>
  </div></div>

</div>

<div class="footer">
  🎥 <strong>PureVid AI</strong> | Family-safe AI video + prompt support <br>
  🔒 No data stored | ✅ Family safe always<br>
  <span style="font-size:.8em;color:#94a3b8">
    ⚠️ AI-generated content may contain errors or unexpected results. This tool is provided
    as-is for creative purposes only. Creators are not responsible for any generated content
    or decisions made based on AI output. Always review content before publishing.
  </span>
</div>

<script>
function g(id){return document.getElementById(id).value;}
function setExample(text){document.getElementById('vp').value=text;document.getElementById('vp').focus();}
function switchTab(tab){const b=document.getElementById('tab-'+tab);if(b)show(tab,b);}
function show(tab,btn){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tabs button').forEach(b=>{b.classList.remove('active');b.setAttribute('aria-selected','false');});
  document.getElementById(tab).classList.add('active');
  btn.classList.add('active');
  btn.setAttribute('aria-selected','true');
}
function _fallbackCopy(text,btn){
  const ta=document.createElement('textarea');
  ta.value=text;ta.style.position='fixed';ta.style.opacity='0';
  document.body.appendChild(ta);ta.focus();ta.select();
  try{const ok=document.execCommand('copy');btn.textContent=ok?'✅ Copied!':'❌ Copy failed';}
  catch(e){btn.textContent='❌ Copy failed';}
  document.body.removeChild(ta);
  setTimeout(()=>btn.textContent='📋 Copy',2000);
}
function cp(id){
  const text=document.getElementById(id).innerText;
  const btn=document.getElementById(id).closest('.output-wrap').querySelector('.copy-btn');
  if(navigator.clipboard&&navigator.clipboard.writeText){
    navigator.clipboard.writeText(text).then(()=>{
      btn.textContent='✅ Copied!';
      setTimeout(()=>btn.textContent='📋 Copy',2000);
    }).catch(()=>_fallbackCopy(text,btn));
  }else{_fallbackCopy(text,btn);}
}
async function call(endpoint,data,outId,btnId,label){
  const out=document.getElementById(outId),btn=document.getElementById(btnId);
  out.textContent='';
  btn.disabled=true;
  btn.innerHTML='<span class="spinner"></span>Generating...';
  out.textContent='⏳ AI is thinking...';
  try{
    const r=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    if(!r.ok){const t=await r.text();out.textContent='❌ Server error: '+t.substring(0,300);return;}
    const j=await r.json();
    out.textContent=j.result;
  }catch(e){
    out.textContent='❌ Error: '+e.message;
  }finally{
    btn.disabled=false;
    btn.textContent=label;
  }
}
async function submitFeedback(){
  const out=document.getElementById('fbo');
  const btn=document.getElementById('fbb');
  const message=g('fb3').trim();
  out.textContent='';
  if(!message){out.textContent='❌ Please write feedback before submitting.';return;}
  btn.disabled=true;
  btn.innerHTML='<span class="spinner"></span>Sending...';
  out.textContent='⏳ Submitting feedback...';
  try{
    const r=await fetch('/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:g('fb1'),email:g('fb2'),message})});
    const j=await r.json();
    out.textContent=j.result || j.error || 'Thanks for your feedback!';
    if(j.ok){document.getElementById('fb3').value='';}
  }catch(e){
    out.textContent='❌ Error: '+e.message;
  }finally{
    btn.disabled=false;
    btn.textContent='📝 Submit Feedback';
  }
}

async function generateVideo(){
  const prompt=g('vp'),ratio=g('va');
  const btn=document.getElementById('vbtn');
  const status=document.getElementById('vstatus');
  const out=document.getElementById('vo');
  const prog=document.getElementById('prog');
  const bar=document.getElementById('progbar');
  const vbox=document.getElementById('vbox');
  out.textContent='';status.textContent='';
  if(!prompt.trim()){out.textContent='❌ Please describe your video first!';return;}
  btn.disabled=true;
  btn.innerHTML='<span class="spinner"></span>Generating...';
  vbox.style.display='none';
  prog.style.display='block';
  bar.style.width='5%';
  let pct=5;
  const ticker=setInterval(()=>{pct=Math.min(pct+1,90);bar.style.width=pct+'%';},3000);
  const steps=[
    '🛡️ Checking safety...',
    '🤖 Enhancing your prompt with AI...',
    '📡 Connecting to video provider...',
    '🎬 Generating video frames... This may take up to 10 minutes, please wait',
    '🎞️ Composing final video...',
    '📦 Almost ready...'
  ];
  let si=0;
  status.textContent=steps[si++];
  const stepTick=setInterval(()=>{if(si<steps.length)status.textContent=steps[si++];},50000);
  const ctrl=new AbortController();
  const timeoutId=setTimeout(()=>ctrl.abort(),720000);
  try{
    const r=await fetch('/generate_video',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({prompt,ratio}),
      signal:ctrl.signal
    });
    clearTimeout(timeoutId);
    const j=await r.json();
    clearInterval(ticker);clearInterval(stepTick);
    bar.style.width='100%';
    if(j.error){
      out.textContent='❌ '+j.error;
      status.textContent='';
    }else{
      status.textContent='✅ Video ready!';
      out.textContent='✅ Generated | Prompt: '+j.prompt_used;
      const src=j.video_b64?'data:video/mp4;base64,'+j.video_b64:j.video_url;
      document.getElementById('vplayer').src=src;
      const dlBtn=document.getElementById('vdownload');
      if(src.startsWith('data:')){
        dlBtn.href=src;
        dlBtn.onclick=null;
      }else{
        dlBtn.href='#';
        dlBtn.onclick=async function(e){
          e.preventDefault();
          try{
            const resp=await fetch(src);
            const blob=await resp.blob();
            const url=URL.createObjectURL(blob);
            const a=document.createElement('a');a.href=url;a.download='purevid.mp4';a.click();
            setTimeout(()=>URL.revokeObjectURL(url),60000);
          }catch(err){window.open(src,'_blank');}
        };
      }
      vbox.style.display='block';
    }
  }catch(e){
    clearTimeout(timeoutId);
    clearInterval(ticker);clearInterval(stepTick);
    if(e.name==='AbortError'){
      out.textContent='❌ Request timed out after 12 minutes. Please try again.';
    }else{
      out.textContent='❌ Error: '+e.message;
    }
    status.textContent='';
  }finally{
    btn.disabled=false;
    btn.textContent='🎬 Generate Video';
    setTimeout(()=>{prog.style.display='none';bar.style.width='0%';},2000);
  }
}
</script>
</body>
</html>"""

def _get_google_auth_context():
    try:
        import google.auth
        from google.auth.transport.requests import Request as GoogleAuthRequest
    except ImportError as exc:
        raise RuntimeError("google-auth package is required for Google Cloud mode.") from exc

    creds, adc_project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(GoogleAuthRequest())
    project_id = VERTEX_PROJECT_ID or adc_project
    return creds.token, project_id


def _vertex_generate_video(prompt, ratio):
    token, project_id = _get_google_auth_context()
    if not project_id:
        raise RuntimeError("Google Cloud project is not configured. Set VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT.")

    ratio_map = {"16:9": "16:9", "9:16": "9:16", "1:1": "1:1"}
    endpoint = (
        f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1/projects/"
        f"{project_id}/locations/{VERTEX_LOCATION}/publishers/google/models/"
        f"{VERTEX_VIDEO_MODEL}:predictLongRunning"
    )
    payload = {
        "instances": [{
            "prompt": prompt,
            "aspectRatio": ratio_map.get(ratio, "16:9")
        }]
    }
    submit = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=40,
    )
    if submit.status_code not in (200, 201):
        raise RuntimeError(f"Google video submission failed: {submit.text[:400]}")

    operation_name = submit.json().get("name")
    if not operation_name:
        raise RuntimeError("Google video operation name was not returned.")

    for i in range(60):
        if i % 5 == 0:
            try:
                token, _ = _get_google_auth_context()
            except Exception as exc:
                logger.warning("_vertex_generate_video: token refresh failed: %s", exc)
        time.sleep(10)
        poll = requests.get(
            f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1/{operation_name}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )
        if poll.status_code != 200:
            continue
        data = poll.json()
        if data.get("done"):
            if data.get("error"):
                raise RuntimeError(f"Google video generation failed: {data['error']}")
            response = data.get("response", {})
            for pred in response.get("predictions", []):
                uri = pred.get("video", {}).get("uri") or pred.get("videoUri")
                if uri:
                    return uri
            raise RuntimeError("Google video generation completed but no video URI was returned.")

    raise RuntimeError("Google video generation timed out.")


# ── ROUTES ───────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/generate_video", methods=["POST"])
def generate_video():
    try:
        d = _json_body()
        raw_prompt = d.get("prompt", "").strip()
        ratio = d.get("ratio", "16:9")

        if not raw_prompt:
            return jsonify(error="Please enter a prompt."), 400
        if not is_safe(raw_prompt):
            return jsonify(error="🚫 Unsafe content detected. Please use family-friendly descriptions."), 400

        if not _check_video_rate_limit():
            return jsonify(error="⏳ Too many requests. Please wait a moment and try again."), 429

        final_prompt = raw_prompt
        if client or VIDEO_PROVIDER == "google":
            try:
                enhanced = llm(
                    "Expert prompt enhancer for safe AI video generation. CogVideoX works best with detailed cinematic scene descriptions.",
                    f"Enhance this for cinematic family-safe video:\n\n{raw_prompt}\n\nAspect ratio: {ratio}\n\nKeep it safe, detailed, and under 200 words."
                )
                if enhanced and not enhanced.startswith("❌"):
                    final_prompt = enhanced
            except Exception:
                final_prompt = raw_prompt

        provider = _normalized_video_provider()
        if provider == "google":
            try:
                video_url = _vertex_generate_video(final_prompt, ratio)
                return jsonify(prompt_used=final_prompt, video_url=video_url, video_b64=None, provider="google")
            except Exception as ge:
                if not (ALLOW_FAL_FALLBACK and FAL_KEY):
                    return jsonify(error=f"Google Cloud video generation is not ready: {str(ge)[:300]}"), 503
                provider = "fal"

        if provider == "fal":
            if not FAL_KEY:
                return jsonify(error="❌ FAL_KEY missing. Set VIDEO_PROVIDER=google to use Google Cloud only, or provide FAL_KEY for fal provider."), 400

            sizes = {
                "16:9": {"width": 1360, "height": 768},
                "9:16": {"width": 768, "height": 1360},
                "1:1":  {"width": 768, "height": 768}
            }
            submit = requests.post(
                "https://queue.fal.run/fal-ai/cogvideox-5b",
                headers={"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"},
                json={
                    "prompt": final_prompt,
                    "num_frames": 49,
                    "guidance_scale": 7.0,
                    "num_inference_steps": 50,
                    "video_size": sizes.get(ratio, sizes["16:9"])
                },
                timeout=30
            )

            if submit.status_code not in [200, 201]:
                return jsonify(error=f"Submission failed: {submit.text[:300]}"), 502

            request_id = submit.json().get("request_id")
            if not request_id:
                return jsonify(error="No request ID returned from fal.ai."), 502

            for _ in range(60):
                time.sleep(10)
                poll = requests.get(
                    f"https://queue.fal.run/fal-ai/cogvideox-5b/requests/{request_id}/status",
                    headers={"Authorization": f"Key {FAL_KEY}"},
                    timeout=15
                )
                status = poll.json().get("status")
                if status == "COMPLETED":
                    final = requests.get(
                        f"https://queue.fal.run/fal-ai/cogvideox-5b/requests/{request_id}",
                        headers={"Authorization": f"Key {FAL_KEY}"},
                        timeout=15
                    ).json()
                    video_url = final.get("video", {}).get("url")
                    if not video_url:
                        return jsonify(error="Video URL not found in response."), 502
                    return jsonify(prompt_used=final_prompt, video_url=video_url, video_b64=None, provider="fal")
                elif status == "FAILED":
                    return jsonify(error="Generation failed on fal.ai. Please try again."), 502

            return jsonify(error="Timed out after 10 minutes. Try a simpler prompt."), 504

        return jsonify(error="Video provider configuration is invalid. Use 'google' or 'fal'."), 400

    except requests.exceptions.Timeout:
        return jsonify(error="Request timed out. Please try again."), 504
    except Exception:
        logger.exception("generate_video: unexpected error")
        return jsonify(error="❌ Something went wrong. Please try again."), 500

@app.route("/gen_prompt", methods=["POST"])
def gen_prompt():
    try:
        d = _json_body()
        return jsonify(result=llm(
            "Professional AI video prompt writer for general users. Family-safe. Optimized for CogVideoX.",
            f"Write a polished AI video prompt. Idea: {d.get('idea', '')}\nStyle: {d.get('style', '')} | Mood: {d.get('mood', '')} | Duration: {d.get('duration', '')}\n\nInclude: style options, shot suggestions, and practical tips.\n\n✨ MAIN PROMPT\n🎨 STYLE TAGS\n🎯 CREATIVE GOAL\n🧩 PRACTICAL USE\n🚫 NEGATIVE PROMPT\n💡 PRO TIP"
        ))
    except Exception:
        logger.exception("gen_prompt: unexpected error")
        return jsonify(result="❌ Something went wrong. Please try again."), 500
@app.route("/story_to_video", methods=["POST"])
def story_to_video():
    try:
        d = _json_body()
        return jsonify(result=llm(
            "Professional video director. Family-safe scene prompts only. Optimized for CogVideoX.",
            f"Break into {d.get('scenes', '')} scenes. Style: {d.get('style', '')}\nStory: {d.get('story', '')}\n\nFor each:\n🎬 SCENE [N]\n📍 Setting\n✨ AI PROMPT\n🎵 Mood"
        ))
    except Exception:
        logger.exception("story_to_video: unexpected error")
        return jsonify(result="❌ Something went wrong. Please try again."), 500

@app.route("/safety_check", methods=["POST"])
def safety_check():
    try:
        d = _json_body()
        return jsonify(result=llm(
            "Content safety expert for AI video generation.",
            f"Audience: {d.get('audience', '')}\nPrompt: {d.get('prompt', '')}\n\n🛡️ RATING (Safe/Caution/Unsafe)\n✅ SAFE ELEMENTS\n⚠️ CONCERNS\n🔧 SAFE ALTERNATIVE"
        ))
    except Exception:
        logger.exception("safety_check: unexpected error")
        return jsonify(result="❌ Something went wrong. Please try again."), 500

@app.route("/enhance_prompt", methods=["POST"])
def enhance_prompt():
    try:
        d = _json_body()
        return jsonify(result=llm(
            "Master AI prompt engineer for cinematic safe video. Optimized for CogVideoX-5b.",
            f"Enhance: {d.get('prompt', '')}\nCamera: {d.get('camera', '')} | Lighting: {d.get('lighting', '')}\n\n✨ ENHANCED PROMPT\n📸 TECHNICAL DETAILS\n🎨 COLORS & MOOD\n🚫 NEGATIVE PROMPT"
        ))
    except Exception:
        logger.exception("enhance_prompt: unexpected error")
        return jsonify(result="❌ Something went wrong. Please try again."), 500

@app.route("/gen_ideas", methods=["POST"])
def gen_ideas():
    try:
        d = _json_body()
        return jsonify(result=llm(
            "Creative content strategist for family-safe AI video.",
            f"10 family-safe video ideas:\nTheme: {d.get('theme', '')} | Platform: {d.get('platform', '')} | Audience: {d.get('audience', '')}\n\nFor each:\n💡 IDEA [N]\n📝 Concept\n🎯 Goal\n✨ AI Prompt\n📈 Why it works"
        ))
    except Exception:
        logger.exception("gen_ideas: unexpected error")
        return jsonify(result="❌ Something went wrong. Please try again."), 500

@app.route("/follow_up", methods=["POST"])
def follow_up():
    try:
        d = _json_body()
        context = d.get("context", "").strip()
        question = d.get("question", "").strip()
        if not question:
            return jsonify(result="❌ Please provide a follow-up question.")
        return jsonify(result=llm(
            "Helpful assistant for follow-up explanations. Keep answers clear, practical, and family-safe.",
            f"Context:\n{context}\n\nFollow-up question:\n{question}\n\nGive a clear answer with simple steps."
        ))
    except Exception:
        logger.exception("follow_up: unexpected error")
        return jsonify(result="❌ Something went wrong. Please try again."), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        d = _json_body()
        message = (d.get("message") or "").strip()
        if not message:
            return jsonify(ok=False, error="Please provide feedback message."), 400
        entry = {
            "time": int(time.time()),
            "name": (d.get("name") or "").strip()[:120],
            "email": (d.get("email") or "").strip()[:200],
            "message": message[:2000],
        }
        with FEEDBACK_LOG_LOCK:
            FEEDBACK_LOG.append(entry)
            if len(FEEDBACK_LOG) > FEEDBACK_LOG_MAX:
                del FEEDBACK_LOG[:-FEEDBACK_LOG_MAX]
        try:
            import json
            with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return jsonify(ok=True, result="✅ Thanks! Your feedback was submitted successfully.")
    except Exception:
        logger.exception("feedback: unexpected error")
        return jsonify(ok=False, error="❌ Something went wrong. Please try again."), 500


@app.route("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    logger.info("🚀 PureVid AI starting...")
    logger.info("Groq: %s", 'configured' if GROQ_KEY else 'not configured')
    logger.info("Cerebras: %s", 'configured' if os.environ.get('CEREBRAS_KEY') else 'not configured')
    logger.info("Gemini: %s", 'configured' if os.environ.get('GEMINI_KEY') else 'not configured')
    logger.info("Cohere: %s", 'configured' if os.environ.get('COHERE_KEY') else 'not configured')
    logger.info("Mistral: %s", 'configured' if os.environ.get('MISTRAL_KEY') else 'not configured')
    logger.info("OpenRouter: %s", 'configured' if os.environ.get('OPENROUTER_KEY') else 'not configured')
    logger.info("HuggingFace: %s", 'configured' if os.environ.get('HF_KEY') else 'not configured')
    logger.info("✅ Video provider: %s", _normalized_video_provider())
    logger.info("✅ Google video model: %s", VERTEX_VIDEO_MODEL if VIDEO_PROVIDER == 'google' else 'n/a')
    logger.info("✅ fal fallback enabled: %s", ALLOW_FAL_FALLBACK)
    logger.info("✅ fal provider key: %s", 'Ready' if FAL_KEY else 'Not configured')
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 8080)), debug=False)
