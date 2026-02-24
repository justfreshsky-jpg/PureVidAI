import os
os.environ['HTTPX_PROXIES'] = 'null'  # Fix Render/httpx proxies bug
import traceback, time, threading
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template_string
from groq import Groq

app = Flask(__name__)
GROQ_KEY = os.environ.get("GROQ_KEY")
FAL_KEY = os.environ.get("FAL_KEY")
client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

UNSAFE = ["nudity","naked","violence","blood","kill","alcohol","drugs","gambling","weapon","gore","nsfw","sexy","adult","explicit","hate","terrorist"]

def is_safe(prompt):
    return not any(w in prompt.lower() for w in UNSAFE)

# ── BACKGROUND SCRAPER ──────────────────────────────────────
_cache = {"content": "", "last": 0}

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

def _fetch_tips():
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
            "Referer": "https://www.google.com/"
        }
        sources = [
            "https://fal.ai/models/fal-ai/cogvideox-5b",
            "https://huggingface.co/THUDM/CogVideoX-5b"
        ]
        combined = ""
        for url in sources:
            try:
                r = requests.get(url, headers=headers, timeout=6)
                soup = BeautifulSoup(r.text, "html.parser")
                for tag in soup(["script","style","nav","header","footer"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)
                combined += text[:1500] + "\n---\n"
            except Exception:
                continue
        if combined.strip():
            _cache["content"] = combined[:5000]
            _cache["last"] = time.time()
    except Exception:
        pass

def _bg_refresh():
    while True:
        _fetch_tips()
        time.sleep(3600)

threading.Thread(target=_bg_refresh, daemon=True).start()

def get_context():
    return _cache["content"] if _cache["content"] else FALLBACK

# ── LLM ─────────────────────────────────────────────────────
def llm(system, user):
    if not client:
        return "❌ GROQ_KEY missing. Add it in Render > Environment Variables."
    full_system = system + "\n\nReference data:\n" + get_context()
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"system","content":full_system},{"role":"user","content":user}],
        max_tokens=1200,
        temperature=0.7
    )
    
    # ** bold temizle + garip karakterler
    text = r.choices[0].message.content
    text = text.replace('**', '')  # **word** → word
    text = ''.join(c for c in text if ord(c) < 128)  # Sadece ASCII
    
    return text.strip()

# ── HTML ─────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<title>🎥 PureVid AI</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Safe AI video generator for everyone. Family-friendly and powered by CogVideoX.">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
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
</style>
</head>
<body>
<div class="header">
  <h1>🎥 PureVid AI</h1>
  <p><b>Safe AI video generator for everyone</b></p>
  <div class="badges">
    <span class="badge">✅ Family Safe</span>
    <span class="badge">🔒 No Data Stored</span>
    <span class="badge">🎬 Real Video Generation</span>
    <span class="badge">⚡ Powered by CogVideoX</span>
  </div>
</div>

<div class="container">
  <div class="tabs">
    <button class="active" onclick="show('generate',this)"><span class="tab-icon">🎬</span>Generate</button>
    <button onclick="show('prompt',this)"><span class="tab-icon">✨</span>Prompts</button>
    <button onclick="show('story',this)"><span class="tab-icon">📖</span>Story</button>
    <button onclick="show('safety',this)"><span class="tab-icon">🛡️</span>Safety</button>
    <button onclick="show('enhance',this)"><span class="tab-icon">⚡</span>Enhance</button>
    <button onclick="show('ideas',this)"><span class="tab-icon">💡</span>Ideas</button>
  </div>

  <!-- GENERATE -->
  <div id="generate" class="tab active"><div class="card">
    <h2>🎬 Generate a Video</h2>
    <p class="hint">Describe what you want → PureVid AI generates a real video using CogVideoX. Always family-safe. Takes 2–4 minutes.</p>
    <hr>
    <div class="field">
      <label>What do you want in your video?</label>
      <textarea id="vp" rows="4" placeholder="e.g. Children playing in a sunny park, golden light, slow motion, cinematic..."></textarea>
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
    <div id="vstatus" style="text-align:center;color:var(--gray);font-size:13px;margin-top:8px"></div>
    <div class="video-box" id="vbox">
      <video id="vplayer" controls autoplay loop></video><br>
      <a id="vdownload" class="download-btn" download="purevid.mp4">⬇️ Download Video</a>
    </div>
    <div class="output-wrap" style="margin-top:10px">
      <div id="vo" class="output" style="min-height:30px"></div>
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

</div>

<div class="footer">
  🎥 <strong>PureVid AI</strong> | Safe AI video generator <br>
  🔒 No data stored | ✅ Family safe always<br>
  <span style="font-size:.8em;color:#94a3b8">
    ⚠️ AI-generated content may contain errors or unexpected results. This tool is provided
    as-is for creative purposes only. Creators are not responsible for any generated content
    or decisions made based on AI output. Always review content before publishing.
  </span>
</div>

<script>
function g(id){return document.getElementById(id).value;}
function show(tab,btn){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tabs button').forEach(b=>b.classList.remove('active'));
  document.getElementById(tab).classList.add('active');
  btn.classList.add('active');
}
function cp(id){
  navigator.clipboard.writeText(document.getElementById(id).innerText).then(()=>{
    const btn=document.getElementById(id).closest('.output-wrap').querySelector('.copy-btn');
    btn.textContent='✅ Copied!';
    setTimeout(()=>btn.textContent='📋 Copy',2000);
  });
}
async function call(endpoint,data,outId,btnId,label){
  const out=document.getElementById(outId),btn=document.getElementById(btnId);
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
async function generateVideo(){
  const prompt=g('vp'),ratio=g('va');
  const btn=document.getElementById('vbtn');
  const status=document.getElementById('vstatus');
  const out=document.getElementById('vo');
  const prog=document.getElementById('prog');
  const bar=document.getElementById('progbar');
  const vbox=document.getElementById('vbox');
  if(!prompt.trim()){out.textContent='❌ Please describe your video first!';return;}
  btn.disabled=true;
  btn.innerHTML='<span class="spinner"></span>Generating...';
  vbox.style.display='none';
  prog.style.display='block';
  bar.style.width='5%';
  out.textContent='';
  let pct=5;
  const ticker=setInterval(()=>{pct=Math.min(pct+1,90);bar.style.width=pct+'%';},3000);
  const steps=[
    '🛡️ Checking safety...',
    '🤖 Enhancing your prompt with AI...',
    '📡 Connecting to CogVideoX...',
    '🎬 Generating video frames... (2–4 min, please wait)',
    '🎞️ Composing final video...',
    '📦 Almost ready...'
  ];
  let si=0;
  status.textContent=steps[si++];
  const stepTick=setInterval(()=>{if(si<steps.length)status.textContent=steps[si++];},50000);
  try{
    const r=await fetch('/generate_video',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({prompt,ratio})
    });
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
      document.getElementById('vdownload').href=src;
      vbox.style.display='block';
    }
  }catch(e){
    clearInterval(ticker);clearInterval(stepTick);
    out.textContent='❌ Error: '+e.message;
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

# ── ROUTES ───────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/generate_video", methods=["POST"])
def generate_video():
    try:
        d = request.json
        raw_prompt = d.get("prompt", "").strip()
        ratio = d.get("ratio", "16:9")

        if not raw_prompt:
            return jsonify(error="Please enter a prompt.")
        if not is_safe(raw_prompt):
            return jsonify(error="🚫 Unsafe content detected. Please use family-friendly descriptions.")

        # Enhance with Groq
        final_prompt = raw_prompt
        if client:
            try:
                final_prompt = llm(
                    "Expert prompt enhancer for safe AI video generation. CogVideoX works best with detailed cinematic scene descriptions.",
                    f"Enhance this for cinematic family-safe video:\n\n{raw_prompt}\n\nAspect ratio: {ratio}\n\nKeep it safe, detailed, and under 200 words."
                )
            except Exception:
                final_prompt = raw_prompt

        if not FAL_KEY:
            return jsonify(error="❌ FAL_KEY missing. Get free credits at fal.ai and add to Render > Environment Variables.")

        # Submit to fal.ai CogVideoX
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
            return jsonify(error=f"Submission failed: {submit.text[:300]}")

        request_id = submit.json().get("request_id")
        if not request_id:
            return jsonify(error="No request ID returned from fal.ai.")

        # Poll max 10 min
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
                    return jsonify(error="Video URL not found in response.")
                return jsonify(prompt_used=final_prompt, video_url=video_url, video_b64=None)
            elif status == "FAILED":
                return jsonify(error="Generation failed on fal.ai. Please try again.")

        return jsonify(error="Timed out after 10 minutes. Try a simpler prompt.")

    except requests.exceptions.Timeout:
        return jsonify(error="Request timed out. Please try again.")
    except Exception as e:
        return jsonify(error=f"Server error: {traceback.format_exc()}")

@app.route("/gen_prompt", methods=["POST"])
def gen_prompt():
    try:
        d = request.json
        return jsonify(result=llm(
            "Professional AI video prompt writer. Always family-safe. Optimized for CogVideoX.",
            f"Write AI video prompt for: {d['idea']}\nStyle: {d['style']} | Mood: {d['mood']} | Duration: {d['duration']}\n\n✨ MAIN PROMPT\n🎨 STYLE TAGS\n🚫 NEGATIVE PROMPT\n💡 PRO TIP"
        ))
    except Exception:
        return jsonify(result=f"❌ {traceback.format_exc()}")

@app.route("/story_to_video", methods=["POST"])
def story_to_video():
    try:
        d = request.json
        return jsonify(result=llm(
            "Professional video director. Family-safe scene prompts only. Optimized for CogVideoX.",
            f"Break into {d['scenes']} scenes. Style: {d['style']}\nStory: {d['story']}\n\nFor each:\n🎬 SCENE [N]\n📍 Setting\n✨ AI PROMPT\n🎵 Mood"
        ))
    except Exception:
        return jsonify(result=f"❌ {traceback.format_exc()}")

@app.route("/safety_check", methods=["POST"])
def safety_check():
    try:
        d = request.json
        return jsonify(result=llm(
            "Content safety expert for AI video generation.",
            f"Audience: {d['audience']}\nPrompt: {d['prompt']}\n\n🛡️ RATING (Safe/Caution/Unsafe)\n✅ SAFE ELEMENTS\n⚠️ CONCERNS\n🔧 SAFE ALTERNATIVE"
        ))
    except Exception:
        return jsonify(result=f"❌ {traceback.format_exc()}")

@app.route("/enhance_prompt", methods=["POST"])
def enhance_prompt():
    try:
        d = request.json
        return jsonify(result=llm(
            "Master AI prompt engineer for cinematic safe video. Optimized for CogVideoX-5b.",
            f"Enhance: {d['prompt']}\nCamera: {d['camera']} | Lighting: {d['lighting']}\n\n✨ ENHANCED PROMPT\n📸 TECHNICAL DETAILS\n🎨 COLORS & MOOD\n🚫 NEGATIVE PROMPT"
        ))
    except Exception:
        return jsonify(result=f"❌ {traceback.format_exc()}")

@app.route("/gen_ideas", methods=["POST"])
def gen_ideas():
    try:
        d = request.json
        return jsonify(result=llm(
            "Creative content strategist for family-safe AI video.",
            f"10 safe video ideas:\nTheme: {d['theme']} | Platform: {d['platform']} | Audience: {d['audience']}\n\nFor each:\n💡 IDEA [N]\n📝 Concept\n✨ AI Prompt\n📈 Why it works"
        ))
    except Exception:
        return jsonify(result=f"❌ {traceback.format_exc()}")

if __name__ == "__main__":
    print("🚀 PureVid AI starting...")
    print(f"✅ Groq: {'Ready' if client else '❌ Missing GROQ_KEY'}")
    print(f"✅ CogVideoX via fal.ai: {'Ready' if FAL_KEY else '❌ Missing FAL_KEY'}")
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)), debug=False)
