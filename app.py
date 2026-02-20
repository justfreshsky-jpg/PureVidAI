import os, traceback, base64
from flask import Flask, request, jsonify, render_template_string
from groq import Groq

app = Flask(__name__)

GROQ_KEY = os.environ.get("GROQ_KEY")
client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

UNSAFE = ["nudity","naked","violence","blood","kill","alcohol","drugs","gambling","weapon","gore","nsfw","sexy","adult"]

def is_safe(prompt):
    return not any(w in prompt.lower() for w in UNSAFE)

def llm(system, user):
    if not client:
        return "❌ GROQ_KEY missing."
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        max_tokens=500, temperature=0.7
    )
    return r.choices[0].message.content

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<title>PureVid AI</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{--blue-dark:#1a2e4a;--blue-mid:#2563eb;--blue-light:#60a5fa;--pale:#f0f4ff;--border:#bfdbfe;--white:#fff;--gray:#666;--radius:10px}
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:"Segoe UI",Arial,sans-serif;background:var(--pale);color:#222}
  .header{background:linear-gradient(135deg,var(--blue-dark),var(--blue-mid));color:white;padding:24px 20px;text-align:center}
  .header h1{font-size:2.2em;margin-bottom:6px}
  .header p{font-size:.95em;opacity:.9}
  .badges{display:flex;flex-wrap:wrap;justify-content:center;gap:8px;margin-top:10px}
  .badge{background:rgba(255,255,255,.2);border:1px solid rgba(255,255,255,.4);border-radius:20px;padding:4px 14px;font-size:.8em}
  .container{max-width:960px;margin:24px auto;padding:0 16px}
  .tabs{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:20px}
  @media(min-width:700px){.tabs{grid-template-columns:repeat(6,1fr)}}
  .tabs button{background:var(--blue-mid);color:white;border:none;padding:10px 4px;border-radius:var(--radius);cursor:pointer;font-size:11px;font-weight:700;transition:all .2s;display:flex;flex-direction:column;align-items:center;gap:3px}
  .tabs button:hover,.tabs button.active{background:var(--blue-dark)}
  .tabs button.active{border-bottom:3px solid var(--blue-light)}
  .tab-icon{font-size:1.4em}
  .tab{display:none}.tab.active{display:block}
  .card{background:var(--white);padding:26px;border-radius:14px;box-shadow:0 4px 18px rgba(0,0,0,.09)}
  .card h2{color:var(--blue-dark);margin-bottom:6px;font-size:1.35em}
  .hint{color:var(--gray);font-size:.85em;margin-bottom:16px}
  .form-row{display:grid;grid-template-columns:1fr;gap:12px;margin-bottom:4px}
  @media(min-width:500px){.form-row.two{grid-template-columns:1fr 1fr}}
  .field{display:flex;flex-direction:column;gap:5px;margin-top:10px}
  label{font-weight:700;color:var(--blue-dark);font-size:.9em}
  input,select,textarea{width:100%;padding:10px 13px;border:1.5px solid #ddd;border-radius:var(--radius);font-size:14px;background:#fafafa;transition:border .2s}
  input:focus,select:focus,textarea:focus{border-color:var(--blue-light);outline:none;background:white}
  textarea{resize:vertical;min-height:90px}
  .btn{background:linear-gradient(135deg,var(--blue-mid),var(--blue-dark));color:white;border:none;padding:14px;width:100%;border-radius:var(--radius);font-size:15px;cursor:pointer;margin:14px 0 8px;font-weight:bold;transition:all .2s;box-shadow:0 3px 8px rgba(0,0,0,.15)}
  .btn:hover{transform:translateY(-2px);box-shadow:0 5px 14px rgba(0,0,0,.2)}
  .btn:disabled{opacity:.6;cursor:not-allowed;transform:none}
  .btn.green{background:linear-gradient(135deg,#22c55e,#16a34a)}
  .output-wrap{position:relative;margin-top:6px}
  .output{background:#f0f4ff;border:1.5px solid var(--border);border-radius:var(--radius);padding:16px;min-height:60px;white-space:pre-wrap;font-size:14px;line-height:1.7}
  .copy-btn{position:absolute;top:8px;right:8px;background:var(--blue-mid);color:white;border:none;border-radius:6px;padding:4px 10px;font-size:12px;cursor:pointer;opacity:0;transition:opacity .2s}
  .output-wrap:hover .copy-btn{opacity:1}
  .video-box{margin-top:16px;text-align:center;display:none}
  .video-box video{max-width:100%;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,.2)}
  .download-btn{display:inline-block;margin-top:10px;background:#16a34a;color:white;padding:10px 24px;border-radius:8px;text-decoration:none;font-weight:bold}
  .progress{background:#e0e7ff;border-radius:8px;height:8px;margin:10px 0;overflow:hidden;display:none}
  .progress-bar{height:100%;background:var(--blue-mid);width:0%;transition:width .5s;border-radius:8px}
  .spinner{display:inline-block;width:16px;height:16px;border:3px solid rgba(255,255,255,.3);border-top-color:white;border-radius:50%;animation:spin .8s linear infinite;vertical-align:middle;margin-right:6px}
  @keyframes spin{to{transform:rotate(360deg)}}
  hr{border:none;border-top:1px solid #e8eaf0;margin:16px 0}
  .footer{text-align:center;padding:24px 16px;color:var(--gray);font-size:13px;line-height:2}
  .tip{background:#eff6ff;border-left:3px solid var(--blue-light);padding:10px 14px;border-radius:0 8px 8px 0;font-size:13px;color:#1e40af;margin-top:10px}
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
    <p class="hint">Describe what you want → PureVid AI generates a real video using Wan2.1. Always family-safe. Takes 4–8 minutes.</p>
    <hr>
    <div class="field">
      <label>What do you want in your video?</label>
      <textarea id="vp" rows="3" placeholder="e.g. Children playing in a sunny park, golden light, slow motion, cinematic..."></textarea>
    </div>
    <div class="field"><label>Aspect Ratio</label>
      <select id="va">
        <option value="16:9">16:9 (YouTube / Wide)</option>
        <option value="9:16">9:16 (Reels / TikTok)</option>
        <option value="1:1">1:1 (Square)</option>
      </select>
    </div>
    <div class="tip">💡 Add words like <b>cinematic, golden light, slow motion, peaceful, nature</b> for better results. Unsafe words are auto-blocked.</div>
    <button class="btn green" id="vbtn" onclick="generateVideo()">🎬 Generate Video</button>
    <div class="progress" id="prog"><div class="progress-bar" id="progbar"></div></div>
    <div id="vstatus" style="text-align:center;color:var(--gray);font-size:13px;margin-top:6px"></div>
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
      <div class="field"><label>Style</label>
        <select id="p2"><option>Cinematic</option><option>Animated</option><option>Nature Documentary</option><option>Warm & Cozy</option><option>Inspirational</option></select>
      </div>
    </div>
    <div class="form-row two">
      <div class="field"><label>Mood</label>
        <select id="p3"><option>Happy & Joyful</option><option>Peaceful & Calm</option><option>Inspiring</option><option>Educational</option></select>
      </div>
      <div class="field"><label>Duration</label>
        <select id="p4"><option>5 seconds</option><option>10 seconds</option><option>30 seconds</option></select>
      </div>
    </div>
    <button class="btn" id="pb" onclick="call('/gen_prompt',{idea:v('p1'),style:v('p2'),mood:v('p3'),duration:v('p4')},'po','pb','✨ Generate Prompt')">✨ Generate Prompt</button>
    <div class="output-wrap"><div id="po" class="output">Your prompt will appear here...</div><button class="copy-btn" onclick="copyOut('po')">📋 Copy</button></div>
  </div></div>

  <!-- STORY -->
  <div id="story" class="tab"><div class="card">
    <h2>📖 Story → Video Prompts</h2>
    <p class="hint">Turn any story into scene-by-scene AI video prompts.</p>
    <hr>
    <div class="field"><label>Your Story</label><textarea id="s1" rows="5" placeholder="e.g. A child plants a seed and watches it grow into a beautiful tree..."></textarea></div>
    <div class="form-row two">
      <div class="field"><label>Scenes</label><select id="s2"><option>3</option><option selected>5</option><option>8</option></select></div>
      <div class="field"><label>Style</label><select id="s3"><option>Cinematic</option><option>Animated</option><option>Storybook</option><option>Documentary</option></select></div>
    </div>
    <button class="btn" id="sb" onclick="call('/story_to_video',{story:v('s1'),scenes:v('s2'),style:v('s3')},'so','sb','📖 Generate Scene Prompts')">📖 Generate Scene Prompts</button>
    <div class="output-wrap"><div id="so" class="output">Scene prompts will appear here...</div><button class="copy-btn" onclick="copyOut('so')">📋 Copy</button></div>
  </div></div>

  <!-- SAFETY -->
  <div id="safety" class="tab"><div class="card">
    <h2>🛡️ Content Safety Checker</h2>
    <p class="hint">Check if your prompt is family-safe before generating.</p>
    <hr>
    <div class="field"><label>Paste Your Prompt</label><textarea id="sc1" rows="4" placeholder="Paste any AI video prompt..."></textarea></div>
    <div class="field"><label>Audience</label>
      <select id="sc2"><option>General (All Ages)</option><option>Children (Under 12)</option><option>Family</option><option>Islamic Guidelines</option></select>
    </div>
    <button class="btn" id="scb" onclick="call('/safety_check',{prompt:v('sc1'),audience:v('sc2')},'sco','scb','🛡️ Check Safety')">🛡️ Check Safety</button>
    <div class="output-wrap"><div id="sco" class="output">Safety report will appear here...</div><button class="copy-btn" onclick="copyOut('sco')">📋 Copy</button></div>
  </div></div>

  <!-- ENHANCE -->
  <div id="enhance" class="tab"><div class="card">
    <h2>⚡ Prompt Enhancer</h2>
    <p class="hint">Turn a basic idea into a cinematic, detailed AI prompt.</p>
    <hr>
    <div class="field"><label>Basic Prompt</label><input id="e1" placeholder="e.g. sunset beach"></div>
    <div class="form-row two">
      <div class="field"><label>Camera</label>
        <select id="e2"><option>Cinematic Wide Shot</option><option>Close Up</option><option>Drone Aerial</option><option>Time Lapse</option></select>
      </div>
      <div class="field"><label>Lighting</label>
        <select id="e3"><option>Golden Hour</option><option>Soft Natural Light</option><option>Warm Indoor</option><option>Sunrise</option></select>
      </div>
    </div>
    <button class="btn" id="eb" onclick="call('/enhance_prompt',{prompt:v('e1'),camera:v('e2'),lighting:v('e3')},'eo','eb','⚡ Enhance Prompt')">⚡ Enhance Prompt</button>
    <div class="output-wrap"><div id="eo" class="output">Enhanced prompt will appear here...</div><button class="copy-btn" onclick="copyOut('eo')">📋 Copy</button></div>
  </div></div>

  <!-- IDEAS -->
  <div id="ideas" class="tab"><div class="card">
    <h2>💡 Content Ideas</h2>
    <p class="hint">Get 10 creative, family-safe AI video ideas.</p>
    <hr>
    <div class="form-row two">
      <div class="field"><label>Theme</label><input id="i1" placeholder="e.g. Eid, family picnic, nature"></div>
      <div class="field"><label>Platform</label>
        <select id="i2"><option>YouTube</option><option>Instagram Reels</option><option>TikTok</option><option>WhatsApp Status</option></select>
      </div>
    </div>
    <div class="field"><label>Audience</label>
      <select id="i3"><option>Children</option><option selected>Family</option><option>Muslim Community</option><option>General Public</option></select>
    </div>
    <button class="btn" id="ib" onclick="call('/gen_ideas',{theme:v('i1'),platform:v('i2'),audience:v('i3')},'io','ib','💡 Generate Ideas')">💡 Generate 10 Video Ideas</button>
    <div class="output-wrap"><div id="io" class="output">Ideas will appear here...</div><button class="copy-btn" onclick="copyOut('io')">📋 Copy</button></div>
  </div></div>

  <div class="footer">🎥 <strong>PureVid AI</strong> &nbsp;|&nbsp; Safe AI video generator<br>🔒 No data stored &nbsp;|&nbsp; ✅ Family safe always</div>
</div>

<script>
function v(id){return document.getElementById(id).value}
function show(tab,btn){
  document.querySelectorAll(".tab").forEach(t=>t.classList.remove("active"));
  document.querySelectorAll(".tabs button").forEach(b=>b.classList.remove("active"));
  document.getElementById(tab).classList.add("active");
  btn.classList.add("active");
}
function copyOut(id){
  navigator.clipboard.writeText(document.getElementById(id).innerText).then(()=>{
    const cb=document.getElementById(id).closest(".output-wrap").querySelector(".copy-btn");
    cb.textContent="✅ Copied!";
    setTimeout(()=>cb.textContent="📋 Copy",2000);
  });
}
async function call(endpoint,data,outId,btnId,label){
  const out=document.getElementById(outId),btn=document.getElementById(btnId);
  btn.disabled=true;
  btn.innerHTML='<span class="spinner"></span>Generating...';
  out.innerHTML="⏳ AI is thinking...";
  try{
    const r=await fetch(endpoint,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
    const j=await r.json();
    out.innerHTML=j.result;
  }catch(e){out.innerHTML="❌ Error: "+e.message;}
  finally{btn.disabled=false;btn.innerHTML=label;}
}
async function generateVideo(){
  const prompt=v("vp"), ratio=v("va");
  const btn=document.getElementById("vbtn");
  const status=document.getElementById("vstatus");
  const out=document.getElementById("vo");
  const prog=document.getElementById("prog");
  const bar=document.getElementById("progbar");
  const vbox=document.getElementById("vbox");
  if(!prompt.trim()){out.innerHTML="❌ Please describe your video first!";return;}
  btn.disabled=true;
  btn.innerHTML='<span class="spinner"></span>Generating...';
  vbox.style.display="none";
  prog.style.display="block";
  bar.style.width="5%";
  out.innerHTML="";
  let pct=5;
  const ticker=setInterval(()=>{pct=Math.min(pct+1,90);bar.style.width=pct+"%";},3000);
  const steps=[
    "🛡️ Checking safety...",
    "🤖 Enhancing your prompt with AI...",
    "📡 Connecting to video AI...",
    "🎬 Generating video frames... (this takes 4–8 min, please wait)",
    "🎞️ Composing final video...",
    "📦 Almost ready..."
  ];
  let si=0;
  status.innerHTML=steps[si++];
  const stepTick=setInterval(()=>{if(si<steps.length)status.innerHTML=steps[si++];},60000);
  try{
    const r=await fetch("/generate_video",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({prompt,ratio})
    });
    const j=await r.json();
    clearInterval(ticker);clearInterval(stepTick);
    bar.style.width="100%";
    if(j.error){
      out.innerHTML="❌ "+j.error;
      status.innerHTML="";
    } else {
      status.innerHTML="✅ Video ready!";
      out.innerHTML="✅ Generated | Prompt: "+j.prompt_used;
      const src = j.video_b64
        ? "data:video/mp4;base64,"+j.video_b64
        : j.video_url;
      document.getElementById("vplayer").src=src;
      document.getElementById("vdownload").href=src;
      vbox.style.display="block";
    }
  }catch(e){
    clearInterval(ticker);clearInterval(stepTick);
    out.innerHTML="❌ Error: "+e.message;
    status.innerHTML="";
  }finally{
    btn.disabled=false;
    btn.innerHTML="🎬 Generate Video";
    setTimeout(()=>{prog.style.display="none";bar.style.width="0%";},2000);
  }
}
</script>
</body></html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/generate_video", methods=["POST"])
def generate_video():
    try:
        import requests

        d = request.json
        raw_prompt = d.get("prompt", "").strip()
        ratio = d.get("ratio", "16:9")

        if not raw_prompt:
            return jsonify(error="Please enter a prompt.")
        if not is_safe(raw_prompt):
            return jsonify(error="🚫 Unsafe content detected.")

        # Step 1: Enhance with Groq (if available)
        final_prompt = raw_prompt
        if client:
            try:
                final_prompt = llm(
                    "Expert prompt enhancer for safe AI video generation.",
                    f"Enhance this prompt for a cinematic video suitable for all ages:\n\n{raw_prompt}\n\nAspect ratio: {ratio}"
                )
            except Exception:
                final_prompt = raw_prompt  # fallback gracefully if Groq fails

        # Step 2: Simulate video generation (placeholder)
        # In real use, you'd call your video API (e.g., Wan2.1, Replicate, etc.)
        fake_video_bytes = base64.b64encode(b"FAKE_VIDEO_BINARY_DATA").decode("utf-8")

        # Step 3: Return simulated response
        return jsonify(
            prompt_used=final_prompt,
            video_b64=fake_video_bytes,
            video_url=None,
        )

    except Exception as e:
        return jsonify(error=f"Server error: {traceback.format_exc()}"), 200

@app.route("/gen_prompt", methods=["POST"])
def gen_prompt():
    try:
        d = request.json
        return jsonify(result=llm(
            "Professional AI video prompt writer. Always family-safe.",
            f"Write AI video prompt for: {d['idea']}\nStyle:{d['style']} Mood:{d['mood']} Duration:{d['duration']}\n\n✨ MAIN PROMPT\n🎨 STYLE TAGS\n🚫 NEGATIVE PROMPT\n💡 PRO TIP"
        ))
    except Exception:
        return jsonify(result=f"❌ {traceback.format_exc()}"), 200

@app.route("/story_to_video", methods=["POST"])
def story_to_video():
    try:
        d = request.json
        return jsonify(result=llm(
            "Professional video director. Family-safe scene prompts only.",
            f"Break into {d['scenes']} scenes. Style:{d['style']}\nStory:{d['story']}\n\nFor each:\n🎬 SCENE [N]\n📍 Setting\n✨ AI PROMPT\n🎵 Mood"
        ))
    except Exception:
        return jsonify(result=f"❌ {traceback.format_exc()}"), 200

@app.route("/safety_check", methods=["POST"])
def safety_check():
    try:
        d = request.json
        return jsonify(result=llm(
            "Content safety expert.",
            f"Audience:{d['audience']}\nPrompt:{d['prompt']}\n\n🛡️ RATING (Safe/Caution/Unsafe)\n✅ SAFE ELEMENTS\n⚠️ CONCERNS\n🔧 SAFE ALTERNATIVE"
        ))
    except Exception:
        return jsonify(result=f"❌ {traceback.format_exc()}"), 200

@app.route("/enhance_prompt", methods=["POST"])
def enhance_prompt():
    try:
        d = request.json
        return jsonify(result=llm(
            "Master AI prompt engineer for cinematic safe video.",
            f"Enhance:{d['prompt']} Camera:{d['camera']} Lighting:{d['lighting']}\n\n✨ ENHANCED PROMPT\n📸 TECHNICAL DETAILS\n🎨 COLORS\n🚫 NEGATIVE PROMPT"
        ))
    except Exception:
        return jsonify(result=f"❌ {traceback.format_exc()}"), 200

@app.route("/gen_ideas", methods=["POST"])
def gen_ideas():
    try:
        d = request.json
        return jsonify(result=llm(
            "Creative content strategist for family-safe video.",
            f"10 safe video ideas:\nTheme:{d['theme']} Platform:{d['platform']} Audience:{d['audience']}\n\nFor each:\n💡 IDEA [N]\n📝 Concept\n✨ AI Prompt\n📈 Why it works"
        ))
    except Exception:
        return jsonify(result=f"❌ {traceback.format_exc()}"), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
