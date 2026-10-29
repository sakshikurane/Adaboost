import os
import pickle
import warnings
 
import pandas as pd
from flask import Flask, jsonify, render_template_string, request
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "adaboost.pkl")
 
# Text shown for the model's output classes (change to match your dataset)
POSITIVE_LABEL = "Likely to be Admitted"
NEGATIVE_LABEL = "Unlikely to be Admitted"
 
app = Flask(__name__)
 
# --------------------------------------------------------------------------
# Load model once at startup
# --------------------------------------------------------------------------
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
 
FEATURE_ORDER = list(model.feature_names_in_)
POS_INDEX = list(model.classes_).index(1) if 1 in list(model.classes_) else -1
 
# --------------------------------------------------------------------------
# Form definition (drives both the HTML form and the server-side validation)
# kind: "slider" | "number" | "select"
# --------------------------------------------------------------------------
GROUPS = [
    {
        "title": "Personal Info",
        "icon": "👤",
        "fields": [
            {"name": "age", "label": "Age", "kind": "slider", "min": 16, "max": 30, "step": 1, "default": 18},
            {"name": "gender", "label": "Gender", "kind": "select", "default": 0,
             "options": [(0, "Female"), (1, "Male"), (2, "Other")]},
            {"name": "state", "label": "State (encoded ID)", "kind": "number", "min": 0, "max": 49, "step": 1, "default": 0,
             "hint": "Use the same numeric code used during training"},
            {"name": "family_income", "label": "Family Income", "kind": "number", "min": 0, "max": 1000000, "step": 1000, "default": 60000},
        ],
    },
    {
        "title": "Academics",
        "icon": "🎓",
        "fields": [
            {"name": "high_school_gpa", "label": "High School GPA", "kind": "slider", "min": 0, "max": 4, "step": 0.01, "default": 3.5},
            {"name": "sat_score", "label": "SAT Score", "kind": "slider", "min": 400, "max": 1600, "step": 10, "default": 1200},
            {"name": "act_score", "label": "ACT Score", "kind": "slider", "min": 1, "max": 36, "step": 1, "default": 26},
            {"name": "attendance_rate", "label": "Attendance Rate (%)", "kind": "slider", "min": 0, "max": 100, "step": 1, "default": 92},
            {"name": "ap_courses", "label": "AP Courses", "kind": "slider", "min": 0, "max": 15, "step": 1, "default": 3},
        ],
    },
    {
        "title": "Activities & Skills",
        "icon": "🚀",
        "fields": [
            {"name": "extracurricular_count", "label": "Extracurriculars", "kind": "slider", "min": 0, "max": 15, "step": 1, "default": 3},
            {"name": "volunteer_hours", "label": "Volunteer Hours", "kind": "slider", "min": 0, "max": 500, "step": 5, "default": 60},
            {"name": "leadership_positions", "label": "Leadership Positions", "kind": "slider", "min": 0, "max": 10, "step": 1, "default": 1},
            {"name": "coding_projects", "label": "Coding Projects", "kind": "slider", "min": 0, "max": 20, "step": 1, "default": 2},
            {"name": "online_certifications", "label": "Online Certifications", "kind": "slider", "min": 0, "max": 20, "step": 1, "default": 2},
            {"name": "social_media_hours", "label": "Social Media (hrs/day)", "kind": "slider", "min": 0, "max": 12, "step": 0.5, "default": 3},
        ],
    },
    {
        "title": "Evaluation Scores",
        "icon": "⭐",
        "fields": [
            {"name": "essay_score", "label": "Essay Score", "kind": "slider", "min": 0, "max": 100, "step": 1, "default": 75},
            {"name": "recommendation_score", "label": "Recommendation Score", "kind": "slider", "min": 0, "max": 100, "step": 1, "default": 75},
            {"name": "interview_score", "label": "Interview Score", "kind": "slider", "min": 0, "max": 100, "step": 1, "default": 75},
        ],
    },
]
 
FIELDS = {f["name"]: f for g in GROUPS for f in g["fields"]}
 
# Top drivers (model-wide feature importance) for the "Key factors" panel
_imp = sorted(zip(FEATURE_ORDER, model.feature_importances_), key=lambda x: x[1], reverse=True)
TOP_FACTORS = [
    {"name": n.replace("_", " ").title(), "value": round(float(v) * 100, 1)}
    for n, v in _imp[:5] if v > 0
]
 
# --------------------------------------------------------------------------
# HTML / CSS / JS
# --------------------------------------------------------------------------
PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Admission Predictor · AdaBoost</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg1:#0f0c29; --bg2:#302b63; --bg3:#24243e;
  --cyan:#22d3ee; --violet:#a78bfa; --pink:#f472b6;
  --good:#34d399; --bad:#fb7185;
  --card:rgba(255,255,255,.07); --border:rgba(255,255,255,.14);
  --text:#f1f5f9; --muted:#a5b0c6;
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{
  font-family:'Poppins',system-ui,sans-serif;color:var(--text);min-height:100vh;
  background:linear-gradient(135deg,var(--bg1),var(--bg2),var(--bg3),var(--bg1));
  background-size:300% 300%;animation:bgShift 18s ease infinite;overflow-x:hidden;
}
@keyframes bgShift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
 
/* floating blobs */
.blob{position:fixed;border-radius:50%;filter:blur(80px);opacity:.35;z-index:0;pointer-events:none;animation:float 14s ease-in-out infinite}
.b1{width:380px;height:380px;background:var(--violet);top:-80px;left:-80px}
.b2{width:320px;height:320px;background:var(--cyan);bottom:-60px;right:-60px;animation-delay:-5s}
.b3{width:260px;height:260px;background:var(--pink);top:45%;left:55%;animation-delay:-9s}
@keyframes float{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(40px,-50px) scale(1.12)}66%{transform:translate(-30px,40px) scale(.92)}}
 
.wrap{position:relative;z-index:1;max-width:1180px;margin:0 auto;padding:40px 20px 60px}
 
header{text-align:center;margin-bottom:38px;animation:fadeDown .9s ease both}
.badge{display:inline-block;padding:6px 16px;border-radius:999px;font-size:.78rem;letter-spacing:.08em;
  background:rgba(34,211,238,.12);border:1px solid rgba(34,211,238,.4);color:var(--cyan);margin-bottom:16px}
h1{font-size:clamp(2rem,5vw,3.2rem);font-weight:700;line-height:1.15;
  background:linear-gradient(90deg,var(--cyan),var(--violet),var(--pink),var(--cyan));background-size:250% auto;
  -webkit-background-clip:text;background-clip:text;color:transparent;animation:shine 6s linear infinite}
@keyframes shine{to{background-position:250% center}}
header p{color:var(--muted);margin-top:12px;font-weight:300}
@keyframes fadeDown{from{opacity:0;transform:translateY(-24px)}to{opacity:1;transform:none}}
@keyframes fadeUp{from{opacity:0;transform:translateY(30px)}to{opacity:1;transform:none}}
 
.layout{display:grid;grid-template-columns:1.6fr 1fr;gap:26px;align-items:start}
@media(max-width:940px){.layout{grid-template-columns:1fr}}
 
.card{
  background:var(--card);border:1px solid var(--border);border-radius:22px;padding:24px;
  backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
  box-shadow:0 10px 40px rgba(0,0,0,.28);animation:fadeUp .8s ease both;
  transition:transform .3s ease,border-color .3s ease,box-shadow .3s ease;
}
.card:hover{transform:translateY(-4px);border-color:rgba(167,139,250,.55);box-shadow:0 16px 50px rgba(124,58,237,.25)}
.group{margin-bottom:22px}
.group:nth-child(1){animation-delay:.1s}.group:nth-child(2){animation-delay:.2s}
.group:nth-child(3){animation-delay:.3s}.group:nth-child(4){animation-delay:.4s}
.group h2{font-size:1.1rem;font-weight:600;margin-bottom:18px;display:flex;align-items:center;gap:10px}
.group h2 .ico{width:38px;height:38px;display:grid;place-items:center;border-radius:12px;font-size:1.15rem;
  background:linear-gradient(135deg,rgba(34,211,238,.25),rgba(167,139,250,.25))}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:18px 22px}
@media(max-width:600px){.grid{grid-template-columns:1fr}}
.field label{display:flex;justify-content:space-between;font-size:.84rem;color:var(--muted);margin-bottom:8px}
.field label b{color:var(--cyan);font-weight:600}
.hint{font-size:.7rem;color:var(--muted);margin-top:4px;opacity:.8}
 
input[type=number],select{
  width:100%;padding:11px 14px;border-radius:12px;border:1px solid var(--border);
  background:rgba(15,12,41,.55);color:var(--text);font:inherit;font-size:.92rem;outline:none;transition:.25s}
input[type=number]:focus,select:focus{border-color:var(--cyan);box-shadow:0 0 0 3px rgba(34,211,238,.18)}
select option{background:#1e1b4b}
 
input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:7px;border-radius:99px;outline:none;cursor:pointer;
  background:linear-gradient(90deg,var(--cyan) var(--p,50%),rgba(255,255,255,.14) var(--p,50%))}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:20px;height:20px;border-radius:50%;
  background:#fff;border:3px solid var(--violet);box-shadow:0 0 12px rgba(167,139,250,.8);transition:transform .2s}
input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.25)}
input[type=range]::-moz-range-thumb{width:16px;height:16px;border-radius:50%;background:#fff;border:3px solid var(--violet)}
 
.btn{
  width:100%;margin-top:8px;padding:15px;border:0;border-radius:16px;cursor:pointer;
  font:inherit;font-weight:600;font-size:1.02rem;color:#0f0c29;position:relative;overflow:hidden;
  background:linear-gradient(90deg,var(--cyan),var(--violet),var(--pink));background-size:200% auto;
  transition:transform .25s,box-shadow .25s,background-position .6s;
  box-shadow:0 8px 28px rgba(34,211,238,.3);
}
.btn:hover{transform:translateY(-2px) scale(1.01);background-position:right center;box-shadow:0 12px 34px rgba(167,139,250,.5)}
.btn:active{transform:scale(.98)}
.btn[disabled]{opacity:.7;cursor:wait}
.btn .spin{display:none;width:18px;height:18px;border:3px solid rgba(15,12,41,.3);border-top-color:#0f0c29;
  border-radius:50%;animation:rot .8s linear infinite;vertical-align:-3px;margin-right:8px}
.btn.loading .spin{display:inline-block}
@keyframes rot{to{transform:rotate(360deg)}}
.reset{background:none;border:0;color:var(--muted);margin-top:14px;cursor:pointer;font:inherit;font-size:.85rem;width:100%;transition:.2s}
.reset:hover{color:var(--text)}
 
/* result side */
.side{position:sticky;top:20px;display:flex;flex-direction:column;gap:22px}
@media(max-width:940px){.side{position:static}}
.result{text-align:center}
.result h3{font-weight:600;font-size:1.05rem;margin-bottom:6px}
.placeholder{color:var(--muted);font-weight:300;font-size:.92rem;padding:30px 0}
.placeholder .big{font-size:3rem;display:block;margin-bottom:10px;animation:bob 2.6s ease-in-out infinite}
@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
.gauge{position:relative;width:210px;height:210px;margin:14px auto 10px}
.gauge svg{transform:rotate(-90deg);width:100%;height:100%}
.gauge .track{fill:none;stroke:rgba(255,255,255,.1);stroke-width:14}
.gauge .bar{fill:none;stroke-width:14;stroke-linecap:round;stroke-dasharray:565.5;stroke-dashoffset:565.5;
  transition:stroke-dashoffset 1.6s cubic-bezier(.22,1,.36,1);filter:drop-shadow(0 0 8px currentColor)}
.gauge .pct{position:absolute;inset:0;display:grid;place-items:center;font-size:2.6rem;font-weight:700}
.gauge .pct small{font-size:1rem;font-weight:400;color:var(--muted)}
.verdict{display:inline-block;margin-top:6px;padding:9px 20px;border-radius:999px;font-weight:600;font-size:.95rem;animation:pop .6s cubic-bezier(.2,1.6,.4,1) both}
.verdict.good{background:rgba(52,211,153,.15);color:var(--good);border:1px solid rgba(52,211,153,.5)}
.verdict.bad{background:rgba(251,113,133,.15);color:var(--bad);border:1px solid rgba(251,113,133,.5)}
@keyframes pop{from{opacity:0;transform:scale(.5)}to{opacity:1;transform:scale(1)}}
.err{color:var(--bad);font-size:.9rem;padding:10px 0}
 
.factors h3{font-size:1rem;font-weight:600;margin-bottom:16px}
.fbar{margin-bottom:14px}
.fbar .row{display:flex;justify-content:space-between;font-size:.8rem;color:var(--muted);margin-bottom:5px}
.fbar .tr{height:8px;border-radius:99px;background:rgba(255,255,255,.1);overflow:hidden}
.fbar .fl{height:100%;width:0;border-radius:99px;background:linear-gradient(90deg,var(--cyan),var(--violet));transition:width 1.4s cubic-bezier(.22,1,.36,1)}
footer{text-align:center;margin-top:40px;color:var(--muted);font-size:.8rem}
 
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
</head>
<body>
<div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div>
 
<div class="wrap">
  <header>
    <span class="badge">⚡ ADABOOST · MACHINE LEARNING</span>
    <h1>Admission Predictor</h1>
    <p>Fill in the student profile and let the model estimate the outcome instantly.</p>
  </header>
 
  <div class="layout">
    <form id="form" class="card" autocomplete="off">
      {% for g in groups %}
      <section class="group">
        <h2><span class="ico">{{ g.icon }}</span>{{ g.title }}</h2>
        <div class="grid">
          {% for f in g.fields %}
          <div class="field">
            {% if f.kind == 'slider' %}
              <label for="{{ f.name }}">{{ f.label }}<b id="v_{{ f.name }}">{{ f.default }}</b></label>
              <input type="range" id="{{ f.name }}" name="{{ f.name }}" min="{{ f.min }}" max="{{ f.max }}" step="{{ f.step }}" value="{{ f.default }}">
            {% elif f.kind == 'number' %}
              <label for="{{ f.name }}">{{ f.label }}</label>
              <input type="number" id="{{ f.name }}" name="{{ f.name }}" min="{{ f.min }}" max="{{ f.max }}" step="{{ f.step }}" value="{{ f.default }}" required>
              {% if f.hint %}<div class="hint">{{ f.hint }}</div>{% endif %}
            {% else %}
              <label for="{{ f.name }}">{{ f.label }}</label>
              <select id="{{ f.name }}" name="{{ f.name }}">
                {% for val, text in f.options %}<option value="{{ val }}" {{ 'selected' if val == f.default }}>{{ text }}</option>{% endfor %}
              </select>
            {% endif %}
          </div>
          {% endfor %}
        </div>
      </section>
      {% endfor %}
      <button type="submit" class="btn" id="btn"><span class="spin"></span><span id="btnText">Predict Now ✨</span></button>
      <button type="button" class="reset" id="reset">↺ Reset to defaults</button>
    </form>
 
    <aside class="side">
      <div class="card result" id="result" style="animation-delay:.3s">
        <h3>Prediction</h3>
        <div id="resultBody">
          <div class="placeholder"><span class="big">🔮</span>Your result will appear here.</div>
        </div>
      </div>
 
      {% if factors %}
      <div class="card factors" style="animation-delay:.5s">
        <h3>📊 What the model weighs most</h3>
        {% for f in factors %}
        <div class="fbar">
          <div class="row"><span>{{ f.name }}</span><span>{{ f.value }}%</span></div>
          <div class="tr"><div class="fl" data-w="{{ f.value }}"></div></div>
        </div>
        {% endfor %}
      </div>
      {% endif %}
    </aside>
  </div>
 
  <footer>Built with Flask · scikit-learn AdaBoostClassifier</footer>
</div>
 
<script>
const POS_LABEL = {{ pos_label|tojson }};
const NEG_LABEL = {{ neg_label|tojson }};
const form = document.getElementById('form');
const btn = document.getElementById('btn');
const btnText = document.getElementById('btnText');
const body = document.getElementById('resultBody');
 
/* slider fill + live value */
function paint(r){
  const p = (r.value - r.min) / (r.max - r.min) * 100;
  r.style.setProperty('--p', p + '%');
  const out = document.getElementById('v_' + r.id);
  if (out) out.textContent = r.value;
}
document.querySelectorAll('input[type=range]').forEach(r => {
  paint(r);
  r.addEventListener('input', () => paint(r));
});
 
/* animate factor bars when they scroll into view / on load */
window.addEventListener('load', () => {
  setTimeout(() => document.querySelectorAll('.fl').forEach(b => b.style.width = Math.min(100, b.dataset.w * 1.6) + '%'), 500);
});
 
/* count-up helper */
function countUp(el, to, ms){
  const start = performance.now();
  (function tick(now){
    const t = Math.min(1, (now - start) / ms);
    el.firstChild.nodeValue = (to * (1 - Math.pow(1 - t, 3))).toFixed(1);
    if (t < 1) requestAnimationFrame(tick);
  })(start);
}
 
function showResult(d){
  const good = d.prediction === 1;
  const color = good ? 'var(--good)' : 'var(--bad)';
  body.innerHTML = `
    <div class="gauge" style="color:${color}">
      <svg viewBox="0 0 200 200">
        <circle class="track" cx="100" cy="100" r="90"></circle>
        <circle class="bar" id="bar" cx="100" cy="100" r="90" stroke="currentColor"></circle>
      </svg>
      <div class="pct"><span><span id="num">0</span><small>%</small></span></div>
    </div>
    <div class="verdict ${good ? 'good' : 'bad'}">${good ? '✅ ' + POS_LABEL : '⚠️ ' + NEG_LABEL}</div>
    <div class="hint" style="margin-top:12px">Model confidence for the positive outcome</div>`;
  const bar = document.getElementById('bar');
  requestAnimationFrame(() => requestAnimationFrame(() => {
    bar.style.strokeDashoffset = 565.5 * (1 - d.probability / 100);
  }));
  countUp(document.getElementById('num'), d.probability, 1600);
}
 
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  btn.disabled = true; btn.classList.add('loading'); btnText.textContent = 'Analyzing…';
  const payload = Object.fromEntries(new FormData(form).entries());
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    await new Promise(r => setTimeout(r, 500)); /* short pause so the loader is visible */
    if (!res.ok) throw new Error(data.error || 'Something went wrong');
    showResult(data);
    if (window.innerWidth <= 940) document.getElementById('result').scrollIntoView({behavior: 'smooth', block: 'center'});
  } catch (err) {
    body.innerHTML = '<div class="err">❌ ' + err.message + '</div>';
  } finally {
    btn.disabled = false; btn.classList.remove('loading'); btnText.textContent = 'Predict Now ✨';
  }
});
 
document.getElementById('reset').addEventListener('click', () => {
  form.reset();
  document.querySelectorAll('input[type=range]').forEach(paint);
  body.innerHTML = '<div class="placeholder"><span class="big">🔮</span>Your result will appear here.</div>';
});
</script>
</body>
</html>
"""
 
 
# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template_string(
        PAGE,
        groups=GROUPS,
        factors=TOP_FACTORS,
        pos_label=POSITIVE_LABEL,
        neg_label=NEGATIVE_LABEL,
    )
 
 
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    row = {}
 
    for name in FEATURE_ORDER:
        spec = FIELDS.get(name)
        if spec is None:
            return jsonify(error=f"Unknown feature in model: {name}"), 500
        raw = data.get(name)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return jsonify(error=f"Please enter a valid number for '{spec['label']}'."), 400
        if spec["kind"] != "select" and not (spec["min"] <= value <= spec["max"]):
            return jsonify(error=f"'{spec['label']}' must be between {spec['min']} and {spec['max']}."), 400
        row[name] = value
 
    # Keep the exact column order used during training
    df = pd.DataFrame([row], columns=FEATURE_ORDER)
 
    proba = model.predict_proba(df)[0]
    p_pos = float(proba[POS_INDEX]) if POS_INDEX >= 0 else float(proba.max())
    prediction = int(model.predict(df)[0])
 
    return jsonify(prediction=prediction, probability=round(p_pos * 100, 1))
 
 
@app.route("/health")
def health():
    return jsonify(status="ok", features=FEATURE_ORDER)
 
 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
 