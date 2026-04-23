"""
app.py — AgroSAT · Interface Streamlit ULTRA
Classification des terres par satellite · EuroSAT · MobileNetV2
PFE Licence d'excellence en IA — 2025/2026
"""

import os, json, sqlite3, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.preprocess import predict, load_model_and_classes

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AgroSAT — Satellite Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════
CROP_EMOJI = {
    "AnnualCrop":           "🌾", "Forest":              "🌲",
    "HerbaceousVegetation": "🌿", "Highway":             "🛣️",
    "Industrial":           "🏭", "Pasture":             "🐄",
    "PermanentCrop":        "🫒", "Residential":         "🏘️",
    "River":                "🌊", "SeaLake":             "🏖️",
}
CROP_FR = {
    "AnnualCrop":           "Culture annuelle",
    "Forest":               "Forêt",
    "HerbaceousVegetation": "Végétation herbacée",
    "Highway":              "Route / Autoroute",
    "Industrial":           "Zone industrielle",
    "Pasture":              "Pâturage",
    "PermanentCrop":        "Culture permanente",
    "Residential":          "Zone résidentielle",
    "River":                "Rivière",
    "SeaLake":              "Mer / Lac",
}
CROP_COLOR = {
    "AnnualCrop":           "#F4C430", "Forest":              "#228B22",
    "HerbaceousVegetation": "#7CFC00", "Highway":             "#A0A0A0",
    "Industrial":           "#FF6B35", "Pasture":             "#90EE90",
    "PermanentCrop":        "#D4A017", "Residential":         "#87CEEB",
    "River":                "#1E90FF", "SeaLake":             "#006994",
}

DB_PATH = "models/predictions_history.db"
os.makedirs("models", exist_ok=True)

# ══════════════════════════════════════════════════════════════
#  CSS — DARK SPACE AESTHETIC
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&family=Unbounded:wght@300;400;700;900&display=swap');

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
  font-family: 'Space Grotesk', sans-serif;
  background: #050a0e !important;
  color: #c9d1d9 !important;
}
.stApp { background: #050a0e !important; }
.main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #21c55d; border-radius: 3px; }

/* ── Header ── */
.hero {
  position: relative;
  background: linear-gradient(135deg, #060f0a 0%, #0a1f12 40%, #071520 100%);
  border: 1px solid #1a3a2a;
  border-radius: 16px;
  padding: 2.5rem 3rem;
  margin-bottom: 2rem;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute;
  top: -50%; left: -50%;
  width: 200%; height: 200%;
  background: radial-gradient(ellipse at 70% 50%, rgba(33,197,93,0.06) 0%, transparent 60%),
              radial-gradient(ellipse at 20% 80%, rgba(56,189,248,0.04) 0%, transparent 50%);
  pointer-events: none;
}
.hero-grid {
  position: absolute; top:0; left:0; right:0; bottom:0;
  background-image:
    linear-gradient(rgba(33,197,93,0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(33,197,93,0.04) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
}
.hero-badge {
  display: inline-block;
  background: rgba(33,197,93,0.12);
  border: 1px solid rgba(33,197,93,0.3);
  color: #21c55d;
  font-family: 'Space Mono', monospace;
  font-size: 0.7rem;
  letter-spacing: 0.15em;
  padding: 0.3rem 0.8rem;
  border-radius: 20px;
  margin-bottom: 1rem;
}
.hero h1 {
  font-family: 'Unbounded', sans-serif;
  font-size: 2.8rem;
  font-weight: 900;
  color: #ffffff;
  margin: 0 0 0.5rem;
  letter-spacing: -0.02em;
  line-height: 1;
}
.hero h1 span { color: #21c55d; }
.hero-sub {
  color: #8b949e;
  font-family: 'Space Mono', monospace;
  font-size: 0.78rem;
  letter-spacing: 0.05em;
}
.hero-stats {
  display: flex; gap: 2rem; margin-top: 1.5rem;
}
.hero-stat { text-align: left; }
.hero-stat .n {
  font-family: 'Unbounded', sans-serif;
  font-size: 1.6rem; font-weight: 900; color: #21c55d;
}
.hero-stat .l { color: #6e7681; font-size: 0.72rem; letter-spacing: 0.08em; }

/* ── Cards ── */
.card {
  background: #0d1117;
  border: 1px solid #1c2128;
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 1rem;
  transition: border-color 0.2s;
}
.card:hover { border-color: #30363d; }
.card-title {
  font-family: 'Space Mono', monospace;
  font-size: 0.72rem; letter-spacing: 0.12em;
  color: #6e7681; text-transform: uppercase;
  margin-bottom: 0.8rem;
}

/* ── Result banner ── */
.result-main {
  background: linear-gradient(135deg, #0a1f12, #060f0a);
  border: 1px solid #21c55d;
  border-radius: 12px;
  padding: 1.5rem 2rem;
  margin: 0.5rem 0 1.2rem;
  position: relative;
  overflow: hidden;
}
.result-main::after {
  content:'';
  position:absolute; top:0; right:0;
  width:150px; height:150px;
  background: radial-gradient(circle, rgba(33,197,93,0.12) 0%, transparent 70%);
  pointer-events:none;
}
.result-main .crop-name {
  font-family: 'Unbounded', sans-serif;
  font-size: 1.8rem; font-weight: 700;
  color: #ffffff; margin-bottom: 0.3rem;
}
.result-main .crop-fr { color: #21c55d; font-size: 1rem; margin-bottom: 0.8rem; }
.conf-bar-wrap { background: #1c2128; border-radius: 20px; height: 8px; margin-top: 0.5rem; }
.conf-bar { height: 8px; border-radius: 20px; background: linear-gradient(90deg, #21c55d, #38bdf8); }

/* ── Metric pills ── */
.metrics-row { display: flex; gap: 0.8rem; margin: 1rem 0; flex-wrap: wrap; }
.metric-pill {
  background: #0d1117;
  border: 1px solid #1c2128;
  border-radius: 8px;
  padding: 0.7rem 1.2rem;
  text-align: center; flex: 1; min-width: 90px;
}
.metric-pill .val {
  font-family: 'Space Mono', monospace;
  font-size: 1.3rem; font-weight: 700; color: #21c55d;
}
.metric-pill .lbl { color: #6e7681; font-size: 0.7rem; margin-top: 0.2rem; }

/* ── Class chip ── */
.chip {
  display: inline-block;
  padding: 0.2rem 0.6rem;
  border-radius: 4px;
  font-family: 'Space Mono', monospace;
  font-size: 0.7rem;
  background: rgba(33,197,93,0.1);
  border: 1px solid rgba(33,197,93,0.25);
  color: #21c55d;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
  background: #0d1117 !important;
  border: 1px dashed #30363d !important;
  border-radius: 12px !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: #21c55d !important;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, #16803c, #21c55d) !important;
  color: white !important; border: none !important;
  border-radius: 8px !important; font-weight: 600 !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 0.9rem !important; padding: 0.6rem 1.5rem !important;
  width: 100% !important; letter-spacing: 0.02em !important;
  transition: all 0.2s !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #21c55d, #4ade80) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 20px rgba(33,197,93,0.3) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  gap: 0.5rem;
  background: transparent !important;
  border-bottom: 1px solid #1c2128;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: #6e7681 !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 500 !important;
  border-radius: 8px 8px 0 0 !important;
  border: none !important;
  padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(33,197,93,0.1) !important;
  color: #21c55d !important;
  border-bottom: 2px solid #21c55d !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: #0a0e13 !important;
  border-right: 1px solid #1c2128 !important;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* ── DataFrame ── */
[data-testid="stDataFrame"] { border: 1px solid #1c2128 !important; border-radius: 8px !important; }

/* ── Section title ── */
.section-title {
  font-family: 'Unbounded', sans-serif;
  font-size: 0.9rem; font-weight: 700;
  color: #ffffff; letter-spacing: -0.01em;
  margin: 1.5rem 0 1rem;
  display: flex; align-items: center; gap: 0.5rem;
}
.section-title::after {
  content: ''; flex: 1; height: 1px; background: #1c2128;
}

/* ── Divider ── */
hr { border-color: #1c2128 !important; }

/* ── Progress ── */
[data-testid="stProgress"] > div { background: #1c2128 !important; }
[data-testid="stProgress"] > div > div { background: #21c55d !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #21c55d !important; }

/* ── Info / Warning / Error ── */
[data-testid="stAlert"] { border-radius: 8px !important; border: 1px solid #30363d !important; }

/* ── History tag ── */
.tag-high { color: #21c55d; font-family: 'Space Mono', monospace; font-size: 0.8rem; font-weight: 700; }
.tag-med  { color: #f0a500; font-family: 'Space Mono', monospace; font-size: 0.8rem; font-weight: 700; }
.tag-low  { color: #f85149; font-family: 'Space Mono', monospace; font-size: 0.8rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, filename TEXT,
            predicted TEXT, confidence REAL, all_probs TEXT
        )""")
    conn.commit(); conn.close()

def save_pred(filename, predicted, confidence, all_probs):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (timestamp,filename,predicted,confidence,all_probs) VALUES (?,?,?,?,?)",
        (datetime.datetime.now().isoformat(timespec="seconds"),
         filename, predicted, confidence, json.dumps(all_probs)))
    conn.commit(); conn.close()

def load_history(limit=100):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM predictions ORDER BY id DESC LIMIT {limit}", conn)
    conn.close()
    return df

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    avg   = conn.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0] or 0
    top   = conn.execute(
        "SELECT predicted, COUNT(*) as c FROM predictions GROUP BY predicted ORDER BY c DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return total, avg, top

init_db()

# ══════════════════════════════════════════════════════════════
#  MATPLOTLIB THEME
# ══════════════════════════════════════════════════════════════
BG    = "#050a0e"
BG2   = "#0d1117"
CARD  = "#161b22"
BORD  = "#1c2128"
GREEN = "#21c55d"
BLUE  = "#38bdf8"
AMBER = "#f0a500"
RED   = "#f85149"
MUTED = "#6e7681"
FG    = "#c9d1d9"

def style_ax(ax, fig=None):
    if fig: fig.patch.set_facecolor(BG2)
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORD)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(FG)
    ax.grid(color=BORD, linewidth=0.5, alpha=0.6)

# ══════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════
total_preds, avg_conf, top_class = get_stats()

st.markdown(f"""
<div class="hero">
  <div class="hero-grid"></div>
  <div class="hero-badge">🛰️ &nbsp; SATELLITE INTELLIGENCE SYSTEM</div>
  <h1>Agro<span>SAT</span></h1>
  <p class="hero-sub">EuroSAT · MobileNetV2 Transfer Learning · PFE 2025–2026</p>
  <div class="hero-stats">
    <div class="hero-stat"><div class="n">10</div><div class="l">LAND CLASSES</div></div>
    <div class="hero-stat"><div class="n">~93%</div><div class="l">ACCURACY</div></div>
    <div class="hero-stat"><div class="n">{total_preds}</div><div class="l">PREDICTIONS MADE</div></div>
    <div class="hero-stat"><div class="n">{avg_conf*100:.0f}%</div><div class="l">AVG CONFIDENCE</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🛰️ AgroSAT")
    st.markdown("---")
    st.markdown("**⚙️ Configuration**")
    MODEL_PATH   = st.text_input("Modèle", value="models/best_model.keras")
    CLASSES_PATH = st.text_input("Classes JSON", value="models/class_indices.json")
    st.markdown("---")
    st.markdown("**📐 Seuil de confiance**")
    threshold = st.slider("", 0.0, 1.0, 0.5, 0.05,
                          format="%.0f%%",
                          help="Avertissement si confiance < seuil") / 100
    st.markdown("---")
    st.markdown("**🔧 Affichage**")
    show_raw_probs  = st.checkbox("Afficher toutes les probabilités", True)
    show_radar      = st.checkbox("Graphique Radar", True)
    show_heatmap    = st.checkbox("Heatmap de confiance", True)
    show_history_s  = st.checkbox("Historique", False)
    show_curves_s   = st.checkbox("Courbes d'apprentissage", False)
    st.markdown("---")
    st.markdown("""
    <div style="color:#6e7681;font-size:.75rem;font-family:'Space Mono',monospace;">
    Dataset · EuroSAT RGB<br>
    Architecture · MobileNetV2<br>
    Input · 224×224×3<br>
    Classes · 10<br>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Classification",
    "⚖️ Comparaison",
    "📊 Analytiques",
    "📋 Historique"
])

# ──────────────────────────────────────────────────────────────
# TAB 1 — CLASSIFICATION
# ──────────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1, 1.1], gap="large")

    with col_left:
        st.markdown('<div class="section-title">📡 Image satellite</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Glissez-déposez une image (JPG / PNG / TIF)",
            type=["jpg","jpeg","png","tif","tiff"],
            key="main_upload"
        )
        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption=f"📄 {uploaded.name}", use_container_width=True)

            # Image metadata
            w, h = image.size
            st.markdown(f"""
            <div style="display:flex;gap:1rem;margin-top:.5rem;">
              <span class="chip">📐 {w}×{h}px</span>
              <span class="chip">🎨 {image.mode}</span>
              <span class="chip">📦 {uploaded.size//1024} KB</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("🚀 Analyser l'image", key="btn_classify")
        else:
            st.info("⬆️ Importez une image satellite EuroSAT pour commencer l'analyse.")
            run_btn = False

    with col_right:
        st.markdown('<div class="section-title">🧠 Résultat de l\'analyse</div>', unsafe_allow_html=True)

        if uploaded and run_btn:
            with st.spinner("Analyse satellitaire en cours…"):
                try:
                    result     = predict(image, MODEL_PATH, CLASSES_PATH)
                    pred       = result["predicted_class"]
                    conf       = result["confidence"]
                    probs      = result["all_probabilities"]
                    emoji      = CROP_EMOJI.get(pred, "🌍")
                    fr         = CROP_FR.get(pred, pred)
                    pred_color = CROP_COLOR.get(pred, GREEN)

                    save_pred(uploaded.name, pred, conf, probs)

                    # ── Result banner ──
                    conf_pct = conf * 100
                    warn = f'<span style="color:{AMBER};font-size:.8rem;">⚠️ Confiance faible</span>' if conf < threshold else f'<span style="color:{GREEN};font-size:.8rem;">✅ Confiance élevée</span>'
                    st.markdown(f"""
                    <div class="result-main">
                      <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                        <div>
                          <div class="crop-name">{emoji} {pred}</div>
                          <div class="crop-fr">{fr}</div>
                          {warn}
                        </div>
                        <div style="font-family:'Space Mono',monospace;font-size:2rem;font-weight:700;color:{pred_color};">
                          {conf_pct:.1f}%
                        </div>
                      </div>
                      <div class="conf-bar-wrap">
                        <div class="conf-bar" style="width:{conf_pct:.1f}%;background:linear-gradient(90deg,{pred_color},{pred_color}88);"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Top-3 pills ──
                    top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    st.markdown('<div class="metrics-row">', unsafe_allow_html=True)
                    for i, (cls, p) in enumerate(top3):
                        rank = ["🥇","🥈","🥉"][i]
                        color = GREEN if i == 0 else (BLUE if i == 1 else MUTED)
                        st.markdown(f"""
                        <div class="metric-pill">
                          <div class="val" style="color:{color};">{p*100:.1f}%</div>
                          <div class="lbl">{rank} {CROP_EMOJI.get(cls,'')}{cls}</div>
                        </div>""", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # ── CHART 1: Horizontal bar all classes ──
                    if show_raw_probs:
                        st.markdown('<div class="section-title">📊 Probabilités par classe</div>', unsafe_allow_html=True)
                        sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
                        labels = [f"{CROP_EMOJI.get(k,'')} {CROP_FR.get(k,k)}" for k in sorted_probs]
                        values = list(sorted_probs.values())
                        keys   = list(sorted_probs.keys())

                        fig, ax = plt.subplots(figsize=(6, 4.5))
                        style_ax(ax, fig)
                        bars = ax.barh(
                            labels[::-1], values[::-1],
                            color=[CROP_COLOR.get(k, BORD) for k in keys[::-1]],
                            edgecolor="none", height=0.65
                        )
                        # Glow effect on top bar
                        top_bar = bars[-1]
                        ax.barh(
                            top_bar.get_y() + top_bar.get_height()/2,
                            top_bar.get_width(),
                            height=top_bar.get_height() * 2.5,
                            color=CROP_COLOR.get(pred, GREEN),
                            alpha=0.08, edgecolor="none"
                        )
                        for bar, val in zip(bars, values[::-1]):
                            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                                    f"{val*100:.1f}%", va="center",
                                    color=FG if val > 0.05 else MUTED, fontsize=7.5,
                                    fontfamily="monospace")
                        ax.set_xlim(0, 1.12)
                        ax.set_xlabel("Probabilité", fontsize=8)
                        ax.set_title("Distribution des probabilités", fontsize=9, pad=10)
                        ax.axvline(x=conf, color=GREEN, linewidth=1, linestyle="--", alpha=0.4)
                        plt.tight_layout(pad=1.2)
                        st.pyplot(fig); plt.close()

                    # ── CHART 2: Radar chart ──
                    if show_radar:
                        st.markdown('<div class="section-title">🕸️ Radar — Toutes classes</div>', unsafe_allow_html=True)
                        classes   = list(probs.keys())
                        vals      = [probs[c] for c in classes]
                        N         = len(classes)
                        angles    = [n / float(N) * 2 * np.pi for n in range(N)]
                        angles   += angles[:1]
                        vals_plot  = vals + vals[:1]

                        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
                        fig.patch.set_facecolor(BG2)
                        ax.set_facecolor(CARD)
                        ax.set_theta_offset(np.pi / 2)
                        ax.set_theta_direction(-1)
                        ax.set_thetagrids(np.degrees(angles[:-1]),
                                          [f"{CROP_EMOJI.get(c,'')} {c}" for c in classes],
                                          fontsize=6.5, color=MUTED)
                        for spine in ax.spines.values():
                            spine.set_edgecolor(BORD)
                        ax.tick_params(colors=MUTED, labelsize=6)
                        ax.yaxis.set_ticklabels([])
                        ax.grid(color=BORD, linewidth=0.5)

                        # Fill
                        ax.fill(angles, vals_plot, color=GREEN, alpha=0.15)
                        ax.plot(angles, vals_plot, color=GREEN, linewidth=2)

                        # Points
                        for angle, val, cls in zip(angles[:-1], vals, classes):
                            c = CROP_COLOR.get(cls, BORD)
                            ax.plot(angle, val, 'o', color=c, markersize=5, zorder=5)

                        ax.set_ylim(0, 1)
                        ax.set_title("Carte radar des probabilités", color=FG, fontsize=9, pad=20)
                        plt.tight_layout()
                        st.pyplot(fig); plt.close()

                except FileNotFoundError as e:
                    st.error(f"🚫 Modèle introuvable : {e}")
                except Exception as e:
                    st.error(f"❌ Erreur : {e}")
                    st.exception(e)

        elif not uploaded:
            # Placeholder illustration
            fig, ax = plt.subplots(figsize=(6, 5))
            style_ax(ax, fig)
            ax.set_xlim(0, 10); ax.set_ylim(0, 10)
            ax.set_xticks([]); ax.set_yticks([])
            # Draw satellite icon via text
            ax.text(5, 5.5, "🛰️", fontsize=60, ha="center", va="center")
            ax.text(5, 2.5, "Importez une image\npour commencer",
                    ha="center", va="center", color=MUTED, fontsize=11,
                    fontfamily="monospace")
            ax.set_title("Prêt pour l'analyse", color=FG, fontsize=10)
            for spine in ax.spines.values(): spine.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

# ──────────────────────────────────────────────────────────────
# TAB 2 — COMPARAISON
# ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">⚖️ Comparaison de deux images</div>', unsafe_allow_html=True)
    colA, colB = st.columns(2, gap="medium")
    imgs = {}

    for col, key, lbl in [(colA, "cmp_a", "Image A"), (colB, "cmp_b", "Image B")]:
        with col:
            f = st.file_uploader(f"📁 {lbl}", type=["jpg","jpeg","png"], key=key)
            if f:
                imgs[key] = (f, Image.open(f))
                st.image(imgs[key][1], caption=f.name, use_container_width=True)

    if len(imgs) == 2:
        if st.button("⚖️ Lancer la comparaison"):
            results = {}
            for key, (f, img) in imgs.items():
                try:
                    results[key] = predict(img, MODEL_PATH, CLASSES_PATH)
                except Exception as e:
                    st.error(f"Erreur ({key}): {e}")

            if len(results) == 2:
                rA, rB = results["cmp_a"], results["cmp_b"]
                pA, pB = rA["predicted_class"], rB["predicted_class"]
                same   = pA == pB

                # Summary pills
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"""<div class="metric-pill">
                  <div class="val">{CROP_EMOJI.get(pA,'')} {pA}</div>
                  <div class="lbl">Image A · {rA['confidence']*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
                c2.markdown(f"""<div class="metric-pill" style="border-color:{'#21c55d' if same else '#f85149'};">
                  <div class="val" style="color:{'#21c55d' if same else '#f85149'};">{'MÊME' if same else 'DIFF.'}</div>
                  <div class="lbl">Résultat</div>
                </div>""", unsafe_allow_html=True)
                c3.markdown(f"""<div class="metric-pill">
                  <div class="val">{CROP_EMOJI.get(pB,'')} {pB}</div>
                  <div class="lbl">Image B · {rB['confidence']*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

                # ── CHART: Side-by-side bars ──
                st.markdown('<div class="section-title">📊 Probabilités comparées</div>', unsafe_allow_html=True)
                all_cls = list(rA["all_probabilities"].keys())
                vA = [rA["all_probabilities"][c] for c in all_cls]
                vB = [rB["all_probabilities"][c] for c in all_cls]
                x  = np.arange(len(all_cls)); w = 0.38

                fig, ax = plt.subplots(figsize=(11, 4.5))
                style_ax(ax, fig)
                barsA = ax.bar(x - w/2, vA, w, label="Image A", color=GREEN,   alpha=0.85, edgecolor="none")
                barsB = ax.bar(x + w/2, vB, w, label="Image B", color=BLUE, alpha=0.85, edgecolor="none")
                ax.set_xticks(x)
                ax.set_xticklabels([f"{CROP_EMOJI.get(c,'')} {c}" for c in all_cls],
                                   rotation=35, ha="right", fontsize=8)
                ax.set_ylabel("Probabilité", fontsize=8)
                ax.set_ylim(0, 1.1)
                ax.legend(facecolor=CARD, edgecolor=BORD, labelcolor=FG, fontsize=8)
                ax.set_title("Comparaison des distributions de probabilités", color=FG, fontsize=9, pad=10)
                # Value labels
                for bar in list(barsA) + list(barsB):
                    h = bar.get_height()
                    if h > 0.05:
                        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                                f"{h*100:.0f}%", ha="center", va="bottom",
                                fontsize=6.5, color=FG, fontfamily="monospace")
                plt.tight_layout(pad=1.2)
                st.pyplot(fig); plt.close()

                # ── CHART: Δ difference ──
                st.markdown('<div class="section-title">📉 Différence A − B</div>', unsafe_allow_html=True)
                diffs = [vA[i] - vB[i] for i in range(len(all_cls))]
                colors_diff = [GREEN if d > 0 else RED for d in diffs]

                fig, ax = plt.subplots(figsize=(11, 3.5))
                style_ax(ax, fig)
                ax.bar(x, diffs, color=colors_diff, edgecolor="none", width=0.6)
                ax.axhline(0, color=BORD, linewidth=1)
                ax.set_xticks(x)
                ax.set_xticklabels([c for c in all_cls], rotation=35, ha="right", fontsize=8)
                ax.set_ylabel("Δ Probabilité", fontsize=8)
                ax.set_title("Écart de probabilité (A − B)", color=FG, fontsize=9, pad=10)
                plt.tight_layout(pad=1.2)
                st.pyplot(fig); plt.close()

# ──────────────────────────────────────────────────────────────
# TAB 3 — ANALYTIQUES
# ──────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">📊 Tableau de bord analytique</div>', unsafe_allow_html=True)
    df_hist = load_history()

    if df_hist.empty:
        st.info("Aucune prédiction encore. Analysez des images pour voir les statistiques.")
    else:
        # ── KPIs ──
        k1, k2, k3, k4 = st.columns(4)
        top_cls  = df_hist["predicted"].value_counts().idxmax()
        low_conf = (df_hist["confidence"] < 0.6).sum()
        high_conf= (df_hist["confidence"] >= 0.9).sum()

        for col, val, lbl in [
            (k1, len(df_hist),              "TOTAL PRÉDICTIONS"),
            (k2, f"{df_hist['confidence'].mean()*100:.1f}%", "CONFIANCE MOYENNE"),
            (k3, f"{CROP_EMOJI.get(top_cls,'')} {top_cls}", "CLASSE DOMINANTE"),
            (k4, f"{high_conf}",             "HAUTE CONFIANCE (≥90%)"),
        ]:
            col.markdown(f"""<div class="metric-pill">
              <div class="val" style="font-size:1.1rem;">{val}</div>
              <div class="lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        # ── CHART A: Class frequency donut ──
        left_c, right_c = st.columns(2, gap="large")

        with left_c:
            st.markdown('<div class="section-title">🥧 Fréquence des classes</div>', unsafe_allow_html=True)
            vc = df_hist["predicted"].value_counts()
            colors_pie = [CROP_COLOR.get(c, MUTED) for c in vc.index]

            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            fig.patch.set_facecolor(BG2)
            ax.set_facecolor(BG2)
            wedges, texts, autotexts = ax.pie(
                vc.values,
                labels=[f"{CROP_EMOJI.get(c,'')} {c}" for c in vc.index],
                autopct="%1.0f%%", startangle=140,
                colors=colors_pie,
                pctdistance=0.75,
                wedgeprops=dict(width=0.55, edgecolor=BG2, linewidth=2)
            )
            for t in texts:     t.set_color(MUTED);    t.set_fontsize(7)
            for t in autotexts: t.set_color(FG);       t.set_fontsize(7); t.set_fontweight("bold")
            ax.set_title("Distribution des classes prédites", color=FG, fontsize=9, pad=10)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        with right_c:
            st.markdown('<div class="section-title">📈 Confiance dans le temps</div>', unsafe_allow_html=True)
            df_hist["ts"] = pd.to_datetime(df_hist["timestamp"])
            df_sorted = df_hist.sort_values("ts")

            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            style_ax(ax, fig)
            ax.fill_between(range(len(df_sorted)), df_sorted["confidence"],
                            alpha=0.12, color=GREEN)
            ax.plot(range(len(df_sorted)), df_sorted["confidence"],
                    color=GREEN, linewidth=1.8)
            ax.scatter(range(len(df_sorted)), df_sorted["confidence"],
                       c=[GREEN if c >= 0.7 else AMBER if c >= 0.5 else RED
                          for c in df_sorted["confidence"]],
                       zorder=5, s=20)
            ax.axhline(0.9, color=GREEN, linewidth=0.8, linestyle="--", alpha=0.5)
            ax.axhline(0.6, color=AMBER, linewidth=0.8, linestyle="--", alpha=0.5)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Prédiction #", fontsize=8)
            ax.set_ylabel("Confiance", fontsize=8)
            ax.set_title("Évolution de la confiance", color=FG, fontsize=9, pad=10)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        # ── CHART B: Confidence distribution histogram ──
        st.markdown('<div class="section-title">📉 Distribution de confiance</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor(BG2)

        # Histogram
        ax = axes[0]
        style_ax(ax)
        n, bins, patches = ax.hist(df_hist["confidence"], bins=20,
                                    edgecolor=BG2, linewidth=0.5)
        # Color by zone
        for patch, left in zip(patches, bins[:-1]):
            if left >= 0.9:   patch.set_facecolor(GREEN)
            elif left >= 0.6: patch.set_facecolor(AMBER)
            else:             patch.set_facecolor(RED)
        ax.axvline(df_hist["confidence"].mean(), color=BLUE, linewidth=1.5,
                   linestyle="--", label=f"Moy. {df_hist['confidence'].mean()*100:.0f}%")
        ax.set_xlabel("Confiance", fontsize=8)
        ax.set_ylabel("Fréquence", fontsize=8)
        ax.set_title("Histogramme des confiances", fontsize=9, color=FG, pad=8)
        ax.legend(facecolor=CARD, edgecolor=BORD, labelcolor=FG, fontsize=7)

        # Per-class avg confidence
        ax2 = axes[1]
        style_ax(ax2)
        cls_conf = df_hist.groupby("predicted")["confidence"].mean().sort_values()
        bars = ax2.barh(cls_conf.index, cls_conf.values,
                        color=[CROP_COLOR.get(c, MUTED) for c in cls_conf.index],
                        edgecolor="none", height=0.6)
        for bar, val in zip(bars, cls_conf.values):
            ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                     f"{val*100:.1f}%", va="center", fontsize=7.5,
                     color=FG, fontfamily="monospace")
        ax2.set_xlim(0, 1.1)
        ax2.set_xlabel("Confiance moyenne", fontsize=8)
        ax2.set_title("Confiance par classe", fontsize=9, color=FG, pad=8)

        plt.tight_layout(pad=1.5)
        st.pyplot(fig); plt.close()

        # ── CHART C: Heatmap confidence matrix ──
        if show_heatmap and len(df_hist) >= 5:
            st.markdown('<div class="section-title">🔥 Heatmap — Confiance × Classe × Temps</div>', unsafe_allow_html=True)
            df_hist["hour"] = df_hist["ts"].dt.hour if "ts" in df_hist.columns else 0
            pivot = df_hist.pivot_table(
                values="confidence",
                index="predicted",
                columns=pd.cut(df_hist.index, bins=min(10, len(df_hist)), labels=False),
                aggfunc="mean"
            ).fillna(0)

            if not pivot.empty and pivot.shape[1] > 1:
                custom_cmap = LinearSegmentedColormap.from_list(
                    "agrosat", [BG, "#1a3a2a", GREEN, "#7ee787"], N=256
                )
                fig, ax = plt.subplots(figsize=(12, max(3, len(pivot)*0.5 + 1.5)))
                style_ax(ax, fig)
                im = ax.imshow(pivot.values, aspect="auto", cmap=custom_cmap, vmin=0, vmax=1)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels([f"{CROP_EMOJI.get(c,'')} {c}" for c in pivot.index],
                                   fontsize=8, color=FG)
                ax.set_xticks(range(pivot.shape[1]))
                ax.set_xticklabels([f"Seg {i+1}" for i in range(pivot.shape[1])],
                                   fontsize=7, color=MUTED)
                ax.set_title("Heatmap confiance (segments temporels)", color=FG, fontsize=9, pad=10)
                cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
                cbar.ax.yaxis.set_tick_params(color=MUTED, labelsize=7)
                plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED)
                plt.tight_layout()
                st.pyplot(fig); plt.close()

# ──────────────────────────────────────────────────────────────
# TAB 4 — HISTORIQUE
# ──────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">📋 Historique des prédictions</div>', unsafe_allow_html=True)
    df_h = load_history()

    if df_h.empty:
        st.info("Aucune prédiction enregistrée pour le moment.")
    else:
        # Filter controls
        fc1, fc2 = st.columns([2, 1])
        with fc1:
            search = st.text_input("🔍 Filtrer par classe", placeholder="ex: Forest, River…")
        with fc2:
            conf_filter = st.selectbox("Confiance", ["Toutes", "≥ 90%", "≥ 70%", "< 60%"])

        df_display = df_h.copy()
        if search:
            df_display = df_display[df_display["predicted"].str.contains(search, case=False)]
        if conf_filter == "≥ 90%":  df_display = df_display[df_display["confidence"] >= 0.9]
        elif conf_filter == "≥ 70%": df_display = df_display[df_display["confidence"] >= 0.7]
        elif conf_filter == "< 60%":  df_display = df_display[df_display["confidence"] < 0.6]

        # Format
        df_show = df_display[["timestamp","filename","predicted","confidence"]].copy()
        df_show["predicted"]  = df_show["predicted"].apply(
            lambda x: f"{CROP_EMOJI.get(x,'')} {CROP_FR.get(x,x)}")
        df_show["confidence"] = (df_show["confidence"] * 100).round(1).astype(str) + "%"
        df_show.columns       = ["Horodatage","Fichier","Classe prédite","Confiance"]

        st.dataframe(df_show, use_container_width=True, height=400)
        st.caption(f"Affichage de {len(df_show)} sur {len(df_h)} prédictions")

        col_dl, col_clear = st.columns([1, 1])
        with col_dl:
            csv = df_show.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Exporter CSV", csv,
                               "historique_predictions.csv", "text/csv")
        with col_clear:
            if st.button("🗑️ Vider l'historique"):
                conn = sqlite3.connect(DB_PATH)
                conn.execute("DELETE FROM predictions")
                conn.commit(); conn.close()
                st.success("Historique vidé.")
                st.rerun()

# ──────────────────────────────────────────────────────────────
# COURBES D'APPRENTISSAGE (sidebar toggle)
# ──────────────────────────────────────────────────────────────
if show_curves_s:
    st.markdown("---")
    st.markdown('<div class="section-title">📈 Courbes d\'apprentissage</div>', unsafe_allow_html=True)
    HISTORY_PATH = "models/history.json"

    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            h = json.load(f)

        epochs = range(1, len(h["accuracy"]) + 1)
        fig = plt.figure(figsize=(14, 5))
        fig.patch.set_facecolor(BG2)
        gs = gridspec.GridSpec(1, 3, figure=fig)

        # Accuracy
        ax1 = fig.add_subplot(gs[0, 0])
        style_ax(ax1)
        ax1.fill_between(epochs, h["accuracy"],     alpha=0.1, color=GREEN)
        ax1.fill_between(epochs, h["val_accuracy"], alpha=0.1, color=BLUE)
        ax1.plot(epochs, h["accuracy"],     color=GREEN, linewidth=2,   label="Train")
        ax1.plot(epochs, h["val_accuracy"], color=BLUE,  linewidth=2,   label="Validation", linestyle="--")
        ax1.set_title("Accuracy", color=FG, fontsize=9); ax1.legend(facecolor=CARD, edgecolor=BORD, labelcolor=FG, fontsize=7)

        # Loss
        ax2 = fig.add_subplot(gs[0, 1])
        style_ax(ax2)
        ax2.fill_between(epochs, h["loss"],     alpha=0.1, color=RED)
        ax2.fill_between(epochs, h["val_loss"], alpha=0.1, color=AMBER)
        ax2.plot(epochs, h["loss"],     color=RED,   linewidth=2,  label="Train")
        ax2.plot(epochs, h["val_loss"], color=AMBER, linewidth=2,  label="Validation", linestyle="--")
        ax2.set_title("Loss", color=FG, fontsize=9); ax2.legend(facecolor=CARD, edgecolor=BORD, labelcolor=FG, fontsize=7)

        # Gap (overfitting indicator)
        ax3 = fig.add_subplot(gs[0, 2])
        style_ax(ax3)
        gap = [abs(a - b) for a, b in zip(h["accuracy"], h["val_accuracy"])]
        ax3.fill_between(epochs, gap, alpha=0.2, color=AMBER)
        ax3.plot(epochs, gap, color=AMBER, linewidth=2)
        ax3.axhline(0.05, color=RED, linewidth=1, linestyle="--", alpha=0.6, label="Seuil overfitting")
        ax3.set_title("Écart Train/Val (Overfitting)", color=FG, fontsize=9)
        ax3.legend(facecolor=CARD, edgecolor=BORD, labelcolor=FG, fontsize=7)

        plt.tight_layout(pad=1.5)
        st.pyplot(fig); plt.close()
    else:
        st.warning(f"⚠️ `{HISTORY_PATH}` introuvable. Ajoutez la sauvegarde de l'historique dans `train.py`.")
        st.code("""
# Ajouter à la fin de train.py :
import json
history_data = {
    "accuracy":     history1.history["accuracy"]     + history2.history["accuracy"],
    "val_accuracy": history1.history["val_accuracy"] + history2.history["val_accuracy"],
    "loss":         history1.history["loss"]         + history2.history["loss"],
    "val_loss":     history1.history["val_loss"]     + history2.history["val_loss"],
}
with open("models/history.json", "w") as f:
    json.dump(history_data, f)
        """, language="python")

# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:1rem 0;color:#6e7681;font-family:'Space Mono',monospace;font-size:.72rem;">
  AgroSAT &nbsp;·&nbsp; PFE Licence d'excellence en IA &nbsp;·&nbsp; 2025–2026<br>
  EuroSAT Dataset &nbsp;·&nbsp; MobileNetV2 &nbsp;·&nbsp; TensorFlow &nbsp;·&nbsp; Streamlit
</div>
""", unsafe_allow_html=True)