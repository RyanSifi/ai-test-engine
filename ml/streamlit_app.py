"""
Dashboard AI Test Engine — Prédicteur de risque de régression + génération de tests.
Usage : streamlit run ml/streamlit_app.py
"""
import base64
import json
import os
import subprocess
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    import matplotlib.pyplot as plt
    HAS_PLOTLY = False

# Config
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"
FEATURES_PATH = ROOT / "models" / "feature_names.json"
METRICS_PATH = ROOT / "models" / "metrics.json"
IMPORTANCE_PATH = ROOT / "models" / "feature_importance.csv"
DATASET_DIR = ROOT.parent / "_dataset"
TEST_ENGINE_URL = os.environ.get("TEST_ENGINE_URL", "http://localhost:8000")
TEST_ENGINE_API_KEY = os.environ.get("TEST_ENGINE_API_KEY", "")

st.set_page_config(
    page_title="AI Test Engine",
    page_icon=Path(__file__).parent / "assets" / "favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — système de design CNAM / Bootstrap 5 ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:ital,wght@0,400;0,600;0,700;0,800;1,400&display=swap');

/* ── Palette SUCRE ──
   --navy   #1b3e6f  sidebar, en-têtes de colonnes
   --blue   #2d5a9e  bandeaux de page
   --bordo  #a23a3a  item actif, badge dev
   --cyan   #0084B2  boutons, accents
   --bg     #EEEEEE  fond                                          */

html, body, [class*="css"], * {
    font-family: 'Source Sans 3','Source Sans Pro','Segoe UI',system-ui,sans-serif !important;
    font-size: 0.92rem;
}

/* ── Restaure la police des icônes Material (le sélecteur * l'écrasait) ── */
[data-testid="stIconMaterial"], .material-symbols-rounded, .material-symbols-outlined,
.material-icons, [data-testid="stSidebarCollapseButton"] span,
[data-testid="stExpandSidebarButton"] span{
    font-family:'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"], .main { background:#EEEEEE !important; }
[data-testid="stHeader"] { background:transparent !important; }
.block-container { padding-top:2rem !important; }

/* ── SIDEBAR — bleu marine façon SUCRE ── */
section[data-testid="stSidebar"]{
    background:#1b3e6f !important;
    border-right:none !important;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] code{ color:#dfe7f2 !important; }
section[data-testid="stSidebar"] code{ background:rgba(255,255,255,.12) !important; }
section[data-testid="stSidebar"] hr{ border-color:rgba(255,255,255,.18) !important; }

/* ── NAVIGATION (radio stylisé en menu vertical) ── */
section[data-testid="stSidebar"] div[role="radiogroup"]{ gap:2px !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label{
    width:100%; margin:0 !important; padding:11px 18px !important;
    display:flex; align-items:center; cursor:pointer;
    border-left:3px solid transparent; transition:background .12s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover{
    background:rgba(255,255,255,.08);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child{
    display:none !important;            /* masque le rond du radio */
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label div{
    color:#dfe7f2 !important; font-size:0.95rem !important; font-weight:500 !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked){
    background:#a23a3a !important; border-left:3px solid #ffffff;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) div{
    color:#ffffff !important; font-weight:700 !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"], [data-testid="metric-container"]{
    background:#FFFFFF; border:1px solid #DEE2E6; border-radius:0; padding:16px 20px;
}
[data-testid="stMetricValue"]{ font-size:1.8rem !important; font-weight:700 !important; color:#1b3e6f !important; }
[data-testid="stMetricLabel"]{ color:#6C757D !important; font-size:0.82rem !important; }

/* ── Titres ── */
h1{ font-size:1.5rem !important; font-weight:700 !important; color:#212529 !important;
    border-bottom:2px solid #2d5a9e; padding-bottom:8px; margin-bottom:16px !important; }
h2{ font-size:1.2rem !important; font-weight:600 !important; color:#212529 !important; }
h3{ font-size:1.05rem !important; font-weight:600 !important; color:#212529 !important; }

/* ── Section headers (barre marine) ── */
.section-header{
    background:#1b3e6f; color:#FFFFFF; padding:8px 16px; border-radius:0;
    font-size:0.9rem; font-weight:600; margin:18px 0 10px 0;
    letter-spacing:.02em; text-transform:uppercase;
}

/* ── Indicateurs statut ── */
.dot-green{ display:inline-block;width:10px;height:10px;border-radius:50%;background:#28d17c;margin-right:6px; }
.dot-red{ display:inline-block;width:10px;height:10px;border-radius:50%;background:#ff6b6b;margin-right:6px; }

/* ── Boutons (cyan institutionnel) ── */
.stButton > button{
    background:#0084B2 !important; color:#FFFFFF !important; border:none !important;
    border-radius:0 !important; font-weight:600 !important; padding:8px 20px !important;
    font-size:0.9rem !important; transition:background .15s ease; width:fit-content !important;
}
.stButton > button:hover{ background:#006A91 !important; color:#FFFFFF !important; }
.stButton > button:focus{ box-shadow:0 0 0 3px rgba(0,132,178,.25) !important; outline:none !important; }
section[data-testid="stSidebar"] .stButton > button{ width:100% !important; }

/* ── Inputs / Textareas / Selects ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
div[data-baseweb="input"] input,
div[data-baseweb="select"] > div{
    border:1px solid #CED4DA !important; border-radius:0 !important;
    font-size:0.9rem !important; color:#212529 !important; background:#FFFFFF !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus{
    border-color:#0084B2 !important; box-shadow:0 0 0 2px rgba(0,132,178,.2) !important; outline:none !important;
}

/* ── Checkbox ── */
[data-testid="stCheckbox"] label{ color:#212529 !important; }
[data-testid="stCheckbox"] svg{ color:#0084B2 !important; }

/* ── Onglets ── */
[data-testid="stTabs"] [data-baseweb="tab-list"]{ border-bottom:2px solid #DEE2E6 !important; background:transparent !important; gap:0 !important; }
[data-testid="stTabs"] [data-baseweb="tab"]{ border-radius:0 !important; font-weight:600 !important; color:#6C757D !important; padding:8px 16px !important; border:none !important; background:transparent !important; }
[data-testid="stTabs"] [aria-selected="true"]{ color:#2d5a9e !important; border-bottom:2px solid #2d5a9e !important; background:transparent !important; }
[data-testid="stTabs"] [data-baseweb="tab-panel"]{ background:#FFFFFF; border:1px solid #DEE2E6; border-top:none; padding:20px 16px !important; }

/* ── Expanders / Forms / Code / DataFrames ── */
[data-testid="stExpander"]{ border:1px solid #DEE2E6 !important; border-radius:0 !important; background:#FFFFFF !important; }
[data-testid="stExpander"] summary{ font-weight:600 !important; color:#212529 !important; }
[data-testid="stForm"]{ background:#FFFFFF; border:1px solid #DEE2E6; border-radius:0; padding:20px !important; }
[data-testid="stCode"]{ border:1px solid #DEE2E6; border-radius:0; }
[data-testid="stDataFrame"]{ border:1px solid #DEE2E6; border-radius:0; }

/* ── st.table — en-tête marine, total cyan (rappel du tableau SUCRE) ── */
[data-testid="stTable"] table{ border-collapse:collapse; }
[data-testid="stTable"] thead th{
    background:#1b3e6f !important; color:#fff !important; font-weight:600 !important;
    border:1px solid #1b3e6f !important; text-align:left;
}
[data-testid="stTable"] tbody td{ border:1px solid #DEE2E6 !important; }

/* ── Alertes Bootstrap sémantiques ── */
[data-testid="stAlert"]{ border-radius:0 !important; font-size:0.9rem !important; }
[data-testid="stSuccess"]{ background:#D1E7DD !important; color:#0A3622 !important; border-left:4px solid #198754 !important; }
[data-testid="stError"]{ background:#F8D7DA !important; color:#58151C !important; border-left:4px solid #DC3545 !important; }
[data-testid="stWarning"]{ background:#FFF3CD !important; color:#664D03 !important; border-left:4px solid #FFC107 !important; }
[data-testid="stInfo"]{ background:#CFF4FC !important; color:#055160 !important; border-left:4px solid #0084B2 !important; }
[data-testid="stSpinner"] p{ color:#212529 !important; }

hr{ border-color:#DEE2E6 !important; }
p, div, span, label{ color:#212529; }
a{ color:#0084B2 !important; }
a:hover{ color:#006A91 !important; }

/* ── Masquer la barre Streamlit (Deploy / menu) + décoration ── */
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"]{
    display:none !important;
}
header[data-testid="stHeader"]{ height:0 !important; min-height:0 !important; background:transparent !important; }

/* ── Sidebar : suppression du vide marine en haut ── */
[data-testid="stSidebarHeader"]{ display:none !important; }
[data-testid="stSidebarCollapseButton"]{ display:none !important; }
[data-testid="stSidebarUserContent"]{ padding-top:0 !important; }
section[data-testid="stSidebar"] > div:first-child{ padding-top:0 !important; }

/* ── Contenu principal remonté sous la top-bar ── */
.block-container{ padding-top:1.1rem !important; }

</style>
""", unsafe_allow_html=True)


# Helpers
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, []
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    return model, features


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_importance():
    if not IMPORTANCE_PATH.exists():
        return None
    return pd.read_csv(IMPORTANCE_PATH)


@st.cache_data
def list_datasets():
    return sorted([f.name for f in DATASET_DIR.glob("dataset*.csv")])


def predict(df: pd.DataFrame, model, features) -> pd.DataFrame:
    X = df[features].fillna(-1.0).values
    if hasattr(model, "predict_proba"):
        df["risk_score"] = model.predict_proba(X)[:, 1]
    else:
        df["risk_score"] = model.predict(X)
    df["risk_pred"] = (df["risk_score"] >= 0.5).astype(int)
    df["risk_label"] = df["risk_score"].apply(
        lambda v: "Élevé" if v >= 0.7 else ("Moyen" if v >= 0.4 else "Faible")
    )
    return df


def api_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if TEST_ENGINE_API_KEY:
        h["X-API-Key"] = TEST_ENGINE_API_KEY
    return h


def check_api() -> bool:
    try:
        r = requests.get(f"{TEST_ENGINE_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _logo_data_uri():
    """Charge ml/assets/logo.{png,svg,jpg} en data-URI base64, sinon None."""
    for ext, mime in (("png", "image/png"), ("svg", "image/svg+xml"),
                      ("jpg", "image/jpeg"), ("jpeg", "image/jpeg")):
        f = ROOT / "assets" / f"logo.{ext}"
        if f.exists():
            b64 = base64.b64encode(f.read_bytes()).decode()
            return f"data:{mime};base64,{b64}"
    return None


def top_bar():
    """En-tête institutionnel façon SUCRE : logo + séparateur + pastille dev."""
    uri = _logo_data_uri()
    if uri:
        logo = f'<img src="{uri}" alt="logo" style="height:58px;width:auto;display:block;">'
    else:
        logo = ('<div style="height:58px;width:150px;border:1px dashed #ADB5BD;'
                'display:flex;align-items:center;justify-content:center;color:#ADB5BD;'
                'font-size:0.78rem;font-weight:600;letter-spacing:.04em;">LOGO</div>')
    st.markdown(f"""
<div style="background:#FFFFFF;padding:16px 28px;margin:-0.4rem -1rem 16px -1rem;
            display:flex;align-items:center;gap:22px;
            border-bottom:1px solid #E3E7EB;box-shadow:0 1px 3px rgba(0,0,0,.04);">
  {logo}
  <div style="width:1px;height:46px;background:#C9D3DE;"></div>
  <span style="background:#a23a3a;color:#fff;font-size:0.8rem;font-weight:700;
               padding:3px 13px;border-radius:14px;">dev</span>
  <span style="font-size:1.45rem;font-weight:800;color:#0084B2;letter-spacing:.01em;">AI&nbsp;TEST&nbsp;ENGINE</span>
</div>
""", unsafe_allow_html=True)


def page_header(icon: str, title: str, subtitle: str = ""):
    """Bandeau de page bleu plein (style SUCRE)."""
    sub = (f'<div style="font-size:0.82rem;color:rgba(255,255,255,.82);'
           f'margin-top:2px;">{subtitle}</div>') if subtitle else ""
    st.markdown(f"""
<div style="background:#2d5a9e;padding:13px 22px;margin:0 -1rem 22px -1rem;
            display:flex;align-items:center;gap:14px;">
  <span style="font-size:1.45rem;color:#fff;line-height:1;">{icon}</span>
  <div>
    <div style="font-size:1.2rem;font-weight:700;color:#fff;">{title}</div>
    {sub}
  </div>
</div>
""", unsafe_allow_html=True)


def flow(steps):
    """Timeline verticale numérotée (pastilles cyan reliées par un rail)."""
    parts = ['<div style="background:#fff;border:1px solid #DEE2E6;padding:18px 20px;">']
    n = len(steps)
    for i, s in enumerate(steps):
        title = s[0]
        sub = s[1] if len(s) > 1 else ""
        sub_html = (f'<div style="font-size:0.78rem;color:#6C757D;margin-top:1px;">{sub}</div>'
                    if sub else "")
        rail = ('<div style="flex:1;width:2px;background:#CFE0EA;min-height:16px;'
                'margin:3px 0 0 0;"></div>') if i < n - 1 else ""
        pad = "0 0 16px 0" if i < n - 1 else "0"
        parts.append(
            f'<div style="display:flex;gap:13px;align-items:stretch;">'
            f'<div style="display:flex;flex-direction:column;align-items:center;flex-shrink:0;">'
            f'<div style="width:25px;height:25px;border-radius:50%;background:#0084B2;color:#fff;'
            f'display:flex;align-items:center;justify-content:center;font-size:0.78rem;'
            f'font-weight:700;">{i + 1}</div>{rail}</div>'
            f'<div style="padding:{pad};">'
            f'<div style="font-weight:600;color:#1b3e6f;font-size:0.9rem;line-height:1.25;">{title}</div>'
            f'{sub_html}</div></div>')
    parts.append('</div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


def tech_card(title, items):
    """Carte technique claire (remplace la liste markdown brute)."""
    lis = "".join(f'<li style="margin:4px 0;color:#212529;">{x}</li>' for x in items)
    st.markdown(
        f'<div style="background:#fff;border:1px solid #DEE2E6;padding:14px 16px;height:100%;">'
        f'<div style="font-weight:700;color:#1b3e6f;text-transform:uppercase;font-size:0.78rem;'
        f'letter-spacing:.03em;margin-bottom:8px;border-bottom:1px solid #DEE2E6;padding-bottom:6px;">'
        f'{title}</div>'
        f'<ul style="margin:0;padding-left:18px;font-size:0.88rem;line-height:1.5;">{lis}</ul></div>',
        unsafe_allow_html=True)


def svc_badge(name):
    return (f'<span style="background:#1b3e6f;color:#fff;padding:1px 7px;'
            f'font-size:0.76rem;font-weight:600;border-radius:2px;">{name}</span>')




# Sidebar
with st.sidebar:
    st.markdown("""
<div style="margin:-1rem -1rem 4px -1rem;padding:18px 20px 14px;
            background:rgba(0,0,0,.18);border-bottom:1px solid rgba(255,255,255,.15);">
  <div style="color:#fff;font-size:1.05rem;font-weight:800;letter-spacing:.02em;">AI&nbsp;TEST&nbsp;ENGINE</div>
  <div style="color:rgba(255,255,255,.7);font-size:0.78rem;margin-top:2px;">Dashboard &amp; g&eacute;n&eacute;ration de tests</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    api_ok = check_api()
    dot = '<span class="dot-green"></span>' if api_ok else '<span class="dot-red"></span>'
    status_txt = "API connectée" if api_ok else "API hors ligne"
    st.markdown(
        f'<div style="padding:8px 18px;border-bottom:1px solid rgba(255,255,255,.15);margin-bottom:6px;">'
        f'{dot}<small style="color:#dfe7f2;">{status_txt}</small></div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        ["Accueil", "Prédiction ML", "Génération de tests", "Performance"],
        label_visibility="collapsed",
    )
    st.markdown("<hr style='margin:12px 0;border-color:#DEE2E6'>", unsafe_allow_html=True)
    st.caption(f"Modèle : `best_model.pkl`")
    st.caption(f"API : `{TEST_ENGINE_URL}`")

model, FEATURES = load_model()

top_bar()


# PAGE : ACCUEIL
if page == "Accueil":
    page_header("AI Test Engine", "Prédicteur de risque de régression · Génération de tests Symfony")

    metrics = load_metrics()
    best_auc = metrics["best"].get("cv_auc_mean", metrics["best"].get("auc", 0)) if metrics else 0
    best_bal = metrics["best"].get("cv_balacc_mean", 0) if metrics else 0
    best_mcc = metrics["best"].get("cv_mcc_mean", 0) if metrics else 0
    best_name = metrics["best"].get("model", "—") if metrics else "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC-ROC (CV)", f"{best_auc:.3f}" if best_auc else "—")
    c2.metric("Balanced acc. (CV)", f"{best_bal:.3f}" if best_bal else "—")
    c3.metric("MCC (CV)", f"{best_mcc:.3f}" if best_mcc else "—")
    c4.metric("Meilleur modèle", best_name)

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Pipeline ML</div>', unsafe_allow_html=True)
        flow([
            ("Git history + code source Symfony",),
            ("extract_dataset.py", "Features PHP + git, nettoyées post-EDA"),
            ("dataset_sucre.csv",),
            ("train.py — GridSearchCV", "LogReg / RF / XGBoost"),
            ("best_model.pkl", "Sélection par CV groupée par fichier"),
            ("Ce dashboard", "Prédiction + export"),
        ])

    with col_right:
        st.markdown('<div class="section-header">Pipeline RAG + LLM</div>', unsafe_allow_html=True)
        flow([
            ("Dépôt Symfony indexé",),
            ("Chunking PHP (AST)",),
            ("Embeddings MiniLM-L12", "paraphrase-multilingual"),
            ("pgvector HNSW (cosine)", "3× plus rapide que seqscan"),
            ("FastAPI /generate-test", "LLM → WebTestCase PHP + validation php -l + retry"),
        ])

    st.markdown('<div class="section-header">Stack technique</div>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        tech_card("ML", ["Python 3.11", "XGBoost / scikit-learn",
                         "pandas / numpy", "Streamlit + Plotly"])
    with col_b:
        tech_card("API", ["FastAPI (uvicorn)", "PostgreSQL 15 + pgvector",
                          "sentence-transformers", "Docker Compose"])
    with col_c:
        tech_card("Déploiement", [
            "3 services Docker",
            f"{svc_badge('vector-db')}&nbsp;:5432",
            f"{svc_badge('brain-api')}&nbsp;:8000",
            f"{svc_badge('streamlit-ui')}&nbsp;:8501",
        ])


# PAGE : PRÉDICTION ML
elif page == "Prédiction ML":
    page_header("Prédiction ML", "Chargez un dataset CSV produit par <code>extract_dataset.py</code>")

    if model is None:
        st.error("Aucun modèle entraîné. Lance d'abord `python ml/train.py _dataset/dataset_sucre.csv`")
        st.stop()

    # ── Extraction dataset ────────────────────────────────────────────────────
    with st.expander("Extraire un dataset depuis un dépôt Symfony git"):
        st.caption("Lance `extract_dataset.py` directement depuis cette interface.")
        ex_col1, ex_col2 = st.columns(2)
        ex_repo = ex_col1.text_input(
            "Chemin du dépôt git (dans le conteneur)",
            value="/app/_dataset/sylius",
            help="Chemin absolu accessible depuis le conteneur Streamlit. "
                 "Ex: /app/_dataset/sylius ou /workspace/_dataset/mon-projet",
        )
        ex_output = ex_col2.text_input(
            "Nom du fichier CSV de sortie",
            value="dataset.csv",
            help="Sera créé dans /app/_dataset/<nom>",
        )
        ex_col3, ex_col4, ex_col5 = st.columns(3)
        ex_since = ex_col3.text_input("--since", value="36 months ago")
        ex_threshold = ex_col4.number_input("--bugfix-threshold", min_value=1, value=1)
        ex_feature_set = ex_col5.selectbox("--feature-set (train)", ["v2", "v1", "v3"])

        output_path = f"/app/_dataset/{ex_output}"
        cmd = (
            f"python /app/extract_dataset.py {ex_repo} {output_path}"
            f' --since "{ex_since}" --bugfix-threshold {ex_threshold}'
        )
        st.code(cmd, language="bash")

        if st.button("Lancer l'extraction", key="btn_extract"):
            with st.spinner("Extraction en cours — peut prendre plusieurs minutes…"):
                try:
                    result = subprocess.run(
                        ["python", "/app/extract_dataset.py", ex_repo, output_path,
                         "--since", ex_since,
                         "--bugfix-threshold", str(ex_threshold)],
                        capture_output=True, text=True, timeout=600,
                        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                    )
                    if result.returncode == 0:
                        st.success(f"Dataset extrait : `{output_path}`")
                        st.text(result.stdout[-3000:] if result.stdout else "")
                        # Proposer d'enchaîner avec l'entraînement
                        st.info(
                            f"Lance maintenant l'entraînement avec :\n"
                            f"`python /app/train.py {output_path} --feature-set {ex_feature_set}`"
                        )
                    else:
                        st.error("Erreur lors de l'extraction.")
                        st.code(result.stderr[-2000:])
                except subprocess.TimeoutExpired:
                    st.error("Timeout (10 min) — le dépôt est peut-être trop grand.")
                except Exception as e:
                    st.error(f"Erreur : {e}")

    st.markdown("---")

    # ── Source de données ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Source de données</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        choice = st.radio("Source", ["Dataset existant", "Upload CSV"], horizontal=True)
    with col2:
        if choice == "Dataset existant":
            datasets = list_datasets()
            if not datasets:
                st.warning("Aucun dataset trouvé dans `_dataset/`")
                st.stop()
            default_idx = next((i for i, n in enumerate(datasets) if n == "dataset_sucre.csv"), 0)
            selected = st.selectbox("Fichier", datasets, index=default_idx)
            df_raw = pd.read_csv(DATASET_DIR / selected)
        else:
            uploaded = st.file_uploader("CSV produit par extract_dataset.py", type="csv")
            if uploaded is None:
                st.info("Charge un CSV pour continuer.")
                st.stop()
            df_raw = pd.read_csv(uploaded)

    missing = set(FEATURES) - set(df_raw.columns)
    if missing:
        st.error(f"Colonnes manquantes : `{missing}`")
        st.stop()

    df = predict(df_raw.copy(), model, FEATURES)
    nb_high = int((df["risk_score"] >= 0.7).sum())
    nb_med  = int(((df["risk_score"] >= 0.4) & (df["risk_score"] < 0.7)).sum())
    nb_low  = int((df["risk_score"] < 0.4).sum())

    st.markdown('<div class="section-header">Vue d\'ensemble</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Méthodes analysées", len(df))
    c2.metric("Risque élevé (>=70%)", nb_high)
    c3.metric("Risque moyen (40-70%)", nb_med)
    c4.metric("Risque faible (<40%)", nb_low)
    c5.metric("Classes uniques", df["class"].nunique() if "class" in df.columns else "—")

    st.caption(
        "Le modèle est réglé pour ne rien rater (class_weight='balanced') : il sur-signale "
        "volontairement, donc le nombre de « risque élevé » est plus grand que la prévalence "
        "réelle des régressions. Pour prioriser, il faut se fier au classement par score ci-dessous "
        "plutôt qu'au simple total des « risque élevé »."
    )

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown('<div class="section-header">Distribution des scores</div>', unsafe_allow_html=True)
        if HAS_PLOTLY:
            fig = px.histogram(
                df, x="risk_score", nbins=30,
                color_discrete_sequence=["#0084B2"],
                labels={"risk_score": "Score de risque"},
            )
            fig.add_vline(x=0.5, line_dash="dash", line_color="#dc2626",
                          annotation_text="Seuil 0.5", annotation_position="top right")
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#212529", font_family="Source Sans 3", height=320,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.hist(df["risk_score"], bins=30, color="#0084B2", edgecolor="#1b3e6f")
            ax.axvline(0.5, color="red", linestyle="--")
            st.pyplot(fig)

    with col_g2:
        st.markdown('<div class="section-header">Top 15 classes à risque</div>', unsafe_allow_html=True)
        if "class" in df.columns and HAS_PLOTLY:
            top_classes = (
                df.groupby("class")["risk_score"].mean()
                .nlargest(15).reset_index()
                .sort_values("risk_score")
            )
            top_classes["color"] = top_classes["risk_score"].apply(
                lambda v: "#dc2626" if v >= 0.7 else ("#d97706" if v >= 0.4 else "#16a34a")
            )
            fig2 = go.Figure(go.Bar(
                x=top_classes["risk_score"],
                y=top_classes["class"],
                orientation="h",
                marker_color=top_classes["color"],
                text=top_classes["risk_score"].apply(lambda v: f"{v:.0%}"),
                textposition="outside",
            ))
            fig2.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#212529", font_family="Source Sans 3", height=320,
                xaxis_title="Score moyen",
                margin=dict(l=10, r=60, t=10, b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Colonne `class` absente — graphique non disponible.")

    st.markdown('<div class="section-header">Méthodes prioritaires à tester</div>', unsafe_allow_html=True)
    top_n = st.slider("Nombre de méthodes à afficher", 5, min(200, len(df)), 30)
    display_cols = [c for c in ["class", "method", "risk_label", "risk_score",
                                "cyclomatic_complexity", "nb_params",
                                "git_bugfix_count", "file"]
                    if c in df.columns]
    top_df = df.nlargest(top_n, "risk_score")[display_cols].copy()
    if "risk_score" in top_df.columns:
        top_df["risk_score"] = top_df["risk_score"].apply(lambda v: f"{v:.1%}")
    st.dataframe(top_df, width="stretch", hide_index=True)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("Telecharger tout (CSV)",
                           df.to_csv(index=False).encode("utf-8"),
                           "predictions.csv", "text/csv")
    with col_dl2:
        prio = df.nlargest(top_n, "risk_score")
        st.download_button(f"Telecharger le top {top_n} prioritaire (CSV)",
                           prio.to_csv(index=False).encode("utf-8"),
                           "top_priorites.csv", "text/csv")

    with st.expander("Generation rapide depuis ici"):
        risky = (
            df[df["risk_pred"] == 1]["class"].drop_duplicates().tolist()
            if "class" in df.columns else []
        )
        if not risky:
            st.info("Aucune classe avec risque >= 50%.")
        else:
            project_id = st.text_input("project_id", value="sucre", key="qgen_pid")
            n_cls = st.slider("Nb classes", 1, len(risky), min(3, len(risky)), key="qgen_n")
            if st.button("Generer les tests", key="qgen_btn"):
                if not api_ok:
                    st.error(f"API hors ligne ({TEST_ENGINE_URL})")
                else:
                    results = []
                    prog = st.progress(0.0)
                    for i, cls in enumerate(risky[:n_cls]):
                        prog.progress((i + 1) / n_cls, text=f"Generation : {cls}")
                        try:
                            r = requests.post(
                                f"{TEST_ENGINE_URL}/generate-test",
                                json={"project_id": project_id,
                                      "description": f"Tests pour {cls} (risque eleve)",
                                      "class_name": cls,
                                      "test_name": f"{cls}Test",
                                      "deterministic": True},
                                headers=api_headers(), timeout=900,
                            )
                            body = r.json() if "application/json" in r.headers.get("content-type", "") else {}
                            results.append({"class": cls, "status": r.status_code,
                                            "file": body.get("file", "—")})
                        except Exception as e:
                            results.append({"class": cls, "status": "ERR", "file": str(e)[:60]})
                    prog.empty()
                    st.success(f"{len(results)} generations lancées.")
                    st.dataframe(pd.DataFrame(results), width="stretch", hide_index=True)


# PAGE : GÉNÉRATION DE TESTS
elif page == "Génération de tests":
    page_header("Génération de tests", "Interface complète vers l'API FastAPI du AI Test Engine")

    if not api_ok:
        st.warning(f"L'API est hors ligne. Lance : `docker compose up -d`")

    tab_projects, tab_index, tab_func, tab_unit = st.tabs([
        "Projets indexés", "Indexer un projet",
        "Test fonctionnel", "Test unitaire",
    ])

    with tab_projects:
        st.markdown('<div class="section-header">Projets dans la base vectorielle</div>',
                    unsafe_allow_html=True)
        if st.button("Actualiser"):
            st.cache_data.clear()
        try:
            r = requests.get(f"{TEST_ENGINE_URL}/projects", headers=api_headers(), timeout=5)
            if r.status_code == 200:
                data = r.json()
                projects = data if isinstance(data, list) else data.get("projects", [])
                if projects:
                    for p in projects:
                        pid = p if isinstance(p, str) else p.get("project_id", str(p))
                        col_p, col_s = st.columns([3, 1])
                        col_p.markdown(f"**{pid}**")
                        if col_s.button("Stats", key=f"stats_{pid}"):
                            rs = requests.get(f"{TEST_ENGINE_URL}/project/{pid}/stats",
                                              headers=api_headers(), timeout=5)
                            st.json(rs.json() if rs.status_code == 200 else {"error": rs.status_code})
                else:
                    st.info("Aucun projet indexé.")
            else:
                st.warning(f"HTTP {r.status_code}")
        except Exception as e:
            st.error(f"Impossible de joindre l'API : {e}")

    with tab_index:
        st.markdown('<div class="section-header">Indexer un nouveau projet</div>',
                    unsafe_allow_html=True)
        with st.form("form_index"):
            repo_path = st.text_input("Chemin du dépôt Symfony",
                                      placeholder="C:/projets/mon-projet-symfony")
            proj_id = st.text_input("project_id", placeholder="mon-projet")
            if st.form_submit_button("Lancer l'indexation"):
                if not repo_path or not proj_id:
                    st.error("Remplis les deux champs.")
                elif not api_ok:
                    st.error("API hors ligne.")
                else:
                    with st.spinner("Indexation en cours…"):
                        try:
                            r = requests.post(
                                f"{TEST_ENGINE_URL}/index-project",
                                json={"project_path": repo_path, "project_id": proj_id},
                                headers=api_headers(), timeout=300,
                            )
                            if r.status_code == 200:
                                st.success("Indexation réussie !")
                                st.json(r.json())
                            else:
                                st.error(f"Erreur HTTP {r.status_code}")
                                st.code(r.text)
                        except Exception as e:
                            st.error(f"Erreur : {e}")

    with tab_func:
        st.markdown('<div class="section-header">Générer un WebTestCase Symfony</div>',
                    unsafe_allow_html=True)
        with st.form("form_func"):
            col_f1, col_f2 = st.columns(2)
            f_project = col_f1.text_input("project_id", value="sucre")
            f_class   = col_f2.text_input("Classe cible (optionnel)",
                                          placeholder="ProductController")
            f_desc = st.text_area("Description du scénario",
                                  placeholder="Tester GET /products avec et sans auth",
                                  height=80)
            f_test_name = st.text_input("Nom du test (optionnel)",
                                        placeholder="ProductControllerTest")
            f_det   = st.checkbox("Mode déterministe (sans LLM)", value=True)
            f_async = st.checkbox("Mode asynchrone (non-bloquant)", value=True,
                                  help="Retourne un job_id immédiatement et poll le résultat.")
            if st.form_submit_button("Générer le test fonctionnel"):
                if not f_project or not f_desc:
                    st.error("project_id et description sont obligatoires.")
                elif not api_ok:
                    st.error("API hors ligne.")
                else:
                    payload = {"project_id": f_project, "description": f_desc,
                               "deterministic": f_det, "async_mode": f_async}
                    if f_class:
                        payload["class_name"] = f_class
                    if f_test_name:
                        payload["test_name"] = f_test_name

                    if f_async:
                        # Mode async : on lance et on stocke le job_id en session
                        try:
                            r = requests.post(f"{TEST_ENGINE_URL}/generate-test",
                                              json=payload, headers=api_headers(), timeout=30)
                            if r.status_code == 200:
                                body = r.json()
                                st.session_state["pending_job"] = body.get("job_id")
                                st.info(f"Job lancé : `{body.get('job_id')}` — résultat dans quelques instants…")
                            else:
                                st.error(f"HTTP {r.status_code} — {r.text[:200]}")
                        except Exception as e:
                            st.error(f"Erreur : {e}")
                    else:
                        with st.spinner("Génération en cours (mode sync)…"):
                            try:
                                r = requests.post(f"{TEST_ENGINE_URL}/generate-test",
                                                  json=payload, headers=api_headers(), timeout=900)
                                if r.status_code == 200:
                                    body = r.json()
                                    st.success(f"Test généré : `{body.get('file', '')}`")
                                    st.info(body.get("note", ""))
                                    with st.expander("Réponse complète"):
                                        st.json(body)
                                else:
                                    st.error(f"HTTP {r.status_code}")
                                    st.code(r.text)
                            except Exception as e:
                                st.error(f"Erreur : {e}")

        # ── Polling du job en cours ────────────────────────────────────────────
        if st.session_state.get("pending_job"):
            job_id = st.session_state["pending_job"]
            st.markdown(f"**Job en cours : `{job_id}`**")
            col_poll, col_cancel = st.columns([1, 1])
            if col_poll.button("Actualiser le statut", key="btn_poll"):
                try:
                    r = requests.get(f"{TEST_ENGINE_URL}/job/{job_id}/status",
                                     headers=api_headers(), timeout=10)
                    if r.status_code == 200:
                        job = r.json()
                        status = job.get("status", "?")
                        if status == "done":
                            st.success("Terminé !")
                            st.info(job.get("result", {}).get("note", ""))
                            st.json(job.get("result", {}))
                            st.session_state.pop("pending_job", None)
                        elif status == "error":
                            st.error(f"Erreur : {job.get('error', '?')}")
                            st.session_state.pop("pending_job", None)
                        else:
                            st.info(f"Statut : **{status}** — réactualise dans quelques secondes.")
                    else:
                        st.warning(f"HTTP {r.status_code}")
                except Exception as e:
                    st.error(f"Erreur polling : {e}")
            if col_cancel.button("Annuler le suivi", key="btn_cancel"):
                st.session_state.pop("pending_job", None)
                st.rerun()

    with tab_unit:
        st.markdown('<div class="section-header">Générer un test unitaire PHPUnit</div>',
                    unsafe_allow_html=True)
        with st.form("form_unit"):
            col_u1, col_u2 = st.columns(2)
            u_project = col_u1.text_input("project_id", value="sucre")
            u_class   = col_u2.text_input("Classe cible (optionnel)",
                                          placeholder="OrderService")
            u_desc = st.text_area("Description du comportement à tester",
                                  placeholder="calculateTotal retourne 0 pour un panier vide",
                                  height=80)
            u_det   = st.checkbox("Mode déterministe (sans LLM)", value=False, key="u_det")
            u_async = st.checkbox("Mode asynchrone (non-bloquant)", value=True, key="u_async",
                                  help="Retourne un job_id immédiatement et poll le résultat.")
            if st.form_submit_button("Générer le test unitaire"):
                if not u_project or not u_desc:
                    st.error("project_id et description sont obligatoires.")
                elif not api_ok:
                    st.error("API hors ligne.")
                else:
                    payload = {"project_id": u_project, "description": u_desc,
                               "deterministic": u_det, "async_mode": u_async}
                    if u_class:
                        payload["class_name"] = u_class

                    if u_async:
                        try:
                            r = requests.post(f"{TEST_ENGINE_URL}/generate-unit-test",
                                              json=payload, headers=api_headers(), timeout=30)
                            if r.status_code == 200:
                                body = r.json()
                                st.session_state["pending_unit_job"] = body.get("job_id")
                                st.info(f"Job lancé : `{body.get('job_id')}` — résultat dans quelques instants…")
                            else:
                                st.error(f"HTTP {r.status_code} — {r.text[:200]}")
                        except Exception as e:
                            st.error(f"Erreur : {e}")
                    else:
                        with st.spinner("Génération en cours (mode sync)…"):
                            try:
                                r = requests.post(f"{TEST_ENGINE_URL}/generate-unit-test",
                                                  json=payload, headers=api_headers(), timeout=900)
                                if r.status_code == 200:
                                    body = r.json()
                                    st.success(f"Test généré : `{body.get('file', '')}`")
                                    st.info(body.get("note", ""))
                                    with st.expander("Réponse complète"):
                                        st.json(body)
                                else:
                                    st.error(f"HTTP {r.status_code}")
                                    st.code(r.text)
                            except Exception as e:
                                st.error(f"Erreur : {e}")

        # ── Polling du job unitaire en cours ──────────────────────────────────
        if st.session_state.get("pending_unit_job"):
            unit_job_id = st.session_state["pending_unit_job"]
            st.markdown(f"**Job en cours : `{unit_job_id}`**")
            col_upoll, col_ucancel = st.columns([1, 1])
            if col_upoll.button("Actualiser le statut", key="btn_upoll"):
                try:
                    r = requests.get(f"{TEST_ENGINE_URL}/job/{unit_job_id}/status",
                                     headers=api_headers(), timeout=10)
                    if r.status_code == 200:
                        job = r.json()
                        status = job.get("status", "?")
                        if status == "done":
                            st.success("Terminé !")
                            st.info(job.get("result", {}).get("note", ""))
                            st.json(job.get("result", {}))
                            st.session_state.pop("pending_unit_job", None)
                        elif status == "error":
                            st.error(f"Erreur : {job.get('error', '?')}")
                            st.session_state.pop("pending_unit_job", None)
                        else:
                            st.info(f"Statut : **{status}** — réactualise dans quelques secondes.")
                    else:
                        st.warning(f"HTTP {r.status_code}")
                except Exception as e:
                    st.error(f"Erreur polling : {e}")
            if col_ucancel.button("Annuler le suivi", key="btn_ucancel"):
                st.session_state.pop("pending_unit_job", None)
                st.rerun()


# PAGE : PERFORMANCE
elif page == "Performance":
    page_header("Performance du modèle", "Métriques de validation croisée et matrices de confusion")

    metrics = load_metrics()
    if metrics is None:
        st.warning("`metrics.json` introuvable. Lance d'abord `python ml/train.py`.")
        st.stop()

    best = metrics["best"]
    auc     = best.get("cv_auc_mean", best.get("auc", 0))
    bal_acc = best.get("cv_balacc_mean", 0)
    mcc     = best.get("cv_mcc_mean", 0)
    gap     = best.get("overfit_gap", 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Meilleur modèle", best.get("model", "—"))
    c2.metric("AUC-ROC (CV)", f"{auc:.3f}")
    c3.metric("Balanced acc. (CV)", f"{bal_acc:.3f}")
    c4.metric("MCC (CV)", f"{mcc:.3f}")

    if gap > 0.1:
        st.warning(f"Ecart overfitting : {gap:.4f} — le modèle surapprentit légèrement.")
    else:
        st.success(f"Ecart overfitting faible : {gap:.4f}")

    st.markdown('<div class="section-header">Comparaison des modèles</div>', unsafe_allow_html=True)
    all_models = metrics.get("all", [])
    if all_models:
        df_metrics = pd.DataFrame(all_models).set_index("model")
        display_cols = [c for c in ["cv_test_f1_mean", "cv_auc_mean",
                                    "cv_balacc_mean", "cv_mcc_mean",
                                    "accuracy", "overfit_gap"]
                        if c in df_metrics.columns]
        df_show = df_metrics[display_cols].copy()

        if HAS_PLOTLY:
            fig_bar = go.Figure()
            for col, color, label in [
                ("cv_auc_mean",    "#0084B2", "AUC (CV)"),
                ("cv_balacc_mean", "#2d5a9e", "Balanced acc (CV)"),
                ("cv_mcc_mean",    "#1b3e6f", "MCC (CV)"),
            ]:
                if col in df_show.columns:
                    fig_bar.add_trace(go.Bar(
                        name=label,
                        x=df_show.index.tolist(),
                        y=df_show[col].tolist(),
                        marker_color=color,
                        text=[f"{v:.3f}" for v in df_show[col]],
                        textposition="outside",
                    ))
            fig_bar.update_layout(
                barmode="group",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#212529", font_family="Source Sans 3", height=360,
                legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0,
                            font=dict(size=12, color="#212529")),
                margin=dict(l=10, r=10, t=50, b=30),
                xaxis=dict(showgrid=False, tickfont=dict(size=12, color="#212529")),
                yaxis=dict(range=[0, 1.15], gridcolor="#E3E7EB",
                           tickfont=dict(size=11, color="#6C757D")),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(df_show.map(lambda v: f"{v:.4f}" if isinstance(v, float) else v),
                     width="stretch")

    st.markdown('<div class="section-header">Importance des features</div>', unsafe_allow_html=True)
    importance = load_importance()
    if importance is not None:
        top15 = importance.head(15).sort_values("importance")
        if HAS_PLOTLY:
            fig_imp = go.Figure(go.Bar(
                x=top15["importance"], y=top15["feature"],
                orientation="h",
                marker_color="#0084B2",
                text=top15["importance"].apply(lambda v: f"{v:.4f}"),
                textposition="outside",
            ))
            fig_imp.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#212529", font_family="Source Sans 3", height=420,
                xaxis_title="Importance",
                xaxis=dict(gridcolor="#E3E7EB", tickfont=dict(color="#6C757D")),
                yaxis=dict(tickfont=dict(size=11, color="#212529")),
                margin=dict(l=10, r=70, t=10, b=10),
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(top15["feature"], top15["importance"], color="#0084B2")
            ax.set_xlabel("Importance")
            st.pyplot(fig)

        with st.expander("Toutes les features"):
            st.dataframe(importance, width="stretch", hide_index=True)

    st.markdown('<div class="section-header">Matrices de confusion</div>', unsafe_allow_html=True)
    cols_cm = st.columns(3)
    for i, name in enumerate(["logreg", "rf", "xgb"]):
        png = ROOT / "models" / f"confusion_{name}.png"
        if png.exists():
            cols_cm[i].image(str(png), caption=name.upper(), width="stretch")

    with st.expander("Note méthodologique"):
        st.markdown("""
**Validation croisée** : StratifiedGroupKFold (5 plis), **groupée par fichier** —
aucun fichier (ni aucune de ses méthodes) ne se trouve à la fois en apprentissage
et en test. Évite la fuite de groupe inhérente à une granularité méthode avec un
label défini au niveau fichier.

**Label** : proxy bugfix-from-git (Hassan 2009 ; Kamei et al. 2013) —
`label_risk = (git_bugfix_count >= seuil)` sur une fenêtre de 12 mois.

**Features** : caractéristiques structurelles du code (complexité, taille,
dépendances, routes, sécurité) + churn git. `git_bugfix_count` et
`git_total_commits` sont **exclus** car ils déterminent le label (fuite de cible).

**Split de test** : GroupShuffleSplit groupé par fichier — pas de découpage
aléatoire au niveau méthode.
        """)
