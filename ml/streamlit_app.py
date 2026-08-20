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
# Le dossier des jeux de données n'est pas au même endroit selon le contexte :
# monté sur /app/_dataset dans le conteneur (cf. docker-compose.yml), et
# à la racine du dépôt en exécution locale. Sans ce test, la page « Prédiction
# ML » cherchait /_dataset et affichait « Aucun dataset trouvé » alors que les
# fichiers étaient bien présents.
DATASET_DIR = (Path("/app/_dataset") if Path("/app/_dataset").is_dir()
               else ROOT.parent / "_dataset")
TEST_ENGINE_URL = os.environ.get("TEST_ENGINE_URL", "http://localhost:8000")
TEST_ENGINE_API_KEY = os.environ.get("TEST_ENGINE_API_KEY", "")

# Clés session_state — définies ici pour éviter les magic strings répétées dans le code
_SK_PENDING_JOB      = "pending_job"       # job /generate-test en cours
_SK_PENDING_UNIT_JOB = "pending_unit_job"  # job /generate-unit-test en cours

# ── Seuils de risque — définis UNE SEULE FOIS ────────────────────────────────
# RISK_PRED_THRESHOLD est la seule frontière ACTIONNABLE : c'est elle qui décide
# des classes proposées à la génération de tests.
#
# Sa valeur, 0.5, est celle de `model.predict()` — donc celle sous laquelle
# TOUTES les métriques publiées sont calculées (MCC, balanced accuracy, matrices
# de confusion). Le dashboard s'y tient pour que les chiffres du mémoire
# décrivent bien le comportement réel du produit.
#
# Un seuil de 0.7 avait été essayé pour coller aux bandes d'affichage. Mesuré en
# validation croisée groupée, il coûtait cher :
#     0.5 → MCC 0.717 | rappel 0.804 | 187 méthodes signalées
#     0.7 → MCC 0.698 | rappel 0.749 | 170 méthodes signalées
# soit 12 méthodes réellement à risque disparues de la priorisation. Pour un
# outil dont le rôle est de NE RIEN RATER, échanger 5,5 points de rappel contre
# 2,4 points de précision va dans le mauvais sens.
#
# BAND_* ne sert qu'à l'AFFICHAGE (Faible / Moyen / Élevé) : ces bandes graduent
# la lecture, elles ne décident de rien.
RISK_PRED_THRESHOLD = 0.5   # décision : « à tester en priorité »
BAND_MED            = 0.4   # affichage : seuil Faible → Moyen
BAND_HIGH           = 0.7   # affichage : seuil Moyen → Élevé

# ── Délais des appels à l'API (secondes) ─────────────────────────────────────
# Lancement d'un job asynchrone : la réponse est immédiate (un identifiant).
API_TIMEOUT_QUICK = 30
# Sondage de statut : lecture d'un dict en mémoire.
API_TIMEOUT_POLL  = 10
# Vérification de disponibilité affichée dans la barre latérale — doit rester
# imperceptible, l'interface l'appelle à chaque rafraîchissement.
API_TIMEOUT_HEALTH = 2
# Génération synchrone : le serveur se plafonne lui-même à 840 s
# (_TOTAL_BUDGET_SEC), on garde une minute de marge au-dessus.
API_TIMEOUT_GENERATION = 900
# Extraction d'un dataset : parcourt tout l'historique git du dépôt.
SUBPROCESS_TIMEOUT_EXTRACT = 600

# ── Confinement de l'extraction de dataset ───────────────────────────────────
# Le formulaire d'extraction expose deux champs texte libres (chemin du dépôt,
# nom du CSV) qui aboutissent à un sous-processus. Aucun risque d'injection de
# commande — subprocess.run reçoit une liste, jamais shell=True — mais sans
# bornes, on pourrait écrire le CSV n'importe où (`../../etc/x`) ou lancer
# l'extraction sur un chemin arbitraire du conteneur.
DATASET_DIR_CONTAINER = "/app/_dataset"
ALLOWED_REPO_ROOTS = ("/app/_dataset", "/workspace")

# ── Affichage ────────────────────────────────────────────────────────────────
TOP_CLASSES_CHART = 15   # barres du graphique « classes à risque »
TABLE_ROWS_MIN    = 5    # bornes du curseur de lignes affichées
TABLE_ROWS_MAX    = 200
TABLE_ROWS_DEFAULT = 30

st.set_page_config(
    page_title="AI Test Engine",
    page_icon=Path(__file__).parent / "assets" / "favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — système de design CNAM / Bootstrap 5 ───────────────────────────────
st.markdown("""
<style>
/* ── Aucune police n'est chargée depuis un service externe ───────────────────
   Un `@import url('https://fonts.googleapis.com/…')` figurait ici. Cet import
   est exécuté PAR LE NAVIGATEUR de l'utilisateur : chaque ouverture du tableau
   de bord transmettait donc à Google l'adresse IP du poste, son User-Agent et
   la page référente.

   Deux raisons de l'avoir retiré :
   1. Une adresse IP est une donnée personnelle au sens du RGPD. Le tribunal
      régional de Munich a jugé en janvier 2022 (aff. 3 O 17493/20) que l'appel
      dynamique à Google Fonts constituait un manquement.
   2. Cohérence : un outil dont l'argument central est « rien ne sort de
      l'infrastructure » ne peut pas appeler un CDN américain au chargement.

   La chaîne de repli ci-dessous s'appuie uniquement sur des polices présentes
   sur le poste. Segoe UI (Windows) et system-ui couvrent tous les cas ;
   l'apparence est quasi identique. Pour retrouver Source Sans 3 à l'identique,
   déposer les .woff2 dans ml/assets/ et déclarer une règle @font-face locale. */

/* ── Palette SUCRE ──
   --navy   #1b3e6f  sidebar, en-têtes de colonnes
   --blue   #2d5a9e  bandeaux de page
   --bordo  #a23a3a  item actif, badge dev
   --cyan   #00769F  boutons, liens, accents
                     (etait #0084B2 : 4,25:1 sur blanc, sous le minimum AA de
                      4,5:1 — echouait a la fois comme fond de bouton et comme
                      couleur de lien. #00769F donne 5,13:1.)
   --bord   #8A9199  bordures des CHAMPS DE SAISIE (3,19:1, critere RGAA 3.3
                     « composants d'interface » : la bordure identifie la zone
                     ou saisir. #CED4DA ne donnait que 1,49:1.)
   --trait  #DEE2E6  bordures DECORATIVES (cartes, panneaux, separateurs).
                     Volontairement inchangees : 1,4.11 ne s'applique qu'aux
                     elements necessaires pour IDENTIFIER un composant, ce
                     qu'un liseré de carte n'est pas. Les assombrir alourdirait
                     l'interface sans gain d'accessibilite.
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
/* Padding du .block-container : défini plus bas avec --topbar-h (bandeau fixe). */

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
    background:#00769F !important; color:#FFFFFF !important; border:none !important;
    border-radius:0 !important; font-weight:600 !important; padding:8px 20px !important;
    font-size:0.9rem !important; transition:background .15s ease; width:fit-content !important;
}
.stButton > button:hover{ background:#006A91 !important; color:#FFFFFF !important; }
.stButton > button:focus{ box-shadow:0 0 0 3px rgba(0,118,159,.25) !important; outline:none !important; }
section[data-testid="stSidebar"] .stButton > button{ width:100% !important; }

/* ── Inputs / Textareas / Selects ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
div[data-baseweb="input"] input,
div[data-baseweb="select"] > div{
    border:1px solid #8A9199 !important; border-radius:0 !important;
    font-size:0.9rem !important; color:#212529 !important; background:#FFFFFF !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus{
    border-color:#00769F !important; box-shadow:0 0 0 2px rgba(0,118,159,.2) !important; outline:none !important;
}

/* ── Checkbox ── */
[data-testid="stCheckbox"] label{ color:#212529 !important; }
[data-testid="stCheckbox"] svg{ color:#00769F !important; }

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
[data-testid="stInfo"]{ background:#CFF4FC !important; color:#055160 !important; border-left:4px solid #00769F !important; }
[data-testid="stSpinner"] p{ color:#212529 !important; }

hr{ border-color:#DEE2E6 !important; }
p, div, span, label{ color:#212529; }
a{ color:#00769F !important; }
a:hover{ color:#006A91 !important; }

/* ── Masquer la barre Streamlit (Deploy / menu) + décoration ── */
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"]{
    display:none !important;
}
header[data-testid="stHeader"]{ height:0 !important; min-height:0 !important; background:transparent !important; }

/* ── Sidebar : décalée sous le bandeau fixe (la nav commence sous la barre
      blanche pleine largeur, comme SUCRE). Le fond marine remplit toute la
      hauteur ; le bandeau blanc (z-index élevé) recouvre juste sa partie haute. ── */
[data-testid="stSidebarHeader"]{ display:none !important; }
[data-testid="stSidebarCollapseButton"]{ display:none !important; }
[data-testid="stSidebarUserContent"]{ padding-top:calc(var(--topbar-h) + 10px) !important; }
section[data-testid="stSidebar"] > div:first-child{ padding-top:0 !important; }

/* ── Bandeau fixe pleine largeur + décalage du contenu ──
      --topbar-h : hauteur du bandeau blanc, qui couvre TOUTE la largeur en haut
      (par-dessus la sidebar, façon SUCRE). Le contenu principal et la sidebar
      sont décalés vers le bas de cette hauteur. ── */
:root{ --pad-x:2.6rem; --topbar-h:86px; }
.block-container{
    padding:calc(var(--topbar-h) + 18px) var(--pad-x) 2.4rem var(--pad-x) !important;
    max-width:100% !important;
}

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
    df["risk_pred"] = (df["risk_score"] >= RISK_PRED_THRESHOLD).astype(int)
    df["risk_label"] = df["risk_score"].apply(
        lambda v: "Élevé" if v >= BAND_HIGH else ("Moyen" if v >= BAND_MED else "Faible")
    )
    return df


def _call_generate(endpoint: str, payload: dict, async_mode: bool, session_key: str) -> None:
    """
    Lance un appel POST vers `endpoint` (generate-test ou generate-unit-test) et
    gère l'affichage du résultat — factorise les deux blocs identiques de la page
    "Génération de tests" pour éviter la duplication de ~140 lignes.

    - async_mode=True  : retourne immédiatement un job_id, stocké dans session_state[session_key]
    - async_mode=False : attend la réponse (timeout 900s) et affiche le résultat directement
    """
    url = f"{TEST_ENGINE_URL}/{endpoint}"
    if async_mode:
        try:
            r = requests.post(url, json=payload, headers=api_headers(), timeout=API_TIMEOUT_QUICK)
            if r.status_code == 200:
                body = r.json()
                st.session_state[session_key] = body.get("job_id")
                st.info(f"Job lancé : `{body.get('job_id')}` — résultat dans quelques instants…")
            else:
                st.error(f"HTTP {r.status_code} — {r.text[:200]}")
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur réseau : {e}")
    else:
        with st.spinner("Génération en cours (mode sync)…"):
            try:
                r = requests.post(url, json=payload, headers=api_headers(), timeout=API_TIMEOUT_GENERATION)
                if r.status_code == 200:
                    body = r.json()
                    st.success(f"Test généré : `{body.get('file', '')}`")
                    st.info(body.get("note", ""))
                    with st.expander("Réponse complète"):
                        st.json(body)
                else:
                    st.error(f"HTTP {r.status_code}")
                    st.code(r.text)
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur réseau : {e}")


def _poll_job(session_key: str, poll_btn_key: str, cancel_btn_key: str) -> None:
    """
    Affiche le widget de polling pour un job async stocké dans session_state[session_key].
    Factorise les deux blocs de polling (fonctionnel / unitaire) identiques à ~40 lignes.
    """
    if not st.session_state.get(session_key):
        return
    job_id = st.session_state[session_key]
    st.markdown(f"**Job en cours : `{job_id}`**")
    col_poll, col_cancel = st.columns([1, 1])
    if col_poll.button("Actualiser le statut", key=poll_btn_key):
        try:
            r = requests.get(f"{TEST_ENGINE_URL}/job/{job_id}/status",
                             headers=api_headers(), timeout=API_TIMEOUT_POLL)
            if r.status_code == 200:
                job = r.json()
                status = job.get("status", "?")
                if status == "done":
                    st.success("Terminé !")
                    st.info(job.get("result", {}).get("note", ""))
                    st.json(job.get("result", {}))
                    st.session_state.pop(session_key, None)
                elif status == "error":
                    st.error(f"Erreur : {job.get('error', '?')}")
                    st.session_state.pop(session_key, None)
                else:
                    st.info(f"Statut : **{status}** — réactualise dans quelques secondes.")
            else:
                st.warning(f"HTTP {r.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur polling : {e}")
    if col_cancel.button("Annuler le suivi", key=cancel_btn_key):
        st.session_state.pop(session_key, None)
        st.rerun()


def api_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if TEST_ENGINE_API_KEY:
        h["X-API-Key"] = TEST_ENGINE_API_KEY
    return h


def check_api() -> bool:
    try:
        r = requests.get(f"{TEST_ENGINE_URL}/health", timeout=API_TIMEOUT_HEALTH)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        # Timeout, connexion refusée, DNS — tous les cas réseau attendus
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
    """Bandeau institutionnel façon SUCRE : FIXE, pleine largeur du viewport,
    au-dessus de la sidebar (la barre blanche traverse tout le haut ; la sidebar
    et le contenu sont décalés dessous via --topbar-h). Logo + séparateur +
    pastille dev + titre bleu."""
    uri = _logo_data_uri()
    if uri:
        logo = f'<img src="{uri}" alt="logo" style="height:56px;width:auto;display:block;">'
    else:
        logo = ('<div style="height:56px;width:150px;border:1px dashed #ADB5BD;'
                'display:flex;align-items:center;justify-content:center;color:#ADB5BD;'
                'font-size:0.78rem;font-weight:600;letter-spacing:.04em;">LOGO</div>')
    # position:fixed + z-index très élevé → la barre recouvre le haut de la sidebar
    # marine et occupe toute la largeur de l'écran, comme dans SUCRE.
    st.markdown(f"""
<div style="position:fixed;top:0;left:0;right:0;height:var(--topbar-h);z-index:2147483000;
            background:#FFFFFF;padding:0 26px;box-sizing:border-box;
            display:flex;align-items:center;gap:20px;
            border-bottom:1px solid #E3E7EB;box-shadow:0 1px 4px rgba(0,0,0,.06);">
  {logo}
  <div style="width:1px;height:46px;background:#C9D3DE;"></div>
  <span style="background:#a23a3a;color:#fff;font-size:0.8rem;font-weight:700;
               padding:3px 13px;border-radius:14px;">dev</span>
  <span style="font-size:1.4rem;font-weight:700;color:#00769F;letter-spacing:.01em;
               white-space:nowrap;">AI&nbsp;TEST&nbsp;ENGINE</span>
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
            f'<div style="width:25px;height:25px;border-radius:50%;background:#00769F;color:#fff;'
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
    # Sidebar façon SUCRE : la navigation démarre directement en haut, sans bloc
    # de titre (le branding « AI TEST ENGINE » vit dans le bandeau du haut, pas ici).
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Accueil", "Prédiction ML", "Génération de tests", "Performance"],
        label_visibility="collapsed",
    )

    # Pied de sidebar : statut API + méta (regroupés en bas, hors de la nav).
    # check_api() reste ici car api_ok est réutilisé par les pages plus bas.
    api_ok = check_api()
    dot = '<span class="dot-green"></span>' if api_ok else '<span class="dot-red"></span>'
    status_txt = "API connectée" if api_ok else "API hors ligne"
    st.markdown("<hr style='margin:14px 0 10px;border-color:rgba(255,255,255,.18)'>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="padding:2px 18px 6px;">{dot}'
        f'<small style="color:#dfe7f2;">{status_txt}</small></div>',
        unsafe_allow_html=True
    )
    st.caption(f"Modèle : `best_model.pkl`")
    st.caption(f"API : `{TEST_ENGINE_URL}`")

model, FEATURES = load_model()

top_bar()


# PAGE : ACCUEIL
if page == "Accueil":
    page_header("AI Test Engine", "Prédicteur de risque de régression · Génération de tests Symfony")

    metrics = load_metrics()
    best = metrics.get("best", {}) if metrics else {}
    # cv_auc_mean = clé actuelle ; "auc" = clé legacy des anciens metrics.json
    best_auc  = best.get("cv_auc_mean", best.get("auc", 0))
    best_bal  = best.get("cv_balacc_mean", 0)
    best_mcc  = best.get("cv_mcc_mean", 0)
    best_name = best.get("model", "—")

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
            ("extract_dataset.py", "Features structurelles, seuil bugfix 5"),
            ("dataset_sucre_t5.csv",),
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
            ("pgvector HNSW (cosine)", "×2,24 vs parcours séquentiel"),
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
            "5 services Docker",
            f"{svc_badge('streamlit-ui')}&nbsp;:8501",
            f"{svc_badge('brain-api')}&nbsp;:8000",
            # vector-db et ollama ne publient aucun port : ils ne sont joignables
            # que depuis le réseau interne. C'est un choix de sécurité, pas un oubli.
            f"{svc_badge('vector-db')}&nbsp;<i>interne</i>",
            f"{svc_badge('ollama')}&nbsp;<i>interne</i>",
            f"{svc_badge('ollama-pull')}&nbsp;<i>éphémère</i>",
        ])


# PAGE : PRÉDICTION ML
elif page == "Prédiction ML":
    page_header("Prédiction ML", "Chargez un dataset CSV produit par <code>extract_dataset.py</code>")

    if model is None:
        st.error("Aucun modèle entraîné. Lance d'abord "
                 "`python ml/train.py _dataset/dataset_sucre_t5.csv --feature-set v3`")
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
        # Défauts alignés sur ceux du code (extract_dataset.py / train.py) après
        # recalibrage : un seuil de 1 ou 2 rend le label dégénéré (82 % de positifs
        # sur SUCRE → le modèle ne priorise plus rien) et v2 n'est plus le jeu de
        # features déployé. Proposer ici les anciennes valeurs revenait à faire
        # refabriquer par l'interface le dataset que le recalibrage a écarté.
        ex_threshold = ex_col4.number_input(
            "--bugfix-threshold", min_value=1, value=5,
            help="5 = valeur calibrée. En dessous, le label devient dégénéré "
                 "(à 2, SUCRE a 82 % de positifs et la priorisation perd son sens).",
        )
        ex_feature_set = ex_col5.selectbox(
            "--feature-set (train)", ["v3", "v2", "v1"],
            help="v3 = 13 features structurelles sans fuite git — le jeu réellement déployé.",
        )

        # ── Assainissement des entrées avant de les passer à un sous-processus ──
        # Il n'y a PAS de risque d'injection de commande : subprocess.run reçoit une
        # LISTE d'arguments et jamais shell=True, donc les métacaractères du shell
        # (;, |, &&, $()) sont transmis littéralement, pas interprétés.
        #
        # Le risque réel est ailleurs : ces deux champs sont libres, et sans contrôle
        # ils permettent d'écrire un CSV n'importe où dans le conteneur (`../../etc/x`)
        # ou de lancer l'extraction sur un dépôt arbitraire. On borne donc les deux.
        nom_sortie = os.path.basename(ex_output.strip()) or "dataset.csv"
        if not nom_sortie.endswith(".csv"):
            nom_sortie += ".csv"
        output_path = os.path.join(DATASET_DIR_CONTAINER, nom_sortie)

        chemin_depot = os.path.normpath(ex_repo.strip())
        depot_autorise = any(
            chemin_depot == racine or chemin_depot.startswith(racine + os.sep)
            for racine in ALLOWED_REPO_ROOTS
        )

        cmd = (
            f"python /app/extract_dataset.py {chemin_depot} {output_path}"
            f' --since "{ex_since}" --bugfix-threshold {ex_threshold}'
        )
        st.code(cmd, language="bash")

        if nom_sortie != ex_output.strip():
            st.caption(f"Nom de sortie normalisé : `{nom_sortie}`")
        if not depot_autorise:
            st.warning(
                f"Chemin hors des racines autorisées ({', '.join(ALLOWED_REPO_ROOTS)}). "
                "L'extraction ne sera pas lancée."
            )

        if st.button("Lancer l'extraction", key="btn_extract", disabled=not depot_autorise):
            with st.spinner("Extraction en cours — peut prendre plusieurs minutes…"):
                try:
                    result = subprocess.run(
                        ["python", "/app/extract_dataset.py", chemin_depot, output_path,
                         "--since", ex_since,
                         "--bugfix-threshold", str(ex_threshold)],
                        capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_EXTRACT,
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
            # Présélection : le dataset qui a SERVI À ENTRAÎNER le modèle chargé
            # (seuil 5, 59,5 % de positifs). L'ancien `dataset_sucre.csv` est au
            # seuil 2 (82 % de positifs) — le proposer par défaut faisait comparer
            # les prédictions à un label que le modèle n'a jamais appris.
            default_idx = next(
                (i for i, n in enumerate(datasets) if n == "dataset_sucre_t5.csv"),
                next((i for i, n in enumerate(datasets) if n == "dataset_sucre.csv"), 0),
            )
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
    nb_high = int((df["risk_score"] >= BAND_HIGH).sum())
    nb_med  = int(((df["risk_score"] >= BAND_MED) & (df["risk_score"] < BAND_HIGH)).sum())
    nb_low  = int((df["risk_score"] < BAND_MED).sum())
    nb_prio = int(df["risk_pred"].sum())

    st.markdown('<div class="section-header">Vue d\'ensemble</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Méthodes analysées", len(df),
              help="Une ligne par méthode. Le score de risque, lui, est au niveau du contrôleur (voir note ci-dessous).")
    c2.metric(f"Risque élevé (>={BAND_HIGH:.0%})", nb_high)
    c3.metric(f"Risque moyen ({BAND_MED:.0%}-{BAND_HIGH:.0%})", nb_med)
    c4.metric(f"À tester (>={RISK_PRED_THRESHOLD:.0%})", nb_prio,
              help="Seuil de décision du modèle — c'est cette liste qui alimente "
                   "la génération de tests, et celle sous laquelle les métriques "
                   "publiées (MCC, balanced accuracy) sont calculées.")
    c5.metric("Classes uniques", df["class"].nunique() if "class" in df.columns else "—")

    st.info(
        "**À lire avant d'interpréter les scores.**\n\n"
        "**Granularité réelle = le contrôleur, pas la méthode.** Le score repose surtout sur "
        "des caractéristiques de *classe* (nombre de dépendances injectées, de méthodes, de "
        "rôles), identiques pour toutes les méthodes d'un même contrôleur, et le label "
        "d'apprentissage est calculé par *fichier*. Deux méthodes du même contrôleur ont donc "
        "un score quasi identique : lis ce tableau comme un **classement de contrôleurs à "
        "tester en priorité**, pas comme une hiérarchie fine entre méthodes.\n\n"
        "**Sur-signalement volontaire** (`class_weight='balanced'`) : le modèle préfère "
        "sur-alerter que rater une régression, donc le total « risque élevé » dépasse la "
        "prévalence réelle. Fie-toi au **classement par score**, pas au total brut."
    )

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown('<div class="section-header">Distribution des scores</div>', unsafe_allow_html=True)
        if HAS_PLOTLY:
            fig = px.histogram(
                df, x="risk_score", nbins=30,
                color_discrete_sequence=["#00769F"],
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
            ax.hist(df["risk_score"], bins=30, color="#00769F", edgecolor="#1b3e6f")
            ax.axvline(RISK_PRED_THRESHOLD, color="red", linestyle="--")
            st.pyplot(fig)

    with col_g2:
        st.markdown('<div class="section-header">Top 15 classes à risque</div>', unsafe_allow_html=True)
        if "class" in df.columns and HAS_PLOTLY:
            top_classes = (
                df.groupby("class")["risk_score"].mean()
                .nlargest(TOP_CLASSES_CHART).reset_index()
                .sort_values("risk_score")
            )
            top_classes["color"] = top_classes["risk_score"].apply(
                lambda v: "#dc2626" if v >= BAND_HIGH else ("#d97706" if v >= BAND_MED else "#16a34a")
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

    st.markdown('<div class="section-header">Contrôleurs prioritaires à tester (détail par méthode)</div>', unsafe_allow_html=True)
    top_n = st.slider("Nombre de lignes à afficher", TABLE_ROWS_MIN,
                      min(TABLE_ROWS_MAX, len(df)), TABLE_ROWS_DEFAULT)
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
            st.info(f"Aucune classe au-dessus du seuil de décision "
                    f"({RISK_PRED_THRESHOLD:.0%}).")
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
                                headers=api_headers(), timeout=API_TIMEOUT_GENERATION,
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
                                headers=api_headers(), timeout=800,
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
            f_class   = col_f2.text_input(
                "Classe cible (recommandé)",
                placeholder="CreanceController",
                help="Nom exact de la classe de contrôleur à tester. C'est CE champ qui "
                     "cible le bon contrôleur (recherche directe en base) — renseigne-le "
                     "pour un résultat fiable.",
            )
            f_desc = st.text_area(
                "Description du scénario",
                placeholder="Ex : Tester la consultation et l'édition des créances, "
                            "l'export CSV, et les contrôles d'accès par rôle.",
                height=90,
                help="Décris EN FRANÇAIS, avec le vocabulaire métier, ce que fait le "
                     "contrôleur. Ce texte sert à retrouver le bon code (recherche "
                     "sémantique) — ce n'est PAS un squelette de test. Les routes, rôles "
                     "et types de réponse sont déjà connus par le moteur.",
            )
            with st.expander("Comment écrire une bonne description ?"):
                st.markdown(
                    "La description **aide le moteur à retrouver le bon code** — elle ne "
                    "dicte pas la structure du test (le squelette est généré automatiquement).\n\n"
                    "**À faire**\n"
                    "- Une phrase courte en **français métier** : "
                    "*« consultation et édition des créances, export CSV, accès par rôle »*.\n"
                    "- **Renseigner la Classe cible** (ex. `CreanceController`) : c'est elle "
                    "qui garantit le bon contrôleur.\n\n"
                    "**À éviter**\n"
                    "- Coller du **code PHP / un squelette** de test → ça dégrade la recherche.\n"
                    "- Rester **vague** (*« teste le contrôleur »*) → contexte mal ciblé."
                )
            f_test_name = st.text_input(
                "Nom du test (optionnel)",
                placeholder="CreanceControllerTest",
                help="Nom de la classe de test générée. Par défaut : <Classe>Test.",
            )
            f_det   = st.checkbox(
                "Mode déterministe (sans LLM)", value=True,
                help="Coché : génération directe depuis le code indexé (rapide, zéro "
                     "hallucination, couverture riche). Décoché : génération par le LLM, "
                     "route par route.",
            )
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

                    _call_generate("generate-test", payload, f_async, _SK_PENDING_JOB)

        # Polling du job en cours
        _poll_job(_SK_PENDING_JOB, "btn_poll", "btn_cancel")

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

                    _call_generate("generate-unit-test", payload, u_async, _SK_PENDING_UNIT_JOB)

        # ── Polling du job unitaire en cours ──────────────────────────────────
        _poll_job(_SK_PENDING_UNIT_JOB, "btn_upoll", "btn_ucancel")


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
                ("cv_auc_mean",    "#00769F", "AUC (CV)"),
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
                marker_color="#00769F",
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
            ax.barh(top15["feature"], top15["importance"], color="#00769F")
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
