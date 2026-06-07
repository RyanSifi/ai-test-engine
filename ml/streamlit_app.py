"""
Dashboard AI Test Engine — Prédicteur de risque de régression + génération de tests.
Usage : streamlit run ml/streamlit_app.py
"""
import json
import os
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
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #2d3250; }

[data-testid="metric-container"] {
    background: #1e2130;
    border: 1px solid #2d3250;
    border-radius: 10px;
    padding: 16px;
}
[data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 700; }

.section-header {
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    color: white;
    padding: 10px 18px;
    border-radius: 8px;
    font-size: 1.1rem;
    font-weight: 600;
    margin: 18px 0 12px 0;
}

.dot-green { display:inline-block; width:10px; height:10px; border-radius:50%; background:#16a34a; margin-right:6px; }
.dot-red   { display:inline-block; width:10px; height:10px; border-radius:50%; background:#dc2626; margin-right:6px; }

.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: .85; }
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


# Sidebar
with st.sidebar:
    st.markdown("## AI Test Engine")
    st.markdown("---")

    api_ok = check_api()
    dot = '<span class="dot-green"></span>' if api_ok else '<span class="dot-red"></span>'
    status_txt = "API connectée" if api_ok else "API hors ligne"
    st.markdown(f"{dot}<small>{status_txt} — {TEST_ENGINE_URL}</small>", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Accueil", "Prédiction ML", "Génération de tests", "Performance"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(f"Modèle : `best_model.pkl`")
    st.caption(f"Dataset : `_dataset/`")

model, FEATURES = load_model()


# PAGE : ACCUEIL
if page == "Accueil":
    st.markdown("# AI Test Engine")
    st.markdown("**Prédicteur de risque de régression + génération de tests Symfony**")
    st.markdown("---")

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
        st.markdown("""
```
Git history + code source Symfony
           ↓
    extract_dataset.py
    (features PHP + git, nettoyées post-EDA)
           ↓
      dataset.csv
           ↓
   train.py  ←  GridSearchCV
   LogReg / RF / XGBoost
           ↓
    best_model.pkl
    (sélection par CV groupée par fichier)
           ↓
   Ce dashboard (prédiction + export)
```
""")

    with col_right:
        st.markdown('<div class="section-header">Pipeline RAG + LLM</div>', unsafe_allow_html=True)
        st.markdown("""
```
Dépôt Symfony indexé
         ↓
  Chunking PHP (AST)
         ↓
  Embeddings MiniLM-L12
  (paraphrase-multilingual)
         ↓
  pgvector HNSW (cosine)
  3x plus rapide que seqscan
         ↓
  FastAPI /generate-test
  LLM -> WebTestCase PHP
  + validation php -l + retry
```
""")

    st.markdown('<div class="section-header">Stack technique</div>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    col_a.markdown("""
**ML**
- Python 3.11
- XGBoost / scikit-learn
- pandas / numpy
- Streamlit + Plotly
""")
    col_b.markdown("""
**API**
- FastAPI (uvicorn)
- PostgreSQL 15 + pgvector
- sentence-transformers
- Docker Compose
""")
    col_c.markdown("""
**Deploiement**
- 3 services Docker
- `vector-db` :5432
- `brain-api` :8000
- `streamlit-ui` :8501
""")


# PAGE : PRÉDICTION ML
elif page == "Prédiction ML":
    st.markdown("# Prédiction ML")
    st.caption("Chargez un dataset CSV produit par `extract_dataset.py` pour obtenir les scores de risque.")

    if model is None:
        st.error("Aucun modèle entraîné. Lance d'abord `python ml/train.py _dataset/dataset.csv`")
        st.stop()

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
            selected = st.selectbox("Fichier", datasets)
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

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown('<div class="section-header">Distribution des scores</div>', unsafe_allow_html=True)
        if HAS_PLOTLY:
            fig = px.histogram(
                df, x="risk_score", nbins=30,
                color_discrete_sequence=["#4f46e5"],
                labels={"risk_score": "Score de risque"},
            )
            fig.add_vline(x=0.5, line_dash="dash", line_color="#dc2626",
                          annotation_text="Seuil 0.5", annotation_position="top right")
            fig.update_layout(
                plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
                font_color="white", height=320,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.hist(df["risk_score"], bins=30, color="#4f46e5", edgecolor="black")
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
                plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
                font_color="white", height=320,
                xaxis_title="Score moyen",
                margin=dict(l=10, r=60, t=10, b=10),
            )
            st.plotly_chart(fig2, width="stretch")
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
        high_risk = df[df["risk_pred"] == 1]
        if len(high_risk):
            st.download_button("Telecharger risques elevés (CSV)",
                               high_risk.to_csv(index=False).encode("utf-8"),
                               "high_risk.csv", "text/csv")

    with st.expander("Generation rapide depuis ici"):
        risky = (
            df[df["risk_score_raw"] >= 0.5]["class"].drop_duplicates().tolist()
            if "risk_score_raw" in df.columns
            else df[df["risk_pred"] == 1]["class"].drop_duplicates().tolist()
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
    st.markdown("# Génération de tests")
    st.caption("Interface complète vers l'API FastAPI du AI Test Engine.")

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
            f_det = st.checkbox("Mode déterministe (sans LLM)", value=False)
            if st.form_submit_button("Générer le test fonctionnel"):
                if not f_project or not f_desc:
                    st.error("project_id et description sont obligatoires.")
                elif not api_ok:
                    st.error("API hors ligne.")
                else:
                    with st.spinner("Génération en cours…"):
                        try:
                            payload = {"project_id": f_project, "description": f_desc,
                                       "deterministic": f_det}
                            if f_class:
                                payload["class_name"] = f_class
                            if f_test_name:
                                payload["test_name"] = f_test_name
                            r = requests.post(f"{TEST_ENGINE_URL}/generate-test",
                                              json=payload, headers=api_headers(), timeout=900)
                            if r.status_code == 200:
                                body = r.json()
                                st.success(f"Test généré : `{body.get('file','')}`")
                                code = body.get("code") or body.get("content", "")
                                if code:
                                    st.code(code, language="php")
                                    st.download_button("Télécharger", code.encode("utf-8"),
                                                       file_name=body.get("file", "test.php"))
                                with st.expander("Réponse complète"):
                                    st.json(body)
                            else:
                                st.error(f"HTTP {r.status_code}")
                                st.code(r.text)
                        except Exception as e:
                            st.error(f"Erreur : {e}")

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
            u_det = st.checkbox("Mode déterministe (sans LLM)", value=False, key="u_det")
            if st.form_submit_button("Générer le test unitaire"):
                if not u_project or not u_desc:
                    st.error("project_id et description sont obligatoires.")
                elif not api_ok:
                    st.error("API hors ligne.")
                else:
                    with st.spinner("Génération en cours…"):
                        try:
                            payload = {"project_id": u_project, "description": u_desc,
                                       "deterministic": u_det}
                            if u_class:
                                payload["class_name"] = u_class
                            r = requests.post(f"{TEST_ENGINE_URL}/generate-unit-test",
                                              json=payload, headers=api_headers(), timeout=900)
                            if r.status_code == 200:
                                body = r.json()
                                st.success(f"Test généré : `{body.get('file','')}`")
                                code = body.get("code") or body.get("content", "")
                                if code:
                                    st.code(code, language="php")
                                    st.download_button("Télécharger", code.encode("utf-8"),
                                                       file_name=body.get("file", "unit_test.php"))
                                with st.expander("Réponse complète"):
                                    st.json(body)
                            else:
                                st.error(f"HTTP {r.status_code}")
                                st.code(r.text)
                        except Exception as e:
                            st.error(f"Erreur : {e}")


# PAGE : PERFORMANCE
elif page == "Performance":
    st.markdown("# Performance du modèle")

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
                ("cv_auc_mean",    "#06b6d4", "AUC (CV)"),
                ("cv_balacc_mean", "#7c3aed", "Balanced acc (CV)"),
                ("cv_mcc_mean",    "#4f46e5", "MCC (CV)"),
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
                plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
                font_color="white", height=350,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=10, r=10, t=40, b=10),
                yaxis=dict(range=[0, 1.15]),
            )
            st.plotly_chart(fig_bar, width="stretch")

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
                marker_color="#4f46e5",
                text=top15["importance"].apply(lambda v: f"{v:.4f}"),
                textposition="outside",
            ))
            fig_imp.update_layout(
                plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
                font_color="white", height=420,
                xaxis_title="Importance",
                margin=dict(l=10, r=70, t=10, b=10),
            )
            st.plotly_chart(fig_imp, width="stretch")
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(top15["feature"], top15["importance"], color="#4f46e5")
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
