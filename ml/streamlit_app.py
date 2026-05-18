"""
Dashboard MOA — Prédicteur de risque de régression + génération de tests prioritaires.

Architecture :
1. Upload d'un dataset CSV ou choix d'un projet déjà extrait
2. Le modèle XGBoost (best_model.pkl) prédit le risque par méthode
3. Affichage trié par risque décroissant
4. Bouton "Générer les tests" → POST vers le AI Test Engine pour les classes à risque

Usage :
    streamlit run ml/streamlit_app.py
"""
import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

# Configuration
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"
FEATURES_PATH = ROOT / "models" / "feature_names.json"
METRICS_PATH = ROOT / "models" / "metrics.json"
IMPORTANCE_PATH = ROOT / "models" / "feature_importance.csv"
DATASET_DIR = ROOT.parent / "_dataset"
TEST_ENGINE_URL = os.environ.get("TEST_ENGINE_URL", "http://localhost:8000")
TEST_ENGINE_API_KEY = os.environ.get("TEST_ENGINE_API_KEY", "")


# Cache resources
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
    """Ajoute les colonnes risk_score et risk_pred au DataFrame."""
    X = df[features].fillna(-1.0).values
    if hasattr(model, "predict_proba"):
        df["risk_score"] = model.predict_proba(X)[:, 1]
    else:
        df["risk_score"] = model.predict(X)
    df["risk_pred"] = (df["risk_score"] >= 0.5).astype(int)
    return df


# Setup page
st.set_page_config(
    page_title="AI Test Engine — Risk Predictor",
    layout="wide",
)

st.title(" AI Test Engine — Risk Predictor")
st.caption(
    "Prédicteur de risque de régression par méthode de controller Symfony. "
    "Permet à la MOA de prioriser les tests fonctionnels lors d'un changement de version."
)

model, FEATURES = load_model()
if model is None:
    st.error(
        "Aucun modèle entraîné trouvé. Lance d'abord :\n\n"
        "```bash\npython ml/extract_dataset.py <repo> _dataset/dataset.csv\n"
        "python ml/train.py _dataset/dataset.csv\n```"
    )
    st.stop()

# Sidebar : navigation
page = st.sidebar.radio(
    "Pages",
    ["Prédiction de risque", "Performance du modèle", "À propos"],
)

# ============================================================================
# PAGE 1 : PRÉDICTION DE RISQUE
# ============================================================================
if page == "Prédiction de risque":
    st.header("Choix du dataset")

    col1, col2 = st.columns([1, 2])
    with col1:
        choice = st.radio("Source", ["Dataset existant", "Upload CSV"])
    with col2:
        if choice == "Dataset existant":
            datasets = list_datasets()
            if not datasets:
                st.warning("Aucun dataset trouvé dans _dataset/")
                st.stop()
            selected = st.selectbox("Fichier", datasets)
            df = pd.read_csv(DATASET_DIR / selected)
        else:
            uploaded = st.file_uploader("CSV produit par extract_dataset.py", type="csv")
            if uploaded is None:
                st.info("Charge un CSV pour continuer")
                st.stop()
            df = pd.read_csv(uploaded)

    # Vérification colonnes
    missing = set(FEATURES) - set(df.columns)
    if missing:
        st.error(f"Colonnes manquantes dans le CSV : {missing}")
        st.stop()

    # Prédiction
    df = predict(df.copy(), model, FEATURES)
    st.success(f"Prédictions calculées sur {len(df)} méthodes")

    # Statistiques globales
    st.header("Vue d'ensemble")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Méthodes analysées", len(df))
    c2.metric("Risque haut (≥0.5)", int(df["risk_pred"].sum()),
              f"{100 * df['risk_pred'].mean():.1f}%")
    c3.metric("Score moyen", f"{df['risk_score'].mean():.2%}")
    c4.metric("Classes uniques", df["class"].nunique())

    # Histogramme des scores
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(df["risk_score"], bins=30, color="steelblue", edgecolor="black")
    ax.axvline(0.5, color="red", linestyle="--", label="Seuil")
    ax.set_xlabel("Score de risque")
    ax.set_ylabel("Nombre de méthodes")
    ax.set_title("Distribution des scores prédits")
    ax.legend()
    st.pyplot(fig)

    # Top routes à tester
    st.header("Routes prioritaires à tester")
    st.caption(
        "Triées par score de risque décroissant. La MOA doit tester ces méthodes "
        "en priorité lors du passage à la version suivante."
    )

    top_n = st.slider("Combien afficher", 5, min(100, len(df)), 20)
    top = df.nlargest(top_n, "risk_score")[
        ["class", "method", "risk_score", "git_bugfix_count",
         "git_total_commits", "cyclomatic_complexity", "file"]
    ].copy()
    top["risk_score"] = top["risk_score"].apply(lambda v: f"{v:.2%}")
    st.dataframe(top, use_container_width=True, hide_index=True)

    # ── Actions vers le AI Test Engine ─────────────────────────────────────
    st.header("Génération de tests prioritaires")
    st.caption(
        "Pour les classes les plus à risque, on déclenche le générateur de tests "
        "fonctionnels (mode déterministe — pas d'appel LLM)."
    )

    risky_classes = (
        df[df["risk_score"] >= 0.5]
        .sort_values("risk_score", ascending=False)["class"]
        .drop_duplicates()
        .tolist()
    )

    if not risky_classes:
        st.info("Aucune classe avec un risque ≥ 0.5 — pas d'action requise.")
    else:
        st.write(f"**{len(risky_classes)} classes** à risque détectées.")

        col_a, col_b = st.columns([1, 2])
        with col_a:
            project_id = st.text_input("project_id (AI Test Engine)", value="sucre")
            n_classes = st.slider("Nb classes à traiter", 1, len(risky_classes),
                                  min(5, len(risky_classes)))
        with col_b:
            st.write("**Cibles :**")
            for c in risky_classes[:n_classes]:
                st.write(f"- `{c}`")

        if st.button("Générer les tests"):
            engine_ok = False
            try:
                r = requests.get(f"{TEST_ENGINE_URL}/health", timeout=3)
                engine_ok = r.status_code == 200
            except Exception:
                pass

            if not engine_ok:
                st.error(
                    f"AI Test Engine injoignable sur {TEST_ENGINE_URL}. "
                    "Vérifie qu'il tourne (`docker compose up -d`)."
                )
            else:
                headers = {"Content-Type": "application/json"}
                if TEST_ENGINE_API_KEY:
                    headers["X-API-Key"] = TEST_ENGINE_API_KEY

                progress = st.progress(0.0, text="Préparation…")
                results = []
                for i, cls in enumerate(risky_classes[:n_classes]):
                    progress.progress((i + 1) / n_classes,
                                      text=f"Génération : {cls}")
                    try:
                        resp = requests.post(
                            f"{TEST_ENGINE_URL}/generate-test",
                            json={
                                "project_id": project_id,
                                "description": f"Tests prioritaires pour {cls} (risque élevé)",
                                "class_name": cls,
                                "test_name": f"{cls}Test",
                                "deterministic": True,
                            },
                            headers=headers,
                            timeout=60,
                        )
                        body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                        results.append({
                            "class": cls,
                            "status": resp.status_code,
                            "file": body.get("file", "—"),
                            "path": body.get("path", "—"),
                        })
                    except Exception as e:
                        results.append({
                            "class": cls, "status": "ERR",
                            "file": "—", "path": str(e)[:80],
                        })
                progress.empty()
                st.success(f"{len(results)} générations lancées")
                st.dataframe(pd.DataFrame(results), use_container_width=True,
                             hide_index=True)

    # ── Distribution par classe ────────────────────────────────────────────
    with st.expander("Risque moyen par classe"):
        per_class = (
            df.groupby("class")
            .agg(nb_methods=("method", "count"),
                 risk_mean=("risk_score", "mean"),
                 risk_max=("risk_score", "max"),
                 file=("file", "first"))
            .sort_values("risk_mean", ascending=False)
            .head(30)
        )
        per_class["risk_mean"] = per_class["risk_mean"].apply(lambda v: f"{v:.2%}")
        per_class["risk_max"] = per_class["risk_max"].apply(lambda v: f"{v:.2%}")
        st.dataframe(per_class, use_container_width=True)


# ============================================================================
# PAGE 2 : PERFORMANCE DU MODÈLE
# ============================================================================
elif page == "Performance du modèle":
    st.header("Métriques sur le jeu de test")
    metrics = load_metrics()
    if metrics is None:
        st.warning("metrics.json absent")
        st.stop()

    df_metrics = pd.DataFrame(metrics["all"]).set_index("model")
    # Colonnes disponibles selon la version du train.py
    display_cols = [c for c in ["train_f1", "test_f1", "cv_test_f1_mean",
                                "accuracy", "auc", "overfit_gap"]
                    if c in df_metrics.columns]
    if not display_cols:
        display_cols = [c for c in ["accuracy", "f1", "auc"] if c in df_metrics.columns]
    df_metrics = df_metrics[display_cols].map(
        lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else v
    )
    st.dataframe(df_metrics, use_container_width=True)

    best = metrics["best"]
    cv_f1 = best.get("cv_test_f1_mean", best.get("f1", 0))
    st.success(f"Meilleur modèle : **{best['model']}** (CV F1 = {cv_f1:.4f})")
    if best.get("overfit_gap", 0) > 0.1:
        st.warning(f"Gap overfitting : {best['overfit_gap']:.4f} — "
                   "le modèle pourrait ne pas généraliser.")

    st.header("Importance des features")
    importance = load_importance()
    if importance is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        top15 = importance.head(15).iloc[::-1]
        ax.barh(top15["feature"], top15["importance"], color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title("Top 15 features (Random Forest)")
        fig.tight_layout()
        st.pyplot(fig)
        with st.expander("Tableau complet"):
            st.dataframe(importance, use_container_width=True, hide_index=True)

    st.header("Matrices de confusion")
    cols = st.columns(3)
    for i, name in enumerate(["logreg", "rf", "xgb"]):
        png = ROOT / "models" / f"confusion_{name}.png"
        if png.exists():
            cols[i].image(str(png), caption=name.upper(), use_container_width=True)


# ============================================================================
# PAGE 3 : À PROPOS
# ============================================================================
else:
    st.header("À propos du projet")
    st.markdown("""
### Contexte

Ce dashboard fait partie du **AI Test Engine**, un système hybride qui combine :

1. **Un classifieur supervisé** (XGBoost) qui prédit le risque de régression
   d'une méthode de controller à partir de :
   - features structurelles (complexité, nb routes, formulaires, voters…)
   - features git (historique de commits et bugfix)
2. **Un générateur de tests RAG + LLM** qui produit des `WebTestCase` Symfony
   pour les routes à risque.

### Méthode

Le label de risque est dérivé du proxy bugfix-from-git, méthode standard en
software defect prediction (Hassan 2009, Kamei et al. 2013). Une méthode est
dite *à risque* si son fichier a été modifié par au moins N commits de bugfix
dans la fenêtre temporelle (12 à 36 mois selon la maturité du projet).

### Pipeline

```
git history + code source
        ↓
  extract_dataset.py
        ↓
  CSV (features + label)
        ↓
   train.py (LR / RF / XGB + GridSearch)
        ↓
   best_model.pkl
        ↓
  Streamlit (cette interface)
        ↓
  AI Test Engine (génération PHP)
```

### Datasets utilisés

- **Sylius** : dataset principal (e-commerce open-source mature, ~30k commits)
- **Sucre** : projet métier interne, utilisé pour la validation et la démo

### Contact / Repo

Voir `README.md` à la racine du projet.
    """)
