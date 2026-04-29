"""
Entraîne et compare 3 algorithmes supervisés pour prédire le risque de
régression d'une méthode de controller à partir de features extraites.

Modèles comparés :
- LogisticRegression (baseline interprétable)
- RandomForestClassifier (robuste sans tuning)
- XGBoost (souvent state-of-the-art sur tabulaire)

Sortie :
- ml/models/best_model.pkl     : le meilleur modèle sérialisé
- ml/models/feature_names.json : ordre des features attendu
- ml/models/metrics.json       : métriques de tous les modèles
- ml/models/confusion_*.png    : matrices de confusion
"""
import argparse
import json
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # backend non interactif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ── Colonnes du dataset ──────────────────────────────────────────────────────
ID_COLS = ["file", "class", "method", "profile"]
LABEL_COL = "label_risk"

NUMERIC_FEATURES = [
    "nb_routes_method", "has_route_attr",
    "nb_params", "has_form", "is_ajax_only",
    "nb_voter_checks", "nb_response_types",
    "has_render", "has_redirect", "has_json", "has_file_download",
    "nb_method_grants", "nb_class_grants", "nb_constructor_deps",
    "nb_methods_in_class", "is_invoke",
    "file_loc", "method_loc", "cyclomatic_complexity",
    "git_total_commits", "git_nb_authors", "git_days_since_change",
]
# git_bugfix_count est exclu : c'est ce qui sert à calculer le label, le mettre
# en feature serait du data leakage.


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"📂 Dataset : {csv_path}")
    print(f"   Lignes  : {len(df)}")
    print(f"   label=1 : {df[LABEL_COL].sum()} ({100 * df[LABEL_COL].mean():.1f}%)")
    print(f"   label=0 : {(df[LABEL_COL] == 0).sum()}")
    return df


def evaluate(name, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics["auc"] = float(roc_auc_score(y_test, y_proba))
        except Exception:
            metrics["auc"] = None
    print(f"\n── {name} ──")
    for k, v in metrics.items():
        if k != "model":
            print(f"   {k:>10} : {v:.4f}" if isinstance(v, float) else f"   {k:>10} : {v}")
    print(classification_report(y_test, y_pred, zero_division=0, digits=3))
    return metrics


def confusion_plot(name, model, X_test, y_test, out_path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(f"Matrice de confusion — {name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", help="Fichier CSV produit par extract_dataset.py")
    ap.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "models"))
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--no-grid", action="store_true",
                    help="Skip GridSearchCV (plus rapide pour itérer)")
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    df = load_dataset(args.dataset)

    feats = [c for c in NUMERIC_FEATURES if c in df.columns]
    missing = set(NUMERIC_FEATURES) - set(feats)
    if missing:
        print(f"⚠️  Colonnes manquantes (ignorées) : {missing}")
    print(f"\n🎯 Features utilisées : {len(feats)}")

    X = df[feats].fillna(-1.0).values
    y = df[LABEL_COL].astype(int).values

    if y.sum() == 0 or y.sum() == len(y):
        raise SystemExit("❌ Le label est constant (que des 0 ou que des 1). "
                         "Ajuste --bugfix-threshold ou la fenêtre temporelle.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state,
    )
    print(f"\n📊 Train : {len(X_train)} | Test : {len(X_test)}")

    all_metrics = []

    # ── 1. Logistic Regression ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("1. Logistic Regression (baseline)")
    print("=" * 60)
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=args.random_state)),
    ])
    if args.no_grid:
        lr_pipe.fit(X_train, y_train)
        best_lr = lr_pipe
    else:
        lr_grid = GridSearchCV(
            lr_pipe,
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
            cv=5, scoring="f1", n_jobs=-1,
        )
        lr_grid.fit(X_train, y_train)
        best_lr = lr_grid.best_estimator_
        print(f"   Best params : {lr_grid.best_params_}")
    all_metrics.append(evaluate("LogisticRegression", best_lr, X_test, y_test))
    confusion_plot("LogReg", best_lr, X_test, y_test,
                   os.path.join(args.output_dir, "confusion_logreg.png"))

    # ── 2. Random Forest ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. Random Forest")
    print("=" * 60)
    rf = RandomForestClassifier(class_weight="balanced", random_state=args.random_state)
    if args.no_grid:
        rf.fit(X_train, y_train)
        best_rf = rf
    else:
        rf_grid = GridSearchCV(
            rf,
            {"n_estimators": [100, 200], "max_depth": [None, 8, 16],
             "min_samples_leaf": [1, 3]},
            cv=5, scoring="f1", n_jobs=-1,
        )
        rf_grid.fit(X_train, y_train)
        best_rf = rf_grid.best_estimator_
        print(f"   Best params : {rf_grid.best_params_}")
    all_metrics.append(evaluate("RandomForest", best_rf, X_test, y_test))
    confusion_plot("RandomForest", best_rf, X_test, y_test,
                   os.path.join(args.output_dir, "confusion_rf.png"))

    # Importance des features
    importances = pd.DataFrame({
        "feature": feats,
        "importance": best_rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    importances.to_csv(os.path.join(args.output_dir, "feature_importance.csv"),
                       index=False)
    print("\n📊 Top 10 features (RandomForest) :")
    print(importances.head(10).to_string(index=False))

    # ── 3. XGBoost (si dispo) ────────────────────────────────────────────────
    if HAS_XGB:
        print("\n" + "=" * 60)
        print("3. XGBoost")
        print("=" * 60)
        # scale_pos_weight pour gérer le déséquilibre
        spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=spw, eval_metric="logloss",
            use_label_encoder=False, random_state=args.random_state,
        )
        xgb.fit(X_train, y_train)
        all_metrics.append(evaluate("XGBoost", xgb, X_test, y_test))
        confusion_plot("XGBoost", xgb, X_test, y_test,
                       os.path.join(args.output_dir, "confusion_xgb.png"))
    else:
        print("\n⚠️  XGBoost non installé — `pip install xgboost` pour l'activer")

    # ── Sélection du best ───────────────────────────────────────────────────
    best = max(all_metrics, key=lambda m: m["f1"])
    print("\n" + "=" * 60)
    print(f"🏆 Meilleur modèle : {best['model']}  (F1={best['f1']:.4f})")
    print("=" * 60)

    # Sauvegarde
    name_to_model = {
        "LogisticRegression": best_lr,
        "RandomForest": best_rf,
    }
    if HAS_XGB:
        name_to_model["XGBoost"] = xgb
    joblib.dump(name_to_model[best["model"]],
                os.path.join(args.output_dir, "best_model.pkl"))
    with open(os.path.join(args.output_dir, "feature_names.json"), "w") as f:
        json.dump(feats, f, indent=2)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({"all": all_metrics, "best": best}, f, indent=2)

    print(f"\n💾 Sauvegardés dans {args.output_dir}/")
    print("   - best_model.pkl")
    print("   - feature_names.json")
    print("   - metrics.json")
    print("   - feature_importance.csv")
    print("   - confusion_*.png")


if __name__ == "__main__":
    main()
