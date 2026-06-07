"""
Évaluation croisée inter-projet (cross-project generalization).

Principe : on entraîne sur un ou plusieurs projets (train) et on évalue
sur un projet totalement distinct (test). Cela borne la généralisation
réelle du modèle, indépendamment du CV intra-projet.

Datasets disponibles :
  - sucre   : 369 lignes, 82% positifs  (projet interne CNAM)
  - sylius  : 183 lignes, 26% positifs  (OSS e-commerce PHP)
  - akeneo  : 726 lignes, ~10% positifs (OSS PIM PHP)

Runs disponibles (--run) :
  sylius→akeneo   : entraîne sur Sylius, prédit sur Akeneo
  sucre→akeneo    : entraîne sur Sucre,  prédit sur Akeneo
  all→akeneo      : entraîne sur Sucre+Sylius, prédit sur Akeneo
  sylius→sucre    : entraîne sur Sylius, prédit sur Sucre
  all              : exécute les 4 runs ci-dessus et compare

Usage :
  python ml/cross_eval.py
  python ml/cross_eval.py --run sylius→akeneo
  python ml/cross_eval.py --run all --feature-set v3
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import RandomForestClassifier

# ─── Feature sets (identiques à train.py) ────────────────────────────────────

SAFE_FEATURES = [
    "nb_routes_method", "has_route_attr",
    "nb_params", "has_form", "is_ajax_only",
    "nb_voter_checks", "nb_response_types",
    "has_render", "has_redirect", "has_json", "has_file_download",
    "nb_method_grants", "nb_class_grants",
    "nb_constructor_deps", "nb_methods_in_class", "is_invoke",
    "file_loc", "method_loc", "cyclomatic_complexity",
    "git_nb_authors", "git_days_since_change",
]

EDA_DROP = [
    "is_invoke", "nb_voter_checks", "has_file_download",
    "has_route_attr", "file_loc", "cyclomatic_complexity",
]
SAFE_FEATURES_V2 = [f for f in SAFE_FEATURES if f not in EDA_DROP]

GIT_WINDOW = ["git_nb_authors", "git_days_since_change"]
SAFE_FEATURES_V3 = [f for f in SAFE_FEATURES_V2 if f not in GIT_WINDOW]

FEATURE_SETS = {"v1": SAFE_FEATURES, "v2": SAFE_FEATURES_V2, "v3": SAFE_FEATURES_V3}

LABEL_COL = "label_risk"

# ─── Chemins ─────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent
_DS   = _HERE / "_dataset"

DATASETS = {
    "sucre":  _DS / "dataset_sucre.csv",
    "sylius": _DS / "dataset_sylius.csv",
    "akeneo": _DS / "dataset_akeneo.csv",
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load(name: str, features: list) -> tuple:
    path = DATASETS[name]
    df = pd.read_csv(path)
    # git_days_since_change = -1 (pas de git) → remplacer par NaN → imputation plus tard
    if "git_days_since_change" in df.columns:
        df["git_days_since_change"] = df["git_days_since_change"].replace(-1.0, np.nan)
    available = [f for f in features if f in df.columns]
    X = df[available].copy()
    y = df[LABEL_COL]
    print(f"  [{name}] {len(df)} lignes | label=1 : {y.sum()} ({100*y.mean():.1f}%)"
          f" | features : {len(available)}")
    return X, y, available


def impute(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """Remplace les NaN par la médiane du train dans train ET test."""
    medians = X_train.median()
    return X_train.fillna(medians), X_test.fillna(medians)


def build_model(y_train: pd.Series):
    pos  = int(y_train.sum())
    neg  = int((y_train == 0).sum())
    spw  = neg / pos if pos > 0 else 1.0
    if HAS_XGB:
        return XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    # fallback si XGBoost pas dispo
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=400, max_depth=8, class_weight="balanced", random_state=42
    )


def evaluate(y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    metrics = {
        "mcc":       round(float(matthews_corrcoef(y_true, pred)),       4),
        "bal_acc":   round(float(balanced_accuracy_score(y_true, pred)), 4),
        "roc_auc":   None,
        "pr_auc":    None,
        "threshold": threshold,
        "n_pred_1":  int(pred.sum()),
        "n_true_1":  int(y_true.sum()),
        "n_total":   len(y_true),
    }
    try:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, proba)), 4)
        metrics["pr_auc"]  = round(float(average_precision_score(y_true, proba)), 4)
    except Exception:
        pass
    return metrics


def _check_labels(name: str, y: pd.Series) -> bool:
    """Retourne False si y ne contient qu'une seule classe (évaluation impossible)."""
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        print(f"  ⚠  [{name}] aucun label positif (label_risk=0 partout) — dataset non annoté.")
        print(f"     Ce dataset ne peut pas servir de test-set ; ignoré.")
        return False
    if n_neg == 0:
        print(f"  ⚠  [{name}] aucun label négatif — dataset déséquilibré à 100%.")
        return False
    return True


def run_cross(train_names: list, test_name: str, features: list) -> dict:
    """
    Entraîne sur la concaténation de train_names et évalue sur test_name.
    Retourne un dict de métriques.
    """
    label = "+".join(train_names) + " → " + test_name
    print(f"\n{'─'*60}")
    print(f"  Run : {label}  (features={len(features)})")
    print(f"{'─'*60}")

    # Chargement train
    X_parts, y_parts = [], []
    common_features = None
    for name in train_names:
        X, y, avail = load(name, features)
        X_parts.append(X)
        y_parts.append(y)
        common_features = avail if common_features is None else [f for f in common_features if f in avail]

    X_train = pd.concat([x[common_features] for x in X_parts], ignore_index=True)
    y_train = pd.concat(y_parts, ignore_index=True)

    # Chargement test
    X_test, y_test, _ = load(test_name, common_features)
    X_test = X_test[common_features]

    # Vérification que le test-set est annoté
    if not _check_labels(test_name, y_test):
        return {
            "run":    label,
            "status": "skipped — test-set non annoté (label=0 partout)",
            "mcc":    None, "roc_auc": None, "pr_auc": None, "bal_acc": None,
        }

    # Imputation NaN (git_days_since_change=-1 → NaN → médiane train)
    X_train, X_test = impute(X_train, X_test)

    # Entraînement
    model = build_model(y_train)
    model.fit(X_train, y_train)

    # Évaluation
    proba = model.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, proba)
    metrics["train_size"]    = len(y_train)
    metrics["test_size"]     = len(y_test)
    metrics["train_pos_pct"] = round(100 * float(y_train.mean()), 1)
    metrics["test_pos_pct"]  = round(100 * float(y_test.mean()),  1)
    metrics["features_used"] = len(common_features)
    metrics["run"]           = label

    print(f"\n  MCC      : {metrics['mcc']}")
    print(f"  ROC-AUC  : {metrics['roc_auc']}")
    print(f"  PR-AUC   : {metrics['pr_auc']}")
    print(f"  Bal. acc : {metrics['bal_acc']}")
    print(f"  Préd. 1  : {metrics['n_pred_1']} / {metrics['n_total']}"
          f"  (vrai 1 = {metrics['n_true_1']})")

    return metrics


RUNS = {
    "sylius→akeneo": (["sylius"], "akeneo"),
    "sucre→akeneo":  (["sucre"],  "akeneo"),
    "all→akeneo":    (["sucre", "sylius"], "akeneo"),
    "sylius→sucre":  (["sylius"], "sucre"),
}

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Cross-project evaluation")
    ap.add_argument(
        "--run",
        choices=list(RUNS.keys()) + ["all"],
        default="all",
        help="Run à exécuter (défaut : all)",
    )
    ap.add_argument(
        "--feature-set",
        choices=["v1", "v2", "v3"],
        default="v2",
        help="Jeu de features (défaut : v2, nettoyé post-EDA)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Fichier JSON de sortie (optionnel)",
    )
    args = ap.parse_args()

    features = FEATURE_SETS[args.feature_set]
    runs_to_exec = list(RUNS.keys()) if args.run == "all" else [args.run]

    print(f"\n{'═'*60}")
    print(f"  Cross-project evaluation  |  feature-set={args.feature_set}  ({len(features)} features)")
    print(f"{'═'*60}")

    results = []
    for run_name in runs_to_exec:
        train_names, test_name = RUNS[run_name]
        m = run_cross(train_names, test_name, features)
        results.append(m)

    # Tableau récapitulatif
    print(f"\n\n{'═'*60}")
    print(f"  RÉCAP  (feature-set={args.feature_set})")
    print(f"{'═'*60}")
    print(f"  {'Run':<30}  {'MCC':>6}  {'ROC-AUC':>8}  {'PR-AUC':>7}  {'Bal.acc':>8}")
    print(f"  {'─'*30}  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*8}")
    for m in results:
        if m.get("status", "").startswith("skipped"):
            print(f"  {m['run']:<30}  (ignoré — dataset test non annoté)")
            continue
        mcc  = str(m['mcc'])     if m['mcc']     is not None else "n/a"
        rauc = str(m['roc_auc']) if m['roc_auc'] is not None else "n/a"
        pauc = str(m['pr_auc'])  if m['pr_auc']  is not None else "n/a"
        bacc = str(m['bal_acc']) if m['bal_acc']  is not None else "n/a"
        print(f"  {m['run']:<30}  {mcc:>6}  {rauc:>8}  {pauc:>7}  {bacc:>8}")

    # Interprétation
    print(f"\n  Interprétation MCC :")
    print(f"    > 0.40 → bonne généralisation inter-projet")
    print(f"    0.20–0.40 → généralisation modérée (acceptable)")
    print(f"    < 0.20 → surapprentissage probable au projet source")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n  Résultats écrits dans : {args.out}")

    return results


if __name__ == "__main__":
    main()
