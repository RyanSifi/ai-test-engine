"""
Entraîne et compare 3 algorithmes supervisés pour prédire le risque de
régression d'une méthode de controller à partir de features extraites.

Corrections v3 (post-EDA) :
 - SAFE_FEATURES_V2 : features nettoyées après analyse exploratoire
   (suppression des quasi-constantes, redondances, VIF explosés)
 - --feature-set v1|v2 pour choisir le jeu de features
 - --compare pour entraîner v1 ET v2 et afficher les deltas

Corrections v2 :
 - Détection d'overfitting (scores train vs test)
 - RepeatedStratifiedKFold pour des métriques fiables sur petit dataset
 - Deux expériences : avec/sans git_total_commits (leakage partiel)
 - RF contraint (max_depth limité) pour éviter la mémorisation
 - Compatible pandas >= 2.1

Modèles comparés :
- LogisticRegression (baseline interprétable)
- RandomForestClassifier (robuste sans tuning)
- XGBoost (souvent state-of-the-art sur tabulaire)

Sortie :
- models/best_model.pkl          : le meilleur modèle sérialisé
- models/feature_names.json      : ordre des features attendu
- models/metrics.json            : métriques de tous les modèles
- models/confusion_*.png         : matrices de confusion
- models/feature_importance.csv
"""
import argparse
import json
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ── Seuils et paramètres d'évaluation ────────────────────────────────────────
# Écart train/test au-delà duquel on signale du sur-apprentissage. En dessous de
# 0,05, l'écart relève du bruit d'échantillonnage sur 54 fichiers.
OVERFIT_GAP_WARNING = 0.05
# Au-delà, l'écart n'est plus explicable par le bruit : il est signalé "FORT".
OVERFIT_GAP_SEVERE  = 0.15
# Seuil à partir duquel on relativise le gap du holdout dans le résumé final
# (le holdout ne porte que quelques fichiers, son gap est peu fiable).
OVERFIT_GAP_HOLDOUT_NOTE = 0.10

# Nombre de plis de la validation croisée. 5 est le compromis habituel ; avec
# seulement 54 fichiers, monter à 10 laisserait des plis de test trop petits.
CV_N_SPLITS = 5

# Valeur d'imputation. Aucune valeur n'est manquante dans le jeu de référence
# (vérifié : 0 sur 10 304 cellules) ; c'est un filet pour un dépôt qui
# produirait des fichiers sans historique git.
IMPUTATION_VALUE = -1.0

# Seuil de décision. C'est celui de model.predict(), donc celui sous lequel
# TOUTES les métriques publiées sont calculées — et celui qu'applique le
# tableau de bord (cf. RISK_PRED_THRESHOLD dans streamlit_app.py).
DECISION_THRESHOLD = 0.5

# Métrique guidant la RECHERCHE D'HYPERPARAMÈTRES.
#
# C'était "f1" jusqu'au 29/07/2026, alors que la SÉLECTION du meilleur modèle se
# fait sur le MCC — on optimisait donc sur une métrique qu'on déclare par
# ailleurs trompeuse sur données déséquilibrées. Contradiction qu'un jury
# connaissant le ML pouvait relever.
#
# Désormais le MCC guide l'intégralité du pipeline : réglage, sélection et
# évaluation. Aucune étape n'utilise plus le F1 comme critère de décision.
TUNING_SCORING = "matthews_corrcoef" 

# Colonnes du dataset
ID_COLS = ["file", "class", "method", "profile"]
LABEL_COL = "label_risk"

# Feature sets
#
# v1 (SAFE_FEATURES) — Jeu initial, 21 features
#   Toutes les features structurelles + git, sans leakage direct.
#   Problèmes identifiés par l'EDA (eda.py) :
#     - 3 features quasi-constantes (variance ~0, aucun signal)
#     - 3 features redondantes (corrélations >0.86, VIF explosés)
#     - Overfitting accru sur petit dataset (+50% de gap train/test)
#
# v2 (SAFE_FEATURES_V2) — Jeu nettoyé post-EDA, 15 features  (HISTORIQUE)
#   6 features retirées. Comparaison v1→v2 conservée ci-dessous à titre de trace ;
#   ATTENTION : les CV F1 de ce tableau sont à lire contre leur baseline triviale
#   (SUCRE 82% de positifs → « tout positif » donne déjà F1 = 0.901), ils ne
#   démontrent donc pas grand-chose seuls. Voir docs/analyse-resultats-ml.md.
#
#   ┌─────────────────┬────────────────────────────┬────────────────────────────┐
#   │                 │ SUCRE (368 lignes, 82% +)  │ Sylius (183 lignes, 26% +) │
#   ├─────────────────┼────────────────────────────┼────────────────────────────┤
#   │ XGB  CV F1 v1   │ 0.9296                     │ 0.8366                     │
#   │ XGB  CV F1 v2   │ 0.9131 (-0.017)            │ 0.8363 (-0.000)            │
#   │ XGB  gap v1→v2  │ —                          │ 0.078 → 0.050 (-36%)       │
#   ├─────────────────┼────────────────────────────┼────────────────────────────┤
#   │ RF   CV F1 v1   │ 0.9922                     │ 0.8101                     │
#   │ RF   CV F1 v2   │ 0.9793 (-0.013)            │ 0.8113 (+0.001)            │
#   │ RF   gap v1→v2  │ —                          │ 0.093 → 0.046 (-50%)       │
#   └─────────────────┴────────────────────────────┴────────────────────────────┘
#
#   Conclusion : CV F1 quasi identique, overfitting réduit de 36-50% sur Sylius.
#
# v3 (SAFE_FEATURES_V3) — 13 features, structure pure  ← DÉFAUT ACTUEL (20/07/2026)
#   v2 moins les 2 dernières features git (GIT_WINDOW_FEATURES), calculées sur la
#   MÊME fenêtre de 12 mois que le label → fuite contemporaine. v3 est le jeu
#   réellement déployé (ml/models/feature_names.json = ces 13 features), donc le
#   défaut de --feature-set et de cross_eval.py sont alignés dessus.
#   Métriques à citer pour v3 : MCC = 0.545 et balanced accuracy = 0.806
#   (le F1 = 0.851 est SOUS la baseline triviale 0.901 — ne pas le mettre en avant).
#
# ─────────────────────────────────────────────────────────────────────────────

# v1 — Toutes les features safe (conservé pour comparaison / --feature-set v1)
# git_bugfix_count  -> exclu (détermine directement le label)
# git_total_commits -> exclu par défaut (contient les bugfix commits)
SAFE_FEATURES = [
    # Structure / routes
    "nb_routes_method", "has_route_attr",
    # Paramètres et comportement
    "nb_params", "has_form", "is_ajax_only",
    "nb_voter_checks", "nb_response_types",
    "has_render", "has_redirect", "has_json", "has_file_download",
    # Sécurité
    "nb_method_grants", "nb_class_grants",
    # Complexité structurelle
    "nb_constructor_deps", "nb_methods_in_class", "is_invoke",
    # Taille / complexité du code
    "file_loc", "method_loc", "cyclomatic_complexity",
    # Git (non liés au label)
    "git_nb_authors", "git_days_since_change",
]

# v2 — Features nettoyées post-EDA (défaut)
# Exclusions basées sur eda.py, validées sur SUCRE + Sylius :
#   - is_invoke            : variance nulle (100% = 0 sur SUCRE, 100% sur Sylius)
#   - nb_voter_checks      : quasi-constante (99.5% = 0 sur SUCRE, 100% sur Sylius)
#   - has_file_download    : quasi-constante (96.7% = 0 sur SUCRE, 100% sur Sylius)
#   - has_route_attr       : redondant avec nb_routes_method (r=0.864)
#   - file_loc             : redondant avec nb_methods_in_class (r=0.983, VIF=53)
#   - cyclomatic_complexity: redondant avec method_loc (r=0.893, VIF=6)
EDA_DROP = [
    "is_invoke",
    "nb_voter_checks",
    "has_file_download",
    "has_route_attr",
    "file_loc",
    "cyclomatic_complexity",
]
SAFE_FEATURES_V2 = [f for f in SAFE_FEATURES if f not in EDA_DROP]

# v3 — Structure pure : on retire aussi les 2 dernières features git, calculées
# sur la MÊME fenêtre de 12 mois que le label (fuite contemporaine légère).
# Modèle sans aucune dépendance à la fenêtre du label → le plus défendable.
GIT_WINDOW_FEATURES = ["git_nb_authors", "git_days_since_change"]
SAFE_FEATURES_V3 = [f for f in SAFE_FEATURES_V2 if f not in GIT_WINDOW_FEATURES]

# Features avec git_total_commits (leakage partiel, à comparer)
FEATURES_WITH_COMMITS = SAFE_FEATURES + ["git_total_commits"]
FEATURES_WITH_COMMITS_V2 = SAFE_FEATURES_V2 + ["git_total_commits"]


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"   Dataset : {csv_path}")
    print(f"   Lignes  : {len(df)}")
    print(f"   label=1 : {df[LABEL_COL].sum()} ({100 * df[LABEL_COL].mean():.1f}%)")
    print(f"   label=0 : {(df[LABEL_COL] == 0).sum()}")
    return df


def evaluate(name, model, X_train, y_train, X_test, y_test) -> dict:
    """Évalue sur train ET test pour détecter l'overfitting."""
    # Train
    y_train_pred = model.predict(X_train)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)

    # Test
    y_pred = model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    test_acc = accuracy_score(y_test, y_pred)

    metrics = {
        "model": name,
        "train_f1": float(train_f1),
        "test_f1": float(test_f1),
        "accuracy": float(test_acc),
        "f1": float(test_f1),
        "overfit_gap": float(train_f1 - test_f1),
    }

    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics["auc"] = float(roc_auc_score(y_test, y_proba))
        except Exception:
            metrics["auc"] = None
    else:
        metrics["auc"] = None

    print(f"\n── {name} ──")
    print(f"   train_f1  : {train_f1:.4f}")
    print(f"   test_f1   : {test_f1:.4f}")
    print(f"   accuracy  : {test_acc:.4f}")
    if metrics["auc"] is not None:
        print(f"   auc       : {metrics['auc']:.4f}")

    # Alerte overfitting
    gap = train_f1 - test_f1
    if gap > OVERFIT_GAP_WARNING:
        severity = "LEGER" if gap < OVERFIT_GAP_SEVERE else "FORT"
        print(f"   {severity} OVERFITTING : train_f1 - test_f1 = {gap:.4f}")
    else:
        print(f"Pas d'overfitting detecte (gap = {gap:.4f})")

    print(classification_report(y_test, y_pred, zero_division=0, digits=3))
    return metrics


def cross_val_report(name, model, X, y, groups, random_state=42):
    """Cross-validation groupée par fichier : aucun fichier (ni aucune de ses
    méthodes) ne peut être à la fois en apprentissage et en test. Évite la fuite
    de groupe propre à une granularité méthode avec un label défini au niveau
    fichier."""
    cv = StratifiedGroupKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=random_state)
    scores = cross_validate(
        model, X, y, groups=groups, cv=cv,
        scoring=["f1", "accuracy", "roc_auc", "balanced_accuracy",
                 "matthews_corrcoef"],
        return_train_score=True,
        n_jobs=-1,
    )
    print(f"\n   Cross-validation (StratifiedGroupKFold, 5 plis groupés par fichier) :")
    print(f"      train_f1 : {scores['train_f1'].mean():.4f} +/- {scores['train_f1'].std():.4f}")
    print(f"      test_f1  : {scores['test_f1'].mean():.4f} +/- {scores['test_f1'].std():.4f}")
    print(f"      test_acc : {scores['test_accuracy'].mean():.4f} +/- {scores['test_accuracy'].std():.4f}")
    print(f"      auc      : {scores['test_roc_auc'].mean():.4f} +/- {scores['test_roc_auc'].std():.4f}")
    print(f"      bal_acc  : {scores['test_balanced_accuracy'].mean():.4f}")
    print(f"      mcc      : {scores['test_matthews_corrcoef'].mean():.4f}")
    gap = scores['train_f1'].mean() - scores['test_f1'].mean()
    if gap > OVERFIT_GAP_WARNING:
        print(f"Overfitting CV gap : {gap:.4f}")
    return {
        "cv_train_f1_mean": float(scores['train_f1'].mean()),
        "cv_test_f1_mean": float(scores['test_f1'].mean()),
        "cv_test_f1_std": float(scores['test_f1'].std()),
        "cv_test_acc_mean": float(scores['test_accuracy'].mean()),
        "cv_auc_mean": float(scores['test_roc_auc'].mean()),
        "cv_auc_std": float(scores['test_roc_auc'].std()),
        "cv_balacc_mean": float(scores['test_balanced_accuracy'].mean()),
        "cv_mcc_mean": float(scores['test_matthews_corrcoef'].mean()),
        # AUC cross-validée exposée comme métrique principale : stable, contrairement
        # à l'AUC d'un unique holdout de quelques fichiers.
        "auc": float(scores['test_roc_auc'].mean()),
    }


def _cv_tuning(random_state):
    """
    Découpage utilisé pour la RECHERCHE D'HYPERPARAMÈTRES.

    C'était `cv=5` — un entier, donc un StratifiedKFold classique, NON groupé.
    Pendant le tuning, des méthodes d'un même contrôleur se retrouvaient donc à
    la fois en apprentissage et en validation : le modèle avait déjà vu la
    réponse, et les hyperparamètres étaient choisis sur des scores optimistes.

    L'évaluation FINALE publiée, elle, était déjà correctement groupée — les
    chiffres annoncés n'étaient pas faussés. Seul le CHOIX des hyperparamètres
    l'était. Avec ce correctif, plus aucune étape ne mélange les fichiers.

    ⚠️ Il faut passer `groups=` à `.fit()` pour que ce découpage serve à
    quelque chose : sans lui, scikit-learn lève une erreur.
    """
    return StratifiedGroupKFold(n_splits=CV_N_SPLITS, shuffle=True,
                                random_state=random_state)


def file_level_report(name, model, X, y, groups, random_state=42):
    """
    Ré-évalue les mêmes prédictions croisées, mais AGRÉGÉES PAR FICHIER.

    Pourquoi c'est nécessaire. Le label est défini au niveau du FICHIER, tandis
    qu'une ligne = une méthode. Les fichiers positifs étant nettement plus gros
    (13,7 méthodes en moyenne contre 3,9), une métrique calculée par ligne leur
    donne un poids démesuré : sur SUCRE, un seul contrôleur de 77 méthodes pèse
    20,9 % du score alors qu'il ne représente qu'1 contrôleur sur 54.

    Or la décision que l'outil fait réellement prendre est « quel CONTRÔLEUR
    tester en priorité ». Cette fonction mesure donc exactement cette
    décision-là : chaque fichier compte pour un, quelle que soit sa taille.

    Attendre un score plus bas que la version par ligne — c'est normal, et c'est
    le chiffre honnête à citer pour l'usage réel.
    """
    cv = StratifiedGroupKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=random_state)
    pred = cross_val_predict(model, X, y, groups=groups, cv=cv, n_jobs=-1)

    agg = (pd.DataFrame({"file": groups, "y": y, "pred": pred})
             .groupby("file")
             .agg(y=("y", "first"), pred=("pred", "mean")))
    y_file = agg["y"].to_numpy()
    p_file = (agg["pred"] >= DECISION_THRESHOLD).astype(int).to_numpy()   # vote majoritaire

    mcc = matthews_corrcoef(y_file, p_file)
    bal = balanced_accuracy_score(y_file, p_file)
    f1v = f1_score(y_file, p_file, zero_division=0)
    # Baseline triviale « tout positif », à citer avec le F1
    baseline_f1 = f1_score(y_file, np.ones_like(y_file), zero_division=0)

    print(f"\n   Agrégé PAR FICHIER ({len(y_file)} contrôleurs, "
          f"{100 * y_file.mean():.1f}% positifs) :")
    print(f"      mcc      : {mcc:.4f}")
    print(f"      bal_acc  : {bal:.4f}")
    print(f"      f1       : {f1v:.4f}   (baseline triviale : {baseline_f1:.4f})")

    return {
        "file_mcc": float(mcc),
        "file_balacc": float(bal),
        "file_f1": float(f1v),
        "file_f1_baseline": float(baseline_f1),
        "n_files": int(len(y_file)),
        "file_positive_rate": float(y_file.mean()),
    }


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
    ap.add_argument("--include-total-commits", action="store_true",
                    help="Inclure git_total_commits dans les features "
                         "(leakage partiel, pour comparaison)")
    ap.add_argument("--feature-set", choices=["v1", "v2", "v3"], default="v3",
                    help="v3 = structure pure sans features git de fenêtre (DÉFAUT, "
                         "leak-free, aligné sur le modèle déployé ml/models/), "
                         "v2 = nettoyées post-EDA (historique), "
                         "v1 = toutes les features safe")
    ap.add_argument("--compare", action="store_true",
                    help="Entraîne avec v1 ET v2 et compare les résultats")
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    df = load_dataset(args.dataset)

    # Choix du jeu de features
    if args.compare:
        # Mode comparaison entraîne avec les deux sets et affiche les deltas
        print("\n" + "=" * 60)
        print("MODE COMPARAISON v1 vs v2")
        print("=" * 60)
        _run_comparison(df, args)
        return

    if args.include_total_commits:
        feature_list = FEATURES_WITH_COMMITS_V2 if args.feature_set in ("v2", "v3") else FEATURES_WITH_COMMITS
        print(f"\nMode avec git_total_commits (leakage partiel, feature-set={args.feature_set})")
    elif args.feature_set == "v3":
        feature_list = SAFE_FEATURES_V3
        print(f"\nMode structure-seule v3 : features git de fenêtre retirées")
        print(f"   Retirées en plus de v2 : {GIT_WINDOW_FEATURES}")
    elif args.feature_set == "v2":
        feature_list = SAFE_FEATURES_V2
        print(f"\nMode safe v2 (post-EDA) : {len(EDA_DROP)} features retirées")
        print(f"   Exclues : {EDA_DROP}")
    else:
        feature_list = SAFE_FEATURES
        print("\nMode safe v1 : git_bugfix_count ET git_total_commits exclus")

    feats = [c for c in feature_list if c in df.columns]
    missing = set(feature_list) - set(feats)
    if missing:
        print(f"   Colonnes manquantes (ignorees) : {missing}")
    print(f"   Features utilisees : {len(feats)}")
    print(f"   -> {feats}")

    X = df[feats].fillna(IMPUTATION_VALUE).values
    y = df[LABEL_COL].astype(int).values
    groups = df["file"].values  # 1 groupe = 1 fichier (et toutes ses méthodes)

    if y.sum() == 0 or y.sum() == len(y):
        raise SystemExit("Le label est constant (que des 0 ou que des 1). "
                         "Ajuste --bugfix-threshold ou la fenêtre temporelle.")

    # Split groupé : aucun fichier à cheval sur train et test (anti-fuite de groupe)
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                            random_state=args.random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"\nTrain : {len(X_train)} | Test : {len(X_test)} "
          f"| Fichiers : {len(set(groups))} "
          f"(train {len(set(groups[train_idx]))} / test {len(set(groups[test_idx]))})")

    all_metrics = []

    # Logistic Regression
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
            cv=_cv_tuning(args.random_state), scoring=TUNING_SCORING, n_jobs=-1,
        )
        lr_grid.fit(X_train, y_train, groups=groups[train_idx])
        best_lr = lr_grid.best_estimator_
        print(f"   Best params : {lr_grid.best_params_}")

    m = evaluate("LogisticRegression", best_lr, X_train, y_train, X_test, y_test)
    # best_lr (et non lr_pipe) : la sélection du meilleur modèle se fait sur
    # cv_mcc_mean, il faut donc cross-valider le modèle OPTIMISÉ — comme pour
    # RandomForest ci-dessous. Cross-valider lr_pipe (non tuné) comparait un RF
    # optimisé à une LogReg par défaut. Vérifié sur SUCRE : l'écart est de
    # 0,0003 de MCC et RandomForest reste devant de 0,068, donc les chiffres
    # déjà publiés restent valides — mais la comparaison est maintenant loyale.
    m.update(cross_val_report("LogisticRegression", best_lr, X, y, groups, args.random_state))
    m.update(file_level_report("LogisticRegression", best_lr, X, y, groups, args.random_state))
    all_metrics.append(m)
    confusion_plot("LogReg", best_lr, X_test, y_test,
                   os.path.join(args.output_dir, "confusion_logreg.png"))

    # Random Forest (contraint)
    print("\n" + "=" * 60)
    print("2. Random Forest")
    print("=" * 60)
    rf = RandomForestClassifier(class_weight="balanced", random_state=args.random_state)
    if args.no_grid:
        rf.set_params(max_depth=8, min_samples_leaf=5)
        rf.fit(X_train, y_train)
        best_rf = rf
    else:
        rf_grid = GridSearchCV(
            rf,
            {
                "n_estimators": [100, 200],
                "max_depth": [4, 8, 12],
                "min_samples_leaf": [3, 5, 10],
            },
            cv=_cv_tuning(args.random_state), scoring=TUNING_SCORING, n_jobs=-1,
        )
        rf_grid.fit(X_train, y_train, groups=groups[train_idx])
        best_rf = rf_grid.best_estimator_
        print(f"   Best params : {rf_grid.best_params_}")

    m = evaluate("RandomForest", best_rf, X_train, y_train, X_test, y_test)
    m.update(cross_val_report("RandomForest", best_rf, X, y, groups, args.random_state))
    m.update(file_level_report("RandomForest", best_rf, X, y, groups, args.random_state))
    all_metrics.append(m)
    confusion_plot("RandomForest", best_rf, X_test, y_test,
                   os.path.join(args.output_dir, "confusion_rf.png"))

    # Importance des features
    importances = pd.DataFrame({
        "feature": feats,
        "importance": best_rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    importances.to_csv(os.path.join(args.output_dir, "feature_importance.csv"),
                       index=False)
    print("\n Top 10 features (RandomForest) :")
    print(importances.head(10).to_string(index=False))

    # XGBoost
    xgb = None
    if HAS_XGB:
        print("\n" + "=" * 60)
        print("3. XGBoost")
        print("=" * 60)
        spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        xgb = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=args.random_state,
        )
        xgb.fit(X_train, y_train)
        m = evaluate("XGBoost", xgb, X_train, y_train, X_test, y_test)
        m.update(cross_val_report("XGBoost", xgb, X, y, groups, args.random_state))
        m.update(file_level_report("XGBoost", xgb, X, y, groups, args.random_state))
        all_metrics.append(m)
        confusion_plot("XGBoost", xgb, X_test, y_test,
                       os.path.join(args.output_dir, "confusion_xgb.png"))
    else:
        print("\n XGBoost non installe — pip install xgboost pour l'activer")

    # Résumé comparatif
    print("\n" + "=" * 60)
    print("RESUME COMPARATIF")
    print("=" * 60)
    print(f"{'Modele':<25} {'CV MCC':>9} {'MCC fichier':>12} {'CV bal-acc':>11} "
          f"{'CV F1':>14} {'Gap':>8} {'AUC':>8}")
    print("-" * 92)
    for m in all_metrics:
        cv_f1 = f"{m.get('cv_test_f1_mean', 0):.4f}+/-{m.get('cv_test_f1_std', 0):.3f}"
        auc = f"{m['auc']:.4f}" if m.get('auc') is not None else "  N/A"
        print(f"{m['model']:<25} {m.get('cv_mcc_mean', 0):>9.4f} "
              f"{m.get('file_mcc', 0):>12.4f} {m.get('cv_balacc_mean', 0):>11.4f} "
              f"{cv_f1:>14} {m['overfit_gap']:>8.4f} {auc:>8}")
    print("\n   MCC fichier = même modèle, décision agrégée PAR CONTRÔLEUR.")
    print("   C'est le chiffre honnête pour l'usage réel (« quel contrôleur tester »),")
    print("   nécessairement plus bas : par ligne, les gros contrôleurs pèsent davantage.")

    # Sélection du meilleur sur le MCC cross-validé : métrique robuste au
    # déséquilibre (le F1 favorise artificiellement le modèle qui colle à la
    # classe majoritaire). Fallback sur CV F1 si le MCC n'est pas disponible.
    best = max(all_metrics, key=lambda m: m.get("cv_mcc_mean",
                                                m.get("cv_test_f1_mean", m["f1"])))
    print(f"\nMeilleur modele : {best['model']}  "
          f"(MCC CV={best.get('cv_mcc_mean', 0):.4f} | "
          f"AUC CV={best.get('cv_auc_mean', 0):.4f} | "
          f"bal-acc={best.get('cv_balacc_mean', 0):.4f})")

    if best.get("overfit_gap", 0) > OVERFIT_GAP_HOLDOUT_NOTE:
        print(f"Note : gap train/test (holdout) = {best['overfit_gap']:.4f} "
              "— à relativiser, le holdout ne fait que quelques fichiers ; "
              "le gap CV est la référence.")

    # Sauvegarde
    name_to_model = {
        "LogisticRegression": best_lr,
        "RandomForest": best_rf,
    }
    if xgb is not None:
        name_to_model["XGBoost"] = xgb

    joblib.dump(name_to_model[best["model"]],
                os.path.join(args.output_dir, "best_model.pkl"))
    with open(os.path.join(args.output_dir, "feature_names.json"), "w") as f:
        json.dump(feats, f, indent=2)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({"all": all_metrics, "best": best}, f, indent=2)

    print(f"\n Sauvegardes dans {args.output_dir}/")
    print("   - best_model.pkl")
    print("   - feature_names.json")
    print("   - metrics.json")
    print("   - feature_importance.csv")
    print("   - confusion_*.png")

    # Conseil
    if not args.include_total_commits:
        print("\n Pour comparer avec git_total_commits, relance avec :")
        print(f"   python {__file__} {args.dataset} --include-total-commits")
        print("   (si le F1 monte beaucoup, c'est que git_total_commits fuit)")
    if args.feature_set != "v3":
        print("\n Le jeu déployé est v3 (leak-free, aligné sur ml/models/) :")
        print(f"   python {__file__} {args.dataset} --feature-set v3")
    print("\n Pour comparer v1 vs v2 côte à côte :")
    print(f"   python {__file__} {args.dataset} --compare")


def _quick_train(df, feature_list, test_size, random_state):
    """Entraîne un XGBoost (ou RF) rapidement et retourne les métriques CV."""
    feats = [c for c in feature_list if c in df.columns]
    X = df[feats].fillna(IMPUTATION_VALUE).values
    y = df[LABEL_COL].astype(int).values
    groups = df["file"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    results = {}
    # XGBoost
    if HAS_XGB:
        spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=random_state,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        train_f1 = f1_score(y_train, model.predict(X_train), zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)

        cv = StratifiedGroupKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=random_state)
        scores = cross_validate(model, X, y, groups=groups, cv=cv, scoring=["f1"], n_jobs=-1)
        cv_f1 = scores["test_f1"].mean()
        cv_std = scores["test_f1"].std()

        try:
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        except Exception:
            auc = None

        results["XGBoost"] = {
            "train_f1": train_f1, "test_f1": test_f1,
            "cv_f1": cv_f1, "cv_std": cv_std,
            "gap": train_f1 - test_f1, "auc": auc,
        }

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=random_state,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    train_f1 = f1_score(y_train, rf.predict(X_train), zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)

    cv = StratifiedGroupKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=random_state)
    scores = cross_validate(rf, X, y, groups=groups, cv=cv, scoring=["f1"], n_jobs=-1)
    cv_f1 = scores["test_f1"].mean()
    cv_std = scores["test_f1"].std()

    try:
        auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    except Exception:
        auc = None

    results["RandomForest"] = {
        "train_f1": train_f1, "test_f1": test_f1,
        "cv_f1": cv_f1, "cv_std": cv_std,
        "gap": train_f1 - test_f1, "auc": auc,
    }

    return results, feats


def _run_comparison(df, args):
    """Entraîne avec v1 et v2, affiche un tableau comparatif."""
    print("\n── Feature set v1 (toutes les features) ──")
    print(f"   {len(SAFE_FEATURES)} features")
    r1, feats1 = _quick_train(df, SAFE_FEATURES, args.test_size, args.random_state)

    print("\n── Feature set v2 (post-EDA, nettoyé) ──")
    print(f"   {len(SAFE_FEATURES_V2)} features ({len(EDA_DROP)} retirées)")
    print(f"   Exclues : {EDA_DROP}")
    r2, feats2 = _quick_train(df, SAFE_FEATURES_V2, args.test_size, args.random_state)

    # Tableau comparatif
    print("\n" + "=" * 80)
    print("COMPARAISON v1 vs v2")
    print("=" * 80)
    header = f"{'Modèle':<18} {'Version':<5} {'Train F1':>10} {'Test F1':>10} {'CV F1':>14} {'Gap':>8} {'AUC':>8}"
    print(header)
    print("-" * len(header))

    for model_name in ["XGBoost", "RandomForest"]:
        if model_name not in r1:
            continue
        for ver, res in [("v1", r1[model_name]), ("v2", r2[model_name])]:
            cv_str = f"{res['cv_f1']:.4f}+/-{res['cv_std']:.3f}"
            auc_str = f"{res['auc']:.4f}" if res['auc'] is not None else "  N/A"
            gap_flag = " " if res['gap'] > 0.05 else ""
            print(f"{model_name:<18} {ver:<5} {res['train_f1']:>10.4f} {res['test_f1']:>10.4f} "
                  f"{cv_str:>14} {res['gap']:>8.4f}{gap_flag} {auc_str:>8}")

        # Delta
        d1 = r1[model_name]
        d2 = r2[model_name]
        delta_cv = d2["cv_f1"] - d1["cv_f1"]
        delta_gap = d2["gap"] - d1["gap"]
        sign = "+" if delta_cv >= 0 else ""
        print(f"{'  Δ (v2-v1)':<23} {'':>10} {'':>10} {'':>4}{sign}{delta_cv:.4f}{'':>6} {delta_gap:>+8.4f}")
        if delta_cv > 0:
            print(f"  → v2 MEILLEUR sur {model_name} (CV F1 {sign}{delta_cv:.4f})")
        elif delta_cv < 0:
            print(f"  → v1 meilleur sur {model_name} (CV F1 {delta_cv:+.4f})")
        else:
            print(f"  → Pas de différence significative sur {model_name}")
        print()

    dropped = set(feats1) - set(feats2)
    print(f"Features retirées en v2 : {sorted(dropped)}")
    print(f"Features restantes en v2 : {len(feats2)}")
    print(f"\nNB : le jeu déployé est v3 (leak-free). Pour entraîner et sauvegarder :")
    print(f"   python {__file__} {args.dataset} --feature-set v3")


if __name__ == "__main__":
    main()
