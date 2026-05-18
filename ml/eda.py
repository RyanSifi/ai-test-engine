"""
Analyse exploratoire complète (EDA) du dataset de risque de régression.

S'exécute ENTRE extract_dataset.py et train.py dans le pipeline :

    extract_dataset.py  →  eda.py  →  train.py

Produit un rapport visuel complet dans un dossier dédié (PNG + résumé texte)
pour guider les décisions de feature engineering avant l'entraînement.

Sections :
 1. Vue d'ensemble (shape, types, missing, describe)
 2. Déséquilibre de classes + analyse par profil
 3. Distributions univariées (histogrammes colorés par label)
 4. Boxplots par label (features continues)
 5. Matrice de corrélation + corrélation avec le label
 6. Features à faible variance / quasi-constantes
 7. Analyse bivariée features binaires ↔ label (taux de risque)
 8. Analyse git (distributions, leakage check)
 9. Multicolinéarité (VIF)
10. Recommandations automatiques

Usage :
    python eda.py _dataset/dataset.csv [--output-dir eda_report]
"""
import argparse
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# Constantes partagées avec train.py
ID_COLS = ["file", "class", "method", "profile"]
LABEL_COL = "label_risk"

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

FEATURES_WITH_COMMITS = SAFE_FEATURES + ["git_total_commits"]

# Sous-groupes sémantiques pour les visualisations
BINARY_FEATURES = [
    "has_route_attr", "has_form", "is_ajax_only",
    "has_render", "has_redirect", "has_json", "has_file_download", "is_invoke",
]
CONTINUOUS_FEATURES = [
    "file_loc", "method_loc", "cyclomatic_complexity",
    "nb_params", "nb_constructor_deps", "nb_methods_in_class",
    "nb_routes_method", "nb_voter_checks", "nb_response_types",
    "nb_method_grants", "nb_class_grants",
    "git_nb_authors", "git_days_since_change",
]
GIT_FEATURES = [
    "git_bugfix_count", "git_total_commits", "git_nb_authors", "git_days_since_change",
]

# Style global
COLORS = {"safe": "#2196F3", "risk": "#F44336", "neutral": "#607D8B"}
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


class EDAReport:
    """Agrège les findings pour un résumé final."""

    def __init__(self):
        self.findings: list[str] = []
        self.warnings: list[str] = []
        self.recommendations: list[str] = []

    def add(self, category: str, message: str):
        getattr(self, category).append(message)

    def dump(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("  EDA REPORT — Résumé automatique\n")
            f.write("=" * 70 + "\n\n")
            for section, items in [
                ("FINDINGS", self.findings),
                ("WARNINGS", self.warnings),
                ("RECOMMENDATIONS", self.recommendations),
            ]:
                f.write(f"── {section} ──\n")
                if not items:
                    f.write("  (aucun)\n")
                for item in items:
                    f.write(f"  • {item}\n")
                f.write("\n")
        print(f"\n  Résumé écrit → {path}")


def save_fig(fig, outdir: str, name: str):
    path = os.path.join(outdir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {name}.png")


# VUE D'ENSEMBLE
def section_overview(df: pd.DataFrame, feats: list, outdir: str, report: EDAReport):
    print("\n" + "=" * 60)
    print("1. VUE D'ENSEMBLE")
    print("=" * 60)

    print(f"   Shape         : {df.shape}")
    print(f"   Colonnes      : {len(df.columns)}")
    print(f"   Features      : {len(feats)} (sur {len(FEATURES_WITH_COMMITS)} définies)")

    # Missing values
    missing = df[feats].isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    if missing.sum() > 0:
        print("\n   Valeurs manquantes :")
        for col in missing[missing > 0].index:
            print(f"      {col}: {missing[col]} ({missing_pct[col]}%)")
            report.add("warnings", f"Colonne '{col}' a {missing_pct[col]}% de valeurs manquantes")
    else:
        print("   Valeurs manquantes : aucune")
        report.add("findings", "Aucune valeur manquante dans les features")

    # Describe
    desc = df[feats].describe().T
    desc_path = os.path.join(outdir, "describe.csv")
    desc.to_csv(desc_path)
    print(f"\n   describe() complet → describe.csv")

    # Valeurs sentinelles (-1.0 utilisé comme fill dans train.py)
    sentinel_counts = (df[feats] == -1.0).sum()
    sentinel_cols = sentinel_counts[sentinel_counts > 0]
    if len(sentinel_cols) > 0:
        print("\n   Valeurs sentinelles (-1.0) :")
        for col in sentinel_cols.index:
            pct = 100 * sentinel_cols[col] / len(df)
            print(f"      {col}: {sentinel_cols[col]} ({pct:.1f}%)")
            if pct > 20:
                report.add("warnings",
                    f"'{col}' a {pct:.0f}% de valeurs à -1.0 (sentinelle) — "
                    "impact potentiel sur le modèle")


# DÉSÉQUILIBRE DE CLASSES
def section_class_balance(df: pd.DataFrame, outdir: str, report: EDAReport):
    print("\n" + "=" * 60)
    print("2. DÉSÉQUILIBRE DE CLASSES")
    print("=" * 60)

    counts = df[LABEL_COL].value_counts().sort_index()
    ratio = counts[1] / counts[0] if counts[0] > 0 else float("inf")
    print(f"   label=0 : {counts[0]}  ({100 * counts[0] / len(df):.1f}%)")
    print(f"   label=1 : {counts[1]}  ({100 * counts[1] / len(df):.1f}%)")
    print(f"   Ratio 1/0 : {ratio:.3f}")

    if ratio < 0.2 or ratio > 5:
        report.add("warnings",
            f"Fort déséquilibre de classes (ratio={ratio:.2f}). "
            "class_weight='balanced' est activé dans train.py — OK, "
            "mais surveiller les métriques sur la classe minoritaire.")
    else:
        report.add("findings", f"Déséquilibre modéré (ratio={ratio:.2f})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bar plot
    ax = axes[0]
    bars = ax.bar(["safe (0)", "risque (1)"], counts.values,
                  color=[COLORS["safe"], COLORS["risk"]], edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + len(df) * 0.01,
                f"{val}\n({100 * val / len(df):.1f}%)", ha="center", fontweight="bold")
    ax.set_ylabel("Nombre de méthodes")
    ax.set_title("Distribution du label")

    # Par profil
    ax = axes[1]
    if "profile" in df.columns:
        profile_risk = df.groupby("profile")[LABEL_COL].agg(["mean", "count"])
        profile_risk = profile_risk.sort_values("mean", ascending=True)
        colors = [COLORS["risk"] if m > 0.5 else COLORS["safe"] for m in profile_risk["mean"]]
        bars = ax.barh(profile_risk.index, profile_risk["mean"], color=colors,
                       edgecolor="black", linewidth=0.5)
        ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="seuil 50%")
        ax.set_xlabel("Taux de risque")
        ax.set_title("Taux de risque par profil")
        ax.legend()

        # Profils 100% risk ou 100% safe
        for prof, row in profile_risk.iterrows():
            if row["mean"] == 1.0 and row["count"] >= 3:
                report.add("warnings",
                    f"Profil '{prof}' est 100% à risque ({int(row['count'])} méthodes) — "
                    "le modèle pourrait simplement mémoriser le profil")
            elif row["mean"] == 0.0 and row["count"] >= 3:
                report.add("findings",
                    f"Profil '{prof}' est 100% safe ({int(row['count'])} méthodes)")
    else:
        ax.text(0.5, 0.5, "Colonne 'profile' absente", transform=ax.transAxes,
                ha="center", va="center")

    fig.suptitle("Analyse du déséquilibre de classes", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, outdir, "02_class_balance")


# DISTRIBUTIONS UNIVARIÉES (histogrammes colorés par label)
def section_distributions(df: pd.DataFrame, feats: list, outdir: str, report: EDAReport):
    print("\n" + "=" * 60)
    print("3. DISTRIBUTIONS UNIVARIÉES")
    print("=" * 60)

    cont_feats = [f for f in CONTINUOUS_FEATURES if f in feats]
    n = len(cont_feats)
    if n == 0:
        print("   Aucune feature continue trouvée")
        return

    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    df_safe = df[df[LABEL_COL] == 0]
    df_risk = df[df[LABEL_COL] == 1]

    for i, feat in enumerate(cont_feats):
        ax = axes[i]
        ax.hist(df_safe[feat].dropna(), bins=30, alpha=0.6,
                color=COLORS["safe"], label="safe", edgecolor="black", linewidth=0.3)
        ax.hist(df_risk[feat].dropna(), bins=30, alpha=0.6,
                color=COLORS["risk"], label="risque", edgecolor="black", linewidth=0.3)
        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)

        # Détection d'outliers (IQR)
        q1, q3 = df[feat].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_outliers = ((df[feat] < q1 - 3 * iqr) | (df[feat] > q3 + 3 * iqr)).sum()
        if n_outliers > 0:
            pct = 100 * n_outliers / len(df)
            ax.set_xlabel(f"({n_outliers} outliers extrêmes)", fontsize=7, color="red")
            if pct > 5:
                report.add("warnings",
                    f"'{feat}' a {n_outliers} outliers extrêmes ({pct:.1f}%)")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distributions par feature (colorées par label)", fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, outdir, "03_distributions")


# BOXPLOTS PAR LABEL
def section_boxplots(df: pd.DataFrame, feats: list, outdir: str, report: EDAReport):
    print("\n" + "=" * 60)
    print("4. BOXPLOTS PAR LABEL")
    print("=" * 60)

    cont_feats = [f for f in CONTINUOUS_FEATURES if f in feats]
    if not cont_feats:
        return

    ncols = 4
    nrows = (len(cont_feats) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, feat in enumerate(cont_feats):
        ax = axes[i]
        data = [df[df[LABEL_COL] == 0][feat].dropna(),
                df[df[LABEL_COL] == 1][feat].dropna()]
        bp = ax.boxplot(data, tick_labels=["safe", "risque"], patch_artist=True,
                        widths=0.6, showfliers=True,
                        flierprops=dict(markersize=3, alpha=0.4))
        bp["boxes"][0].set_facecolor(COLORS["safe"])
        bp["boxes"][1].set_facecolor(COLORS["risk"])
        for box in bp["boxes"]:
            box.set_alpha(0.6)
        ax.set_title(feat, fontsize=9, fontweight="bold")

        # Test de Mann-Whitney
        if len(data[0]) > 5 and len(data[1]) > 5:
            stat, pval = stats.mannwhitneyu(data[0], data[1], alternative="two-sided")
            if pval < 0.01:
                ax.set_xlabel(f"p={pval:.2e} ***", fontsize=7, color="green")
            elif pval < 0.05:
                ax.set_xlabel(f"p={pval:.3f} *", fontsize=7, color="orange")
            else:
                ax.set_xlabel(f"p={pval:.3f} (ns)", fontsize=7, color="gray")
                report.add("findings",
                    f"'{feat}' ne discrimine pas significativement le label (p={pval:.3f})")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Boxplots par label (Mann-Whitney U test)", fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, outdir, "04_boxplots")


# MATRICE DE CORRÉLATION + CORRÉLATION AVEC LE LABEL
def section_correlations(df: pd.DataFrame, feats: list, outdir: str, report: EDAReport):
    print("\n" + "=" * 60)
    print("5. CORRÉLATIONS")
    print("=" * 60)

    corr_cols = feats + [LABEL_COL]
    corr = df[corr_cols].corr()

    # Heatmap complète
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, ax=ax,
                annot_kws={"size": 7}, linewidths=0.5)
    ax.set_title("Matrice de corrélation (Pearson)", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, outdir, "05a_correlation_matrix")

    # Corrélation avec le label (bar chart trié)
    label_corr = corr[LABEL_COL].drop(LABEL_COL).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [COLORS["risk"] if v > 0 else COLORS["safe"] for v in label_corr.values]
    ax.barh(label_corr.index, label_corr.values, color=colors,
            edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.3, color="red", linestyle="--", alpha=0.3, label="|r| = 0.3")
    ax.axvline(-0.3, color="red", linestyle="--", alpha=0.3)
    ax.set_xlabel("Corrélation de Pearson avec label_risk")
    ax.set_title("Corrélation feature ↔ label", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, outdir, "05b_correlation_with_label")

    # Paires très corrélées entre elles (multicolinéarité)
    print("\n   Paires de features très corrélées (|r| > 0.7) :")
    feat_corr = df[feats].corr()
    high_corr_pairs = []
    for i_idx in range(len(feats)):
        for j_idx in range(i_idx + 1, len(feats)):
            r = feat_corr.iloc[i_idx, j_idx]
            if abs(r) > 0.7:
                pair = (feats[i_idx], feats[j_idx], r)
                high_corr_pairs.append(pair)
                print(f"      {pair[0]} ↔ {pair[1]} : r={r:.3f}")
                report.add("warnings",
                    f"Forte corrélation entre '{pair[0]}' et '{pair[1]}' (r={r:.3f}) — "
                    "envisager d'en retirer une")

    if not high_corr_pairs:
        print("      (aucune)")
        report.add("findings", "Pas de multicolinéarité forte (|r| > 0.7) entre features")

    # Features faiblement corrélées au label
    weak = label_corr[label_corr.abs() < 0.05]
    if len(weak) > 0:
        names = ", ".join(weak.index.tolist())
        report.add("findings",
            f"Features quasi non corrélées au label (|r| < 0.05) : {names}")


# FEATURES À FAIBLE VARIANCE
def section_low_variance(df: pd.DataFrame, feats: list, outdir: str, report: EDAReport):
    print("\n" + "=" * 60)
    print("6. FEATURES À FAIBLE VARIANCE / QUASI-CONSTANTES")
    print("=" * 60)

    results = []
    for feat in feats:
        vc = df[feat].value_counts(normalize=True)
        dominant_pct = vc.iloc[0] * 100
        n_unique = df[feat].nunique()
        results.append({
            "feature": feat,
            "n_unique": n_unique,
            "dominant_value": vc.index[0],
            "dominant_pct": round(dominant_pct, 1),
            "variance": round(df[feat].var(), 4) if df[feat].var() is not None else 0,
        })

    results_df = pd.DataFrame(results).sort_values("dominant_pct", ascending=False)
    results_df.to_csv(os.path.join(outdir, "low_variance.csv"), index=False)

    quasi_constant = results_df[results_df["dominant_pct"] >= 95]
    if len(quasi_constant) > 0:
        print("   Features quasi-constantes (>= 95% même valeur) :")
        for _, row in quasi_constant.iterrows():
            print(f"      {row['feature']}: {row['dominant_pct']}% = {row['dominant_value']} "
                  f"({row['n_unique']} valeurs uniques)")
            report.add("warnings",
                f"'{row['feature']}' est quasi-constante ({row['dominant_pct']}% = {row['dominant_value']}) — "
                "apporte très peu de signal, candidat à l'exclusion")
    else:
        print("   Aucune feature quasi-constante (>= 95%)")

    # Graphique
    fig, ax = plt.subplots(figsize=(10, 6))
    data = results_df.sort_values("dominant_pct", ascending=True)
    colors = ["#F44336" if p >= 95 else "#FF9800" if p >= 90 else "#4CAF50"
              for p in data["dominant_pct"]]
    ax.barh(data["feature"], data["dominant_pct"], color=colors,
            edgecolor="black", linewidth=0.3)
    ax.axvline(95, color="red", linestyle="--", alpha=0.5, label="seuil 95%")
    ax.axvline(90, color="orange", linestyle="--", alpha=0.5, label="seuil 90%")
    ax.set_xlabel("% de la valeur dominante")
    ax.set_title("Dominance de la valeur la plus fréquente par feature", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, outdir, "06_low_variance")


# ANALYSE BIVARIÉE : FEATURES BINAIRES ↔ LABEL
def section_binary_features(df: pd.DataFrame, feats: list, outdir: str, report: EDAReport):
    print("\n" + "=" * 60)
    print("7. FEATURES BINAIRES ↔ LABEL")
    print("=" * 60)

    bin_feats = [f for f in BINARY_FEATURES if f in feats]
    if not bin_feats:
        print("   Aucune feature binaire trouvée")
        return

    results = []
    for feat in bin_feats:
        for val in [0, 1]:
            subset = df[df[feat] == val]
            if len(subset) == 0:
                continue
            risk_rate = subset[LABEL_COL].mean()
            results.append({
                "feature": feat,
                "value": val,
                "count": len(subset),
                "risk_rate": risk_rate,
            })

    results_df = pd.DataFrame(results)

    ncols = 4
    nrows = (len(bin_feats) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten() if len(bin_feats) > 1 else [axes]

    for i, feat in enumerate(bin_feats):
        ax = axes[i]
        sub = results_df[results_df["feature"] == feat]
        if len(sub) < 2:
            ax.set_visible(False)
            continue
        rates = sub.set_index("value")["risk_rate"]
        counts = sub.set_index("value")["count"]
        bars = ax.bar([f"0\n(n={counts.get(0, 0)})", f"1\n(n={counts.get(1, 0)})"],
                      [rates.get(0, 0), rates.get(1, 0)],
                      color=[COLORS["safe"], COLORS["risk"]],
                      edgecolor="black", linewidth=0.5, alpha=0.7)
        for bar, val in zip(bars, [rates.get(0, 0), rates.get(1, 0)]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.1%}", ha="center", fontsize=8, fontweight="bold")
        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.set_ylabel("Taux de risque")
        ax.set_ylim(0, 1)

        # Test chi2
        ct = pd.crosstab(df[feat], df[LABEL_COL])
        if ct.shape == (2, 2) and ct.min().min() >= 5:
            chi2, pval, _, _ = stats.chi2_contingency(ct)
            sig = "***" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            ax.set_xlabel(f"χ²={chi2:.1f}, p={pval:.3f} {sig}", fontsize=7)

            diff = abs(rates.get(1, 0) - rates.get(0, 0))
            if pval < 0.05 and diff > 0.1:
                report.add("findings",
                    f"'{feat}' discrimine le label : taux risque = "
                    f"{rates.get(0, 0):.1%} (=0) vs {rates.get(1, 0):.1%} (=1)")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Taux de risque par feature binaire (test χ²)", fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, outdir, "07_binary_vs_label")


# ANALYSE GIT + LEAKAGE CHECK
def section_git_analysis(df: pd.DataFrame, outdir: str, report: EDAReport):
    print("\n" + "=" * 60)
    print("8. ANALYSE GIT + LEAKAGE CHECK")
    print("=" * 60)

    git_cols = [c for c in GIT_FEATURES if c in df.columns]
    if not git_cols:
        print("   Aucune feature git trouvée")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Distribution git_bugfix_count (c'est le label source)
    if "git_bugfix_count" in df.columns:
        ax = axes[0, 0]
        ax.hist(df["git_bugfix_count"], bins=30, color=COLORS["neutral"],
                edgecolor="black", linewidth=0.3)
        ax.set_title("Distribution git_bugfix_count\n(source du label)", fontweight="bold")
        ax.set_xlabel("Nombre de bugfix commits")

    # Scatter leakage : git_total_commits vs git_bugfix_count
    if "git_total_commits" in df.columns and "git_bugfix_count" in df.columns:
        ax = axes[0, 1]
        ax.scatter(df["git_total_commits"], df["git_bugfix_count"],
                   c=df[LABEL_COL].map({0: COLORS["safe"], 1: COLORS["risk"]}),
                   alpha=0.4, s=15, edgecolors="black", linewidths=0.2)
        r = df["git_total_commits"].corr(df["git_bugfix_count"])
        ax.set_title(f"Leakage check : total_commits vs bugfix_count\nr={r:.3f}",
                     fontweight="bold")
        ax.set_xlabel("git_total_commits")
        ax.set_ylabel("git_bugfix_count")

        if abs(r) > 0.5:
            report.add("warnings",
                f"Corrélation forte git_total_commits ↔ git_bugfix_count (r={r:.3f}) — "
                "confirme le leakage partiel. Bon choix de l'exclure par défaut dans train.py")
        report.add("findings",
            f"Corrélation git_total_commits ↔ git_bugfix_count : r={r:.3f}")

    # git_nb_authors par label
    if "git_nb_authors" in df.columns:
        ax = axes[1, 0]
        for label, color, name in [(0, COLORS["safe"], "safe"), (1, COLORS["risk"], "risque")]:
            ax.hist(df[df[LABEL_COL] == label]["git_nb_authors"], bins=20,
                    alpha=0.6, color=color, label=name, edgecolor="black", linewidth=0.3)
        ax.set_title("git_nb_authors par label", fontweight="bold")
        ax.legend()

    # git_days_since_change : % de valeurs sentinelles
    if "git_days_since_change" in df.columns:
        ax = axes[1, 1]
        sentinel_pct = 100 * (df["git_days_since_change"] == -1.0).sum() / len(df)
        valid = df[df["git_days_since_change"] > 0]["git_days_since_change"]
        if len(valid) > 0:
            ax.hist(valid, bins=30, color=COLORS["neutral"],
                    edgecolor="black", linewidth=0.3)
        ax.set_title(f"git_days_since_change\n({sentinel_pct:.0f}% valeurs sentinelles à -1)",
                     fontweight="bold")
        ax.set_xlabel("Jours depuis dernier changement")
        if sentinel_pct > 30:
            report.add("warnings",
                f"git_days_since_change a {sentinel_pct:.0f}% de valeurs à -1.0 — "
                "ces fichiers n'ont pas de commit dans la fenêtre")

    fig.suptitle("Analyse des features git", fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, outdir, "08_git_analysis")


# VIF (VARIANCE INFLATION FACTOR)
def section_vif(df: pd.DataFrame, feats: list, outdir: str, report: EDAReport):
    print("\n" + "=" * 60)
    print("9. VIF (MULTICOLINÉARITÉ)")
    print("=" * 60)

    from sklearn.linear_model import LinearRegression

    X = df[feats].fillna(-1.0).values
    vifs = []
    for i, feat in enumerate(feats):
        mask = list(range(len(feats)))
        mask.remove(i)
        X_other = X[:, mask]
        y_i = X[:, i]
        if np.std(y_i) == 0:
            vifs.append({"feature": feat, "VIF": float("inf")})
            continue
        lr = LinearRegression().fit(X_other, y_i)
        r2 = lr.score(X_other, y_i)
        vif_val = 1 / (1 - r2) if r2 < 1 else float("inf")
        vifs.append({"feature": feat, "VIF": round(vif_val, 2)})

    vif_df = pd.DataFrame(vifs).sort_values("VIF", ascending=False)
    vif_df.to_csv(os.path.join(outdir, "vif.csv"), index=False)

    print("   Top VIF :")
    for _, row in vif_df.head(10).iterrows():
        flag = " ⚠" if row["VIF"] > 5 else ""
        print(f"      {row['feature']:<30} VIF = {row['VIF']:>8.2f}{flag}")

    high_vif = vif_df[vif_df["VIF"] > 10]
    if len(high_vif) > 0:
        names = ", ".join(high_vif["feature"].tolist())
        report.add("warnings",
            f"VIF > 10 (multicolinéarité forte) : {names}. "
            "Impact faible sur RF/XGB, mais dégrade la LogisticRegression.")
    elif len(vif_df[vif_df["VIF"] > 5]) > 0:
        names = ", ".join(vif_df[vif_df["VIF"] > 5]["feature"].tolist())
        report.add("findings", f"VIF modéré (5-10) : {names}")
    else:
        report.add("findings", "Pas de multicolinéarité problématique (tous VIF < 5)")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    data = vif_df.sort_values("VIF", ascending=True)
    colors = ["#F44336" if v > 10 else "#FF9800" if v > 5 else "#4CAF50"
              for v in data["VIF"]]
    ax.barh(data["feature"], data["VIF"], color=colors,
            edgecolor="black", linewidth=0.3)
    ax.axvline(5, color="orange", linestyle="--", alpha=0.5, label="VIF = 5")
    ax.axvline(10, color="red", linestyle="--", alpha=0.5, label="VIF = 10")
    ax.set_xlabel("Variance Inflation Factor")
    ax.set_title("VIF par feature", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, outdir, "09_vif")


# RECOMMANDATIONS
def section_recommendations(df: pd.DataFrame, feats: list, report: EDAReport):
    print("\n" + "=" * 60)
    print("10. RECOMMANDATIONS AUTOMATIQUES")
    print("=" * 60)

    # Ratio de classes
    ratio = df[LABEL_COL].mean()
    if ratio < 0.1 or ratio > 0.9:
        report.add("recommendations",
            "Envisager SMOTE ou un sous-échantillonnage en plus de class_weight='balanced'")

    # Taille du dataset
    n = len(df)
    n_feats = len(feats)
    if n < 10 * n_feats:
        report.add("recommendations",
            f"Dataset petit ({n} lignes pour {n_feats} features, ratio={n / n_feats:.0f}). "
            "Risque d'overfitting élevé — la RepeatedStratifiedKFold dans train.py est un bon choix.")

    # Features candidates à l'exclusion
    report.add("recommendations",
        "Après cet EDA, relancer train.py et comparer les métriques "
        "avec/sans les features flaggées (quasi-constantes, non significatives)")

    report.add("recommendations",
        "Pipeline recommandé : extract_dataset.py → eda.py (ce script) → "
        "ajustements features → train.py → streamlit_app.py")

    for rec in report.recommendations:
        print(f"   → {rec}")


# MAIN
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="CSV produit par extract_dataset.py")
    ap.add_argument("--output-dir", default="eda_report",
                    help="Dossier de sortie pour les graphiques et le rapport")
    ap.add_argument("--include-total-commits", action="store_true",
                    help="Inclure git_total_commits dans l'analyse")
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    report = EDAReport()

    print("=" * 60)
    print("  EDA — Analyse exploratoire du dataset de risque")
    print("=" * 60)

    df = pd.read_csv(args.dataset)
    print(f"   Fichier : {args.dataset}")
    print(f"   Shape   : {df.shape}")

    # Déterminer les features présentes
    feature_list = FEATURES_WITH_COMMITS if args.include_total_commits else SAFE_FEATURES
    feats = [c for c in feature_list if c in df.columns]
    missing = set(feature_list) - set(feats)
    if missing:
        print(f"   Colonnes manquantes : {missing}")

    # Lancer toutes les sections
    section_overview(df, feats, args.output_dir, report)
    section_class_balance(df, args.output_dir, report)
    section_distributions(df, feats, args.output_dir, report)
    section_boxplots(df, feats, args.output_dir, report)
    section_correlations(df, feats, args.output_dir, report)
    section_low_variance(df, feats, args.output_dir, report)
    section_binary_features(df, feats, args.output_dir, report)
    section_git_analysis(df, args.output_dir, report)
    section_vif(df, feats, args.output_dir, report)
    section_recommendations(df, feats, report)

    # Sauvegarder le rapport texte
    report.dump(os.path.join(args.output_dir, "eda_summary.txt"))

    print("\n" + "=" * 60)
    print(f"  EDA terminé — {len(os.listdir(args.output_dir))} fichiers dans {args.output_dir}/")
    print("=" * 60)
    print("   Fichiers produits :")
    for f in sorted(os.listdir(args.output_dir)):
        print(f"     - {f}")
    print(f"\n   Prochaine étape : python train.py {args.dataset}")


if __name__ == "__main__":
    main()
