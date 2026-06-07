"""
EDA du dataset de risque de régression.

Tourne entre extract_dataset.py et train.py :

    extract_dataset.py  ->  eda.py  ->  train.py

Sort un dossier avec les graphes (PNG), un describe.csv et un résumé texte.
L'idée c'est de regarder les features avant l'entraînement pour décider
quoi garder, quoi virer, et repérer un éventuel leakage.

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

# Doivent rester alignées avec train.py
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

# git_total_commits est exclu par défaut : il fuite vers le label (voir section 8)
FEATURES_WITH_COMMITS = SAFE_FEATURES + ["git_total_commits"]

# Regroupements pour adapter le type de graphe
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

COLORS = {"safe": "#2196F3", "risk": "#F44336", "neutral": "#607D8B"}
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


# Petits helpers réutilisés un peu partout
def banner(title: str):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def grid(n: int, ncols: int = 4, cell_w: float = 5, cell_h: float = 4):
    """Crée une grille de subplots pour n graphes et renvoie les axes à plat."""
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell_w * ncols, cell_h * nrows))
    return fig, np.atleast_1d(axes).flatten()


def hide_unused(axes, used: int):
    """Cache les cases vides de la grille (quand n features < n cases)."""
    for ax in axes[used:]:
        ax.set_visible(False)


def save_fig(fig, outdir: str, name: str):
    path = os.path.join(outdir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {name}.png")


class EDAReport:
    """Collecte les findings au fil des sections pour le résumé final."""

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
                for item in items or ["(aucun)"]:
                    f.write(f"  • {item}\n")
                f.write("\n")
        print(f"\n  Résumé écrit -> {path}")


# Vue d'ensemble
def section_overview(df: pd.DataFrame, feats: list, outdir: str, report: EDAReport):
    banner("1. VUE D'ENSEMBLE")

    print(f"   Shape    : {df.shape}")
    print(f"   Colonnes : {len(df.columns)}")
    print(f"   Features : {len(feats)} (sur {len(FEATURES_WITH_COMMITS)} définies)")

    missing = df[feats].isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print("\n   Valeurs manquantes :")
        for col, n in missing.items():
            pct = round(100 * n / len(df), 1)
            print(f"      {col}: {n} ({pct}%)")
            report.add("warnings", f"Colonne '{col}' a {pct}% de valeurs manquantes")
    else:
        print("   Valeurs manquantes : aucune")
        report.add("findings", "Aucune valeur manquante dans les features")

    df[feats].describe().T.to_csv(os.path.join(outdir, "describe.csv"))
    print("\n   describe() complet -> describe.csv")

    # -1.0 sert de valeur de remplissage dans train.py on regarde combien il y en a
    sentinels = (df[feats] == -1.0).sum()
    sentinels = sentinels[sentinels > 0]
    if len(sentinels) > 0:
        print("\n   Valeurs sentinelles (-1.0) :")
        for col, n in sentinels.items():
            pct = 100 * n / len(df)
            print(f"      {col}: {n} ({pct:.1f}%)")
            if pct > 20:
                report.add("warnings",
                    f"'{col}' a {pct:.0f}% de valeurs à -1.0 (sentinelle) — "
                    "impact potentiel sur le modèle")


# Déséquilibre de classes
def section_class_balance(df: pd.DataFrame, outdir: str, report: EDAReport):
    banner("2. DÉSÉQUILIBRE DE CLASSES")

    counts = df[LABEL_COL].value_counts().sort_index()
    ratio = counts[1] / counts[0] if counts[0] > 0 else float("inf")
    print(f"   label=0 : {counts[0]}  ({100 * counts[0] / len(df):.1f}%)")
    print(f"   label=1 : {counts[1]}  ({100 * counts[1] / len(df):.1f}%)")
    print(f"   Ratio 1/0 : {ratio:.3f}")

    if ratio < 0.2 or ratio > 5:
        report.add("warnings",
            f"Fort déséquilibre de classes (ratio={ratio:.2f}). "
            "class_weight='balanced' est déjà activé dans train.py, mais bien "
            "surveiller les métriques sur la classe minoritaire.")
    else:
        report.add("findings", f"Déséquilibre modéré (ratio={ratio:.2f})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    bars = axes[0].bar(["safe (0)", "risque (1)"], counts.values,
                       color=[COLORS["safe"], COLORS["risk"]],
                       edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + len(df) * 0.01,
                     f"{val}\n({100 * val / len(df):.1f}%)", ha="center", fontweight="bold")
    axes[0].set_ylabel("Nombre de méthodes")
    axes[0].set_title("Distribution du label")

    # Taux de risque par profil — repère les profils trop "déterministes"
    ax = axes[1]
    if "profile" in df.columns:
        profile_risk = (df.groupby("profile")[LABEL_COL]
                          .agg(["mean", "count"])
                          .sort_values("mean"))
        colors = [COLORS["risk"] if m > 0.5 else COLORS["safe"] for m in profile_risk["mean"]]
        ax.barh(profile_risk.index, profile_risk["mean"], color=colors,
                edgecolor="black", linewidth=0.5)
        ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="seuil 50%")
        ax.set_xlabel("Taux de risque")
        ax.set_title("Taux de risque par profil")
        ax.legend()

        # Un profil 100% risk ou 100% safe = le modèle peut juste mémoriser le profil
        for prof, row in profile_risk.iterrows():
            if row["count"] < 3:
                continue
            if row["mean"] == 1.0:
                report.add("warnings",
                    f"Profil '{prof}' est 100% à risque ({int(row['count'])} méthodes) — "
                    "le modèle pourrait juste mémoriser le profil")
            elif row["mean"] == 0.0:
                report.add("findings",
                    f"Profil '{prof}' est 100% safe ({int(row['count'])} méthodes)")
    else:
        ax.text(0.5, 0.5, "Colonne 'profile' absente", transform=ax.transAxes,
                ha="center", va="center")

    fig.suptitle("Analyse du déséquilibre de classes", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, outdir, "02_class_balance")


# Distributions univariées
def section_distributions(df: pd.DataFrame, feats: list, outdir: str, report: EDAReport):
    banner("3. DISTRIBUTIONS UNIVARIÉES")

    cont_feats = [f for f in CONTINUOUS_FEATURES if f in feats]
    if not cont_feats:
        print("   Aucune feature continue trouvée")
        return

    fig, axes = grid(len(cont_feats))
    df_safe = df[df[LABEL_COL] == 0]
    df_risk = df[df[LABEL_COL] == 1]

    for ax, feat in zip(axes, cont_feats):
        ax.hist(df_safe[feat].dropna(), bins=30, alpha=0.6,
                color=COLORS["safe"], label="safe", edgecolor="black", linewidth=0.3)
        ax.hist(df_risk[feat].dropna(), bins=30, alpha=0.6,
                color=COLORS["risk"], label="risque", edgecolor="black", linewidth=0.3)
        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)

        # Outliers très éloignés (règle des 3*IQR plutôt que 1.5 pour ne garder que les extrêmes)
        q1, q3 = df[feat].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_outliers = ((df[feat] < q1 - 3 * iqr) | (df[feat] > q3 + 3 * iqr)).sum()
        if n_outliers > 0:
            pct = 100 * n_outliers / len(df)
            ax.set_xlabel(f"({n_outliers} outliers extrêmes)", fontsize=7, color="red")
            if pct > 5:
                report.add("warnings", f"'{feat}' a {n_outliers} outliers extrêmes ({pct:.1f}%)")

    hide_unused(axes, len(cont_feats))
    fig.suptitle("Distributions par feature (colorées par label)", fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, outdir, "03_distributions")


# Boxplots par label
def section_boxplots(df: pd.DataFrame, feats: list, outdir: str, report: EDAReport):
    banner("4. BOXPLOTS PAR LABEL")

    cont_feats = [f for f in CONTINUOUS_FEATURES if f in feats]
    if not cont_feats:
        return

    fig, axes = grid(len(cont_feats))
    for ax, feat in zip(axes, cont_feats):
        safe = df[df[LABEL_COL] == 0][feat].dropna()
        risk = df[df[LABEL_COL] == 1][feat].dropna()
        bp = ax.boxplot([safe, risk], tick_labels=["safe", "risque"], patch_artist=True,
                        widths=0.6, flierprops=dict(markersize=3, alpha=0.4))
        for box, color in zip(bp["boxes"], [COLORS["safe"], COLORS["risk"]]):
            box.set_facecolor(color)
            box.set_alpha(0.6)
        ax.set_title(feat, fontsize=9, fontweight="bold")

        # Mann-Whitney : test non paramétrique, on ne suppose pas la normalité
        if len(safe) > 5 and len(risk) > 5:
            _, pval = stats.mannwhitneyu(safe, risk, alternative="two-sided")
            if pval < 0.01:
                ax.set_xlabel(f"p={pval:.2e} ***", fontsize=7, color="green")
            elif pval < 0.05:
                ax.set_xlabel(f"p={pval:.3f} *", fontsize=7, color="orange")
            else:
                ax.set_xlabel(f"p={pval:.3f} (ns)", fontsize=7, color="gray")
                report.add("findings",
                    f"'{feat}' ne discrimine pas significativement le label (p={pval:.3f})")

    hide_unused(axes, len(cont_feats))
    fig.suptitle("Boxplots par label (Mann-Whitney U test)", fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, outdir, "04_boxplots")


# Corrélations
def section_correlations(df: pd.DataFrame, feats: list, outdir: str, report: EDAReport):
    banner("5. CORRÉLATIONS")

    corr = df[feats + [LABEL_COL]].corr()

    # Heatmap, triangle inférieur seulement (le reste est redondant)
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, ax=ax,
                annot_kws={"size": 7}, linewidths=0.5)
    ax.set_title("Matrice de corrélation (Pearson)", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, outdir, "05a_correlation_matrix")

    # Corrélation de chaque feature avec le label, triée
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

    # Paires de features trop corrélées entre elles -> redondance
    print("\n   Paires de features très corrélées (|r| > 0.7) :")
    feat_corr = df[feats].corr()
    found = False
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            r = feat_corr.iloc[i, j]
            if abs(r) > 0.7:
                found = True
                print(f"      {feats[i]} ↔ {feats[j]} : r={r:.3f}")
                report.add("warnings",
                    f"Forte corrélation entre '{feats[i]}' et '{feats[j]}' (r={r:.3f}) — "
                    "envisager d'en retirer une")
    if not found:
        print("      (aucune)")
        report.add("findings", "Pas de multicolinéarité forte (|r| > 0.7) entre features")

    weak = label_corr[label_corr.abs() < 0.05]
    if len(weak) > 0:
        report.add("findings",
            f"Features quasi non corrélées au label (|r| < 0.05) : {', '.join(weak.index)}")


# Features à faible variance
def section_low_variance(df: pd.DataFrame, feats: list, outdir: str, report: EDAReport):
    banner("6. FEATURES À FAIBLE VARIANCE / QUASI-CONSTANTES")

    rows = []
    for feat in feats:
        vc = df[feat].value_counts(normalize=True)
        rows.append({
            "feature": feat,
            "n_unique": df[feat].nunique(),
            "dominant_value": vc.index[0],
            "dominant_pct": round(vc.iloc[0] * 100, 1),
            "variance": round(df[feat].var() or 0, 4),
        })

    results_df = pd.DataFrame(rows).sort_values("dominant_pct", ascending=False)
    results_df.to_csv(os.path.join(outdir, "low_variance.csv"), index=False)

    # Une valeur qui domine à 95%+ n'apporte presque aucun signal
    quasi_constant = results_df[results_df["dominant_pct"] >= 95]
    if len(quasi_constant) > 0:
        print("   Features quasi-constantes (>= 95% même valeur) :")
        for _, row in quasi_constant.iterrows():
            print(f"      {row['feature']}: {row['dominant_pct']}% = {row['dominant_value']} "
                  f"({row['n_unique']} valeurs uniques)")
            report.add("warnings",
                f"'{row['feature']}' est quasi-constante "
                f"({row['dominant_pct']}% = {row['dominant_value']}) — candidat à l'exclusion")
    else:
        print("   Aucune feature quasi-constante (>= 95%)")

    fig, ax = plt.subplots(figsize=(10, 6))
    data = results_df.sort_values("dominant_pct")
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


# Features binaires vs label
def section_binary_features(df: pd.DataFrame, feats: list, outdir: str, report: EDAReport):
    banner("7. FEATURES BINAIRES ↔ LABEL")

    bin_feats = [f for f in BINARY_FEATURES if f in feats]
    if not bin_feats:
        print("   Aucune feature binaire trouvée")
        return

    fig, axes = grid(len(bin_feats), cell_w=4)
    for ax, feat in zip(axes, bin_feats):
        # Taux de risque quand la feature vaut 0 vs 1
        rates = df.groupby(feat)[LABEL_COL].mean()
        counts = df[feat].value_counts()
        if 0 not in rates or 1 not in rates:
            ax.set_visible(False)
            continue

        bars = ax.bar([f"0\n(n={counts.get(0, 0)})", f"1\n(n={counts.get(1, 0)})"],
                      [rates[0], rates[1]],
                      color=[COLORS["safe"], COLORS["risk"]],
                      edgecolor="black", linewidth=0.5, alpha=0.7)
        for bar, val in zip(bars, [rates[0], rates[1]]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.1%}", ha="center", fontsize=8, fontweight="bold")
        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.set_ylabel("Taux de risque")
        ax.set_ylim(0, 1)

        # Chi2 seulement si chaque case du tableau de contingence a >= 5 obs
        ct = pd.crosstab(df[feat], df[LABEL_COL])
        if ct.shape == (2, 2) and ct.min().min() >= 5:
            chi2, pval, _, _ = stats.chi2_contingency(ct)
            sig = "***" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            ax.set_xlabel(f"χ²={chi2:.1f}, p={pval:.3f} {sig}", fontsize=7)
            if pval < 0.05 and abs(rates[1] - rates[0]) > 0.1:
                report.add("findings",
                    f"'{feat}' discrimine le label : taux risque = "
                    f"{rates[0]:.1%} (=0) vs {rates[1]:.1%} (=1)")

    hide_unused(axes, len(bin_feats))
    fig.suptitle("Taux de risque par feature binaire (test χ²)", fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, outdir, "07_binary_vs_label")


# Analyse git + leakage
def section_git_analysis(df: pd.DataFrame, outdir: str, report: EDAReport):
    banner("8. ANALYSE GIT + LEAKAGE CHECK")

    if not [c for c in GIT_FEATURES if c in df.columns]:
        print("   Aucune feature git trouvée")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # git_bugfix_count : c'est lui qui sert à construire le label
    if "git_bugfix_count" in df.columns:
        ax = axes[0, 0]
        ax.hist(df["git_bugfix_count"], bins=30, color=COLORS["neutral"],
                edgecolor="black", linewidth=0.3)
        ax.set_title("Distribution git_bugfix_count\n(source du label)", fontweight="bold")
        ax.set_xlabel("Nombre de bugfix commits")

    # Le vrai test de leakage : total_commits est-il corrélé au bugfix_count ?
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
                "confirme le leakage partiel, bien de l'exclure par défaut dans train.py")
        report.add("findings",
            f"Corrélation git_total_commits ↔ git_bugfix_count : r={r:.3f}")

    if "git_nb_authors" in df.columns:
        ax = axes[1, 0]
        for label, color, name in [(0, COLORS["safe"], "safe"), (1, COLORS["risk"], "risque")]:
            ax.hist(df[df[LABEL_COL] == label]["git_nb_authors"], bins=20,
                    alpha=0.6, color=color, label=name, edgecolor="black", linewidth=0.3)
        ax.set_title("git_nb_authors par label", fontweight="bold")
        ax.legend()

    # git_days_since_change vaut -1 quand le fichier n'a aucun commit dans la fenêtre
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


# VIF (multicolinéarité)
def section_vif(df: pd.DataFrame, feats: list, outdir: str, report: EDAReport):
    banner("9. VIF (MULTICOLINÉARITÉ)")

    from sklearn.linear_model import LinearRegression

    # VIF_i = 1 / (1 - R²) où R² vient de la régression de feature_i sur les autres
    X = df[feats].fillna(-1.0).values
    vifs = []
    for i, feat in enumerate(feats):
        if np.std(X[:, i]) == 0:
            vifs.append({"feature": feat, "VIF": float("inf")})
            continue
        others = np.delete(X, i, axis=1)
        r2 = LinearRegression().fit(others, X[:, i]).score(others, X[:, i])
        vif_val = 1 / (1 - r2) if r2 < 1 else float("inf")
        vifs.append({"feature": feat, "VIF": round(vif_val, 2)})

    vif_df = pd.DataFrame(vifs).sort_values("VIF", ascending=False)
    vif_df.to_csv(os.path.join(outdir, "vif.csv"), index=False)

    print("   Top VIF :")
    for _, row in vif_df.head(10).iterrows():
        flag = "  (élevé)" if row["VIF"] > 5 else ""
        print(f"      {row['feature']:<30} VIF = {row['VIF']:>8.2f}{flag}")

    high_vif = vif_df[vif_df["VIF"] > 10]
    moderate_vif = vif_df[(vif_df["VIF"] > 5) & (vif_df["VIF"] <= 10)]
    if len(high_vif) > 0:
        report.add("warnings",
            f"VIF > 10 (multicolinéarité forte) : {', '.join(high_vif['feature'])}. "
            "Peu d'impact sur RF/XGB, mais dégrade la LogisticRegression.")
    elif len(moderate_vif) > 0:
        report.add("findings", f"VIF modéré (5-10) : {', '.join(moderate_vif['feature'])}")
    else:
        report.add("findings", "Pas de multicolinéarité problématique (tous VIF < 5)")

    fig, ax = plt.subplots(figsize=(10, 7))
    data = vif_df.sort_values("VIF")
    colors = ["#F44336" if v > 10 else "#FF9800" if v > 5 else "#4CAF50" for v in data["VIF"]]
    ax.barh(data["feature"], data["VIF"], color=colors, edgecolor="black", linewidth=0.3)
    ax.axvline(5, color="orange", linestyle="--", alpha=0.5, label="VIF = 5")
    ax.axvline(10, color="red", linestyle="--", alpha=0.5, label="VIF = 10")
    ax.set_xlabel("Variance Inflation Factor")
    ax.set_title("VIF par feature", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, outdir, "09_vif")


# Recommandations
def section_recommendations(df: pd.DataFrame, feats: list, report: EDAReport):
    banner("10. RECOMMANDATIONS AUTOMATIQUES")

    ratio = df[LABEL_COL].mean()
    if ratio < 0.1 or ratio > 0.9:
        report.add("recommendations",
            "Envisager SMOTE ou un sous-échantillonnage en plus de class_weight='balanced'")

    n, n_feats = len(df), len(feats)
    if n < 10 * n_feats:
        report.add("recommendations",
            f"Dataset petit ({n} lignes pour {n_feats} features, ratio={n / n_feats:.0f}). "
            "Risque d'overfitting — la RepeatedStratifiedKFold de train.py aide là-dessus.")

    report.add("recommendations",
        "Relancer train.py et comparer les métriques avec/sans les features flaggées")
    report.add("recommendations",
        "Pipeline : extract_dataset.py -> eda.py -> ajustements features -> "
        "train.py -> streamlit_app.py")

    for rec in report.recommendations:
        print(f"   -> {rec}")


# Point d'entrée
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="CSV produit par extract_dataset.py")
    ap.add_argument("--output-dir", default="eda_report",
                    help="Dossier de sortie pour les graphiques et le rapport")
    ap.add_argument("--include-total-commits", action="store_true",
                    help="Inclure git_total_commits dans l'analyse (attention au leakage)")
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    report = EDAReport()

    banner("  EDA — Analyse exploratoire du dataset de risque")
    df = pd.read_csv(args.dataset)
    print(f"   Fichier : {args.dataset}")
    print(f"   Shape   : {df.shape}")

    # On ne garde que les features réellement présentes dans le CSV
    feature_list = FEATURES_WITH_COMMITS if args.include_total_commits else SAFE_FEATURES
    feats = [c for c in feature_list if c in df.columns]
    if missing := set(feature_list) - set(feats):
        print(f"   Colonnes manquantes : {missing}")

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

    report.dump(os.path.join(args.output_dir, "eda_summary.txt"))

    banner(f"  EDA terminé — {len(os.listdir(args.output_dir))} fichiers dans {args.output_dir}/")
    print("   Fichiers produits :")
    for f in sorted(os.listdir(args.output_dir)):
        print(f"     - {f}")
    print(f"\n   Prochaine étape : python train.py {args.dataset}")


if __name__ == "__main__":
    main()