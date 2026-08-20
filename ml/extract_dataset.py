"""
Extrait un dataset (features + label de risque de régression) depuis un projet
Symfony, pour entraîner un classifieur supervisé.

Le label est un proxy basé sur l'historique git : une route est dite "à risque"
si le fichier qui la contient a été modifié par >= N commits de bugfix dans
la fenêtre temporelle (défaut : 12 derniers mois). Méthodologie issue de la
littérature en software defect prediction (Hassan 2009, Kamei et al. 2013).

Usage :
    python extract_dataset.py /path/to/symfony/repo output.csv \
        [--since "12 months ago"] [--bugfix-threshold 5]
"""
import argparse
import csv
import logging
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional

# Le parser PHP est partagé avec l'AI Test Engine
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from code_parser import _extract_balanced_block, _parse_file_content, _read_php_file  # noqa: E402


# Patterns commits "bugfix"
BUGFIX_PATTERNS = [
    r"^\s*\[?(?:fix|bugfix|hotfix|patch|correctif)",
    r"^\s*fixes?\s*[:#]",
    r"^\s*corrige",
    r"\[fix\]",
    r"#\d+\s+fix",
    r"\bbug\s*fix",
]
BUGFIX_RE = re.compile("|".join(BUGFIX_PATTERNS), re.IGNORECASE)

# Mots-clés de complexité cyclomatique (approximation)
COMPLEXITY_KEYWORDS = [
    r"\bif\b", r"\belseif\b", r"\belse\s+if\b", r"\bwhile\b",
    r"\bfor\b", r"\bforeach\b", r"\bcase\b", r"\bcatch\b",
    r"&&", r"\|\|", r"\?",
]
COMPLEXITY_RE = re.compile("|".join(COMPLEXITY_KEYWORDS))

# Colonnes du CSV qui ne sont PAS des features : traçabilité (pour retrouver la
# méthode d'origine et pour grouper le split par fichier) et cible.
IDENTITY_COLS = ("file", "class", "method", "profile")
LABEL_COL = "label_risk"


def git(args: List[str], cwd: str) -> str:
    """Wrapper git avec gestion d'erreur silencieuse."""
    try:
        return subprocess.check_output(
            ["git"] + args,
            cwd=cwd,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return ""


def list_controllers(repo_path: str) -> List[str]:
    """
    Trouve tous les fichiers PHP qui ressemblent à des controllers ou actions.
    Inclut :
    - *Controller.php (convention classique)
    - *Action.php (single-action controllers, Sylius style)
    - Fichiers dans /Controller/ ou /Action/
    """
    controllers = []
    skip_parts = ("vendor", ".git", "var", "node_modules", "Tests", "tests",
                  "Spec", "spec", "Fixtures", "fixtures")
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in skip_parts]
        for f in files:
            if not f.endswith(".php"):
                continue
            if f.endswith("Controller.php") or f.endswith("Action.php"):
                controllers.append(os.path.join(root, f))
                continue
            # Fichiers dans un dossier /Controller/ ou /Action/
            parts = root.replace("\\", "/").split("/")
            if "Controller" in parts or "Action" in parts:
                controllers.append(os.path.join(root, f))
    return sorted(set(controllers))


def get_git_metrics_bulk(repo_path: str, since: str) -> Dict[str, Dict]:
    """
    Calcule les métriques git pour TOUS les fichiers du repo en un seul appel
    `git log`, au lieu d'un appel par fichier (qui scale mal : 150-700+ process
    git séparés selon le projet, cf. cross_eval.py qui compare sucre/sylius/akeneo).

    Utilise `--name-only` pour associer chaque commit aux fichiers qu'il touche.
    Format de sortie par commit :
        <ligne d'en-tête : sujet\tem@il\ttimestamp>
        <ligne vide>
        fichier1.php
        fichier2.php
        <ligne vide séparant du commit suivant>

    Retourne {chemin_relatif (slashes, posix) : {bugfix_count, total_commits,
    nb_authors, days_since_change}}.
    """
    # Séparateur de commit explicite (\x00) pour ne pas confondre une ligne
    # vide de fin de commit avec une ligne vide de séparation de fichiers.
    raw = git(
        ["log", f"--since={since}", "--name-only",
         "--format=%x00%s%x09%ae%x09%ct"],
        repo_path,
    )

    now = time.time()
    # file_rel -> {"bugfix": int, "total": int, "authors": set, "last_ts": int|None}
    acc: Dict[str, Dict] = {}

    commits = raw.split("\x00")
    for commit_block in commits:
        if not commit_block.strip():
            continue
        lines = commit_block.splitlines()
        header = lines[0]
        parts = header.split("\t")
        if len(parts) < 3:
            continue
        msg, email, ts = parts[0], parts[1], parts[2]
        try:
            ts_int = int(ts)
        except ValueError:
            continue
        is_bugfix = bool(BUGFIX_RE.search(msg))

        # Fichiers touchés par ce commit (lignes suivantes, jusqu'à la fin du bloc)
        files_touched = [l.strip().replace("\\", "/") for l in lines[1:] if l.strip()]
        for f in files_touched:
            entry = acc.setdefault(f, {"bugfix": 0, "total": 0, "authors": set(), "last_ts": None})
            entry["total"] += 1
            entry["authors"].add(email)
            if is_bugfix:
                entry["bugfix"] += 1
            if entry["last_ts"] is None or ts_int > entry["last_ts"]:
                entry["last_ts"] = ts_int

    result: Dict[str, Dict] = {}
    for f, entry in acc.items():
        days_since = (now - entry["last_ts"]) / 86400 if entry["last_ts"] else None
        result[f] = {
            "bugfix_count": entry["bugfix"],
            "total_commits": entry["total"],
            "nb_authors": len(entry["authors"]),
            "days_since_change": round(days_since, 1) if days_since is not None else -1.0,
        }
    return result


_EMPTY_GIT_METRICS = {
    "bugfix_count": 0, "total_commits": 0, "nb_authors": 0, "days_since_change": -1.0,
}


def cyclomatic_complexity(text: Optional[str]) -> int:
    """Approximation rapide de la complexité cyclomatique."""
    if not text:
        return 1
    return 1 + len(COMPLEXITY_RE.findall(text))


def extract_method_body(content: str, method_name: str) -> str:
    """Récupère le corps d'une méthode (best effort).

    Localise le '{' d'ouverture par regex, puis délègue le comptage
    d'accolades à _extract_balanced_block() (code_parser.py) qui gère
    strings et commentaires PHP, plutôt qu'un compteur naïf qui coupe
    au mauvais endroit si '{' ou '}' apparaît dans une string/commentaire.
    """
    pattern = re.compile(
        r"function\s+" + re.escape(method_name) + r"\s*\([^)]*\)[^{]*\{",
        re.DOTALL,
    )
    m = pattern.search(content)
    if not m:
        return ""
    block = _extract_balanced_block(content, m.end() - 1)
    if block is None:
        return ""
    return block[1:-1]


def extract_features(file_path: str, repo_path: str, git_metrics_by_file: Dict[str, Dict],
                     bugfix_threshold: int) -> List[Dict]:
    """
    Une ligne par (controller, méthode).
    Granularité méthode plutôt que route, car beaucoup de projets Symfony
    déclarent leurs routes en YAML hors du code (Sylius, Akeneo).

    git_metrics_by_file cache pré-calculé par get_git_metrics_bulk(), un seul
    appel git pour tout le repo au lieu d'un appel par fichier ici.
    """
    try:
        content = _read_php_file(file_path)
        parsed = _parse_file_content(content)
    except Exception as e:
        logging.warning(f"Skip {file_path}: {e}")
        return []
    if not parsed.get("class_name"):
        return []

    rel_path = os.path.relpath(file_path, repo_path).replace("\\", "/")
    git_metrics = git_metrics_by_file.get(rel_path, _EMPTY_GIT_METRICS)
    file_loc = content.count("\n") + 1

    rows = []
    for method in parsed.get("methods", []):
        if method["name"] == "__construct":
            continue
        body = extract_method_body(content, method["name"])
        method_loc = body.count("\n") + 1 if body else 0
        complexity = cyclomatic_complexity(body)
        routes = method.get("routes", [])
        response_types = method.get("response_types", [])

        row = {
            # Identifiants
            "file": rel_path,
            "class": parsed["class_name"],
            "method": method["name"],
            "profile": parsed.get("controller_profile", "unknown"),

            # Features de routes (zéro si pas de #[Route] dans le code)
            "nb_routes_method": len(routes),
            "has_route_attr": int(bool(routes)),

            # Features structurelles (extraites du parser)
            "nb_params": len(method.get("params", [])),
            "has_form": int(bool(method.get("has_form", False))),
            "is_ajax_only": int(bool(method.get("is_ajax_only", False))),
            "nb_voter_checks": len(method.get("voter_checks", [])),
            "nb_response_types": len(response_types),
            "has_render": int(any("render" in r for r in response_types)),
            "has_redirect": int(any("redirect" in r for r in response_types)),
            "has_json": int(any("json" in r for r in response_types)),
            "has_file_download": int(any(
                "file_download" in r or "binary" in r for r in response_types
            )),
            "nb_method_grants": len(method.get("method_grants", [])),
            "nb_class_grants": len(parsed.get("class_grants", [])),
            "nb_constructor_deps": len(parsed.get("constructor_params", [])),
            "nb_methods_in_class": len(parsed.get("methods", [])),
            "is_invoke": int(method["name"] == "__invoke"),

            # Features de complexité / taille
            "file_loc": file_loc,
            "method_loc": method_loc,
            "cyclomatic_complexity": complexity,

            # Features git (au niveau fichier)
            "git_bugfix_count": git_metrics["bugfix_count"],
            "git_total_commits": git_metrics["total_commits"],
            "git_nb_authors": git_metrics["nb_authors"],
            "git_days_since_change": git_metrics["days_since_change"],

            # LABEL
            # 1 = à risque (≥ threshold bugfix dans la fenêtre)
            "label_risk": int(git_metrics["bugfix_count"] >= bugfix_threshold),
        }
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_path", help="Chemin du repo Symfony")
    parser.add_argument("output", help="Fichier CSV de sortie")
    parser.add_argument("--since", default="12 months ago",
                        help='Fenêtre git (ex: "12 months ago")')
    parser.add_argument("--bugfix-threshold", type=int, default=5,
                        help="Seuil min de bugfix pour label_risk=1 (défaut: 5). "
                             "Un seuil trop bas rend le label dégénéré (à 2, SUCRE a "
                             "82%% de positifs et le modèle ne priorise plus rien) ; "
                             "trop haut, il ne reste plus assez de fichiers positifs "
                             "pour entraîner. Voir docs/analyse-resultats-ml.md.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    repo_path = os.path.abspath(args.repo_path)
    if not os.path.isdir(os.path.join(repo_path, ".git")):
        sys.exit(f"{repo_path} n'est pas un repo git.")

    print(f"Repo : {repo_path}")
    print(f"Fenêtre git : {args.since}")
    print(f"Seuil bugfix pour label_risk=1 : ≥ {args.bugfix_threshold}")
    print()

    controllers = list_controllers(repo_path)
    print(f"Trouvé {len(controllers)} fichiers Controller.php")
    if not controllers:
        sys.exit("Aucun controller trouvé.")

    print("Calcul des métriques git (un seul passage sur tout l'historique)...")
    t_git = time.time()
    git_metrics_by_file = get_git_metrics_bulk(repo_path, args.since)
    print(f"  {len(git_metrics_by_file)} fichiers avec historique git ({time.time() - t_git:.1f}s)")

    all_rows = []
    t0 = time.time()
    for i, c in enumerate(controllers):
        if i % 25 == 0:
            elapsed = time.time() - t0
            print(f"  [{i:>4}/{len(controllers)}] {elapsed:>5.0f}s — {os.path.basename(c)}")
        rows = extract_features(c, repo_path, git_metrics_by_file, args.bugfix_threshold)
        all_rows.extend(rows)

    if not all_rows:
        sys.exit("Aucune ligne extraite (pas de routes trouvées dans les controllers ?).")

    fieldnames = list(all_rows[0].keys())
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    n_pos = sum(1 for r in all_rows if r["label_risk"] == 1)
    elapsed = time.time() - t0
    print(f"\n {len(all_rows)} lignes écrites → {args.output}  ({elapsed:.1f}s)")
    print(f"   label_risk=1 : {n_pos:>4}  ({100 * n_pos / len(all_rows):.1f}%)")
    print(f"   label_risk=0 : {len(all_rows) - n_pos:>4}")
    n_features = len([c for c in fieldnames if c not in IDENTITY_COLS and c != LABEL_COL])
    print(f"   Features     : {n_features} (hors identifiants et label)")


if __name__ == "__main__":
    main()
