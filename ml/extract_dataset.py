"""
Extrait un dataset (features + label de risque de régression) depuis un projet
Symfony, pour entraîner un classifieur supervisé.

Le label est un proxy basé sur l'historique git : une route est dite "à risque"
si le fichier qui la contient a été modifié par >= N commits de bugfix dans
la fenêtre temporelle (défaut : 12 derniers mois). Méthodologie issue de la
littérature en software defect prediction (Hassan 2009, Kamei et al. 2013).

Usage :
    python extract_dataset.py /path/to/symfony/repo output.csv \
        [--since "12 months ago"] [--bugfix-threshold 2]
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
from code_parser import _parse_file_content, _read_php_file  # noqa: E402


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


def get_git_metrics(file_path: str, repo_path: str, since: str) -> Dict:
    """Métriques git pour un fichier dans la fenêtre temporelle."""
    rel = os.path.relpath(file_path, repo_path)
    log = git(
        ["log", f"--since={since}", "--format=%s%x09%ae%x09%ct", "--", rel],
        repo_path,
    )
    bugfix = total = authors_set_len = 0
    last_ts = None
    authors = set()
    for line in log.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        msg, email, ts = parts[0], parts[1], parts[2]
        total += 1
        authors.add(email)
        if BUGFIX_RE.search(msg):
            bugfix += 1
        try:
            ts_int = int(ts)
            if last_ts is None or ts_int > last_ts:
                last_ts = ts_int
        except ValueError:
            pass
    days_since = (time.time() - last_ts) / 86400 if last_ts else None
    return {
        "bugfix_count": bugfix,
        "total_commits": total,
        "nb_authors": len(authors),
        "days_since_change": round(days_since, 1) if days_since is not None else -1.0,
    }


def cyclomatic_complexity(text: Optional[str]) -> int:
    """Approximation rapide de la complexité cyclomatique."""
    if not text:
        return 1
    return 1 + len(COMPLEXITY_RE.findall(text))


def extract_method_body(content: str, method_name: str) -> str:
    """Récupère le corps d'une méthode (best effort)."""
    pattern = re.compile(
        r"function\s+" + re.escape(method_name) + r"\s*\([^)]*\)[^{]*\{",
        re.DOTALL,
    )
    m = pattern.search(content)
    if not m:
        return ""
    # Compteur de braces équilibrées
    depth = 1
    start = m.end()
    i = start
    while i < len(content) and depth > 0:
        if content[i] == "{":
            depth += 1
        elif content[i] == "}":
            depth -= 1
        i += 1
    return content[start:i - 1]


def extract_features(file_path: str, repo_path: str, since: str,
                     bugfix_threshold: int) -> List[Dict]:
    """
    Une ligne par (controller, méthode).
    Granularité méthode plutôt que route, car beaucoup de projets Symfony
    déclarent leurs routes en YAML hors du code (Sylius, Akeneo).
    """
    try:
        content = _read_php_file(file_path)
        parsed = _parse_file_content(content)
    except Exception as e:
        logging.warning(f"Skip {file_path}: {e}")
        return []
    if not parsed.get("class_name"):
        return []

    git_metrics = get_git_metrics(file_path, repo_path, since)
    rel_path = os.path.relpath(file_path, repo_path).replace("\\", "/")
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
    parser.add_argument("--bugfix-threshold", type=int, default=2,
                        help="Seuil min de bugfix pour label_risk=1 (défaut: 2)")
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

    all_rows = []
    t0 = time.time()
    for i, c in enumerate(controllers):
        if i % 25 == 0:
            elapsed = time.time() - t0
            print(f"  [{i:>4}/{len(controllers)}] {elapsed:>5.0f}s — {os.path.basename(c)}")
        rows = extract_features(c, repo_path, args.since, args.bugfix_threshold)
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
    print(f"   Features     : {len(fieldnames) - 7} (hors identifiants et label)")


if __name__ == "__main__":
    main()
