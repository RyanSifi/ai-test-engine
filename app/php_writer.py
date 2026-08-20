"""
Écriture sécurisée de fichiers PHP générés dans le workspace, et validation
de syntaxe via `php -l`.

Trois garanties assurées ici :

1. **Confinement** — `_safe_join()` interdit toute écriture hors du workspace
   (path traversal, chemin absolu, remontées `..`).
2. **Non-destruction** — un fichier écrit à la main n'est jamais perdu :
   `_write_php_file()` refuse d'écraser par défaut, et sauvegarde sinon.
3. **Atomicité** — l'écriture passe par un fichier temporaire renommé à la fin.
   Une interruption (timeout, arrêt du conteneur) ne laisse jamais un PHP tronqué
   à la place d'un fichier valide.
"""
import os
import re
import shutil
import subprocess
import logging
import tempfile
import time
from typing import Optional
from fastapi import HTTPException
from config import settings
from prompt_safety import scan_generated_code, format_findings

# Marqueur inséré en tête de chaque fichier produit par le moteur. Il sert à
# distinguer « fichier généré, réécrasable » de « fichier écrit par un humain,
# à ne pas détruire » — cf. _is_generated_file().
GENERATED_MARKER = "@ai-test-engine-generated"

# Suffixe des sauvegardes créées quand on écrase un fichier non généré.
BACKUP_SUFFIX = ".bak"

# Taille lue pour chercher le marqueur. Il est dans l'en-tête du fichier ; lire
# un test de plusieurs milliers de lignes en entier serait inutile.
MARKER_SCAN_BYTES = 2048


def validate_php_syntax(code: str) -> Optional[str]:
    """Valide la syntaxe PHP. Retourne le message d'erreur ou None si OK."""
    try:
        proc = subprocess.Popen(
            ["php", "-l"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        stdout, stderr = proc.communicate(input=code)
        return (stderr or stdout).strip() if proc.returncode != 0 else None
    except FileNotFoundError:
        logging.warning("PHP non disponible — validation de syntaxe ignorée.")
        return None


def _safe_join(base: str, *parts: str) -> str:
    """
    Joint et résout un chemin relatif sous `base`.
    Lève HTTPException(400) si le résultat sort de `base` (path traversal).
    """
    base_real = os.path.realpath(base)
    # L'antislash est un séparateur sous Windows mais un caractère de nom de
    # fichier ordinaire sous Linux, où tourne le conteneur. Sans cette
    # normalisation, « ..\..\etc\passwd » ne serait pas vu comme une remontée :
    # il produirait un fichier au nom aberrant, certes confiné dans le workspace,
    # mais impossible à récupérer sur un poste Windows. On traite donc les deux
    # séparateurs de la même façon, quelle que soit la plateforme.
    parts = [p.replace("\\", "/") for p in parts]
    candidate = os.path.realpath(os.path.join(base_real, *parts))
    # Comparaison avec un séparateur final pour empêcher /workspace-evil de matcher /workspace
    if candidate != base_real and not candidate.startswith(base_real + os.sep):
        raise HTTPException(
            status_code=400,
            detail=f"Chemin invalide (hors workspace) : {os.path.join(*parts)}",
        )
    return candidate


def _sanitize_path_component(name: str) -> str:
    """Garde uniquement les caractères sûrs pour un nom de fichier ou dossier."""
    return re.sub(r"[^a-zA-Z0-9_-]", "", name) or "Unnamed"


def _is_generated_file(path: str) -> bool:
    """
    Vrai si le fichier porte le marqueur du moteur — donc s'il a été produit par
    une génération précédente et peut être remplacé sans rien détruire.

    On ne lit que le début du fichier : le marqueur est dans l'en-tête, et un
    fichier de test peut être volumineux.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return GENERATED_MARKER in f.read(MARKER_SCAN_BYTES)
    except OSError:
        return False


def _stamp(code: str) -> str:
    """
    Insère le marqueur de génération juste après l'ouverture `<?php`.

    Placé là (et non en tête de fichier) pour rester du PHP valide : un
    commentaire avant `<?php` serait envoyé tel quel au navigateur.
    """
    if GENERATED_MARKER in code:
        return code
    banniere = (
        f"\n/**\n"
        f" * {GENERATED_MARKER}\n"
        f" *\n"
        f" * Fichier produit automatiquement par AI Test Engine — BROUILLON à relire.\n"
        f" * Généré le {time.strftime('%d/%m/%Y à %H:%M')}.\n"
        f" *\n"
        f" * Toute regénération de ce test ÉCRASERA ce fichier. Si vous le modifiez à la\n"
        f" * main et souhaitez le conserver, supprimez la ligne {GENERATED_MARKER}\n"
        f" * ci-dessus : le moteur refusera alors de l'écraser.\n"
        f" */\n"
    )
    if code.lstrip().startswith("<?php"):
        i = code.index("<?php") + len("<?php")
        return code[:i] + banniere + code[i:]
    return f"<?php{banniere}\n{code}"


def _write_php_file(relative_path: str, code: str, overwrite: bool = False,
                    allow_unsafe: bool = False) -> str:
    """
    Écrit un fichier PHP dans le workspace et retourne son chemin absolu.

    **Contrôle du contenu** — le code vient d'un modèle de langage nourri avec le
    code source du projet analysé. Avant écriture, il est relu par
    `scan_generated_code()` : exécution de commandes, évaluation dynamique,
    appels réseau sortants ou suppression de fichiers font échouer l'écriture
    (`HTTPException 422`). C'est la seule barrière qui ne dépend pas du bon
    vouloir du modèle — voir prompt_safety.py.

    `allow_unsafe=True` court-circuite ce contrôle. Réservé aux cas où un motif
    interdit est un faux positif avéré ; à n'utiliser qu'après avoir lu le code.

    Protection contre la perte de travail (le moteur écrit dans `tests/`, où se
    trouvent aussi les tests écrits à la main) :

    - fichier **absent** → écriture directe ;
    - fichier **déjà généré** par le moteur (marqueur présent) → remplacé, c'est
      le cas nominal d'une regénération ;
    - fichier **écrit à la main** → refus (`HTTPException 409`) sauf
      `overwrite=True`, et dans ce cas une sauvegarde `.bak` est créée d'abord.

    L'écriture est **atomique** : on écrit dans un fichier temporaire du même
    dossier, puis `os.replace()`. Sur les deux plateformes cette opération est
    atomique, donc une interruption laisse soit l'ancien fichier intact, soit le
    nouveau complet — jamais un PHP à moitié écrit.
    """
    if not allow_unsafe:
        verdict = scan_generated_code(code)
        for avertissement in verdict["warnings"]:
            logging.warning(
                f"[write] '{relative_path}' — construction suspecte "
                f"ligne {avertissement['line']} : {avertissement['reason']}"
            )
        if not verdict["safe"]:
            details = format_findings(verdict["blocking"])
            logging.error(f"[write] '{relative_path}' REFUSÉ — {details}")
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Le code généré contient des constructions interdites dans un "
                    f"test et n'a pas été écrit : {details}. "
                    f"Un test appelle un contrôleur et pose des assertions — il "
                    f"n'exécute pas de commandes et ne contacte pas l'extérieur. "
                    f"Cela peut signaler une injection de prompt via un commentaire "
                    f"du code analysé. Relancez la génération ; si le motif est un "
                    f"faux positif avéré, passez allow_unsafe=true."
                ),
            )

    full_path = _safe_join(settings.container_project_root, relative_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    code = _stamp(code)

    if os.path.exists(full_path) and not _is_generated_file(full_path):
        if not overwrite:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"'{relative_path}' existe déjà et n'a pas été produit par le moteur "
                    f"(marqueur {GENERATED_MARKER} absent) : il s'agit vraisemblablement "
                    f"d'un test écrit à la main. Écriture refusée pour ne pas le détruire. "
                    f"Renvoyez la requête avec overwrite=true pour le remplacer — une "
                    f"sauvegarde {BACKUP_SUFFIX} sera créée."
                ),
            )
        backup = full_path + BACKUP_SUFFIX
        shutil.copy2(full_path, backup)
        logging.warning(
            f"[write] '{relative_path}' écrit à la main — écrasement demandé, "
            f"sauvegarde : {os.path.basename(backup)}"
        )

    # Écriture atomique : temporaire dans le MÊME dossier (os.replace ne peut pas
    # être atomique d'un système de fichiers à un autre), puis renommage.
    dossier = os.path.dirname(full_path)
    fd, tmp = tempfile.mkstemp(dir=dossier, prefix=".tmp_", suffix=".php")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, full_path)
    except BaseException:
        # Y compris KeyboardInterrupt / SystemExit : on ne laisse jamais de .tmp_ traîner.
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return full_path
