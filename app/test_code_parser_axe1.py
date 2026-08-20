"""
Adaptateur pytest pour check_code_parser_axe1.py.

check_code_parser_axe1.py est un exécuteur autonome : il lance ses ~45
vérifications au niveau module puis appelle sys.exit(). Un tel fichier ne peut
pas être collecté par pytest (le sys.exit() casse la collecte et empêche TOUS
les tests du dossier de tourner). On l'exécute donc en sous-processus et on
vérifie son code de retour.

Avantage annexe : un plantage du script (PHP absent, vendor/ manquant) devient
un échec de test lisible au lieu d'une erreur de collecte pytest.
"""
import os
import subprocess
import sys

import pytest

_SCRIPT = os.path.join(os.path.dirname(__file__), "check_code_parser_axe1.py")


def _php_available() -> bool:
    try:
        subprocess.run(["php", "--version"], capture_output=True, timeout=10)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.mark.skipif(not _php_available(),
                    reason="php-cli absent — le pont AST ne peut pas tourner")
def test_code_parser_axe1_suite():
    """Les vérifications du parseur AST passent toutes (code de retour 0)."""
    proc = subprocess.run(
        [sys.executable, _SCRIPT],
        capture_output=True, text=True, encoding="utf-8", timeout=300,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    # La sortie du script n'est affichée que si l'assertion échoue.
    assert proc.returncode == 0, (
        f"check_code_parser_axe1.py a échoué (code {proc.returncode})\n\n"
        f"--- stdout ---\n{proc.stdout[-4000:]}\n"
        f"--- stderr ---\n{proc.stderr[-2000:]}"
    )
