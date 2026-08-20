"""
Helpers de mise en forme et de filtrage des chunks RAG pour le prompt LLM
(contexte, résumé des routes, vérification de couverture post-génération).

Ce module centralise aussi les **paramètres de récupération** (§ ci-dessous) :
combien de chunks aller chercher, lesquels garder. Ils étaient auparavant écrits
en dur dans `main.py`, ce qui rendait impossible de les ajuster sans relire le
code des endpoints.
"""
import re
from typing import Dict, List

from chunk_format import (
    _RE_HTTP_VERBS, _RE_AJAX_ONLY, _RE_FORM_TYPE, _RE_METHOD_ROLE, _RE_VOTER,
)
from prompt_safety import wrap_untrusted_context

# ─────────────────────────────────────────────────────────────────────────────
# PARAMÈTRES DE RÉCUPÉRATION (RAG)
#
# Ce que ces nombres arbitrent : plus de chunks = plus de contexte pour le
# modèle, mais aussi plus de tokens consommés et plus de bruit. Le modèle
# déployé (qwen2.5-coder:3b) a une fenêtre de 8192 tokens ; au-delà de ~6500
# tokens de prompt, ses réponses commencent à être tronquées.
#
# ⚠️ Portée réelle, à connaître avant de les ajuster : depuis le passage à la
# génération PAR ROUTE (le mode par défaut), chaque appel au modèle ne reçoit
# que **le chunk de la classe + le chunk d'UNE route**. Ces limites ne pilotent
# donc plus la taille des prompts — elles déterminent surtout **combien de
# routes seront couvertes**. Les augmenter allonge la génération (un appel LLM
# de plus par route) sans dégrader la qualité de chaque test.
# ─────────────────────────────────────────────────────────────────────────────

# Nombre de chunks récupérés par similarité sémantique.
# Deux valeurs, car les deux situations n'ont rien à voir :
#  - classe connue → la recherche vectorielle n'est qu'un complément, les chunks
#    de la classe sont de toute façon ajoutés directement par nom (get_code_by_class_name) ;
#  - classe inconnue → la recherche est la SEULE source, il en faut davantage.
RAG_LIMIT_WITH_CLASS    = 6
RAG_LIMIT_WITHOUT_CLASS = 10

# Chunks de templates Twig ajoutés en complément, et uniquement si le contrôleur
# rend des vues. Ils servent à récupérer les titres H1 et les champs de
# formulaire réels, pour que le modèle n'invente pas de sélecteur CSS.
TEMPLATE_LIMIT_WITH_CLASS    = 3
TEMPLATE_LIMIT_WITHOUT_CLASS = 5

# Filtre post-récupération (filter_chunks_by_class) : un chunk qui ne mentionne
# pas la classe demandée n'est gardé que s'il est très proche sémantiquement.
# 0.65 en similarité cosinus est déjà exigeant sur des embeddings normalisés.
MIN_SECONDARY_SIMILARITY = 0.65

# Et au plus N chunks secondaires, quelle que soit leur similarité : au-delà,
# on a observé que le contexte se remplit de contrôleurs voisins sans rapport.
MAX_SECONDARY_CHUNKS = 3


def _build_context_str(context_chunks: List[Dict]) -> str:
    """
    Formate les chunks RAG en bloc de texte pour le prompt.

    Le contenu vient du code du projet analysé — donc d'une source que le moteur
    n'a pas écrite. Il est neutralisé et explicitement délimité avant d'entrer
    dans le prompt : voir prompt_safety.py pour le détail du risque.
    """
    if not context_chunks:
        return "CONTEXTE : Aucun contexte trouvé pour ce projet.\n"
    lines = ["CONTEXTE RÉEL DU PROJET (NE PAS INVENTER) :"]
    for c in context_chunks:
        lines.append(f"- {c['content']} (Fichier: {c['file_path']})")
    return wrap_untrusted_context("\n".join(lines)) + "\n"

def _build_routes_summary(context_chunks: List[Dict]) -> str:
    """Extrait la liste des routes depuis les chunks pour la passer au LLM."""
    lines = []
    for c in context_chunks:
        content = c.get("content", "")
        if "Route:" not in content:
            continue
        method = re.search(r"Méthode '([^']+)'", content)
        route  = re.search(r"Route:\s*([^\s—\n]+)", content)
        rtype  = re.search(r"→ Type de réponse:\s*(.+)", content)
        if method and route:
            rtype_str = rtype.group(1).strip() if rtype else "(type non détecté)"
            line = f"  - {method.group(1)}: {route.group(1)} → {rtype_str}"

            # Enrichir le résumé avec les métadonnées
            verb_m = _RE_HTTP_VERBS.search(content)
            if verb_m:
                line += f" [{verb_m.group(1).strip()}]"

            if _RE_AJAX_ONLY.search(content):
                line += " [AJAX]"

            form_m = _RE_FORM_TYPE.search(content)
            if form_m:
                line += f" [Form: {form_m.group(1).strip()}]"

            role_m = _RE_METHOD_ROLE.search(content)
            if role_m:
                line += f" [Rôle: {role_m.group(1).strip()}]"

            voter_m = _RE_VOTER.search(content)
            if voter_m:
                line += f" [Voter: {voter_m.group(1)}]"
            lines.append(line)
    if not lines:
        return ""
    return "ROUTES À TESTER (exhaustif, ne pas en oublier) :\n" + "\n".join(lines)

def _validate_coverage(generated_code: str, context_chunks: List[Dict]) -> List[str]:
    """
    Retourne la liste des routes présentes dans les chunks
    mais absentes du code généré.

    Les paramètres dynamiques {id} sont remplacés par un wildcard non-quote,
    de sorte que /foo/{id} matche /foo/1 ou /foo/dupont dans le code généré.
    """
    missing = []
    for c in context_chunks:
        route_m = re.search(r"Route:\s*([^\s—\n]+)", c["content"])
        if not route_m:
            continue
        route = route_m.group(1)
        # Découper la route sur les paramètres {…}, échapper les segments littéraux
        # puis joindre avec le wildcard. Évite l'écueil de re.escape qui transforme
        # {id} en \{id\} et casse une substitution naïve.
        literal_parts = re.split(r"\{[^}]+\}", route)
        route_pattern = r"[^'\"\s]+".join(re.escape(p) for p in literal_parts)
        if not re.search(route_pattern, generated_code):
            missing.append(route)
    return missing

def filter_chunks_by_class(chunks: List[Dict], class_name: str) -> List[Dict]:
    """
    Filtre post-retrieval pour réduire le bruit contextuel.
    """
    if not class_name:
        return chunks
    name_lower = class_name.lower()
    primary   = [c for c in chunks if name_lower in c["content"].lower()]
    secondary = [
        c for c in chunks
        if name_lower not in c["content"].lower()
        and c.get("chunk_type") != "template_info"  # exclut les templates hors-sujet
        and c.get("similarity", 0.0) >= MIN_SECONDARY_SIMILARITY
    ]
    return primary + secondary[:MAX_SECONDARY_CHUNKS]
