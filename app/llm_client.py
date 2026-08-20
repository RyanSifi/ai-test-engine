"""
Client Ollama : construction de prompt, appel LLM, chargement du golden dataset
(few-shots) pour la génération de tests via RAG + LLM.
"""
import json
import logging
import os
import time
from typing import Dict, List, Optional

import requests

from config import settings

MAX_SHOT_CHARS = 3500


def _load_golden_dataset(filename: str, profile: str = "") -> List[Dict]:
    """
    Charge le golden dataset et sélectionne l'exemple le plus pertinent
    selon le profil du contrôleur (web_crud, api, mixed, etc.).
    Retourne toujours 1 seul exemple pour garder le prompt court.
    """
    path = os.path.join(os.path.dirname(__file__), filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception as e:
        logging.warning(f"Impossible de charger {filename}: {e}")
        return []

    for ex in dataset:
        ti = ex.get("test_ideal", "")
        if len(ti) > MAX_SHOT_CHARS:
            truncated = ti[:MAX_SHOT_CHARS]
            # Équilibrage approximatif des accolades : on referme autant de blocs
            # ouverts que nécessaire au lieu d'ajouter un `}` aveugle qui pouvait
            # produire du PHP invalide en few-shot.
            missing = max(0, truncated.count('{') - truncated.count('}'))
            ex["test_ideal"] = truncated + "\n// [tronqué]\n" + ("}" * missing)

    if not dataset:
        return []

    # Keywords disjoints — un même mot-clé ne doit pas appartenir à 2 profils
    # (sinon le premier dans l'ordre d'itération gagne arbitrairement).
    PROFILE_KEYWORDS = {
        "api":      ["xhr", "api", "ajax"],
        "mixed":    ["mixte", "mixed"],
        "web_crud": ["crud", "formulaire", "form", "redirect"],
    }

    keywords = PROFILE_KEYWORDS.get(profile, [])
    if keywords:
        for ex in dataset:
            demand = ex.get("demande_utilisateur", "").lower()
            if any(kw in demand for kw in keywords):
                logging.info(f"[few-shot] Exemple sélectionné pour profil '{profile}' : {demand[:60]}")
                return [ex]

    logging.info(f"[few-shot] Fallback premier exemple (profil='{profile}')")
    return [dataset[0]]


# ── Budgets de prompt ────────────────────────────────────────────────────────
# Plafond de longueur du prompt, en caractères. Au-delà, le prompt est tronqué.
MAX_PROMPT_CHARS = 12_000

# Estimation grossière du nombre de tokens : ~3 caractères par token en français
# et sur du code. Sert uniquement à journaliser un avertissement, jamais à
# décider — d'où l'approximation assumée.
CHARS_PER_TOKEN = 3

# Seuil d'alerte, en tokens estimés. La fenêtre de contexte configurée est de
# 8192 (LLM_NUM_CTX) ; au-delà de ~6500 tokens en entrée, il ne reste plus assez
# de place en sortie et les réponses commencent à être tronquées.
TOKEN_WARNING_THRESHOLD = 6_500

# ── Délais réseau (secondes) ─────────────────────────────────────────────────
# Le health-check est une simple requête GET sur /api/tags : s'il n'a pas répondu
# en quelques secondes, le service est en réalité indisponible.
HEALTH_CHECK_TIMEOUT = 5

# Délai d'établissement de la connexion TCP. Distinct du délai de lecture :
# se connecter est instantané ou échoue, alors que générer prend des minutes.
CONNECT_TIMEOUT = 10

# Délai de lecture par défaut. Généreux car sur CPU le modèle produit ~2,5
# tokens/seconde : 1500 tokens demandent déjà une dizaine de minutes.
# La génération par route impose sa propre valeur, bien plus courte.
DEFAULT_READ_TIMEOUT = 600

# ── Paramètres d'inférence ───────────────────────────────────────────────────
# Température basse : on veut du code reproductible et conforme au contexte
# fourni, pas de la créativité. À 0, le modèle boucle parfois sur des motifs
# répétitifs ; 0,1 laisse juste assez de variation pour l'éviter.
LLM_TEMPERATURE = 0.1


def _check_ollama_alive(base_url: str, timeout: int = HEALTH_CHECK_TIMEOUT) -> bool:
    """Vérifie qu'Ollama répond avant d'envoyer un prompt."""
    try:
        health_url = base_url.replace("/api/generate", "") + "/api/tags"
        r = requests.get(health_url, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _call_llm_meta(
    prompt: str,
    timeout: int = DEFAULT_READ_TIMEOUT,
    num_predict: Optional[int] = None,
    _skip_health: bool = False,
) -> Dict:
    """
    Appelle Ollama et retourne un dict de métadonnées :
        {
          "text":        str,   # texte généré, nettoyé des balises markdown
          "done_reason": str,   # "stop" (fin normale) | "length" (tronqué) | ...
          "eval_count":  int,   # nb de tokens générés (rapporté par Ollama)
          "elapsed":     float, # durée de l'appel en secondes
        }

    Version « riche » de _call_llm : nécessaire pour la génération par route
    (retry sur done_reason == "length", logs par route). `_call_llm` en dessous
    reste un wrapper qui ne renvoie que le texte, pour les appelants existants.

    `num_predict` : override par appel (ex: 1024 pour une route). Si None, on
    garde la valeur globale de config (settings.llm_num_predict).
    `_skip_health` : zappe le health-check (retries après un appel déjà réussi).
    """
    if not _skip_health and not _check_ollama_alive(settings.ollama_url):
        raise RuntimeError(
            f"Ollama injoignable — vérifiez que le service tourne sur {settings.ollama_url}"
        )
    if len(prompt) > MAX_PROMPT_CHARS:
        logging.warning(f"[_call_llm] Prompt tronqué : {len(prompt)} → {MAX_PROMPT_CHARS} chars")
        prompt = prompt[:MAX_PROMPT_CHARS]

    n_predict = num_predict if num_predict is not None else settings.llm_num_predict
    estimated_tokens = len(prompt) // CHARS_PER_TOKEN
    logging.info(
        f"[_call_llm] ~{estimated_tokens} tokens estimés, "
        f"num_ctx={settings.llm_num_ctx}, num_predict={n_predict}"
    )
    if estimated_tokens > TOKEN_WARNING_THRESHOLD:
        logging.warning(
            f"[_call_llm] Prompt estimé à {estimated_tokens} tokens "
            f"(seuil {TOKEN_WARNING_THRESHOLD}) — risque de troncature de la réponse"
        )

    resp = requests.post(
        settings.ollama_url,
        json={
            "model":  settings.default_llm_model,
            "prompt": prompt,
            "stream": True,
            # keep_alive est un champ de premier niveau (pas dans options) : il
            # garde le modèle en RAM entre deux appels → pas de rechargement.
            "keep_alive": settings.llm_keep_alive,
            "options": {
                "temperature":  LLM_TEMPERATURE,
                "stop":         ["<|im_end|>"],
                "num_ctx":      settings.llm_num_ctx,
                "num_predict":  n_predict,
            },
        },
        stream=True,
        timeout=(CONNECT_TIMEOUT, timeout),
    )
    resp.raise_for_status()

    code_parts = []
    done_reason = None
    eval_count = 0
    start = time.time()

    for line in resp.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            # Ollama renvoie occasionnellement des lignes non-JSON (logs internes,
            # message d'erreur). On les ignore plutôt que de tuer la génération.
            logging.debug(f"[_call_llm] ligne ignorée (non-JSON) : {line[:100]!r}")
            continue
        code_parts.append(chunk.get("response", ""))
        if chunk.get("done"):
            # Le chunk final porte done_reason ("stop"/"length") et eval_count
            # (nb de tokens réellement générés, plus fiable que compter les lignes).
            done_reason = chunk.get("done_reason")
            eval_count = chunk.get("eval_count", eval_count)
            break

    elapsed = round(time.time() - start, 1)
    logging.info(f"[_call_llm] {eval_count} tokens en {elapsed}s (done_reason={done_reason})")

    code = "".join(code_parts).strip().replace("```php", "").replace("```", "").strip()
    return {
        "text":        code,
        "done_reason": done_reason,
        "eval_count":  eval_count,
        "elapsed":     elapsed,
    }


def _call_llm(prompt: str, timeout: int = DEFAULT_READ_TIMEOUT, _skip_health: bool = False) -> str:
    """
    Appelle Ollama et retourne uniquement le texte généré (nettoyé des balises
    markdown). Wrapper mince sur _call_llm_meta — conservé tel quel pour les
    appelants existants (generate_test, generate_unit_test).
    """
    return _call_llm_meta(prompt, timeout=timeout, _skip_health=_skip_health)["text"]


def _build_prompt(system: str, few_shots: List[Dict], user: str) -> str:
    """
    Construit le prompt selon le format du modèle configuré.
    - Qwen / Mistral / Phi : format <|im_start|> / <|im_end|>
    - Gemma / autres       : format texte brut ### Instructions / ### Tâche
    """
    model = settings.default_llm_model.lower()
    is_chatml = any(k in model for k in ("qwen", "mistral", "phi"))

    if is_chatml:
        shots = ""
        for ex in few_shots:
            shots += (
                f"<|im_end|>\n<|im_start|>user\nScénario: {ex['demande_utilisateur']}\n"
                f"<|im_end|>\n<|im_start|>assistant\n{ex['test_ideal']}\n"
            )
        return (
            f"<|im_start|>system\n{system}\n"
            f"{shots}"
            f"<|im_end|>\n<|im_start|>user\nScénario: {user}\n"
            f"<|im_end|>\n<|im_start|>assistant\n"
        )
    else:
        shots = ""
        for ex in few_shots:
            shots += (
                f"\n### Exemple\nScénario: {ex['demande_utilisateur']}\n"
                f"Réponse:\n{ex['test_ideal']}\n"
            )
        return (
            f"### Instructions\n{system}\n"
            f"{shots}"
            f"\n### Tâche\nScénario: {user}\n"
            f"Réponse:\n"
        )
