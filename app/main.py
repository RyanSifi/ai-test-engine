from fastapi import FastAPI, HTTPException, Depends
import requests
import re
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import subprocess
import time
from contextlib import asynccontextmanager
from functools import lru_cache
import logging
import json
from code_parser import analyze_project_code, extract_code_for_symbol
from config import settings
from brain import SemanticEngine
from db import KnowledgeDB

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# DÉPENDANCES (singletons mis en cache)
# ---------------------------------------------------------------------------

@lru_cache
def get_db() -> KnowledgeDB:
    return KnowledgeDB(os.getenv("DATABASE_URL"))


def _allowed_embedding_models() -> set:
    """Set des modèles autorisés (default toujours inclus)."""
    raw = settings.allowed_embedding_models or ""
    allowed = {m.strip() for m in raw.split(",") if m.strip()}
    allowed.add(settings.default_embedding_model)
    return allowed


def _check_model_allowed(model_name: str) -> None:
    """Refuse les noms de modèle hors allowlist (anti-DL HuggingFace arbitraire)."""
    if model_name not in _allowed_embedding_models():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Modèle '{model_name}' non autorisé. "
                f"Liste : {sorted(_allowed_embedding_models())}"
            ),
        )


# maxsize plafonné pour éviter qu'un attaquant n'épuise la RAM en variant model_name
@lru_cache(maxsize=4)
def get_brain(model_name: str) -> SemanticEngine:
    return SemanticEngine(model_name=model_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Démarrage de l'application.")
    db = get_db()
    brain = get_brain(settings.default_embedding_model)
    # Initialise le schéma si les tables n'existent pas encore
    try:
        # On encode un texte vide pour connaître la dimension du modèle
        sample_vec = brain.encode(["init"])[0]
        db.init_schema(vector_size=len(sample_vec))
        logging.info(f"Schéma DB prêt (dim={len(sample_vec)}).")
    except Exception as e:
        logging.error(f"Impossible d'initialiser le schéma : {e}")
    yield
    logging.info("Arrêt de l'application.")


# ---------------------------------------------------------------------------
# APPLICATION FASTAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Moteur de Test IA",
    description=(
        "Microservice pour analyser un projet Symfony et générer automatiquement "
        "des tests fonctionnels (WebTestCase) et unitaires (PHPUnit) via RAG + LLM."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# MODÈLES PYDANTIC
# ---------------------------------------------------------------------------

class RouteDefinition(BaseModel):
    path: str
    method: str
    name: str


class ScenarioDefinition(BaseModel):
    description: str
    input_json: Optional[str] = "{}"


class LearnFromCodeRequest(BaseModel):
    project_id: str
    model_name: Optional[str] = settings.default_embedding_model
    model_config = {"protected_namespaces": ()}


class GenerateTestRequest(BaseModel):
    project_id: str
    description: str
    test_name: Optional[str] = None
    class_name: Optional[str] = None
    model_name: Optional[str] = settings.default_embedding_model
    deterministic: bool = False  # True = bypass LLM, génération depuis les chunks
    model_config = {"protected_namespaces": ()}


class GenerateUnitTestRequest(BaseModel):
    project_id: str
    file_path: str
    class_name: str
    method_name: Optional[str] = None
    description: str
    test_name: Optional[str] = None
    model_name: Optional[str] = settings.default_embedding_model
    model_config = {"protected_namespaces": ()}


class ResetSchemaRequest(BaseModel):
    confirm: bool = False



# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def validate_php_syntax(code: str) -> Optional[str]:
    """Valide la syntaxe PHP. Retourne le message d'erreur ou None si OK."""
    try:
        proc = subprocess.Popen(
            ["php", "-l"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate(input=code)
        return (stderr or stdout).strip() if proc.returncode != 0 else None
    except FileNotFoundError:
        logging.warning("PHP non disponible — validation de syntaxe ignorée.")
        return None


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
        if len(ex.get("test_ideal", "")) > MAX_SHOT_CHARS:
            ex["test_ideal"] = ex["test_ideal"][:MAX_SHOT_CHARS] + "\n// [tronqué]\n}"

    if not dataset:
        return []

    PROFILE_KEYWORDS = {
        "api":      ["ajax", "xhr", "api"],
        "mixed":    ["mixte", "mixed", "ajax"],
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


MAX_PROMPT_CHARS = 12_000

def _check_ollama_alive(base_url: str, timeout: int = 5) -> bool:
    """Vérifie qu'Ollama répond avant d'envoyer un prompt."""
    try:
        health_url = base_url.replace("/api/generate", "") + "/api/tags"
        r = requests.get(health_url, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False

def _call_llm(prompt: str, timeout: int = 600) -> str:
    """Appelle Ollama et retourne le texte généré (nettoyé des balises markdown)."""
    if not _check_ollama_alive(settings.ollama_url):
        raise RuntimeError("Ollama injoignable — vérifiez que le service tourne sur host.docker.internal:11434")
    if len(prompt) > MAX_PROMPT_CHARS:
        logging.warning(f"[_call_llm] Prompt tronqué : {len(prompt)} → {MAX_PROMPT_CHARS} chars")
        prompt = prompt[:MAX_PROMPT_CHARS]

    estimated_tokens = len(prompt) // 3
    logging.info(f"[_call_llm] ~{estimated_tokens} tokens estimés, num_ctx=8192")
    if estimated_tokens > 6500:
        logging.warning("[_call_llm] Prompt dépasse 6500 tokens — risque de troncature LLM")

    resp = requests.post(
        settings.ollama_url,
        json={
            "model":  settings.default_llm_model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature":  0.1,
                "stop":         ["<|im_end|>"],
                "num_ctx":      8192,
                "num_predict":  1500,
            },
        },
        stream=True,
        timeout=(10, timeout),
    )
    resp.raise_for_status()

    code_parts = []
    eval_count = 0
    start = time.time()

    for line in resp.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        code_parts.append(chunk.get("response", ""))
        eval_count += 1
        if chunk.get("done"):
            break

    elapsed = round(time.time() - start, 1)
    logging.info(f"[_call_llm] {eval_count} tokens en {elapsed}s")

    code = "".join(code_parts).strip()
    return code.replace("```php", "").replace("```", "").strip()


def _safe_join(base: str, *parts: str) -> str:
    """
    Joint et résout un chemin relatif sous `base`.
    Lève HTTPException(400) si le résultat sort de `base` (path traversal).
    """
    base_real = os.path.realpath(base)
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


def _write_php_file(relative_path: str, code: str) -> str:
    """Écrit un fichier PHP dans le workspace et retourne son chemin absolu."""
    full_path = _safe_join(settings.container_project_root, relative_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(code)
    return full_path


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


def _build_context_str(context_chunks: List[Dict]) -> str:
    """Formate les chunks RAG en bloc de texte pour le prompt."""
    if not context_chunks:
        return "CONTEXTE : Aucun contexte trouvé pour ce projet.\n"
    lines = ["CONTEXTE RÉEL DU PROJET (NE PAS INVENTER) :"]
    for c in context_chunks:
        lines.append(f"- {c['content']} (Fichier: {c['file_path']})")
    return "\n".join(lines) + "\n"

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
        and c.get("similarity", 0.0) >= 0.65
    ]
    return primary + secondary[:3]


# ---------------------------------------------------------------------------
# GÉNÉRATEUR DÉTERMINISTE — bypass LLM pour les WebTestCase
# ---------------------------------------------------------------------------

# Regex pour parser le contenu des chunks de méthode enrichis
_RE_METHOD_NAME   = re.compile(r"Méthode '([^']+)'")
_RE_ROUTE         = re.compile(r"Route:\s*([^\s—\n]+)")
_RE_RESPONSE_TYPE = re.compile(r"→ Type de réponse:\s*(.+)")
_RE_H1            = re.compile(r"→ H1:\s*(.+)")
_RE_HIDDEN_IDS    = re.compile(r"→ IDs cachés:\s*(.+)")
_RE_CONSTRUCTOR   = re.compile(r"constructeur injecte\s*:\s*\(([^)]+)\)")
_RE_ROUTE_PARAM   = re.compile(r"\{(\w+)\}")

# Regex enrichies
_RE_HTTP_VERBS    = re.compile(r"→ Verbes HTTP:\s*(.+)")
_RE_AJAX_ONLY     = re.compile(r"→ AJAX uniquement")
_RE_FORM_TYPE     = re.compile(r"→ Formulaire:\s*(.+)")
_RE_VOTER         = re.compile(r"→ Voter:\s*denyAccessUnlessGranted\('([^']+)'")
_RE_CLASS_ROLES   = re.compile(r"→ Rôles classe:\s*(.+)")
_RE_METHOD_ROLE   = re.compile(r"→ Rôle requis \(méthode\):\s*(.+)")
_RE_PROFILE       = re.compile(r"→ Profil:\s*(\w+)")
_RE_INTERNAL      = re.compile(r"→ Pas de route HTTP")

# Valeurs de substitution réalistes pour les paramètres de route courants
_PARAM_DEFAULTS: Dict[str, str] = {
    "id":   "1",
    "nom":  "dupont",
    "name": "dupont",
    "slug": "exemple",
    "page": "1",
    "type": "test",
    "code": "001",
    "tab":  "general",
    "filename": "export.csv",
}


def _resolve_route(route: str) -> str:
    """Remplace les paramètres dynamiques {param} par des valeurs de test."""
    def replace(m: re.Match) -> str:
        param = m.group(1)
        return _PARAM_DEFAULTS.get(param.lower(), "test")
    return _RE_ROUTE_PARAM.sub(replace, route)


def _role_to_factory_key(role: str) -> str:
    """
    Mappe un rôle Symfony (ex: ROLE_PARCOURS) vers la clé attendue par TestUserFactory
    (ex: PARCOURS).  Le mapping est défini dans settings.auth_role_key_map sous la forme
    "ROLE_A:KEY_A,ROLE_B:KEY_B".  Sans mapping, le rôle est retourné tel quel.
    """
    if settings.auth_role_key_map:
        for pair in settings.auth_role_key_map.split(","):
            if ":" in pair:
                src, dst = pair.strip().split(":", 1)
                if src.strip() == role:
                    return dst.strip()
    return role


def _detect_controller_profile(chunks: List[Dict]) -> str:
    """
    Détecte le profil du contrôleur depuis les chunks indexés.
    Retourne : 'web_crud', 'api', 'internal', 'mixed'.
    """
    for c in chunks:
        m = _RE_PROFILE.search(c.get("content", ""))
        if m:
            return m.group(1)
    # Fallback : si aucun chunk de classe n'a de profil, déduire des méthodes
    has_routes   = any(_RE_ROUTE.search(c.get("content", "")) for c in chunks)
    has_internal = any(_RE_INTERNAL.search(c.get("content", "")) for c in chunks)
    if not has_routes and has_internal:
        return "internal"
    return "web_crud"


def _detect_class_role(chunks: List[Dict]) -> str:
    """
    Trouve le rôle requis au niveau classe depuis les chunks.
    Retourne le premier rôle trouvé ou 'ADMIN' par défaut.
    """
    for c in chunks:
        m = _RE_CLASS_ROLES.search(c.get("content", ""))
        if m:
            roles = [r.strip() for r in m.group(1).split(",")]
            return roles[0]  # Premier rôle = rôle principal
    return "ADMIN"


def _extract_method_role(content: str, class_role: str) -> str:
    """
    Retourne le rôle à utiliser pour tester une méthode :
    le rôle spécifique de la méthode s'il existe, sinon le rôle de la classe.
    """
    m = _RE_METHOD_ROLE.search(content)
    if m:
        return m.group(1).strip()
    return class_role


def _extract_http_verb(content: str) -> str:
    """Extrait le verbe HTTP depuis un chunk, ou 'GET' par défaut."""
    m = _RE_HTTP_VERBS.search(content)
    if m:
        verbs = [v.strip() for v in m.group(1).split(",")]
        # Si GET et POST → POST pour les formulaires, sinon premier verbe
        if "POST" in verbs:
            return "POST"
        return verbs[0]
    return "GET"


def _generate_php_test_from_chunks(
    chunks: List[Dict],
    class_name: str,
    test_class_name: str,
) -> str:
    """
    Génère un fichier de test WebTestCase PHP directement depuis les chunks indexés,
    sans passer par le LLM. Élimine toute hallucination structurelle.

    Architecture « scénarios dynamiques » :
    Chaque méthode est analysée et accumule une liste de scénarios de test.
    Chaque scénario est un dict qui décrit un test PHP à générer.
    Ajouter un nouveau pattern = ajouter un builder dans SCENARIO_BUILDERS.
    """
    class_role        = _detect_class_role(chunks)
    class_factory_key = _role_to_factory_key(class_role)
    all_roles         = [r.strip() for r in settings.auth_test_roles.split(",")]
    secondary_roles   = [r for r in all_roles if r != class_factory_key]
    fw            = settings.auth_firewall_name
    redirect_path = settings.auth_redirect_path
    redirect_code = settings.auth_redirect_status
    methods_seen: set    = set()
    test_methods: list   = []
    skipped_private: list = []

    for chunk in chunks:
        content = chunk.get("content", "")
        if not content.startswith("Méthode '"):
            continue

        method_m = _RE_METHOD_NAME.search(content)
        if not method_m:
            continue
        method_name = method_m.group(1)

        if method_name in methods_seen or method_name == "__construct":
            continue
        methods_seen.add(method_name)

        route_m = _RE_ROUTE.search(content)
        if not route_m:
            skipped_private.append(method_name)
            continue

        # Extraire les métadonnées du chunk
        ctx = _parse_chunk_metadata(content, route_m.group(1), class_role)

        # Accumuler les scénarios applicables
        scenarios = []
        for builder in SCENARIO_BUILDERS:
            scenarios.extend(builder(ctx, fw, redirect_path, redirect_code, secondary_roles))

        # Générer le code PHP pour chaque scénario
        for sc in scenarios:
            test_methods.append(_render_scenario(sc))

    # Commentaires pour méthodes sans route
    private_comment = ""
    if skipped_private:
        names = ", ".join(skipped_private)
        private_comment = f"\n    // Méthodes sans route HTTP (non testées ici) : {names}\n"

    # Assemblage final
    body          = "\n\n".join(test_methods)
    factory       = settings.auth_test_class
    sso_user      = settings.auth_sso_user_class
    factory_short = factory.split("\\")[-1]
    sso_short     = sso_user.split("\\")[-1]

    php = (
        "<?php\n\n"
        "namespace App\\Tests\\Functional\\Controller;\n\n"
        f"use {factory};\n"
        f"use {sso_user};\n"
        "use Symfony\\Bundle\\FrameworkBundle\\KernelBrowser;\n"
        "use Symfony\\Bundle\\FrameworkBundle\\Test\\WebTestCase;\n"
        "use Symfony\\Component\\HttpFoundation\\Response;\n\n"
        f"final class {test_class_name} extends WebTestCase\n"
        "{\n"
        "    protected KernelBrowser $client;\n"
        f"    private {factory_short} $testUserFactory;\n\n"
        "    protected function setUp(): void\n"
        "    {\n"
        "        $this->client = self::createClient();\n"
        f"        $this->testUserFactory = $this->client->getContainer()->get({factory_short}::class);\n"
        "    }\n\n"
        f"    private function getTestUser(string $key): {sso_short}\n"
        "    {\n"
        "        return $this->testUserFactory->create($key);\n"
        "    }\n"
        f"{private_comment}"
        f"\n{body}\n"
        "}\n"
    )
    return php


# ---------------------------------------------------------------------------
# MOTEUR DE SCÉNARIOS
# ---------------------------------------------------------------------------

def _parse_chunk_metadata(content: str, raw_route: str, class_role: str) -> Dict:
    """
    Parse toutes les métadonnées d'un chunk de méthode en un dict plat
    réutilisable par tous les builders de scénarios.
    """
    method_m   = _RE_METHOD_NAME.search(content)
    method_name = method_m.group(1) if method_m else "unknown"
    cap         = method_name[0].upper() + method_name[1:]

    rtype_matches  = _RE_RESPONSE_TYPE.findall(content)
    response_types = [r.strip() for r in rtype_matches]

    h1_m       = _RE_H1.search(content)
    hidden_m   = _RE_HIDDEN_IDS.search(content)
    form_m     = _RE_FORM_TYPE.search(content)
    voter_m    = _RE_VOTER.search(content)

    method_role = _extract_method_role(content, class_role)

    return {
        "method_name":    method_name,
        "cap":            cap,
        "raw_route":      raw_route,
        "route":          _resolve_route(raw_route),
        "http_verb":      _extract_http_verb(content),
        "method_role":    method_role,
        "class_role":     class_role,
        # Clés mappées pour getTestUser() — différentes de method_role si auth_role_key_map est défini
        "factory_key":       _role_to_factory_key(method_role),
        "class_factory_key": _role_to_factory_key(class_role),
        "role_label":        _role_to_factory_key(method_role).replace("ROLE_", "").title(),
        "response_types": response_types,
        "has_render":     any("render"   in r for r in response_types),
        "has_redirect":   any("redirect" in r for r in response_types),
        "has_json":       any("json"     in r for r in response_types),
        "has_file":       any("file_download" in r or "binary" in r for r in response_types),
        "has_export":     any("export"   in r for r in response_types),
        "is_ajax":        bool(_RE_AJAX_ONLY.search(content)),
        "has_form":       form_m is not None,
        "form_type":      form_m.group(1).strip() if form_m else None,
        "has_voter":      voter_m is not None,
        "voter_attr":     voter_m.group(1) if voter_m else None,
        "h1":             h1_m.group(1).strip().split(",")[0].strip() if h1_m else "",
        "hidden_ids":     [i.strip() for i in hidden_m.group(1).split(",")] if hidden_m else [],
    }


def _render_scenario(sc: Dict) -> str:
    """Transforme un scénario en méthode de test PHP."""
    lines = []
    if sc.get("comment"):
        lines.append(f"    /** {sc['comment']} */")
    lines.append(f"    public function {sc['func_name']}(): void")
    lines.append("    {")
    for line in sc.get("body", []):
        lines.append(f"        {line}")
    lines.append("    }")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# BUILDERS DE SCÉNARIOS — chaque builder retourne 0..N scénarios
#
# Pour ajouter un nouveau pattern de test :
#   1. Écrire une fonction (ctx, fw, redirect, code, sec_roles) -> List[Dict]
#   2. L'ajouter à SCENARIO_BUILDERS en bas de cette section
# ---------------------------------------------------------------------------

def _scenario_noauth(ctx, fw, redirect_path, redirect_code, _sec_roles):
    """Non authentifié → redirect vers le SSO."""
    return [{
        "comment":   f"{ctx['raw_route']} — non authentifié → WebSSO",
        "func_name": f"test{ctx['cap']}RedirectsWhenNotAuthenticated",
        "body": [
            f"$this->client->request('{ctx['http_verb']}', '{ctx['route']}');",
            f"self::assertResponseStatusCodeSame({redirect_code});",
            f"self::assertResponseRedirects('{redirect_path}');",
        ],
    }]


def _scenario_auth_form(ctx, fw, _rp, _rc, _sec_roles):
    """Formulaire : GET affiche le form, POST soumet et redirige."""
    if not ctx["has_form"]:
        return []

    scenarios = []
    rl  = ctx["role_label"]
    fk  = ctx["factory_key"]

    # GET → affichage
    get_body = [
        f"$this->client->loginUser($this->getTestUser('{fk}'), '{fw}');",
        f"$this->client->request('GET', '{ctx['route']}');",
        "self::assertResponseIsSuccessful();",
    ]
    if ctx["h1"]:
        get_body.append(f"$this->assertSelectorTextContains('h1', '{ctx['h1']}');")

    scenarios.append({
        "comment":   f"{ctx['raw_route']} — {fk} — affichage formulaire {ctx['form_type'] or ''}",
        "func_name": f"test{ctx['cap']}DisplaysFormWith{rl}Role",
        "body":      get_body,
    })

    # POST → soumission
    post_body = [
        f"$this->client->loginUser($this->getTestUser('{fk}'), '{fw}');",
        f"$this->client->request('POST', '{ctx['route']}');",
    ]
    if ctx["has_redirect"]:
        post_body.append("self::assertResponseRedirects();")
    else:
        post_body.append("self::assertResponseIsSuccessful();")

    scenarios.append({
        "comment":   f"{ctx['raw_route']} — {fk} — soumission formulaire",
        "func_name": f"test{ctx['cap']}SubmitWith{rl}Role",
        "body":      post_body,
    })

    return scenarios


def _scenario_auth_simple(ctx, fw, _rp, _rc, _sec_roles):
    """Authentifié — réponse simple (pas de formulaire)."""
    if ctx["has_form"]:
        return []

    rl = ctx["role_label"]
    fk = ctx["factory_key"]

    # Construire la requête
    if ctx["is_ajax"]:
        request_line = (
            f"$this->client->request('{ctx['http_verb']}', '{ctx['route']}', "
            "[], [], ['HTTP_X-Requested-With' => 'XMLHttpRequest']);"
        )
    else:
        request_line = f"$this->client->request('{ctx['http_verb']}', '{ctx['route']}');"

    body = [
        f"$this->client->loginUser($this->getTestUser('{fk}'), '{fw}');",
        request_line,
    ]

    # Assertions selon le type de réponse
    primary = ctx["response_types"][0] if ctx["response_types"] else ""

    if ctx["has_render"] and ctx["has_redirect"] and not ctx["has_form"]:
        # Deux branches possibles (ex: edit qui peut rediriger si erreur)
        suffix = f"IsReachedWith{rl}Role"
        body += [
            "$statusCode = $this->client->getResponse()->getStatusCode();",
            "self::assertContains($statusCode, [200, 302],",
            "    sprintf('Expected 200 or 302, got %d', $statusCode));",
        ]
    elif "redirect (302)" in primary:
        suffix = f"RedirectsWith{rl}Role"
        body.append("self::assertResponseRedirects();")
    elif "json (200)" in primary:
        suffix = f"ReturnsJsonWith{rl}Role"
        body.append("self::assertResponseIsSuccessful();")
        body.append("$this->assertJson($this->client->getResponse()->getContent());")
    elif ctx["has_file"]:
        suffix = f"ReturnsFileWith{rl}Role"
        body.append("self::assertResponseIsSuccessful();")
    elif ctx["has_export"]:
        suffix = f"ReturnsExportWith{rl}Role"
        body.append("self::assertResponseIsSuccessful();")
    else:
        suffix = f"IsReachedWith{rl}Role"
        body.append("self::assertResponseIsSuccessful();")
        if ctx["h1"]:
            body.append(f"$this->assertSelectorTextContains('h1', '{ctx['h1']}');")
        for hid in ctx["hidden_ids"][:1]:
            body.append(f"$this->assertSelectorExists('#{hid}');")

    return [{
        "comment":   f"{ctx['raw_route']} — authentifié {fk}",
        "func_name": f"test{ctx['cap']}{suffix}",
        "body":      body,
    }]


def _scenario_ajax_no_xhr(ctx, fw, _rp, _rc, _sec_roles):
    """Route AJAX appelée sans header XHR → 404."""
    if not ctx["is_ajax"]:
        return []
    fk = ctx["factory_key"]
    return [{
        "comment":   f"{ctx['raw_route']} — {fk} — sans header XHR → 404",
        "func_name": f"test{ctx['cap']}WithoutXhrReturns404",
        "body": [
            f"$this->client->loginUser($this->getTestUser('{fk}'), '{fw}');",
            f"$this->client->request('{ctx['http_verb']}', '{ctx['route']}');",
            "self::assertResponseStatusCodeSame(404);",
        ],
    }]


def _scenario_role_insufficient(ctx, fw, _rp, _rc, _sec_roles):
    """Rôle de la classe mais pas le rôle requis par la méthode → 403."""
    if ctx["method_role"] == ctx["class_role"]:
        return []
    cfk      = ctx["class_factory_key"]
    cr_label = cfk.replace("ROLE_", "").title()
    return [{
        "comment":   f"{ctx['raw_route']} — {cfk} (rôle insuffisant) → 403",
        "func_name": f"test{ctx['cap']}ForbiddenWith{cr_label}Role",
        "body": [
            f"$this->client->loginUser($this->getTestUser('{cfk}'), '{fw}');",
            f"$this->client->request('{ctx['http_verb']}', '{ctx['route']}');",
            "self::assertResponseStatusCodeSame(403);",
        ],
    }]


def _scenario_voter(ctx, fw, _rp, _rc, _sec_roles):
    """Voter sur entité — tester l'accès refusé."""
    if not ctx["has_voter"]:
        return []
    fk = ctx["factory_key"]
    rl = ctx["role_label"]
    return [{
        "comment":   f"{ctx['raw_route']} — voter '{ctx['voter_attr']}' — accès entité",
        "func_name": f"test{ctx['cap']}VoterDeniesAccessWith{rl}Role",
        "body": [
            f"$this->client->loginUser($this->getTestUser('{fk}'), '{fw}');",
            "// ID inexistant ou entité interdite → adapter selon les fixtures",
            f"$this->client->request('{ctx['http_verb']}', '{ctx['route']}');",
            "self::assertContains(",
            "    $this->client->getResponse()->getStatusCode(),",
            "    [403, 404],",
            "    'Le voter devrait refuser ou l\\'entité ne devrait pas exister'",
            ");",
        ],
    }]


def _scenario_secondary_role(ctx, fw, _rp, _rc, sec_roles):
    """Vérifier qu'un rôle secondaire a aussi accès."""
    if not sec_roles:
        return []
    if ctx["is_ajax"] or ctx["has_form"] or ctx["has_voter"]:
        return []
    if ctx["method_role"] != ctx["class_role"]:
        return []

    results = []
    for sr in sec_roles[:1]:
        sr_fk    = _role_to_factory_key(sr)
        sr_label = sr_fk.replace("ROLE_", "").title()
        results.append({
            "comment":   f"{ctx['raw_route']} — {sr_fk} (rôle secondaire)",
            "func_name": f"test{ctx['cap']}IsReachedWith{sr_label}Role",
            "body": [
                f"$this->client->loginUser($this->getTestUser('{sr_fk}'), '{fw}');",
                f"$this->client->request('{ctx['http_verb']}', '{ctx['route']}');",
                "self::assertResponseIsSuccessful();",
            ],
        })
    return results


# ── Registre des builders ─────────────────────────────────────────────────
# L'ordre n'a PAS d'importance fonctionnelle (il détermine juste l'ordre
# des tests dans le fichier PHP). Ajouter un builder ici = nouveau pattern.

SCENARIO_BUILDERS = [
    _scenario_noauth,
    _scenario_auth_form,
    _scenario_auth_simple,
    _scenario_ajax_no_xhr,
    _scenario_role_insufficient,
    _scenario_voter,
    _scenario_secondary_role,
]


# ---------------------------------------------------------------------------
# ENDPOINTS — SANTÉ ET ADMINISTRATION
# ---------------------------------------------------------------------------

@app.get("/health", summary="Vérification de santé")
async def health_check(db: KnowledgeDB = Depends(get_db)):
    """Vérifie que l'API et la base de données sont opérationnelles."""
    try:
        projects = db.list_projects()
        return {
            "status": "ok",
            "projects_indexed": len(projects),
            "projects": projects,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB non disponible : {e}")


@app.get("/project/{project_id}/stats", summary="Statistiques d'un projet")
async def project_stats(project_id: str, db: KnowledgeDB = Depends(get_db)):
    """Retourne le nombre de chunks et de routes indexés pour un projet."""
    try:
        return db.get_project_stats(project_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/project/{project_id}", summary="Supprime les données d'un projet")
async def delete_project(project_id: str, db: KnowledgeDB = Depends(get_db)):
    """Supprime toutes les données indexées pour un projet donné."""
    try:
        db.clear_project(project_id)
        return {"status": "deleted", "project_id": project_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/reset-schema", summary="Réinitialise toute la base de données")
async def reset_schema(
    body: ResetSchemaRequest,
    db: KnowledgeDB = Depends(get_db),
):
    """
    Supprime et recrée toutes les tables.
    À utiliser uniquement lors d'un changement de modèle d'embedding.
    Requiert `confirm: true` dans le body pour éviter les fausses manœuvres.
    """
    if not body.confirm:
        raise HTTPException(
            status_code=400,
            detail='Passez {"confirm": true} pour confirmer la réinitialisation.',
        )
    try:
        brain = get_brain(settings.default_embedding_model)
        sample_vec = brain.encode(["init"])[0]
        db.reset_schema(vector_size=len(sample_vec))
        return {"status": "schema_reset", "vector_size": len(sample_vec)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# ENDPOINT — APPRENTISSAGE
# ---------------------------------------------------------------------------

@app.post("/learn-from-code", summary="Indexe le code source du projet Symfony")
async def learn_from_code(
    data: LearnFromCodeRequest,
    db: KnowledgeDB = Depends(get_db),
):
    """
    Analyse récursivement le code PHP + les templates Twig du projet monté
    dans /workspace, et stocke les embeddings dans PostgreSQL.
    À appeler après chaque modification structurelle du projet.
    """
    project_id   = data.project_id
    project_path = settings.container_project_root
    logging.info(f"[learn-from-code] Début pour project_id={project_id}")

    if not os.path.isdir(project_path):
        raise HTTPException(
            status_code=404,
            detail=f"Workspace introuvable : {project_path}",
        )

    _check_model_allowed(data.model_name)
    brain = get_brain(model_name=data.model_name)

    try:
        analysis = analyze_project_code(project_path)
        total_files = sum(len(v) for v in analysis.values())
        logging.info(f"[learn-from-code] {total_files} fichiers analysés.")

        chunks: List[Dict] = []

        # Pré-calcul du lookup templates : nom_fichier → données du template
        template_lookup: Dict[str, Dict] = {
            tpl["file"]: tpl for tpl in analysis.get("templates", [])
        }

        # ── Chunks PHP (contrôleurs, entités, services, etc.) ──────────────
        for category, items in analysis.items():
            if category == "templates":
                continue
            for item in items:
                # Chunk décrivant la classe entière
                constructor_info = ""
                if item.get("constructor_params"):
                    params_str = ", ".join(
                        f"{p['type'] or 'mixed'} ${p['name']}"
                        for p in item["constructor_params"]
                    )
                    constructor_info = f" Son constructeur injecte : ({params_str})."

                # Profil + rôles classe
                profile_info = ""
                profile = item.get("controller_profile", "")
                if profile:
                    profile_info += f"\n  → Profil: {profile}"
                class_grants = item.get("class_grants", [])
                if class_grants:
                    roles = ", ".join(g["role"] for g in class_grants)
                    profile_info += f"\n  → Rôles classe: {roles}"

                chunks.append({
                    "chunk_type": f"{category}_class",
                    "file_path":  item["file"],
                    "class_name": item["class"],
                    "content": (
                        f"La classe PHP {item['class']} (type: {category}) "
                        f"dans {item['file']}.{constructor_info}{profile_info}"
                    ),
                })

                # Un chunk par méthode (sauf __construct)
                for method in item.get("methods", []):
                    if method["name"] == "__construct":
                        continue

                    # Multi-routes → un chunk par route
                    routes = method.get("routes", [])

                    # Fallback legacy si pas de routes (compat ancien parseur)
                    if not routes and method.get("route"):
                        routes = [{"path": method["route"], "name": None, "http_methods": []}]

                    # Base commune de la description
                    base_info = f"Méthode '{method['name']}' ({item['class']})"
                    if method.get("renders"):
                        base_info += f" — Template: {method['renders']}"
                    if method.get("description"):
                        base_info += f" — {method['description']}"
                    if method.get("params"):
                        p_list = ", ".join(
                            f"{p['type'] or '?'} ${p['name']}"
                            for p in method["params"]
                        )
                        base_info += f" — Params: ({p_list})"

                    # Si pas de route du tout (méthode interne)
                    if not routes:
                        info = base_info
                        info += "\n  → Pas de route HTTP (méthode interne)"

                        # Voter checks
                        for vc in method.get("voter_checks", []):
                            info += f"\n  → Voter: denyAccessUnlessGranted('{vc['attribute']}'"
                            if vc.get("subject"):
                                info += f", ${vc['subject']}"
                            info += ")"

                        chunks.append({
                            "chunk_type": f"{category}_method",
                            "file_path":  item["file"],
                            "class_name": item["class"],
                            "content":    info,
                        })
                        continue

                    # Un chunk par route de la méthode
                    for route_info in routes:
                        info = base_info
                        info += f" — Route: {route_info['path']}"
                        if route_info.get("name"):
                            info += f" (name: {route_info['name']})"

                        # Verbes HTTP
                        http_methods = route_info.get("http_methods", [])
                        if http_methods:
                            info += f"\n  → Verbes HTTP: {', '.join(http_methods)}"

                        # IsGranted méthode
                        for g in method.get("method_grants", []):
                            info += f"\n  → Rôle requis (méthode): {g['role']}"

                        # Types de réponse enrichis
                        renders = method.get("renders")
                        response_types = method.get("response_types", [])

                        # Fallback legacy
                        if not response_types:
                            rtype = method.get("response_type")
                            if rtype:
                                response_types = [rtype]

                        if renders:
                            tpl_data = template_lookup.get(renders)
                            if tpl_data:
                                if tpl_data.get("h1"):
                                    info += f"\n  → H1: {', '.join(tpl_data['h1'][:3])}"
                                if tpl_data.get("inputs"):
                                    info += f"\n  → Champs formulaire: {', '.join(tpl_data['inputs'][:8])}"
                                if tpl_data.get("hidden_ids"):
                                    info += f"\n  → IDs cachés: {', '.join(tpl_data['hidden_ids'][:8])}"

                        for rt in response_types:
                            info += f"\n  → Type de réponse: {rt}"

                        # AJAX-only
                        if method.get("is_ajax_only"):
                            info += "\n  → AJAX uniquement (isXmlHttpRequest)"

                        # Formulaire
                        if method.get("has_form"):
                            form_type = method.get("form_type") or "?"
                            info += f"\n  → Formulaire: {form_type}"

                        # Voter checks
                        for vc in method.get("voter_checks", []):
                            info += f"\n  → Voter: denyAccessUnlessGranted('{vc['attribute']}'"
                            if vc.get("subject"):
                                info += f", ${vc['subject']}"
                            info += ")"

                        chunks.append({
                            "chunk_type": f"{category}_method",
                            "file_path":  item["file"],
                            "class_name": item["class"],
                            "content":    info,
                        })

        # Chunks Twig
        for tpl in analysis.get("templates", []):
            parts = [f"Le template {tpl['file']}."]
            if tpl.get("h1"):
                parts.append(f"Titres H1 : {', '.join(tpl['h1'])}.")
            if tpl.get("buttons"):
                parts.append(f"Boutons/Liens : {', '.join(tpl['buttons'])}.")
            if tpl.get("hidden_ids"):
                parts.append(f"IDs éléments cachés : {', '.join(tpl['hidden_ids'])}.")
            if tpl.get("links"):
                parts.append(f"Routes liées : {', '.join(tpl['links'])}.")
            if tpl.get("inputs"):
                parts.append(f"Champs de formulaire (name) : {', '.join(tpl['inputs'])}.")
            chunks.append({
                "chunk_type": "template_info",
                "file_path":  tpl["file"],
                "class_name": "N/A",
                "content":    " ".join(parts),
            })

        if not chunks:
            return {
                "status":  "warning",
                "message": "Aucun fichier PHP ou Twig trouvé dans le workspace.",
                "project_id": project_id,
            }

        # ── Encodage + sauvegarde ───────────────────────────────────────────
        vectors = brain.encode([c["content"] for c in chunks])
        db.init_schema(vector_size=len(vectors[0]))
        # Supprime les anciennes données du projet avant de ré-indexer
        db.clear_project(project_id)
        db.save_code_context(project_id, chunks, vectors)

        logging.info(
            f"[learn-from-code] {len(chunks)} chunks indexés pour '{project_id}'."
        )
        return {
            "project_id":   project_id,
            "status":       "success",
            "total_files":  total_files,
            "total_chunks": len(chunks),
        }

    except Exception as e:
        logging.error(f"[learn-from-code] Erreur : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# ENDPOINT — GÉNÉRATION DE TEST FONCTIONNEL
# ---------------------------------------------------------------------------

@app.post("/generate-test", summary="Génère un test fonctionnel Symfony (WebTestCase)")
async def generate_test(
    data: GenerateTestRequest,
    db: KnowledgeDB = Depends(get_db),
):
    """
    Génère un WebTestCase PHP à partir d'une description en langage naturel,
    en utilisant le contexte RAG (routes, templates, formulaires) du projet.
    """
    start_time = time.time()
    _check_model_allowed(data.model_name)
    brain = get_brain(model_name=data.model_name)
    query_vec = brain.encode([data.description])[0]

    # RAG : recherche du contexte pertinent
    # Quand class_name est fourni, on réduit la limite RAG générique (le filtre
    # post-retrieval garde les chunks de la classe + max 3 secondaires).
    rag_limit = 6 if data.class_name else 10
    context_chunks = db.find_closest_code_context(data.project_id, query_vec, limit=rag_limit)

    if data.class_name:
        direct_chunks = db.get_code_by_class_name(data.project_id, data.class_name)
        existing_contents = {c["content"] for c in context_chunks}
        for c in direct_chunks:
            if c["content"] not in existing_contents:
                context_chunks.insert(0, c)

    # Lookup templates : toujours injecter des chunks Twig dans le contexte
    tpl_limit = 3 if data.class_name else 5
    template_vec = brain.encode(["template twig h1 bouton lien formulaire champ"])[0]
    template_chunks = db.find_closest_code_context(data.project_id, template_vec, limit=tpl_limit)
    seen = {c["content"] for c in context_chunks}
    for c in template_chunks:
        if c["content"] not in seen:
            context_chunks.append(c)

    # Filtre post-retrieval pour réduire le bruit
    if data.class_name:
        context_chunks = filter_chunks_by_class(context_chunks, data.class_name)

    logging.info(
        f"[generate-test] {len(context_chunks)} chunks trouvés pour '{data.description[:60]}'"
    )

    # Détection du profil contrôleur
    controller_profile = _detect_controller_profile(context_chunks)
    class_role         = _detect_class_role(context_chunks)
    logging.info(
        f"[generate-test] Profil détecté : {controller_profile}, "
        f"rôle classe : {class_role}"
    )

    # Contrôleur interne → rediriger vers /generate-unit-test
    if controller_profile == "internal":
        logging.info(
            f"[generate-test] Contrôleur interne détecté — "
            f"suggestion de test unitaire."
        )
        return {
            "status":       "redirect_to_unit",
            "message": (
                f"Le contrôleur '{data.class_name or '?'}' n'a aucune route HTTP "
                f"(profil: internal). Utilisez POST /generate-unit-test à la place "
                f"pour générer des tests unitaires avec mocks."
            ),
            "controller_profile": controller_profile,
            "context_used": _build_context_str(context_chunks),
            "time_sec":     round(time.time() - start_time, 2),
        }

    # Chemin déterministe (bypass LLM)
    if data.deterministic and data.class_name:
        safe_name = re.sub(r"[^a-zA-Z0-9]", "", data.test_name or f"{data.class_name}Test")
        filename  = f"{safe_name}.php"
        rel_path  = f"tests/Functional/Controller/{filename}"
        code = _generate_php_test_from_chunks(
            chunks=context_chunks,
            class_name=data.class_name,
            test_class_name=safe_name,
        )
        _write_php_file(rel_path, code)
        logging.info(f"[generate-test] Fichier généré (déterministe) : {rel_path}")
        return {
            "status":             "success",
            "mode":               "deterministic",
            "controller_profile": controller_profile,
            "file":               filename,
            "path":               rel_path,
            "context_used":       _build_context_str(context_chunks),
            "time_sec":           round(time.time() - start_time, 2),
        }

    context_str = _build_context_str(context_chunks)

    # Prompt adaptatif selon le profil

    fw       = settings.auth_firewall_name
    redirect = settings.auth_redirect_path
    code_redir = settings.auth_redirect_status
    factory  = settings.auth_test_class
    sso      = settings.auth_sso_user_class

    # Règles de base communes
    base_rules = f"""STRUCTURE OBLIGATOIRE de chaque fichier de test généré :

namespace App\\Tests\\Functional\\Controller;

use {factory};
use {sso};
use Symfony\\Bundle\\FrameworkBundle\\KernelBrowser;
use Symfony\\Bundle\\FrameworkBundle\\Test\\WebTestCase;
use Symfony\\Component\\HttpFoundation\\Response;

final class XxxTest extends WebTestCase
{{
    protected KernelBrowser $client;
    private {factory.split(chr(92))[-1]} $testUserFactory;

    protected function setUp(): void
    {{
        $this->client = self::createClient();
        $this->testUserFactory = $this->client->getContainer()->get({factory.split(chr(92))[-1]}::class);
    }}

    private function getTestUser(string $key): {sso.split(chr(92))[-1]}
    {{
        return $this->testUserFactory->create($key);
    }}
    // méthodes de test ici
}}

RÈGLES GÉNÉRALES :
1. Chaque route → 2 tests minimum :
   - Sans auth : assertResponseStatusCodeSame({code_redir}) + assertResponseRedirects('{redirect}')
   - Avec auth : loginUser puis assertion selon le type de réponse.
2. loginUser utilise le RÔLE indiqué dans le contexte (→ Rôle requis ou → Rôles classe), PAS toujours ADMIN.
   Format : $this->client->loginUser($this->getTestUser('ROLE_XXX'), '{fw}').
3. JAMAIS appeler followRedirects() sauf si le test doit lire le contenu de la page finale.
4. Réponses attendues après auth :
   - "→ render (200)"    → assertResponseIsSuccessful(). Ajouter assertSelectorTextContains SEULEMENT si "→ H1:" est présent.
   - "→ redirect (302)"  → assertResponseRedirects() SANS argument.
   - "→ json (200)"      → assertResponseIsSuccessful() + assertJson(...)
   - "→ file_download"   → assertResponseIsSuccessful() + vérifier Content-Disposition.
   - "→ export (200)"    → assertResponseIsSuccessful().
   - pas de type indiqué → assertResponseIsSuccessful() UNIQUEMENT.
5. INTERDIT d'inventer un texte H1, sélecteur CSS ou URL.
6. Paramètre de route {{param}} → valeur réaliste (ex: dupont, 1).
7. Méthode sans route → commentaire (pas de test HTTP).
8. Retourne UNIQUEMENT le code PHP brut commençant par <?php, fichier COMPLET."""

    # Règles spécifiques selon le profil
    profile_rules = ""
    if controller_profile == "api" or controller_profile == "mixed":
        profile_rules = """

RÈGLES SPÉCIFIQUES (contrôleur API / mixte) :
9. Routes marquées [AJAX] → ajouter un test SANS header XHR qui vérifie le status 404.
   Pour le test authentifié AVEC header XHR : $this->client->request('GET', '/url', [], [], ['HTTP_X-Requested-With' => 'XMLHttpRequest']).
10. Utiliser le verbe HTTP indiqué dans [Verbes HTTP:] (POST, PUT, DELETE...), PAS toujours GET.
11. Routes avec [Form: XxxType] → tester au minimum : soumission vide (erreur attendue) + soumission valide.
12. Routes avec [Voter: xxx] → si possible ajouter un test d'accès refusé (403)."""

    elif controller_profile == "web_crud":
        profile_rules = f"""

RÈGLES SPÉCIFIQUES (contrôleur web CRUD) :
9. Si un rôle spécifique est indiqué pour une méthode (→ Rôle requis), tester aussi qu'un utilisateur
   avec seulement le rôle de la classe ({class_role}) reçoit un 403 sur cette méthode.
10. Utiliser le verbe HTTP indiqué dans [Verbes HTTP:], PAS toujours GET.
11. Routes avec [Form: XxxType] → tester la soumission POST.
12. Routes marquées [AJAX] → ajouter un test sans header XHR → 404."""

    system_message = f"""Tu es un expert Symfony senior spécialisé dans les tests fonctionnels PHPUnit / WebTestCase.

{base_rules}{profile_rules}

{context_str}"""

    few_shots = _load_golden_dataset("golden_dataset_functional.json", profile=controller_profile)
    routes_summary = _build_routes_summary(context_chunks)
    user_prompt = f"{routes_summary}\n\n{data.description}" if routes_summary else data.description
    prompt = _build_prompt(system_message, few_shots, user_prompt)

    try:
        logging.info(f"[generate-test] Taille du prompt : {len(prompt)} chars")
        code = _call_llm(prompt)
        gen_time = time.time() - start_time

        missing_routes = _validate_coverage(code, context_chunks)
        if missing_routes:
            logging.warning(f"[generate-test] Routes non couvertes : {missing_routes}")
            fix_prompt = _build_prompt(
                system_message,
                few_shots,
                f"Routes manquantes : {missing_routes}\n\n"
                f"Complète le fichier suivant en ajoutant les tests manquants. "
                f"Retourne TOUT le fichier corrigé commençant par <?php, sans markdown.\n\n"
                f"Code actuel :\n{code}"
            )
            code = _call_llm(fix_prompt)

        # Validation + correction syntaxique
        error = validate_php_syntax(code)
        if error:
            logging.warning(f"[generate-test] Erreur syntaxe PHP : {error}")
            fix_system = "Corrige l'erreur de syntaxe PHP suivante et retourne TOUT le code corrigé commençant par <?php, sans markdown."
            fix_prompt = _build_prompt(fix_system, [], f"Erreur : {error}\n\nCode à corriger :\n{code}")
            code = _call_llm(fix_prompt)

        # Écriture du fichier dans le projet
        safe_name = re.sub(r"[^a-zA-Z0-9]", "", data.test_name or "GeneratedTest")
        filename  = f"{safe_name}.php"
        rel_path  = f"tests/Functional/Controller/{filename}"
        _write_php_file(rel_path, code)

        return {
            "status":             "success",
            "controller_profile": controller_profile,
            "file":               filename,
            "path":               rel_path,
            "context_used":       context_str,
            "time_sec":           round(gen_time, 2),
        }

    except Exception as e:
        logging.error(f"[generate-test] Erreur : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# ENDPOINT — GÉNÉRATION DE TEST UNITAIRE
# ---------------------------------------------------------------------------

@app.post("/generate-unit-test", summary="Génère un test unitaire PHPUnit")
async def generate_unit_test(
    data: GenerateUnitTestRequest,
    db: KnowledgeDB = Depends(get_db),
):
    """
    Génère un test unitaire PHPUnit pour une classe ou méthode donnée,
    en injectant son code source + le contexte des classes dépendantes (RAG).
    """
    start_time = time.time()

    # Extraction du code source à tester (path traversal protégé via _safe_join)
    abs_file_path = _safe_join(settings.container_project_root, data.file_path)
    code_lines = extract_code_for_symbol(abs_file_path, data.class_name, data.method_name)
    if not code_lines:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Classe '{data.class_name}'"
                + (f" / méthode '{data.method_name}'" if data.method_name else "")
                + f" introuvable dans '{data.file_path}'."
            ),
        )

    extracted_code = "\n".join(code_lines)

    # RAG : classes dépendantes (injection de dépendances, FormTypes, etc.)
    potential_classes = re.findall(r"([A-Z][a-zA-Z0-9]+)::class", extracted_code)
    constructor_types = re.findall(
        r"private\s+(?:readonly\s+)?([A-Z][a-zA-Z0-9]+)\s+\$", extracted_code
    )
    ignored_classes = {
        "Request", "Response", "AbstractController", "Yaml",
        "EntityManager", "FormView",
    }
    classes_to_fetch = set(potential_classes + constructor_types) - ignored_classes

    related_context = ""
    for cls in classes_to_fetch:
        results = db.get_code_by_class_name(data.project_id, cls)
        if results:
            related_context += f"\nStructure de '{cls}':\n{results[0]['content']}\n"

    system_message = f"""Tu es un expert Symfony (PHPUnit + Mocking).
IMPORTANT : Ne génère JAMAIS de test de login sauf si le mot "login" ou "auth" apparaît explicitement dans le code ci-dessous.

CONTEXTE DES CLASSES LIÉES :
{related_context or "(aucun)"}

CODE À TESTER :
{extracted_code}

DIRECTIVES :
1. Identifie les dépendances injectées dans le constructeur.
2. Utilise $this->createMock() pour simuler les services injectés.
3. Utilise PHPUnit\\Framework\\TestCase (PAS WebTestCase).
4. Retourne UNIQUEMENT le code PHP, sans markdown, commençant par <?php.
"""

    few_shots = _load_golden_dataset("golden_dataset.json")
    prompt = _build_prompt(system_message, few_shots, data.description)

    try:
        code = _call_llm(prompt)
        gen_time = time.time() - start_time

        # Validation + correction syntaxique (même logique que /generate-test)
        error = validate_php_syntax(code)
        if error:
            logging.warning(f"[generate-unit-test] Erreur syntaxe PHP : {error}")
            fix_prompt = _build_prompt(
                system_message,
                few_shots,
                f"Erreur : {error}\n\nCode à corriger :\n{code}"
            )
            code = _call_llm(fix_prompt)

        # Détermine le sous-dossier selon le type de classe
        # class_short est sanitizé pour empêcher tout `..` ou `/` injecté via class_name
        class_short = _sanitize_path_component(data.class_name.split("\\")[-1])
        category    = "Service" if "Service" in data.file_path else (
                      "Controller" if "Controller" in data.file_path else "Unit"
                  )
        rel_path    = f"tests/Unit/{category}/{class_short}Test.php"
        _write_php_file(rel_path, code)

        return {
            "status":   "success",
            "file":     f"{class_short}Test.php",
            "path":     rel_path,
            "time_sec": round(gen_time, 2),
        }

    except Exception as e:
        logging.error(f"[generate-unit-test] Erreur : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# ENDPOINTS LEGACY (dépréciés, conservés pour compatibilité)
# ---------------------------------------------------------------------------

@app.post("/learn", include_in_schema=False)
async def legacy_learn():
    return {
        "status":  "deprecated",
        "message": "Utilisez POST /learn-from-code",
    }


@app.post("/predict", include_in_schema=False)
async def legacy_predict():
    return {
        "status":  "deprecated",
        "message": "Utilisez POST /generate-test",
    }
