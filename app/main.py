from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
# requests n'est plus appelé directement ici (déplacé dans llm_client.py),
# mais l'import doit rester : test_main.py patche `main.requests.post`.
import requests
import re
from typing import List, Optional, Dict
import os
import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
import logging
from code_parser import analyze_project_code, extract_code_for_symbol
from chunk_format import MAX_METHOD_H1, MAX_METHOD_FIELDS
from config import settings
from brain import SemanticEngine
from db import KnowledgeDB

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# DÉPENDANCES (singletons mis en cache)

@lru_cache
def get_db() -> KnowledgeDB:
    # settings.database_url et NON os.getenv("DATABASE_URL") : os.getenv ne lit que
    # les variables du processus, jamais le fichier .env. Une installation locale
    # (hors Docker) configurée uniquement par .env — le mode décrit dans le README —
    # recevait donc None et ne pouvait pas se connecter. En Docker rien ne change :
    # docker-compose.yml pose de vraies variables d'environnement, que pydantic-settings
    # lit en priorité sur le fichier.
    return KnowledgeDB(settings.database_url)


def _allowed_embedding_models() -> set:
    """Set des modèles autorisés (default toujours inclus)."""
    raw = settings.allowed_embedding_models or ""
    allowed = {m.strip() for m in raw.split(",") if m.strip()}
    allowed.add(settings.default_embedding_model)
    return allowed


def require_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> None:
    """
    Dépendance d'auth : si settings.api_key est défini, exige le header X-API-Key
    avec la même valeur. Sinon (dev), laisse passer.
    """
    expected = settings.api_key
    if not expected:
        return
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="API key invalide ou manquante")


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

    # ── Avertissements de configuration ──────────────────────────────────────
    if not settings.api_key:
        logging.warning(
            "API_KEY non défini — tous les endpoints d'écriture sont accessibles "
            "sans authentification. Définir API_KEY dans docker-compose.yml (ou .env) "
            "avant tout déploiement en production."
        )
    else:
        logging.info("API_KEY configuré — authentification activée sur les endpoints d'écriture.")

    if not settings.cors_origins:
        logging.info("CORS_ORIGINS vide — middleware CORS désactivé (mode dev).")
    else:
        logging.info(f"CORS activé pour : {settings.cors_origins}")

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


# APPLICATION FASTAPI

app = FastAPI(
    title="Moteur de Test IA",
    description=(
        "Microservice pour analyser un projet Symfony et générer automatiquement "
        "des tests fonctionnels (WebTestCase) et unitaires (PHPUnit) via RAG + LLM."
    ),
    version="1.2.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "santé",        "description": "Health check & monitoring."},
        {"name": "projets",      "description": "Stats et nettoyage des projets indexés."},
        {"name": "indexation",   "description": "Apprentissage RAG depuis le code Symfony."},
        {"name": "génération",   "description": "Génération de tests fonctionnels et unitaires."},
        {"name": "admin",        "description": "Opérations destructrices (reset schema)."},
    ],
)

# CORS — activé uniquement si CORS_ORIGINS est défini. Vide = désactivé
# (cohérent avec le README et le message loggé au démarrage dans lifespan()).
# La console HTML (/ui) est servie en same-origin par brain-api : elle n'a pas
# besoin de CORS pour fonctionner, contrairement à un appelant cross-origin.
_cors_origins = [o.strip() for o in (settings.cors_origins or "").split(",") if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["X-API-Key", "Content-Type"],
    )

# Console HTML
_STATIC_DIR = Path(__file__).parent / "static"

@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def console_ui():
    """Console web interactive pour l'API."""
    return (_STATIC_DIR / "index.html").read_text(encoding="utf-8")


# MODÈLES PYDANTIC (voir models.py)
from models import (  # noqa: E402
    LearnFromCodeRequest,
    GenerateTestRequest,
    GenerateUnitTestRequest,
    ResetSchemaRequest,
    GenerateTestsBatchRequest,
)

# HELPERS

from php_writer import (  # noqa: E402
    validate_php_syntax, _safe_join, _sanitize_path_component, _write_php_file,
)

from llm_client import (  # noqa: E402
    _load_golden_dataset, _check_ollama_alive, _call_llm, _build_prompt,
)


from rag_context import (  # noqa: E402
    _build_context_str, _build_routes_summary, _validate_coverage,
    filter_chunks_by_class,
    RAG_LIMIT_WITH_CLASS, RAG_LIMIT_WITHOUT_CLASS,
    TEMPLATE_LIMIT_WITH_CLASS, TEMPLATE_LIMIT_WITHOUT_CLASS,
)


# GÉNÉRATEUR DÉTERMINISTE + MOTEUR DE SCÉNARIOS — voir deterministic_generator.py
# (couplage fragile avec chunk_format.py documenté là-bas)
from deterministic_generator import (  # noqa: E402
    _detect_controller_profile,
    _detect_class_role,
    _generate_php_test_from_chunks,
    _extract_entity_types_from_chunks,
    _generate_fixtures_skeleton,
)
# Génération LLM par route (anti-troncature) — voir per_route_generator.py
from per_route_generator import generate_functional_test_per_route  # noqa: E402

# JOB STORE — suivi des tâches asynchrones (voir job_store.py)
from job_store import _new_job, _update_job, _get_job  # noqa: E402


# ENDPOINTS — SANTÉ ET ADMINISTRATION

@app.get("/health", summary="Vérification de santé", tags=["santé"])
async def health_check(request: Request, db: KnowledgeDB = Depends(get_db)):
    """
    Vérifie que l'API et la base de données sont opérationnelles.

    Volontairement sans authentification : la sonde du conteneur doit pouvoir
    l'interroger. En revanche la LISTE des projets indexés — qui révèle des noms
    d'applications internes — n'est renvoyée qu'à un appelant authentifié. Sans
    clé, on ne donne que le compte.
    """
    try:
        projects = db.list_projects()
        reponse = {"status": "ok", "projects_indexed": len(projects)}
        attendue = settings.api_key
        if not attendue or request.headers.get("X-API-Key") == attendue:
            reponse["projects"] = projects
        return reponse
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB non disponible : {e}")


@app.get(
    "/projects",
    summary="Liste les projets indexés",
    tags=["projets"],
    dependencies=[Depends(require_api_key)],
)
async def list_projects(db: KnowledgeDB = Depends(get_db)):
    """Retourne la liste des project_id présents dans la base vectorielle."""
    try:
        return db.list_projects()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/project/{project_id}/stats",
    summary="Statistiques d'un projet",
    tags=["projets"],
    dependencies=[Depends(require_api_key)],
)
async def project_stats(project_id: str, db: KnowledgeDB = Depends(get_db)):
    """Retourne le nombre de chunks et de routes indexés pour un projet."""
    try:
        return db.get_project_stats(project_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete(
    "/project/{project_id}",
    summary="Supprime les données d'un projet",
    tags=["projets"],
    dependencies=[Depends(require_api_key)],
)
async def delete_project(project_id: str, db: KnowledgeDB = Depends(get_db)):
    """Supprime toutes les données indexées pour un projet donné."""
    try:
        db.clear_project(project_id)
        return {"status": "deleted", "project_id": project_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/admin/reset-schema",
    summary="Réinitialise toute la base de données",
    tags=["admin"],
    dependencies=[Depends(require_api_key)],
)
def reset_schema(
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


@app.post(
    "/index-project",
    summary="Alias pour /learn-from-code",
    tags=["indexation"],
    dependencies=[Depends(require_api_key)],
    include_in_schema=False,
)
def index_project_alias(
    data: LearnFromCodeRequest,
    db: KnowledgeDB = Depends(get_db),
):
    return learn_from_code(data, db)


# ENDPOINT — GÉNÉRATION DE TEST FONCTIONNEL

@app.post(
    "/learn-from-code",
    summary="Indexe le code source du projet Symfony",
    tags=["indexation"],
    dependencies=[Depends(require_api_key)],
)
def learn_from_code(
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

        # Chunks PHP (contrôleurs, entités, services, etc.)
        for category, items in analysis.items():
            if category == "templates":
                continue
            for item in items:
                # Chunk décrivant la classe entière
                constructor_info = ""
                if item.get("constructor_params"):
                    params_str = ", ".join(
                        f"{p['type'] or 'mixed'} ${p['name'] or '?'}"
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
                    roles = ", ".join(g["role"] or "?" for g in class_grants)
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
                            f"{p['type'] or '?'} ${p['name'] or '?'}"
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
                                    info += f"\n  → H1: {', '.join(tpl_data['h1'][:MAX_METHOD_H1])}"
                                if tpl_data.get("inputs"):
                                    info += f"\n  → Champs formulaire: {', '.join(tpl_data['inputs'][:MAX_METHOD_FIELDS])}"
                                if tpl_data.get("hidden_ids"):
                                    info += f"\n  → IDs cachés: {', '.join(tpl_data['hidden_ids'][:MAX_METHOD_FIELDS])}"

                        for rt in response_types:
                            info += f"\n  → Type de réponse: {rt}"

                        # AJAX-only
                        if method.get("is_ajax_only"):
                            info += "\n  → AJAX uniquement (isXmlHttpRequest)"

                        # Formulaire
                        if method.get("has_form"):
                            form_type = method.get("form_type") or "?"
                            info += f"\n  → Formulaire: {form_type}"

                        # Verbe HTTP inféré du body (lit POST data, isMethod, upload…)
                        if method.get("body_inferred_verb"):
                            info += f"\n  → Verbe HTTP inféré (body): {method['body_inferred_verb']}"
                        if method.get("body_reads"):
                            info += f"\n  → Lit: {', '.join(r for r in method['body_reads'] if r)}"

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

        # Encodage + sauvegarde
        vectors = brain.encode([c["content"] for c in chunks])
        db.init_schema(vector_size=len(vectors[0]))
        # Ré-indexation atomique : delete + insert dans la même transaction
        # → pas de fenêtre où le projet apparaît vide aux requêtes concurrentes.
        db.reindex_project(project_id, chunks, vectors)

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

# ENDPOINT — GÉNÉRATION DE TEST FONCTIONNEL

@app.post(
    "/generate-test",
    summary="Génère un test fonctionnel Symfony (WebTestCase)",
    tags=["génération"],
    dependencies=[Depends(require_api_key)],
)
def generate_test(
    data: GenerateTestRequest,
    background_tasks: BackgroundTasks,
    db: KnowledgeDB = Depends(get_db),
):
    """
    Génère un WebTestCase PHP à partir d'une description en langage naturel,
    en utilisant le contexte RAG (routes, templates, formulaires) du projet.
    Avec async_mode=true : retourne un job_id immédiatement, poll GET /job/{id}/status.
    """
    if data.async_mode:
        job_id = _new_job(
            type="functional",
            project_id=data.project_id,
            class_name=data.class_name or "?",
        )
        background_tasks.add_task(_bg_generate_test, job_id, data, db)
        return {
            "status":   "accepted",
            "job_id":   job_id,
            "poll_url": f"/job/{job_id}/status",
        }

    start_time = time.time()
    _check_model_allowed(data.model_name)
    brain = get_brain(model_name=data.model_name)
    query_vec = brain.encode([data.description])[0]

    # RAG : recherche du contexte pertinent
    # Quand class_name est fourni, on réduit la limite RAG générique (le filtre
    # post-retrieval garde les chunks de la classe + max 3 secondaires).
    rag_limit = RAG_LIMIT_WITH_CLASS if data.class_name else RAG_LIMIT_WITHOUT_CLASS
    context_chunks = db.find_closest_code_context(data.project_id, query_vec, limit=rag_limit)

    if data.class_name:
        direct_chunks = db.get_code_by_class_name(data.project_id, data.class_name)
        existing_contents = {c["content"] for c in context_chunks}
        for c in direct_chunks:
            if c["content"] not in existing_contents:
                context_chunks.insert(0, c)

    # Lookup templates : seulement si le contrôleur rend des vues (render).
    # Pour un contrôleur API pur (json), c'est du bruit qui consomme le budget tokens.
    needs_templates = any(
        "render" in c.get("content", "").lower() or "template:" in c.get("content", "").lower()
        for c in context_chunks
    )
    if needs_templates:
        tpl_limit = TEMPLATE_LIMIT_WITH_CLASS if data.class_name else TEMPLATE_LIMIT_WITHOUT_CLASS
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
            test_class_name=safe_name,
        )
        _write_php_file(rel_path, code, overwrite=data.overwrite, allow_unsafe=data.allow_unsafe)
        logging.info(f"[generate-test] Fichier généré (déterministe) : {rel_path}")

        # Génère aussi un squelette de DataFixtures (TODO à compléter par la MOA)
        entity_types = _extract_entity_types_from_chunks(context_chunks)
        fixtures_path = None
        if entity_types:
            fixtures_code = _generate_fixtures_skeleton(entity_types, safe_name)
            fixtures_filename = f"{safe_name}Fixtures.php"
            fixtures_path = f"tests/Functional/Fixtures/{fixtures_filename}"
            _write_php_file(fixtures_path, fixtures_code, overwrite=data.overwrite, allow_unsafe=data.allow_unsafe)
            logging.info(f"[generate-test] Fixtures squelette : {fixtures_path}")

        return {
            "status":             "success",
            "mode":               "deterministic",
            "controller_profile": controller_profile,
            "file":               filename,
            "path":               rel_path,
            "fixtures_path":      fixtures_path,
            "entities_detected":  entity_types,
            "context_used":       _build_context_str(context_chunks),
            "time_sec":           round(time.time() - start_time, 2),
            "note":               "Brouillon généré automatiquement — à relire et adapter avant exécution (fixtures, valeurs de test, cas limites).",
        }

    # Chemin LLM PAR ROUTE (défaut) — un appel Ollama par route, anti-troncature.
    # L'ancien chemin monolithique (ci-dessous) reste atteignable via per_route=false.
    if data.per_route:
        safe_name = re.sub(r"[^a-zA-Z0-9]", "", data.test_name or f"{data.class_name or 'Generated'}Test")
        filename  = f"{safe_name}.php"
        rel_path  = f"tests/Functional/Controller/{filename}"
        try:
            result = generate_functional_test_per_route(
                chunks=context_chunks,
                class_name=data.class_name or "",
                test_class_name=safe_name,
            )
        except Exception as e:
            logging.error(f"[generate-test] Erreur génération par route : {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        _write_php_file(rel_path, result["code"], overwrite=data.overwrite, allow_unsafe=data.allow_unsafe)
        logging.info(
            f"[generate-test] Par route : {result['routes_generated']}/{result['routes_total']} "
            f"routes OK, {len(result['routes_failed'])} en échec, php_valid={result['php_valid']}"
        )

        note = "Brouillon généré automatiquement (par route) — à relire et adapter avant exécution (fixtures, valeurs de test, cas limites)."
        if result["routes_failed"]:
            note += (
                f" ⚠ {len(result['routes_failed'])} route(s) en échec de génération, "
                f"marquée(s) markTestIncomplete : {', '.join(result['routes_failed'])}."
            )
        if result.get("routes_skipped"):
            note += (
                f" ⏱ Budget de temps ({result['elapsed_sec']}s) atteint : "
                f"{len(result['routes_skipped'])} route(s) non tentée(s) — "
                f"{', '.join(result['routes_skipped'])}. Relancez la génération "
                f"sur ces routes, ou utilisez deterministic=true (instantané)."
            )

        # Le statut doit refléter ce qui s'est réellement passé. Renvoyer
        # "success" alors qu'aucune route n'a abouti — fichier entièrement
        # composé de markTestIncomplete — laissait croire à une génération
        # réussie. On distingue désormais trois issues.
        if result["routes_generated"] == 0 and result["routes_total"] > 0:
            statut = "failed"
        elif result["routes_failed"] or result.get("routes_skipped"):
            statut = "partial"
        else:
            statut = "success"

        return {
            "status":             statut,
            "mode":               "llm_per_route",
            "controller_profile": controller_profile,
            "file":               filename,
            "path":               rel_path,
            "context_used":       _build_context_str(context_chunks),
            "time_sec":           round(time.time() - start_time, 2),
            "note":               note,
            # Clés ajoutées (rétro-compatibles — Streamlit ignore les clés inconnues) :
            "php_valid":          result["php_valid"],
            "routes_total":       result["routes_total"],
            "routes_generated":   result["routes_generated"],
            "routes_failed":      result["routes_failed"],
            "per_route":          result["per_route"],
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
        if missing_routes and len(missing_routes) > settings.llm_coverage_retry_max:
            logging.warning(
                f"[generate-test] {len(missing_routes)} routes non couvertes — "
                f"retry de couverture ignoré (> {settings.llm_coverage_retry_max}). "
                "Génère par méthode/sous-ensemble pour une couverture complète."
            )
        elif missing_routes:
            logging.warning(f"[generate-test] Routes non couvertes : {missing_routes}")
            fix_prompt = _build_prompt(
                system_message,
                few_shots,
                f"Routes manquantes : {missing_routes}\n\n"
                f"Complète le fichier suivant en ajoutant les tests manquants. "
                f"Retourne TOUT le fichier corrigé commençant par <?php, sans markdown.\n\n"
                f"Code actuel :\n{code}"
            )
            code = _call_llm(fix_prompt, _skip_health=True)
            # Re-validation : on logge si le retry n'a toujours pas comblé le manque
            still_missing = _validate_coverage(code, context_chunks)
            if still_missing:
                logging.warning(
                    f"[generate-test] Après retry, routes toujours manquantes : {still_missing}"
                )

        # Validation + correction syntaxique
        error = validate_php_syntax(code)
        if error:
            logging.warning(f"[generate-test] Erreur syntaxe PHP : {error}")
            fix_system = "Corrige l'erreur de syntaxe PHP suivante et retourne TOUT le code corrigé commençant par <?php, sans markdown."
            fix_prompt = _build_prompt(fix_system, [], f"Erreur : {error}\n\nCode à corriger :\n{code}")
            code = _call_llm(fix_prompt, _skip_health=True)

        # Écriture du fichier dans le projet
        safe_name = re.sub(r"[^a-zA-Z0-9]", "", data.test_name or "GeneratedTest")
        filename  = f"{safe_name}.php"
        rel_path  = f"tests/Functional/Controller/{filename}"
        _write_php_file(rel_path, code, overwrite=data.overwrite, allow_unsafe=data.allow_unsafe)

        return {
            "status":             "success",
            "mode":               "llm",
            "controller_profile": controller_profile,
            "file":               filename,
            "path":               rel_path,
            "context_used":       context_str,
            "time_sec":           round(gen_time, 2),
            "note":               "Brouillon généré automatiquement — à relire et adapter avant exécution (fixtures, valeurs de test, cas limites).",
        }

    except Exception as e:
        logging.error(f"[generate-test] Erreur : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ENDPOINT — GÉNÉRATION DE TEST UNITAIRE

@app.post(
    "/generate-unit-test",
    summary="Génère un test unitaire PHPUnit",
    tags=["génération"],
    dependencies=[Depends(require_api_key)],
)
def generate_unit_test(
    data: GenerateUnitTestRequest,
    background_tasks: BackgroundTasks,
    db: KnowledgeDB = Depends(get_db),
):
    """
    Génère un test unitaire PHPUnit pour une classe ou méthode donnée,
    en injectant son code source + le contexte des classes dépendantes (RAG).
    Avec async_mode=true : retourne un job_id immédiatement, poll GET /job/{id}/status.
    """
    if data.async_mode:
        job_id = _new_job(
            type="unit",
            project_id=data.project_id,
            class_name=data.class_name,
        )
        background_tasks.add_task(_bg_generate_unit_test, job_id, data, db)
        return {
            "status":   "accepted",
            "job_id":   job_id,
            "poll_url": f"/job/{job_id}/status",
        }

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
            code = _call_llm(fix_prompt, _skip_health=True)

        # Détermine le sous-dossier selon le type de classe
        # class_short est sanitizé pour empêcher tout `..` ou `/` injecté via class_name
        class_short = _sanitize_path_component(data.class_name.split("\\")[-1])
        category    = "Service" if "Service" in data.file_path else (
                      "Controller" if "Controller" in data.file_path else "Unit"
                  )
        rel_path    = f"tests/Unit/{category}/{class_short}Test.php"
        _write_php_file(rel_path, code, overwrite=data.overwrite, allow_unsafe=data.allow_unsafe)

        return {
            "status":   "success",
            "file":     f"{class_short}Test.php",
            "path":     rel_path,
            "time_sec": round(gen_time, 2),
            "note":     "Brouillon généré automatiquement — à relire et adapter avant exécution (mocks, cas limites, assertions métier).",
        }

    except Exception as e:
        logging.error(f"[generate-unit-test] Erreur : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# WRAPPERS ASYNCHRONES
# Appelés par BackgroundTasks — exécutés APRÈS l'envoi de la réponse HTTP.
# Toutes les exceptions sont capturées et stockées dans le job store.

def _bg_generate_test(job_id: str, data: GenerateTestRequest, db: KnowledgeDB) -> None:
    """Lance generate_test en arrière-plan et met à jour le job store."""
    try:
        _update_job(job_id, status="running")
        # Appel synchrone avec async_mode=False pour éviter la récursion
        sync_data = data.model_copy(update={"async_mode": False})
        # On simule l'appel direct à la logique métier via BackgroundTasks
        from fastapi import BackgroundTasks as _BT
        dummy_bt = _BT()
        result = generate_test(sync_data, dummy_bt, db)
        _update_job(job_id, status="done", result=result)
    except Exception as e:
        logging.error(f"[job:{job_id}] Erreur generate_test : {e}", exc_info=True)
        _update_job(job_id, status="error", error=str(e))


def _bg_generate_unit_test(job_id: str, data: GenerateUnitTestRequest, db: KnowledgeDB) -> None:
    """Lance generate_unit_test en arrière-plan et met à jour le job store."""
    try:
        _update_job(job_id, status="running")
        sync_data = data.model_copy(update={"async_mode": False})
        from fastapi import BackgroundTasks as _BT
        dummy_bt = _BT()
        result = generate_unit_test(sync_data, dummy_bt, db)
        _update_job(job_id, status="done", result=result)
    except Exception as e:
        logging.error(f"[job:{job_id}] Erreur generate_unit_test : {e}", exc_info=True)
        _update_job(job_id, status="error", error=str(e))


# ENDPOINT — STATUT D'UN JOB ASYNCHRONE

@app.get(
    "/job/{job_id}/status",
    summary="Statut d'une génération asynchrone",
    tags=["génération"],
    dependencies=[Depends(require_api_key)],
)
def job_status(job_id: str):
    """
    Retourne l'état d'une tâche lancée avec async_mode=true.
    Statuts : pending → running → done | error.
    Quand status=done, le champ `result` contient la réponse complète.
    """
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' introuvable ou expiré (TTL 2h).")
    # On n'expose pas le timestamp interne
    job.pop("ts", None)
    # On s'assure que job_id est toujours dans la réponse (pratique côté client)
    job.setdefault("job_id", job_id)
    return job


# ENDPOINT — GÉNÉRATION EN LOT (éclatement par classe)

@app.post(
    "/generate-tests-batch",
    # « en tâche de fond » et non « en parallèle » : BackgroundTasks exécute les
    # classes l'une après l'autre. C'est la requête qui rend la main aussitôt,
    # pas la génération qui se parallélise.
    summary="Génère des tests pour plusieurs classes en tâche de fond (async)",
    tags=["génération"],
    dependencies=[Depends(require_api_key)],
)
def generate_tests_batch(
    data: GenerateTestsBatchRequest,
    background_tasks: BackgroundTasks,
    db: KnowledgeDB = Depends(get_db),
):
    """
    Lance la génération de tests pour chaque classe listée en arrière-plan.
    Retourne un batch_id et un job_id par classe.
    Poll GET /job/{job_id}/status pour suivre chaque classe individuellement.
    """
    batch_id = uuid.uuid4().hex[:8]
    jobs = []

    for class_name in data.class_names:
        safe = re.sub(r"[^a-zA-Z0-9]", "", class_name)
        req = GenerateTestRequest(
            project_id=data.project_id,
            description=f"{data.description_prefix} {class_name}",
            class_name=class_name,
            test_name=f"{safe}Test",
            deterministic=data.deterministic,
            async_mode=False,  # exécuté directement dans le thread du lot
        )
        job_id = _new_job(type="functional", project_id=data.project_id, class_name=class_name)
        background_tasks.add_task(_bg_generate_test, job_id, req, db)
        jobs.append({"class_name": class_name, "job_id": job_id, "poll_url": f"/job/{job_id}/status"})

    logging.info(f"[batch:{batch_id}] {len(jobs)} jobs lancés pour project_id={data.project_id}")
    return {
        "status":   "accepted",
        "batch_id": batch_id,
        "jobs":     jobs,
        "note":     f"{len(jobs)} générations lancées en arrière-plan — poll chaque poll_url pour le résultat.",
    }
