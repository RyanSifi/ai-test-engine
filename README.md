# AI Test Engine

Microservice FastAPI qui génère automatiquement des tests PHP (PHPUnit / WebTestCase Symfony) pour des applications Symfony, en utilisant un LLM local (Ollama) et un index vectoriel (PostgreSQL + pgvector) pour le RAG.

## Stack

- **API** : FastAPI + Uvicorn (Python 3.11)
- **LLM local** : Ollama — `qwen2.5-coder:7b` par défaut
- **Embeddings** : `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dims)
- **Vector store** : PostgreSQL 15 + pgvector (cosinus)
- **Conteneurs** : Docker Compose

## Démarrage rapide

```bash
# 1. Configuration
cp .env.example .env
# Éditer .env (DATABASE_URL est obligatoire, le reste a des défauts)

# 2. Démarrage
docker compose up -d

# 3. Vérifier que tout est OK
curl http://localhost:8000/health

# 4. Indexer un projet Symfony (placé dans le dossier parent)
curl -X POST http://localhost:8000/learn-from-code \
  -H "Content-Type: application/json" \
  -d '{"project_id": "mon-projet"}'

# 5. Générer un test
curl -X POST http://localhost:8000/generate-test \
  -H "Content-Type: application/json" \
  -d '{
    "project_id":  "mon-projet",
    "description": "Tester le contrôleur Creance",
    "class_name":  "CreanceController",
    "test_name":   "CreanceControllerTest",
    "deterministic": true
  }'
```

## Structure attendue

```
mon-projet-symfony/         ← dossier parent (monté dans /workspace)
├── ai-test-engine/         ← ce dépôt
│   ├── docker-compose.yml
│   └── app/
├── src/                    ← code Symfony à analyser
├── templates/              ← templates Twig
└── tests/                  ← les tests générés sont écrits ici
```

## Endpoints

| Méthode  | URL                          | Tag         | Auth |
|----------|------------------------------|-------------|------|
| GET      | `/health`                    | santé       | non  |
| GET      | `/project/{id}/stats`        | projets     | oui  |
| DELETE   | `/project/{id}`              | projets     | oui  |
| POST     | `/admin/reset-schema`        | admin       | oui  |
| POST     | `/learn-from-code`           | indexation  | oui  |
| POST     | `/generate-test`             | génération  | oui  |
| POST     | `/generate-unit-test`        | génération  | oui  |

OpenAPI complet : http://localhost:8000/docs

## Authentification

Si `API_KEY` est défini dans `.env`, tous les endpoints d'écriture exigent le header `X-API-Key: <valeur>`. Si vide (mode dev), l'auth est désactivée.

## Modes de génération

- **LLM (par défaut)** : `/generate-test` envoie le contexte RAG + few-shot au LLM Ollama. Risque d'hallucination, plus créatif.
- **Déterministe** : `deterministic: true` dans le body. Génération directe depuis les chunks indexés, sans LLM. Pas d'hallucination, moins de variété.

## Tests

```bash
docker compose exec brain-api pytest /app/test_main.py -v
```

Le fichier `test_code_parser_axe1.py` nécessite des fixtures `CreanceController.php` / `EtatImportController.php`. Surcharge des chemins via `AXE1_CREANCE_PATH` / `AXE1_ETAT_PATH` / `AXE1_FIXTURES_DIR`.

## Configuration — variables clés

Voir `.env.example` pour la liste complète.

| Variable                    | Défaut                                                | Description |
|-----------------------------|-------------------------------------------------------|-------------|
| `DATABASE_URL`              | (obligatoire)                                         | Connexion PG |
| `OLLAMA_URL`                | `http://host.docker.internal:11434/api/generate`      | Endpoint LLM |
| `DEFAULT_EMBEDDING_MODEL`   | `paraphrase-multilingual-MiniLM-L12-v2`               | Modèle d'embedding |
| `ALLOWED_EMBEDDING_MODELS`  | (3 modèles standard)                                  | Allowlist anti-DL arbitraire |
| `API_KEY`                   | (vide = auth off)                                     | Header `X-API-Key` requis si défini |
| `CORS_ORIGINS`              | (vide = CORS off)                                     | Origines CORS, CSV |
| `AUTH_FIREWALL_NAME`        | `secured_area`                                        | Nom du firewall Symfony cible |
| `AUTH_TEST_ROLES`           | `ADMIN,CONSULT`                                       | Rôles disponibles dans TestUserFactory |
