# AI Test Engine

Système hybride qui automatise la génération de tests fonctionnels Symfony pour **SUCRE**
(application de gestion du recouvrement de créances, CNAM Hauts-de-Seine).

Il répond à deux questions que se pose l'équipe MOA à chaque montée de version :

| Question | Réponse |
|---|---|
| *Qu'est-ce que je dois tester en priorité ?* | Un modèle supervisé estime le risque de régression **par contrôleur** |
| *Comment obtenir un premier jet de test sans coder ?* | Un moteur analyse le code Symfony et génère des `WebTestCase` PHPUnit |

**Tout tourne en local.** Le code analysé est du code métier de l'Assurance Maladie : aucune
donnée n'est transmise à un service externe.

---

## Sommaire

| Section | Contenu |
|---|---|
| [1. Démarrage rapide](#1-démarrage-rapide) | Installer et lancer |
| [2. Architecture](#2-architecture) | Les services et leurs échanges |
| [3. API](#3-api) | Endpoints, authentification, modes de génération |
| [4. Sécurité de l'écriture](#4-sécurité-de-lécriture-des-tests) | Où le moteur écrit, et ce qu'il ne détruira pas |
| [5. Module ML](#5-module-ml--prédiction-du-risque) | Méthode, résultats, limites |
| [6. Base de données](#6-base-de-données) | Schéma, restauration, identifiants |
| [7. Tests](#7-tests) | Comment les lancer, ce qu'ils couvrent |
| [8. Accès et identifiants](#8-accès-et-identifiants) | Interfaces, comptes, jeu d'essai |
| [9. Compatibilité navigateurs](#9-compatibilité-navigateurs) | |
| [10. Configuration](#10-configuration) | Variables d'environnement |
| [11. Documentation détaillée](#11-documentation-détaillée) | Les autres documents |

> 📖 **Installation pas à pas sur un poste neuf** (Windows, 10 étapes, dépannage) :
> [`docs/installation.md`](docs/installation.md).

---

## 1. Démarrage rapide

### Prérequis

**Docker Desktop uniquement.** Rien d'autre à installer sur le poste.

Allouer au moins **8 Go de RAM à Docker** (Settings → Resources) : le modèle de langage tourne
désormais dans un conteneur et occupe ~2 Go à lui seul.

> 📌 **Ollama est conteneurisé** depuis le 29/07/2026. Il tournait auparavant sur l'hôte, ce qui
> imposait de l'installer séparément **et** de penser à `setx OLLAMA_HOST 0.0.0.0` — sans quoi les
> conteneurs ne l'atteignaient pas. Le symptôme était trompeur : tout démarrait normalement, seule
> la génération partait en timeout. C'était la première cause d'échec d'installation.
>
> Le modèle (~2 Go) est téléchargé automatiquement au premier `docker compose up` par le service
> éphémère `ollama-pull`, puis conservé dans un volume nommé.
>
> **Pour utiliser un Ollama installé sur l'hôte** (par exemple afin d'exploiter un GPU) : la
> marche à suivre est en commentaire en tête du service `ollama` dans `docker-compose.yml`.

### ⚠️ L'arborescence — l'erreur la plus fréquente

`docker-compose.yml` monte le **dossier parent** en tant que `/workspace`. Ce dépôt doit donc être
placé **à l'intérieur** du projet Symfony, pas à côté :

```
mon-projet-symfony/         ← devient /workspace
├── ai-test-engine/         ← CE dépôt
│   ├── docker-compose.yml
│   ├── app/
│   └── ml/
├── src/                    ← code Symfony analysé
├── templates/
└── tests/                  ← les tests générés arrivent ici
```

Placé à côté, l'indexation répond « succès » avec **0 fichier trouvé**.

### Lancement

```bash
# 1. Configuration
cp .env.example .env

# 2. Dépendances PHP du pont AST (nikic/php-parser)
#    Requis même en Docker : le montage ./app:/app masque le vendor/ de l'image.
(cd app/php_ast && composer install --no-dev)

# 3. Démarrage — le 1er lancement télécharge le modèle (~2 Go), comptez 10-20 min
docker compose up -d

# 4. Vérifier
docker compose ps                       # ollama-pull doit être "Exited (0)" : c'est normal
curl http://localhost:8000/health
```

| Service | Attendu | Rôle |
|---|---|---|
| `vector-db` | `healthy` | PostgreSQL + pgvector |
| `ollama` | `healthy` | Modèle de langage |
| `ollama-pull` | **`Exited (0)`** | Éphémère — a téléchargé le modèle puis s'est arrêté |
| `brain-api` | `healthy` | API |
| `streamlit-ui` | `running` | Dashboard |

### Premier usage

```bash
# Indexer le projet Symfony
curl -X POST http://localhost:8000/learn-from-code \
  -H "Content-Type: application/json" -d '{"project_id": "sucre"}'

# Vérifier (doit renvoyer chunks > 0)
curl http://localhost:8000/project/sucre/stats

# Générer un test — mode déterministe : instantané, sans Ollama
curl -X POST http://localhost:8000/generate-test \
  -H "Content-Type: application/json" \
  -d '{"project_id":"sucre","description":"Tester le controleur Creance",
       "class_name":"CreanceController","test_name":"CreanceControllerTest",
       "deterministic":true}'
```

Dashboard MOA : <http://localhost:8501>

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Docker Compose — réseau interne                                 │
│                                                                  │
│   streamlit-ui  ──HTTP──►   brain-api   ──vecteurs──►  vector-db │
│     :8501                     :8000                  PostgreSQL 15│
│   Dashboard MOA             FastAPI                   + pgvector  │
│                                 │                                │
│                                 │ prompts                        │
│                                 ▼                                │
│                              ollama                              │
│                          qwen2.5-coder:3b                        │
│                       (port non publié)                          │
└─────────────────────────────────┼────────────────────────────────┘
                                  │ lit le code / écrit les tests
                        ┌─────────▼─────────┐
                        │ Projet Symfony    │
                        │ monté en volume   │
                        └───────────────────┘
```

Streamlit ne parle **jamais** directement à la base ni à Ollama : tout passe par `brain-api`.

| Couche | Technologie | Rôle |
|---|---|---|
| Interface | Streamlit | Dashboard MOA |
| API | FastAPI + Uvicorn (Python 3.11) | Orchestration, validation Pydantic |
| Base | PostgreSQL 15 + pgvector, index **HNSW** | Chunks de code et leurs vecteurs |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | Code → vecteurs de 384 dimensions |
| LLM | Ollama — `qwen2.5-coder:3b` (conteneurisé) | Corps des méthodes de test |
| Parsing PHP | `nikic/php-parser` v5, via sous-processus | Routes, méthodes, rôles |
| ML | scikit-learn, XGBoost | Prédiction du risque |

**Deux réseaux de neurones profonds** sont au cœur du système, encapsulés dans leurs
bibliothèques : MiniLM (BERT, 12 couches, **117 653 760 paramètres**) pour la vectorisation, et
qwen2.5-coder (**≈ 3 milliards**) pour la génération. Socle PyTorch 2.12.

---

## 3. API

OpenAPI complet : <http://localhost:8000/docs> · Console web : <http://localhost:8000/ui>

| Méthode | URL | Auth | Rôle |
|---|---|---|---|
| GET | `/health` | non | État de l'API et de la base |
| GET | `/projects` | oui | Liste des projets indexés |
| GET | `/project/{id}/stats` | oui | Nombre de chunks indexés |
| DELETE | `/project/{id}` | oui | Purge un projet |
| POST | `/learn-from-code` | oui | Indexe le code Symfony |
| POST | `/generate-test` | oui | Génère un test fonctionnel |
| POST | `/generate-unit-test` | oui | Génère un test unitaire |
| POST | `/generate-tests-batch` | oui | Plusieurs classes en tâche de fond |
| GET | `/job/{id}/status` | oui | Suivi d'une génération asynchrone |
| POST | `/admin/reset-schema` | oui | Réinitialise la base ⚠️ |

### Authentification

`API_KEY` vide (défaut) = authentification désactivée. Si elle est définie, tous les endpoints
d'écriture exigent l'en-tête `X-API-Key`.

### Modes de génération

| Mode | Déclencheur | Caractéristiques |
|---|---|---|
| **Par route** *(défaut)* | `per_route: true` | Un appel LLM par route, borné à 2-3 méthodes. Squelette de classe déterministe : le LLM n'écrit **que** le corps des méthodes |
| **Déterministe** | `deterministic: true` | Aucun appel LLM. Instantané, zéro hallucination, moins de variété |
| **Monolithique** | `per_route: false` | Ancien chemin, fichier complet en un appel. Conservé pour comparaison |

### Budget de temps

La génération par route est plafonnée à **840 secondes au total**. Passé ce délai, les routes
restantes reçoivent un `markTestIncomplete` et le fichier est produit quand même.

Sans ce plafond, un contrôleur à 10 routes en échec pouvait occuper le serveur **une heure**,
pendant que le client abandonnait à 15 minutes — l'utilisateur voyait une erreur, puis un fichier
apparaissait tout seul bien plus tard.

Le statut renvoyé reflète le résultat réel :

| Statut | Signification |
|---|---|
| `success` | Toutes les routes générées |
| `partial` | Certaines en échec ou non tentées (détail dans `routes_failed` / `routes_skipped`) |
| `failed` | Aucune route générée — le fichier n'est que des stubs |

---

## 4. Sécurité de l'écriture des tests

Le moteur écrit dans le projet analysé, **à côté des tests écrits à la main**. Trois garanties,
chacune verrouillée par des tests automatisés.

### Confinement

Toutes les écritures sont confinées à `tests/`, sous des préfixes en dur
(`tests/Functional/Controller/`, `tests/Functional/Fixtures/`, `tests/Unit/{catégorie}/`).

Les noms fournis par l'appelant sont réduits aux caractères alphanumériques, puis `_safe_join()`
vérifie que le chemin résolu reste sous le workspace. Les tentatives de remontée (`../`), les
chemins absolus et les octets nuls sont rejetés par une **400**.

### Non-destruction

Chaque fichier produit porte un marqueur `@ai-test-engine-generated` dans son en-tête.

| Situation | Comportement |
|---|---|
| Fichier absent | Écriture directe |
| Fichier **déjà généré** (marqueur présent) | Remplacé — cas nominal d'une regénération |
| Fichier **écrit à la main** | **Refus (409)**. Avec `overwrite: true`, une sauvegarde `.bak` est créée avant remplacement |

> 💡 Pour protéger définitivement un test généré que vous avez modifié à la main : **supprimez la
> ligne du marqueur**. Le moteur refusera alors de l'écraser.

### Atomicité

L'écriture passe par un fichier temporaire du même dossier, renommé par `os.replace()` à la fin.
Une interruption (timeout, arrêt du conteneur, disque plein) laisse donc soit l'ancien fichier
intact, soit le nouveau complet — **jamais un PHP à moitié écrit**.

### Injection SQL

Toutes les requêtes utilisent des paramètres liés (`%s` / `execute_values`). Une seule exception
inévitable : la dimension du vecteur dans le `CREATE TABLE`, qui fait partie du *type* de colonne
et n'accepte pas de paramètre. Elle est validée en amont (entier, bornes pgvector) par
`_validate_vector_size()`.

---

## 5. Module ML — prédiction du risque

### Méthode

**Label** — calculé **par fichier** : un fichier est « à risque » s'il a été corrigé par au moins
**5 commits de bugfix** sur la fenêtre observée. Méthodologie issue de la littérature en *software
defect prediction* (Hassan 2009, Kamei et al. 2013).

> Le seuil de 5 n'est pas arbitraire. À 2, **82 %** des fichiers étaient étiquetés « à risque » :
> un outil qui répond « tout est prioritaire » ne priorise rien.

**Features** — `extract_dataset.py` produit 23 colonnes, mais le modèle déployé n'en utilise que
**13** (jeu `v3`). Sont exclues :

- les **features git**, calculées sur la même fenêtre temporelle que le label → **fuite de
  données** (l'AUC atteignait 1,000) ;
- les **quasi-constantes et redondantes**, écartées après analyse exploratoire.

**Validation** — croisée et **groupée par fichier** (`StratifiedGroupKFold`) : un contrôleur
entier est soit en apprentissage, soit en test, jamais réparti entre les deux.

### Résultats (SUCRE — 368 lignes, 54 fichiers, 59,5 % de positifs)

> **Lire toute métrique avec sa baseline.** Sur données déséquilibrées, le F1 est trompeur : un
> classifieur trivial « tout positif » l'atteint presque. Les métriques de référence sont le
> **MCC** et la **balanced accuracy**, dont les baselines sont 0 et 0,50.

**Par méthode :**

| Modèle | MCC | bal. acc. | F1 *(baseline 0,746)* | AUC |
|---|---|---|---|---|
| **RandomForest** — retenu | **0,758** | **0,861** | 0,840 | 0,920 |
| LogisticRegression | 0,690 | 0,823 | 0,823 | 0,936 |
| XGBoost | 0,593 | 0,778 | 0,761 | 0,827 |

**Par contrôleur** — c'est la décision que l'outil fait réellement prendre :

| Modèle | MCC | bal. acc. | F1 *(baseline 0,457)* |
|---|---|---|---|
| **RandomForest** — retenu | **0,529** | **0,706** | 0,583 |
| LogisticRegression | 0,477 | 0,711 | 0,593 |
| XGBoost | 0,317 | 0,635 | 0,462 |

Le chiffre baisse, et c'est voulu : par ligne, les contrôleurs à risque pèsent trop lourd — ils
sont **3,5 × plus gros** (13,7 méthodes en moyenne contre 3,9), et un seul contrôleur de
77 méthodes représente **20,9 %** du score. Par fichier, chacun compte pour un.

**Top features** — 83 % du pouvoir prédictif sur 3 features **de classe** :

| Feature | Poids |
|---|---|
| `nb_methods_in_class` | 41,0 % |
| `nb_constructor_deps` | 26,8 % |
| `nb_class_grants` | 15,2 % |

→ **Conséquence directe** : le modèle discrimine entre *contrôleurs*, pas entre méthodes. Le
dashboard le dit explicitement.

### Pipeline

```bash
pip install -r ml/requirements.txt

# 1. Extraire le dataset depuis l'historique git
PYTHONIOENCODING=utf-8 python ml/extract_dataset.py <repo> _dataset/dataset_sucre_t5.csv \
    --since "36 months ago" --bugfix-threshold 5

# 2. Entraîner et comparer les 3 modèles
PYTHONIOENCODING=utf-8 python ml/train.py _dataset/dataset_sucre_t5.csv --feature-set v3

# 3. Dashboard
streamlit run ml/streamlit_app.py
```

> ⚠️ `PYTHONIOENCODING=utf-8` est **obligatoire sous Windows** : sans lui, les accents des
> messages de commit corrompent la sortie.
>
> ⚠️ Le réentraînement **écrase** `ml/models/`. Sauvegarder avant si besoin.

### Datasets

| Projet | Lignes | Positifs | Usage |
|---|---|---|---|
| **SUCRE** | 368 / 54 fichiers | 59,5 % | **Référence** — a entraîné le modèle déployé |
| Sylius | 183 / 72 fichiers | 26,2 % | Comparaison inter-projets |
| Akeneo PIM | 726 / 234 fichiers | **0 %** | **Inexploitable** — aucun label positif |

### Limites assumées

- **Petit échantillon** — 54 fichiers indépendants seulement, d'où une variance élevée.
- **Label proxy** — « corrigé souvent par le passé » n'est pas « va casser demain ».
- **Granularité = contrôleur**, pas méthode.
- **Généralisation inter-projets non établie** — Akeneo est inexploitable, Sylius a un profil
  trop différent pour conclure.

---

## 6. Base de données

### Table `project_code_context`

| Colonne | Type | Rôle |
|---|---|---|
| `id` | `SERIAL` (PK) | Identifiant |
| `project_id` | `VARCHAR(100)` | Cloisonnement par projet |
| `chunk_type` | `VARCHAR(50)` | `controllers_class`, `controllers_method`, `template_info`… |
| `file_path` | `VARCHAR(500)` | Fichier source |
| `class_name` | `VARCHAR(255)` | Classe PHP |
| `content` | `TEXT` | Texte du fragment — c'est lui qui est vectorisé |
| `embedding` | `vector(384)` | Vecteur sémantique |

### Index

| Index | Type | Rôle |
|---|---|---|
| `..._embedding_idx` | **HNSW** (`vector_cosine_ops`) | Recherche par similarité cosinus |
| `..._project_id_idx` | B-tree | Filtrage et purge par projet |
| `..._pkey` | B-tree | Clé primaire |

**Gain mesuré de l'index HNSW** sur 20 000 vecteurs : 22,97 ms → **10,28 ms** (× 2,24). Le plan
d'exécution passe d'un `Seq Scan` sur 20 000 lignes à un `Index Scan` remontant 8 lignes.
Protocole : [`docs/benchmark-bdd.md`](docs/benchmark-bdd.md).

### Restauration

Le schéma se crée automatiquement au démarrage de `brain-api`. Manuellement :

```bash
docker compose exec -T vector-db psql -U admin -d knowledge_base < sql/schema.sql
```

Remise à plat complète :

```bash
docker compose exec -T vector-db psql -U admin -d knowledge_base \
  -c "DROP TABLE IF EXISTS project_code_context CASCADE;"
docker compose restart brain-api
```

> ⚠️ **Piège** : les index ne sont créés **que si la table est absente**. Une table créée à la
> main sans ses index ne les recevra jamais — la recherche fonctionnera, mais en parcours
> séquentiel. Toujours faire un `DROP TABLE` complet.

### Produire un dump

```bash
# Structure seule — à versionner
docker compose exec -T vector-db pg_dump -U admin --schema-only --no-owner \
  --no-privileges knowledge_base > sql/schema.sql

# Structure + données — ⚠️ contient du code source CNAM, ne pas versionner
docker compose exec -T vector-db pg_dump -U admin --no-owner \
  --no-privileges knowledge_base > backup_complet.sql
```

---

## 7. Tests

```bash
docker compose exec brain-api pytest /app -v
```

**104 tests, tous verts.** Aucune dépendance externe : base, LLM et Ollama sont simulés.

| Fichier | Contenu |
|---|---|
| `app/test_main.py` | Endpoints, générateur déterministe, routes multiples, contrat chunk ↔ regex, sécurité d'écriture, budget de temps |
| `app/test_code_parser_axe1.py` | Adaptateur pytest — lance le script ci-dessous en sous-processus |
| `app/check_code_parser_axe1.py` | ~45 vérifications du parseur AST. Exécutable seul : `python app/check_code_parser_axe1.py` |

> `check_code_parser_axe1.py` ne s'appelle pas `test_*` volontairement : il exécute ses
> vérifications au niveau module et se termine par `sys.exit()`, ce qui interrompait la collecte
> pytest et empêchait **tous** les tests du dossier de tourner.

**Non couvert** : le module `ml/`, et la section d'intégration du parseur qui nécessite des
fixtures absentes de ce dépôt (chemins surchargeables via `AXE1_CREANCE_PATH` /
`AXE1_ETAT_PATH` / `AXE1_FIXTURES_DIR`).

---

## 8. Accès et identifiants

> Environnement de **développement local**. Rien n'est exposé hors de la machine hôte ; à durcir
> avant tout déploiement.

### Interfaces

| Interface | URL | Authentification |
|---|---|---|
| Dashboard MOA | <http://localhost:8501> | aucune |
| API — documentation | <http://localhost:8000/docs> | aucune |
| API — console d'admin | <http://localhost:8000/ui> | aucune |
| API — santé | <http://localhost:8000/health> | aucune |

Il n'existe **pas de back-office avec comptes** : l'outil est mono-utilisateur, sans session. Le
seul contrôle d'accès est la clé API.

### Base de données

| Paramètre | Valeur |
|---|---|
| Base / utilisateur | `knowledge_base` / `admin` |
| Mot de passe | `secret_ai_password` |
| Hôte / port | `vector-db:5432` — **non publié sur l'hôte** |

```bash
docker compose exec vector-db psql -U admin -d knowledge_base
```

### Jeu de test

Aucun compte requis. Le parcours de validation complet figure au [§1](#premier-usage).
La page **Prédiction ML** du dashboard fonctionne **sans indexation** : elle lit un CSV de
`_dataset/` et le modèle livré.

---

## 9. Compatibilité navigateurs

Streamlit 1.53 (React + WebSocket) et une console HTML/JS sans dépendance externe. Aucun code
spécifique à un navigateur.

| Navigateur | Minimum | Statut |
|---|---|---|
| Chrome / Edge | 90+ | **Testé** — référence |
| Firefox | 90+ | Testé |
| Safari | 15+ | Compatible |
| Internet Explorer 11 | — | **Non supporté** (exclu par Streamlit) |

Résolution confortable : **1280 × 720**. JavaScript et WebSocket doivent être autorisés — un
proxy qui coupe les WebSocket bloque le rafraîchissement de l'interface.

---

## 10. Configuration

Liste complète dans `.env.example`.

> En Docker, les variables de `docker-compose.yml` sont **prioritaires** sur `.env`.

| Variable | Défaut | Rôle |
|---|---|---|
| `DATABASE_URL` | *(obligatoire)* | Connexion PostgreSQL |
| `OLLAMA_URL` | `http://ollama:11434/api/generate` | Endpoint du LLM (nom du service Docker) |
| `DEFAULT_EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Modèle d'embedding |
| `DEFAULT_LLM_MODEL` | `qwen2.5-coder:3b` | Modèle de génération |
| `ALLOWED_EMBEDDING_MODELS` | 3 modèles | Liste blanche anti-téléchargement arbitraire |
| `API_KEY` | *(vide = auth off)* | En-tête `X-API-Key` |
| `CORS_ORIGINS` | *(vide = CORS off)* | Origines autorisées, séparées par des virgules |
| `LLM_NUM_CTX` | `8192` | Fenêtre de contexte |
| `LLM_NUM_PREDICT` | `1500` | Tokens maximum en sortie |
| `LLM_KEEP_ALIVE` | `30m` | Durée de maintien du modèle en RAM |

**Bloc authentification** — décrit le Symfony analysé, à adapter au projet cible :

| Variable | Défaut |
|---|---|
| `AUTH_FIREWALL_NAME` | `secured_area` |
| `AUTH_REDIRECT_PATH` | `/connect/web-sso` |
| `AUTH_REDIRECT_STATUS` | `307` |
| `AUTH_TEST_ROLES` | `ADMIN,CONSULT` |
| `AUTH_TEST_CLASS` | `App\Tests\Security\TestUserFactory` |

---

## 11. Documentation détaillée

| Document | Contenu |
|---|---|
| [`docs/installation.md`](docs/installation.md) | Installation pas à pas sur poste neuf, dépannage, checklist |
| [`docs/doc-tech.md`](docs/doc-tech.md) | **Documentation technique de passation** — architecture, décisions, historique |
| [`docs/presentation-projet.md`](docs/presentation-projet.md) | Guide de présentation du projet, avec déroulé minuté |
| [`docs/audit-code.md`](docs/audit-code.md) | Audit complet du code, vérifié par exécution |
| [`docs/analyse-resultats-ml.md`](docs/analyse-resultats-ml.md) | Analyse critique des résultats ML |
| [`docs/dictionnaire-donnees.md`](docs/dictionnaire-donnees.md) | Les 28 colonnes du dataset |
| [`docs/rapport-qualite-donnees.md`](docs/rapport-qualite-donnees.md) | Valeurs manquantes, doublons, cohérence |
| [`docs/benchmark-bdd.md`](docs/benchmark-bdd.md) | Mesures de performance des requêtes |
| [`docs/conformite-examen.md`](docs/conformite-examen.md) | Conformité au référentiel RNCP 37137 |
| [`docs/complements-memoire.md`](docs/complements-memoire.md) | RGPD, accessibilité, suivi des problématiques |
| [`TODO.md`](TODO.md) | Points techniques restants |
