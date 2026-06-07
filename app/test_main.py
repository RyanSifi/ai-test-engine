"""
Tests unitaires du moteur de test IA.

Couvre :
  - GET  /health
  - POST /learn-from-code
  - POST /generate-test  (profil web_crud, interne, déterministe)
  - POST /generate-unit-test
  - POST /admin/reset-schema
  - DELETE /project/{id}
  - _generate_php_test_from_chunks (générateur déterministe, sans réseau)

Tous les appels à la DB, au LLM et à Ollama sont mockés — aucune dépendance externe requise.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from main import app, get_db


# ---------------------------------------------------------------------------
# HELPER — mock réponse Ollama streaming
# ---------------------------------------------------------------------------

def _llm_response(code: str) -> MagicMock:
    """
    Crée un mock de réponse requests compatible avec le mode streaming de _call_llm.
    _call_llm appelle resp.iter_lines() et json.loads() sur chaque ligne,
    puis vérifie chunk.get("done").
    """
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines.return_value = [
        json.dumps({"response": code, "done": True}).encode()
    ]
    return mock_resp


# ---------------------------------------------------------------------------
# CHUNKS DE RÉFÉRENCE — format enrichi Axe 1/2
# ---------------------------------------------------------------------------

CHUNK_CLASS = {
    "chunk_type": "controllers_class",
    "file_path":  "Controller/FooController.php",
    "class_name": "FooController",
    "content": (
        "La classe PHP FooController (type: controllers) dans Controller/FooController.php."
        " → Profil: web_crud → Rôles classe: ADMIN"
    ),
    "similarity": 0.90,
}

CHUNK_METHOD_RENDER = {
    "chunk_type": "controllers_method",
    "file_path":  "Controller/FooController.php",
    "class_name": "FooController",
    "content": (
        "Méthode 'index' (FooController) — Route: /foo/ — Template: foo/index.html.twig\n"
        "  → Type de réponse: render (200)\n"
        "  → H1: Liste des items"
    ),
    "similarity": 0.95,
}

CHUNK_METHOD_AJAX = {
    "chunk_type": "controllers_method",
    "file_path":  "Controller/FooController.php",
    "class_name": "FooController",
    "content": (
        "Méthode 'getData' (FooController) — Route: /foo/data\n"
        "  → Type de réponse: json (200)\n"
        "  → AJAX uniquement\n"
        "  → Verbes HTTP: GET"
    ),
    "similarity": 0.88,
}

CHUNK_METHOD_FORM = {
    "chunk_type": "controllers_method",
    "file_path":  "Controller/FooController.php",
    "class_name": "FooController",
    "content": (
        "Méthode 'edit' (FooController) — Route: /foo/{id}/edit — Template: foo/edit.html.twig\n"
        "  → Type de réponse: render (200), redirect (302)\n"
        "  → Formulaire: FooType\n"
        "  → Verbes HTTP: GET, POST\n"
        "  → H1: Modifier"
    ),
    "similarity": 0.87,
}

CHUNK_METHOD_DELETE_HIGH_ROLE = {
    "chunk_type": "controllers_method",
    "file_path":  "Controller/FooController.php",
    "class_name": "FooController",
    "content": (
        # Rôle méthode SUPÉRIEUR au rôle classe (ADMIN) → doit générer un test 403
        # pour un utilisateur qui n'a que le rôle classe.
        "Méthode 'delete' (FooController) — Route: /foo/{id}/delete\n"
        "  → Type de réponse: redirect (302)\n"
        "  → Rôle requis (méthode): SUPERADMIN\n"
        "  → Verbes HTTP: DELETE"
    ),
    "similarity": 0.85,
}

CHUNK_CLASS_INTERNAL = {
    "chunk_type": "controllers_class",
    "file_path":  "Controller/InternalController.php",
    "class_name": "InternalController",
    "content": (
        "La classe PHP InternalController (type: controllers) dans Controller/InternalController.php."
        " → Profil: internal → Pas de route HTTP"
    ),
    "similarity": 0.90,
}


# ---------------------------------------------------------------------------
# FIXTURE
# ---------------------------------------------------------------------------

@pytest.fixture
def client_with_mocks():
    """
    Retourne (client, mock_db, mock_brain) avec toutes les dépendances externes mockées :
    - DB via dependency_overrides (get_db est utilisé avec Depends())
    - get_brain via patch (appelé directement dans les endpoints, pas via Depends())
    - _check_ollama_alive via patch (évite un vrai appel réseau vers Ollama)
    """
    mock_db    = MagicMock()
    mock_brain = MagicMock()

    mock_brain.encode.return_value = [[0.1] * 384]
    mock_db.list_projects.return_value     = ["proj_test"]
    mock_db.get_project_stats.return_value = {
        "project_id": "proj_test", "chunks": 5, "routes": 2
    }
    mock_db.find_closest_code_context.return_value = [CHUNK_CLASS, CHUNK_METHOD_RENDER]
    mock_db.get_code_by_class_name.return_value    = []

    app.dependency_overrides[get_db] = lambda: mock_db

    with patch("main.get_brain", return_value=mock_brain), \
         patch("main._check_ollama_alive", return_value=True):
        client = TestClient(app, raise_server_exceptions=False)
        yield client, mock_db, mock_brain

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self, client_with_mocks):
        client, _, _ = client_with_mocks
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "projects_indexed" in body

    def test_health_db_down(self, client_with_mocks):
        client, mock_db, _ = client_with_mocks
        mock_db.list_projects.side_effect = Exception("Connection refused")
        r = client.get("/health")
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# /project/{id}
# ---------------------------------------------------------------------------

class TestProjectEndpoints:
    def test_stats(self, client_with_mocks):
        client, mock_db, _ = client_with_mocks
        r = client.get("/project/proj_test/stats")
        assert r.status_code == 200
        assert r.json()["project_id"] == "proj_test"
        mock_db.get_project_stats.assert_called_once_with("proj_test")

    def test_delete_project(self, client_with_mocks):
        client, mock_db, _ = client_with_mocks
        r = client.delete("/project/proj_test")
        assert r.status_code == 200
        assert r.json()["status"] == "deleted"
        mock_db.clear_project.assert_called_once_with("proj_test")


# ---------------------------------------------------------------------------
# /admin/reset-schema
# ---------------------------------------------------------------------------

class TestResetSchema:
    def test_reset_requires_confirm(self, client_with_mocks):
        client, _, _ = client_with_mocks
        r = client.post("/admin/reset-schema", json={"confirm": False})
        assert r.status_code == 400

    def test_reset_with_confirm(self, client_with_mocks):
        client, mock_db, mock_brain = client_with_mocks
        mock_brain.encode.return_value = [[0.0] * 384]
        r = client.post("/admin/reset-schema", json={"confirm": True})
        assert r.status_code == 200
        assert r.json()["status"] == "schema_reset"
        mock_db.reset_schema.assert_called_once()


# ---------------------------------------------------------------------------
# /learn-from-code
# ---------------------------------------------------------------------------

class TestLearnFromCode:
    def test_workspace_not_found(self, client_with_mocks):
        client, _, _ = client_with_mocks
        with patch("main.settings") as mock_settings:
            mock_settings.container_project_root  = "/nonexistent/path"
            mock_settings.default_embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"
            mock_settings.api_key                 = None  # désactive l'auth pour ce test
            mock_settings.allowed_embedding_models = "paraphrase-multilingual-MiniLM-L12-v2"
            r = client.post("/learn-from-code", json={"project_id": "proj_test"})
        assert r.status_code == 404

    def test_learn_indexes_chunks(self, client_with_mocks):
        client, mock_db, mock_brain = client_with_mocks

        fake_analysis = {
            "controllers": [
                {
                    "file":   "Controller/FooController.php",
                    "class":  "FooController",
                    "methods": [
                        {
                            "name":          "index",
                            "route":         "/foo/",
                            "description":   "Liste.",
                            "params":        [],
                            "return_type":   "Response",
                            "renders":       None,
                            "response_type": "render (200)",
                            "http_methods":  ["GET"],
                            "is_ajax_only":  False,
                            "form_type":     None,
                            "voter_checks":  [],
                            "method_grants": [],
                        }
                    ],
                    "properties":        [],
                    "constructor_params": [],
                    "class_grants":      [{"role": "ROLE_ADMIN", "message": None}],
                    "controller_profile": "web_crud",
                }
            ],
            "entities": [], "services": [], "repositories": [],
            "commands": [], "forms":    [], "securitys":    [],
            "templates": [
                {
                    "file":    "foo/index.html.twig",
                    "h1":      ["Liste des items"],
                    "buttons": ["Créer"],
                    "links":   ["/foo/new"],
                    "inputs":  [],
                }
            ],
        }

        mock_brain.encode.return_value = [[0.1] * 384] * 20

        with patch("main.analyze_project_code", return_value=fake_analysis), \
             patch("main.os.path.isdir", return_value=True):
            r = client.post("/learn-from-code", json={"project_id": "proj_test"})

        assert r.status_code == 200
        body = r.json()
        assert body["status"]       == "success"
        assert body["total_chunks"] >= 2
        mock_db.reindex_project.assert_called_once()
        args, _ = mock_db.reindex_project.call_args
        assert args[0] == "proj_test"

    def test_learn_warning_when_empty(self, client_with_mocks):
        client, _, _ = client_with_mocks
        empty = {k: [] for k in [
            "controllers", "entities", "services", "repositories",
            "commands", "forms", "securitys", "templates",
        ]}
        with patch("main.analyze_project_code", return_value=empty), \
             patch("main.os.path.isdir", return_value=True):
            r = client.post("/learn-from-code", json={"project_id": "proj_test"})
        assert r.status_code == 200
        assert r.json()["status"] == "warning"


# ---------------------------------------------------------------------------
# /generate-test
# ---------------------------------------------------------------------------

class TestGenerateTest:
    def test_generates_php_file(self, client_with_mocks):
        client, _, _ = client_with_mocks

        generated_php = (
            "<?php\n\nnamespace App\\Tests\\Functional\\Controller;\n\n"
            "use Symfony\\Bundle\\FrameworkBundle\\Test\\WebTestCase;\n\n"
            "final class FooTest extends WebTestCase {\n"
            "    public function testIndexRedirectsWhenNotAuthenticated(): void {\n"
            "        $this->client->request('GET', '/foo/');\n"
            "        self::assertResponseStatusCodeSame(307);\n"
            "    }\n}\n"
        )

        with patch("main.requests.post", return_value=_llm_response(generated_php)), \
             patch("main.validate_php_syntax", return_value=None), \
             patch("main._write_php_file") as mock_write, \
             patch("main._load_golden_dataset", return_value=[]):

            r = client.post("/generate-test", json={
                "project_id":  "proj_test",
                "description": "Tester la page liste",
                "test_name":   "FooTest",
            })

        assert r.status_code == 200
        body = r.json()
        assert body["status"]             == "success"
        assert body["file"]               == "FooTest.php"
        assert body["controller_profile"] == "web_crud"
        mock_write.assert_called_once()

    def test_internal_controller_redirects_to_unit(self, client_with_mocks):
        """Un contrôleur interne (sans route) doit retourner redirect_to_unit."""
        client, mock_db, _ = client_with_mocks
        mock_db.find_closest_code_context.return_value = [CHUNK_CLASS_INTERNAL]

        r = client.post("/generate-test", json={
            "project_id":  "proj_test",
            "description": "Tester InternalController",
            "class_name":  "InternalController",
        })

        assert r.status_code == 200
        body = r.json()
        assert body["status"]             == "redirect_to_unit"
        assert body["controller_profile"] == "internal"

    def test_deterministic_mode(self, client_with_mocks):
        """Mode déterministe : pas d'appel LLM, fichier PHP généré directement."""
        client, _, _ = client_with_mocks

        with patch("main._write_php_file") as mock_write, \
             patch("main.requests.post") as mock_post:

            r = client.post("/generate-test", json={
                "project_id":   "proj_test",
                "description":  "Tests FooController",
                "class_name":   "FooController",
                "test_name":    "FooControllerTest",
                "deterministic": True,
            })

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["mode"]   == "deterministic"
        mock_write.assert_called_once()
        mock_post.assert_not_called()  # pas d'appel LLM

    def test_syntax_error_triggers_llm_correction(self, client_with_mocks):
        client, _, _ = client_with_mocks

        # On neutralise le retry de couverture (_validate_coverage → []) pour
        # se concentrer sur le retry de syntaxe : 2 appels LLM exactement.
        with patch("main.requests.post") as mock_post, \
             patch("main.validate_php_syntax", side_effect=[
                 "Parse error: unexpected end",
                 None,
             ]), \
             patch("main._validate_coverage", return_value=[]), \
             patch("main._write_php_file"), \
             patch("main._load_golden_dataset", return_value=[]):

            mock_post.side_effect = [
                _llm_response("<?php class Broken { "),
                _llm_response("<?php class Fixed extends WebTestCase {}"),
            ]

            r = client.post("/generate-test", json={
                "project_id":  "proj_test",
                "description": "test quelconque",
                "test_name":   "BrokenTest",
            })

        assert mock_post.call_count == 2


# ---------------------------------------------------------------------------
# /generate-unit-test
# ---------------------------------------------------------------------------

class TestGenerateUnitTest:
    def test_class_not_found_returns_404(self, client_with_mocks):
        client, _, _ = client_with_mocks
        with patch("main.extract_code_for_symbol", return_value=None):
            r = client.post("/generate-unit-test", json={
                "project_id":  "proj_test",
                "file_path":   "src/Service/FooService.php",
                "class_name":  "FooService",
                "description": "Tester la méthode bar",
            })
        assert r.status_code == 404

    def test_generates_unit_test_file(self, client_with_mocks):
        client, _, _ = client_with_mocks

        code_lines    = [
            "public function computeTotal(array $items): float",
            "{ return array_sum(array_column($items, 'price')); }",
        ]
        generated_php = (
            "<?php\nnamespace App\\Tests\\Unit\\Service;\n\n"
            "use PHPUnit\\Framework\\TestCase;\n\n"
            "class FooServiceTest extends TestCase {\n"
            "    public function testComputeTotal(): void {}\n}\n"
        )

        with patch("main.extract_code_for_symbol", return_value=code_lines), \
             patch("main.requests.post", return_value=_llm_response(generated_php)), \
             patch("main._write_php_file") as mock_write, \
             patch("main._load_golden_dataset", return_value=[]):

            r = client.post("/generate-unit-test", json={
                "project_id":  "proj_test",
                "file_path":   "src/Service/FooService.php",
                "class_name":  "FooService",
                "method_name": "computeTotal",
                "description": "Tester computeTotal",
            })

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert "FooServiceTest.php" in body["file"]
        mock_write.assert_called_once()


# ---------------------------------------------------------------------------
# Générateur déterministe — tests directs sans réseau
# ---------------------------------------------------------------------------

class TestDeterministicGenerator:
    """Teste _generate_php_test_from_chunks directement avec des chunks simulés."""

    def _run(self, chunks):
        from main import _generate_php_test_from_chunks
        return _generate_php_test_from_chunks(chunks, "FooController", "FooControllerTest")

    def test_render_route_generates_noauth_and_auth(self):
        php = self._run([CHUNK_CLASS, CHUNK_METHOD_RENDER])
        assert "<?php" in php
        assert "testIndexRedirectsWhenNotAuthenticated" in php
        assert "assertResponseStatusCodeSame(307)" in php
        assert "testIndexIsReachedWith" in php
        assert "assertResponseIsSuccessful" in php
        assert "assertSelectorTextContains('h1', 'Liste des items')" in php

    def test_ajax_route_generates_xhr_and_noXhr(self):
        php = self._run([CHUNK_CLASS, CHUNK_METHOD_AJAX])
        assert "HTTP_X-Requested-With" in php          # test avec XHR
        assert "testGetDataWithoutXhrReturns404" in php  # test sans XHR
        assert "assertResponseStatusCodeSame(404)" in php

    def test_form_route_generates_get_and_post(self):
        php = self._run([CHUNK_CLASS, CHUNK_METHOD_FORM])
        assert "testEditDisplaysFormWith" in php   # GET — affichage
        assert "testEditSubmitWith" in php          # POST — soumission
        assert "request('GET'" in php
        assert "request('POST'" in php

    def test_role_insufficient_generates_403(self):
        """Méthode avec rôle supérieur → test 403 pour le rôle de la classe."""
        php = self._run([CHUNK_CLASS, CHUNK_METHOD_DELETE_HIGH_ROLE])
        assert "assertResponseStatusCodeSame(403)" in php
        assert "ForbiddenWith" in php

    def test_private_method_without_route_is_commented(self):
        chunk_private = {
            "chunk_type": "controllers_method",
            "file_path":  "Controller/FooController.php",
            "class_name": "FooController",
            "content":    "Méthode 'buildQuery' (FooController) — méthode interne sans route",
            "similarity": 0.70,
        }
        php = self._run([CHUNK_CLASS, chunk_private])
        assert "buildQuery" in php
        assert "// Méthodes sans route HTTP" in php
        # Aucun test HTTP pour cette méthode
        assert "testBuildQuery" not in php

    def test_output_has_correct_namespace_and_structure(self):
        php = self._run([CHUNK_CLASS, CHUNK_METHOD_RENDER])
        assert "namespace App\\Tests\\Functional\\Controller;" in php
        assert "final class FooControllerTest extends WebTestCase" in php
        assert "protected function setUp(): void" in php
        assert "private function getTestUser" in php
        assert "loginUser($this->getTestUser(" in php


# ---------------------------------------------------------------------------
# Tests directs du code_parser (pas de réseau requis)
# ---------------------------------------------------------------------------

class TestCodeParser:
    def test_parse_controller_full(self):
        from code_parser import _parse_file_content

        php = """<?php
namespace App\\Controller;

/** Contrôleur produits. */
#[Route("/product")]
class ProductController
{
    public function __construct(
        private readonly ProductRepository $repo,
        private readonly EntityManagerInterface $em
    ) {}

    /** Liste. */
    #[Route("/", name: "product_index")]
    public function index(): Response { return $this->render("index.html.twig"); }

    #[Route("/new", name: "product_new", methods: ["GET", "POST"])]
    public function new(): Response { return $this->render("new.html.twig"); }

    #[Route("/{id}", name: "product_show")]
    public function show(int $id): Response { return $this->render("show.html.twig"); }
}
"""
        result = _parse_file_content(php)
        assert result["class_name"] == "ProductController"
        assert result["constructor_params"] == [
            {"type": "ProductRepository",      "name": "repo"},
            {"type": "EntityManagerInterface", "name": "em"},
        ]
        names = [m["name"] for m in result["methods"]]
        assert "index" in names
        assert "new"   in names
        assert "show"  in names

        index = next(m for m in result["methods"] if m["name"] == "index")
        assert index["route"]       == "/product/"
        assert index["description"] == "Liste."

        show = next(m for m in result["methods"] if m["name"] == "show")
        assert show["route"]  == "/product/{id}"
        assert show["params"] == [{"type": "int", "name": "id"}]

        new_m = next(m for m in result["methods"] if m["name"] == "new")
        assert new_m["route"] == "/product/new"

    def test_extract_balanced_block(self):
        from code_parser import _extract_balanced_block

        src   = "{ int $x = 1; if ($x) { return $x; } return 0; }"
        block = _extract_balanced_block(src, 0)
        assert block == src

    def test_find_class_block_extracts_full_class(self):
        from code_parser import _find_class_block

        php = """<?php
class Foo extends Bar
{
    private int $x;
    public function bar(): int {
        if ($this->x > 0) { return $this->x; }
        return 0;
    }
}
"""
        block = _find_class_block(php, "Foo")
        assert block is not None
        assert "private int $x"      in block
        assert "public function bar" in block
        assert block.count("{") == block.count("}")


# ---------------------------------------------------------------------------
# Endpoints asynchrones — job store, batch
# ---------------------------------------------------------------------------

class TestAsyncEndpoints:
    """Teste async_mode, /job/{id}/status et /generate-tests-batch."""

    def test_generate_test_async_returns_job_id(self, client_with_mocks):
        """async_mode=True → réponse immédiate avec status=accepted et job_id."""
        client, mock_db, _ = client_with_mocks
        mock_db.find_closest_code_context.return_value = [CHUNK_CLASS, CHUNK_METHOD_RENDER]

        with patch("main._write_php_file"), \
             patch("main._load_golden_dataset", return_value=[]):
            r = client.post("/generate-test", json={
                "project_id":  "proj_test",
                "description": "Tester la page liste en async",
                "async_mode":  True,
            })

        assert r.status_code == 200
        body = r.json()
        assert body["status"]   == "accepted"
        assert "job_id"         in body
        assert "poll_url"       in body
        assert body["poll_url"] == f"/job/{body['job_id']}/status"

    def test_generate_unit_test_async_returns_job_id(self, client_with_mocks):
        """async_mode=True sur generate-unit-test → job_id immédiat."""
        client, _, _ = client_with_mocks

        with patch("main.extract_code_for_symbol", return_value=["class FooService {}"]):
            r = client.post("/generate-unit-test", json={
                "project_id":  "proj_test",
                "file_path":   "src/Service/FooService.php",
                "class_name":  "FooService",
                "description": "test async unit",
                "async_mode":  True,
            })

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "accepted"
        assert "job_id"       in body

    def test_job_status_unknown_returns_404(self, client_with_mocks):
        """Un job_id inexistant retourne 404."""
        client, _, _ = client_with_mocks
        r = client.get("/job/doesnotexist000/status")
        assert r.status_code == 404

    def test_job_status_known_job_is_reachable(self, client_with_mocks):
        """Un job créé via async_mode est immédiatement pollable."""
        client, mock_db, _ = client_with_mocks
        mock_db.find_closest_code_context.return_value = [CHUNK_CLASS, CHUNK_METHOD_RENDER]

        with patch("main._write_php_file"), \
             patch("main._load_golden_dataset", return_value=[]):
            r = client.post("/generate-test", json={
                "project_id":  "proj_test",
                "description": "async poll test",
                "async_mode":  True,
            })

        job_id = r.json().get("job_id")
        assert job_id, "job_id manquant dans la réponse"

        r2 = client.get(f"/job/{job_id}/status")
        assert r2.status_code == 200
        body2 = r2.json()
        assert body2["job_id"] == job_id
        assert body2["status"] in ("pending", "running", "done", "error")

    def test_batch_launches_jobs(self, client_with_mocks):
        """Le batch crée un job par classe et retourne leur poll_url."""
        client, mock_db, _ = client_with_mocks
        mock_db.find_closest_code_context.return_value = [CHUNK_CLASS, CHUNK_METHOD_RENDER]

        with patch("main._write_php_file"), \
             patch("main._load_golden_dataset", return_value=[]):
            r = client.post("/generate-tests-batch", json={
                "project_id":    "proj_test",
                "class_names":   ["FooController", "BarController"],
                "deterministic": True,
            })

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "accepted"
        assert len(body["jobs"]) == 2
        for job in body["jobs"]:
            assert "job_id"   in job
            assert "poll_url" in job

    def test_batch_empty_class_names_returns_422(self, client_with_mocks):
        """Un batch sans classe retourne une erreur de validation (422)."""
        client, _, _ = client_with_mocks
        r = client.post("/generate-tests-batch", json={
            "project_id":  "proj_test",
            "class_names": [],
        })
        # Pydantic min_items=1 → 422 Unprocessable Entity
        assert r.status_code == 422
