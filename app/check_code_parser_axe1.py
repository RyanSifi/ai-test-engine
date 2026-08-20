"""
Exécuteur de tests autonome (sans pytest) pour le code_parser enrichi Axe 1.

⚠️  Ce fichier s'appelle `check_*` et NON `test_*` volontairement.

Il exécute ses vérifications au niveau module et se termine par `sys.exit()`.
Sous le nom `test_code_parser_axe1.py`, pytest le collectait, exécutait le
`sys.exit()` pendant la phase de collecte et s'arrêtait sur `INTERNALERROR:
SystemExit` — plus AUCUN test du dossier ne tournait, y compris ceux de
test_main.py.

Deux façons de le lancer, les deux valides :
  - en direct  : python check_code_parser_axe1.py
  - via pytest : test_code_parser_axe1.py l'exécute en sous-processus et
                 vérifie son code de retour (isole aussi les plantages, ex.
                 PHP absent, en échec de test au lieu d'erreur de collecte).
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(__file__))
from code_parser import (
    _parse_file_content,
    _analyze_method_body,
    _classify_controller,
    _extract_balanced_block,
    _find_class_block,
)

PASSED  = 0
FAILED  = 0
ERRORS  = []


def test(name, condition, detail=""):
    global PASSED, FAILED, ERRORS
    if condition:
        PASSED += 1
        print(f"  {name}")
    else:
        FAILED += 1
        msg = f"  {name}"
        if detail:
            msg += f"  →  {detail}"
        print(msg)
        ERRORS.append(name)


def _method(content: str, name: str = "foo"):
    """Helper : parse un extrait PHP et retourne la méthode `name` (ou None)."""
    parsed = _parse_file_content(content)
    return next((m for m in parsed.get("methods", []) if m["name"] == name), None)


# 1. TESTS D'INTÉGRATION LÉGERS : extraction des routes/attributs via AST
#
# Depuis le passage à un vrai parseur PHP (nikic/php-parser, voir
# code_parser.py et app/php_ast/parse.php), les anciennes fonctions
# unitaires _extract_http_methods / _extract_isgranted_from_attrs /
# _extract_all_routes / _find_last_attribute_run n'existent plus : elles
# opéraient sur du texte d'attribut brut, remplacé par de l'AST structuré.
# Les mêmes comportements sont donc vérifiés ici via _parse_file_content()
# sur des extraits PHP minimaux plutôt que sur des chaînes de texte d'attribut.

print("\n── Extraction des verbes HTTP (#[Route(methods: ...)]) ──")

php = """<?php
class C {
    #[Route('/foo', methods: ['POST'])]
    public function foo() {}
}
"""
m = _method(php)
test("single POST php8", m["routes"][0]["http_methods"] == ["POST"], f"got {m['routes']}")

php = """<?php
class C {
    #[Route('/foo', methods: ['GET', 'POST'])]
    public function foo() {}
}
"""
m = _method(php)
result = m["routes"][0]["http_methods"]
test("multiple methods php8", "GET" in result and "POST" in result, f"got {result}")

php = """<?php
class C {
    /**
     * @Route("/foo", methods={"DELETE"})
     */
    public function foo() {}
}
"""
m = _method(php)
test("phpdoc legacy syntax", m["routes"][0]["http_methods"] == ["DELETE"], f"got {m['routes']}")

php = """<?php
class C {
    #[Route('/foo', name: 'foo_index')]
    public function foo() {}
}
"""
m = _method(php)
test("no methods → empty list", m["routes"][0]["http_methods"] == [], f"got {m['routes']}")

php = """<?php
class C {
    #[Route("/foo", methods: ["PUT", "PATCH"])]
    public function foo() {}
}
"""
m = _method(php)
test("double quotes", m["routes"][0]["http_methods"] == ["PUT", "PATCH"], f"got {m['routes']}")

# 2. TESTS D'INTÉGRATION LÉGERS : #[IsGranted]

print("\n── Extraction #[IsGranted] ──")

php = """<?php
class C {
    #[IsGranted('ROLE_ADMIN')]
    public function foo() {}
}
"""
m = _method(php)
test("simple role", len(m["method_grants"]) == 1 and m["method_grants"][0]["role"] == "ROLE_ADMIN")

php = """<?php
class C {
    #[IsGranted('ROLE_ADMIN', message: 'accès restreint au rôle ROLE_ADMIN')]
    public function foo() {}
}
"""
m = _method(php)
g = m["method_grants"]
test("role with message",
     len(g) == 1 and g[0]["role"] == "ROLE_ADMIN" and "accès restreint" in (g[0]["message"] or ""),
     f"got {g}")

php = """<?php
#[IsGranted('ROLE_PARCOURS')]
class C {
    #[IsGranted('ROLE_ADMIN')]
    public function foo() {}
}
"""
parsed = _parse_file_content(php)
class_roles = [g["role"] for g in parsed["class_grants"]]
method_roles = [g["role"] for g in parsed["methods"][0]["method_grants"]]
test("class + method IsGranted",
     class_roles == ["ROLE_PARCOURS"] and method_roles == ["ROLE_ADMIN"],
     f"got class={class_roles} method={method_roles}")

php = """<?php
class C {
    /** @IsGranted('ROLE_EDITOR') */
    public function foo() {}
}
"""
m = _method(php)
test("phpdoc @IsGranted", len(m["method_grants"]) == 1 and m["method_grants"][0]["role"] == "ROLE_EDITOR")

php = """<?php
class C {
    #[Route('/foo')]
    public function foo() {}
}
"""
m = _method(php)
test("no IsGranted → empty", m["method_grants"] == [])

# 3. TESTS D'INTÉGRATION LÉGERS : #[Route] (chemin, nom, multi-route, imbrication)

print("\n── Extraction #[Route] ──")

php = """<?php
class C {
    #[Route('/foo', name: 'foo_index')]
    public function foo() {}
}
"""
m = _method(php)
test("single route", len(m["routes"]) == 1 and m["routes"][0]["path"] == "/foo" and m["routes"][0]["name"] == "foo_index")

php = """<?php
#[Route('/creance')]
class C {
    #[Route('/', name: 'creance', options: ['expose' => true])]
    #[Route('/parcours', name: 'creance_parcours', options: ['expose' => true])]
    public function index() {}
}
"""
m = _method(php, "index")
paths = {r["path"] for r in m["routes"]}
test("multi-route with prefix",
     len(m["routes"]) == 2 and "/creance/" in paths and "/creance/parcours" in paths,
     f"got {paths}")

php = """<?php
class C {
    #[Route('/export', name: 'export', methods: ['POST'])]
    public function foo() {}
}
"""
m = _method(php)
test("route with methods", m["routes"][0]["http_methods"] == ["POST"])

php = """<?php
#[Route('/creance')]
class C {
    #[Route('/{id}/edit', name: 'edit')]
    public function foo() {}
}
"""
m = _method(php)
test("route with prefix", m["routes"][0]["path"] == "/creance/{id}/edit", f"got {m['routes']}")

php = """<?php
class C {
    #[IsGranted('ROLE_ADMIN')]
    public function foo() {}
}
"""
m = _method(php)
test("no route → empty", m["routes"] == [])

# Cas piège : tableau imbriqué à 2 niveaux dans requirements (ex: sous-liste).
# Avant le passage à l'AST, la regex à profondeur fixe faisait retourner []
# silencieusement dès qu'un argument d'attribut contenait un tableau imbriqué
# à 2 niveaux — cas réel rencontré sur des routes Symfony.
php = """<?php
class C {
    #[Route('/foo/{id}', name: 'foo_show', requirements: ['id' => '\\d+', 'sub' => ['a', 'b']], methods: ['GET'])]
    public function foo() {}
}
"""
m = _method(php)
test("route with 2-level nested array in requirements",
     len(m["routes"]) == 1 and m["routes"][0]["path"] == "/foo/{id}"
     and m["routes"][0]["name"] == "foo_show" and m["routes"][0]["http_methods"] == ["GET"],
     f"got {m['routes']}")

# Cas AST spécifiques : ce que le vrai parseur apporte par rapport aux regex
# (arguments nommés dans un ordre arbitraire, constante de classe en valeur,
# méthode sans attribut, attributs multi-lignes).

print("\n── Cas spécifiques AST ──")

php = """<?php
class C {
    #[Route(name: 'weird_order', methods: ['GET'], path: '/weird')]
    public function foo() {}
}
"""
m = _method(php)
test("named args in arbitrary order",
     m["routes"][0]["path"] == "/weird" and m["routes"][0]["name"] == "weird_order",
     f"got {m['routes']}")

php = """<?php
class C {
    public function noAttr(int $x) { return $x; }
}
"""
m = _method(php, "noAttr")
test("method without any attribute", m["routes"] == [] and m["method_grants"] == [])

php = """<?php
class C {
    #[Route(
        '/multi',
        name: 'multi_line',
        methods: ['GET']
    )]
    public function foo() {}
}
"""
m = _method(php)
test("multi-line attribute", m["routes"][0]["path"] == "/multi" and m["routes"][0]["name"] == "multi_line")

# 4. TESTS UNITAIRES : _analyze_method_body

print("\n── _analyze_method_body ──")

r = _analyze_method_body("{ return $this->render('foo/index.html.twig', ['title' => 'Hello']); }")
test("render response", r["rendered_template"] == "foo/index.html.twig" and "render (200)" in r["response_types"])

r = _analyze_method_body("{ return $this->redirectToRoute('foo_index'); }")
test("redirect response", "redirect (302)" in r["response_types"])

r = _analyze_method_body("{ return $this->json(['data' => $items]); }")
test("json ->json()", "json (200)" in r["response_types"])

r = _analyze_method_body("{ return new JsonResponse(['ok' => true]); }")
test("json JsonResponse", "json (200)" in r["response_types"])

body = """{
    $content = file_get_contents($path);
    $response = new Response();
    $response->headers->set('Content-Disposition', 'attachment; filename=' . $filename);
    return $response;
}"""
r = _analyze_method_body(body)
test("file download", "file_download (200)" in r["response_types"], f"got {r['response_types']}")

r = _analyze_method_body("{ return new ExportResponse($data); }")
test("export response", "export (200)" in r["response_types"])

r = _analyze_method_body("{ return new BinaryFileResponse($path); }")
test("binary response", "binary (200)" in r["response_types"])

body = """{
    if ($form->isSubmitted() && $form->isValid()) {
        return $this->redirectToRoute('foo_index');
    }
    return $this->render('foo/new.html.twig', ['form' => $form]);
}"""
r = _analyze_method_body(body)
test("mixed render+redirect", "render (200)" in r["response_types"] and "redirect (302)" in r["response_types"])
test("form detected in mixed", r["has_form"] is True)

body = """{
    if (!$request->isXmlHttpRequest()) {
        return $this->json([], Response::HTTP_NOT_FOUND);
    }
    return $this->json($data);
}"""
r = _analyze_method_body(body)
test("AJAX check", r["is_ajax_only"] is True)

body = """{
    $form = $this->createForm(CommentairesOrdonnateurType::class);
    $form->handleRequest($request);
}"""
r = _analyze_method_body(body)
test("form type extraction", r["has_form"] is True and r["form_type"] == "CommentairesOrdonnateurType")

body = """{
    $this->denyAccessUnlessGranted('view', $creance, 'accès interdit');
    return $this->render('show.html.twig');
}"""
r = _analyze_method_body(body)
test("voter check", len(r["voter_checks"]) == 1 and r["voter_checks"][0]["attribute"] == "view" and r["voter_checks"][0]["subject"] == "creance")

body = "{ set_time_limit(0); return $result; }"
r = _analyze_method_body(body)
test("no response type for internal", r["response_types"] == [])

# 5. _classify_controller

print("\n── _classify_controller ──")

test("all json → api",
     _classify_controller([
         {"name": "a", "routes": [{"path": "/a"}], "response_types": ["json (200)"]},
         {"name": "b", "routes": [{"path": "/b"}], "response_types": ["json (200)"]},
         {"name": "c", "routes": [{"path": "/c"}], "response_types": ["json (200)"]},
     ], []) == "api")

test("all render → web_crud",
     _classify_controller([
         {"name": "a", "routes": [{"path": "/a"}], "response_types": ["render (200)"]},
         {"name": "b", "routes": [{"path": "/b"}], "response_types": ["render (200)"]},
     ], []) == "web_crud")

test("no routes → internal",
     _classify_controller([
         {"name": "a", "routes": [], "response_types": []},
     ], []) == "internal")

test("mixed render+json → mixed",
     _classify_controller([
         {"name": "a", "routes": [{"path": "/a"}], "response_types": ["render (200)"]},
         {"name": "b", "routes": [{"path": "/b"}], "response_types": ["json (200)"]},
         {"name": "c", "routes": [{"path": "/c"}], "response_types": ["render (200)"]},
         {"name": "d", "routes": [{"path": "/d"}], "response_types": ["json (200)"]},
     ], []) == "mixed")

# 6. NON-RÉGRESSION : _extract_balanced_block / _find_class_block (inchangés)

print("\n── Non-régression ──")

test("balanced block",
     _extract_balanced_block("{ if ($x) { return $x; } }", 0) == "{ if ($x) { return $x; } }")

# Tests du fix commentaires
test("comment // avec apostrophe",
     _extract_balanced_block("{ // l'input est 'final'\n return 1; }", 0) is not None)

test("comment // ne corrompt pas les braces",
     _extract_balanced_block("{ // commentaire { ouvert\n $x = 1; }", 0) == "{ // commentaire { ouvert\n $x = 1; }")

test("comment /* */ avec apostrophe",
     _extract_balanced_block("{ /* c'est l'état */ return 1; }", 0) is not None)

test("comment /* */ avec brace",
     _extract_balanced_block("{ /* { non compté } */ return 1; }", 0) == "{ /* { non compté } */ return 1; }")

test("comment # (PHP alt syntax)",
     _extract_balanced_block("{ # l'apostrophe\n return 1; }", 0) is not None)

test("string avec brace échappée",
     _extract_balanced_block('{ $s = "a{b}c"; }', 0) == '{ $s = "a{b}c"; }')

test("mixed comments and strings",
     _extract_balanced_block("{ // 'test'\n $x = 'val'; /* } */ }", 0) is not None)

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
test("find_class_block", block is not None and "private int $x" in block and block.count("{") == block.count("}"))

# 7. INTÉGRATION SUR FIXTURES RÉELLES (CreanceController / EtatImportController)
#
# Ces fixtures ne sont pas versionnées dans ce repo (contiennent du code
# client). Chemins contrôlables via env vars (CI / autre poste).
# Le check d'existence est placé ICI (pas en haut du fichier) pour que les
# ~40 tests ci-dessus, qui n'en dépendent pas, tournent toujours — avant ce
# fix, sys.exit(0) en tête de fichier faisait sauter TOUS les tests dès que
# les fixtures étaient absentes, y compris ceux qui n'en avaient pas besoin.

_FIXTURES_DIR = os.environ.get(
    "AXE1_FIXTURES_DIR",
    os.path.join(os.path.dirname(__file__), "tests", "fixtures"),
)
_CREANCE_PATH = os.environ.get(
    "AXE1_CREANCE_PATH",
    os.path.join(_FIXTURES_DIR, "CreanceController.php"),
)
_ETAT_PATH = os.environ.get(
    "AXE1_ETAT_PATH",
    os.path.join(_FIXTURES_DIR, "EtatImportController.php"),
)

if not (os.path.isfile(_CREANCE_PATH) and os.path.isfile(_ETAT_PATH)):
    print(
        "\nFixtures introuvables — tests d'intégration sur fixtures réelles ignorés.\n"
        f"  CreanceController.php attendu à : {_CREANCE_PATH}\n"
        f"  EtatImportController.php attendu à : {_ETAT_PATH}\n"
        "  Surcharge possible : AXE1_CREANCE_PATH / AXE1_ETAT_PATH / AXE1_FIXTURES_DIR"
    )
else:
    CREANCE_PHP = open(_CREANCE_PATH, "r", encoding="utf-8").read()
    ETAT_PHP    = open(_ETAT_PATH,    "r", encoding="utf-8").read()

    print("\n── CreanceController (intégration) ──")

    cr = _parse_file_content(CREANCE_PHP)

    test("class_name", cr["class_name"] == "CreanceController")
    test("class_route_prefix", cr["class_route_prefix"] == "/creance")
    test("controller_profile", cr["controller_profile"] in ("web_crud", "mixed"), f"got {cr['controller_profile']}")

    # IsGranted classe
    roles = [g["role"] for g in cr["class_grants"]]
    test("class IsGranted ROLE_PARCOURS", "ROLE_PARCOURS" in roles, f"got {roles}")

    # index : 2 routes
    index = next(m for m in cr["methods"] if m["name"] == "index")
    test("index has 2 routes", len(index["routes"]) == 2, f"got {len(index['routes'])} routes: {index['routes']}")
    if index["routes"]:
        paths = {r["path"] for r in index["routes"]}
        test("index paths correct", "/creance/" in paths and "/creance/parcours" in paths, f"got {paths}")

    # creanceNotifiees : 2 routes
    cn = next(m for m in cr["methods"] if m["name"] == "creanceNotifiees")
    test("creanceNotifiees has 2 routes", len(cn["routes"]) == 2, f"got {len(cn['routes'])}")

    # edit : IsGranted méthode
    edit = next(m for m in cr["methods"] if m["name"] == "edit")
    edit_roles = [g["role"] for g in edit["method_grants"]]
    test("edit has ROLE_MENU_CONSULTATION", "ROLE_MENU_CONSULTATION" in edit_roles, f"got {edit_roles}")

    # creancesSansWfExportAction : POST method
    exp = next(m for m in cr["methods"] if m["name"] == "creancesSansWfExportAction")
    test("export action has POST",
         len(exp["routes"]) >= 1 and "POST" in exp["routes"][0]["http_methods"],
         f"got {exp['routes']}")

    # getData : AJAX-only
    gd = next(m for m in cr["methods"] if m["name"] == "getData")
    test("getData is AJAX-only", gd["is_ajax_only"] is True)

    # suspendusMotifAnvList : AJAX
    sma = next(m for m in cr["methods"] if m["name"] == "suspendusMotifAnvList")
    test("suspendusMotifAnvList is AJAX", sma["is_ajax_only"] is True)

    # ajoutCommentaire : formulaire
    ac = next(m for m in cr["methods"] if m["name"] == "ajoutCommentaire")
    test("ajoutCommentaire has form", ac["has_form"] is True)
    test("ajoutCommentaire form_type", ac["form_type"] == "CommentairesOrdonnateurType")
    test("ajoutCommentaire mixed response",
         "render (200)" in ac["response_types"] and "redirect (302)" in ac["response_types"],
         f"got {ac['response_types']}")

    # parcours : voter check
    par = next(m for m in cr["methods"] if m["name"] == "parcours")
    test("parcours has voter check",
         len(par["voter_checks"]) >= 1 and par["voter_checks"][0]["attribute"] == "view",
         f"got {par['voter_checks']}")

    # exportDownload : file download
    ed = next(m for m in cr["methods"] if m["name"] == "exportDownload")
    test("exportDownload is file_download",
         "file_download (200)" in ed["response_types"],
         f"got {ed['response_types']}")

    # Legacy compat
    test("legacy route field", index["route"] is not None)
    test("legacy response_type field", index["response_type"] is not None)

    print("\n── EtatImportController (intégration) ──")

    et = _parse_file_content(ETAT_PHP)

    test("class_name", et["class_name"] == "EtatImportController")
    test("classified as internal", et["controller_profile"] == "internal")

    roles = [g["role"] for g in et["class_grants"]]
    test("class has ROLE_ADMIN", "ROLE_ADMIN" in roles, f"got {roles}")

    for m in et["methods"]:
        if m["name"] != "__construct":
            test(f"{m['name']} has no routes", m["routes"] == [], f"got {m['routes']}")

    params = et["constructor_params"]
    ptypes = [p["type"] for p in params]
    test("constructor has ManagerRegistry", "ManagerRegistry" in ptypes, f"got {ptypes}")
    test("constructor has DatesEcheancesManager", "DatesEcheancesManager" in ptypes)
    test("constructor has MouvEtatService", "MouvEtatService" in ptypes)
    test("constructor has EtatImportService", "EtatImportService" in ptypes)

    st = next(m for m in et["methods"] if m["name"] == "startTreatment")
    test("startTreatment no response_types", st["response_types"] == [], f"got {st['response_types']}")
    test("startTreatment return_type is int", st["return_type"] == "int")

# RÉSULTAT

print(f"\n{'='*60}")
print(f"RÉSULTAT :  {PASSED} passés,  {FAILED} échoués")
if ERRORS:
    print(f"Tests en échec :")
    for e in ERRORS:
        print(f"  - {e}")
print(f"{'='*60}")
sys.exit(1 if FAILED else 0)
