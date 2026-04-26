"""
Exécuteur de tests autonome (sans pytest) pour le code_parser enrichi Axe 1.
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(__file__))
from code_parser import (
    _parse_file_content,
    _extract_http_methods,
    _extract_isgranted_from_attrs,
    _extract_all_routes,
    _analyze_method_body,
    _classify_controller,
    _extract_balanced_block,
    _find_class_block,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CREANCE_PHP = open("/mnt/user-data/uploads/CreanceController.php", "r", encoding="utf-8").read()
ETAT_PHP    = open("/mnt/user-data/uploads/EtatImportController.php", "r", encoding="utf-8").read()

PASSED  = 0
FAILED  = 0
ERRORS  = []


def test(name, condition, detail=""):
    global PASSED, FAILED, ERRORS
    if condition:
        PASSED += 1
        print(f"  ✓ {name}")
    else:
        FAILED += 1
        msg = f"  ✗ {name}"
        if detail:
            msg += f"  →  {detail}"
        print(msg)
        ERRORS.append(name)


# ═══════════════════════════════════════════════════════════════════════════
# 1. TESTS UNITAIRES : _extract_http_methods
# ═══════════════════════════════════════════════════════════════════════════

print("\n── _extract_http_methods ──")

test("single POST php8",
     _extract_http_methods("#[Route('/foo', methods: ['POST'])]") == ["POST"])

result = _extract_http_methods("#[Route('/foo', methods: ['GET', 'POST'])]")
test("multiple methods php8",
     "GET" in result and "POST" in result, f"got {result}")

test("phpdoc legacy syntax",
     _extract_http_methods('@Route("/foo", methods={"DELETE"})') == ["DELETE"])

test("no methods → empty list",
     _extract_http_methods("#[Route('/foo', name: 'foo_index')]") == [])

test("double quotes",
     _extract_http_methods('#[Route("/foo", methods: ["PUT", "PATCH"])]') == ["PUT", "PATCH"])

# ═══════════════════════════════════════════════════════════════════════════
# 2. TESTS UNITAIRES : _extract_isgranted_from_attrs
# ═══════════════════════════════════════════════════════════════════════════

print("\n── _extract_isgranted_from_attrs ──")

r = _extract_isgranted_from_attrs("#[IsGranted('ROLE_ADMIN')]")
test("simple role", len(r) == 1 and r[0]["role"] == "ROLE_ADMIN")

r = _extract_isgranted_from_attrs("#[IsGranted('ROLE_ADMIN', message: 'accès restreint au rôle ROLE_ADMIN')]")
test("role with message", len(r) == 1 and r[0]["role"] == "ROLE_ADMIN" and "accès restreint" in (r[0]["message"] or ""))

attr = "#[IsGranted('ROLE_PARCOURS', message: 'accès restreint')]\n#[IsGranted('ROLE_ADMIN')]"
r = _extract_isgranted_from_attrs(attr)
test("multiple IsGranted", len(r) == 2 and {x["role"] for x in r} == {"ROLE_PARCOURS", "ROLE_ADMIN"}, f"got {r}")

r = _extract_isgranted_from_attrs("/** @IsGranted('ROLE_EDITOR') */")
test("phpdoc @IsGranted", len(r) == 1 and r[0]["role"] == "ROLE_EDITOR")

test("no IsGranted → empty",
     _extract_isgranted_from_attrs("#[Route('/foo')]") == [])

# ═══════════════════════════════════════════════════════════════════════════
# 3. TESTS UNITAIRES : _extract_all_routes
# ═══════════════════════════════════════════════════════════════════════════

print("\n── _extract_all_routes ──")

r = _extract_all_routes("#[Route('/foo', name: 'foo_index')]", "")
test("single route", len(r) == 1 and r[0]["path"] == "/foo" and r[0]["name"] == "foo_index")

attr = "#[Route('/', name: 'creance', options: ['expose' => true])]\n    #[Route('/parcours', name: 'creance_parcours', options: ['expose' => true])]"
r = _extract_all_routes(attr, "/creance")
paths = {x["path"] for x in r}
test("multi-route with prefix", len(r) == 2 and "/creance/" in paths and "/creance/parcours" in paths, f"got {paths}")

r = _extract_all_routes("#[Route('/export', name: 'export', methods: ['POST'])]", "")
test("route with methods", len(r) == 1 and r[0]["http_methods"] == ["POST"])

r = _extract_all_routes("#[Route('/{id}/edit', name: 'edit')]", "/creance")
test("route with prefix", r[0]["path"] == "/creance/{id}/edit", f"got {r[0]['path']}")

test("no route → empty",
     _extract_all_routes("#[IsGranted('ROLE_ADMIN')]", "/foo") == [])

# ═══════════════════════════════════════════════════════════════════════════
# 4. TESTS UNITAIRES : _analyze_method_body
# ═══════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════
# 5. INTÉGRATION : CreanceController réel
# ═══════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════
# 6. INTÉGRATION : EtatImportController réel
# ═══════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════
# 7. _classify_controller
# ═══════════════════════════════════════════════════════════════════════════

print("\n── _classify_controller ──")

test("all json → api",
     _classify_controller("Api", [
         {"name": "a", "routes": [{"path": "/a"}], "response_types": ["json (200)"]},
         {"name": "b", "routes": [{"path": "/b"}], "response_types": ["json (200)"]},
         {"name": "c", "routes": [{"path": "/c"}], "response_types": ["json (200)"]},
     ], []) == "api")

test("all render → web_crud",
     _classify_controller("Web", [
         {"name": "a", "routes": [{"path": "/a"}], "response_types": ["render (200)"]},
         {"name": "b", "routes": [{"path": "/b"}], "response_types": ["render (200)"]},
     ], []) == "web_crud")

test("no routes → internal",
     _classify_controller("Batch", [
         {"name": "a", "routes": [], "response_types": []},
     ], []) == "internal")

test("mixed render+json → mixed",
     _classify_controller("Mix", [
         {"name": "a", "routes": [{"path": "/a"}], "response_types": ["render (200)"]},
         {"name": "b", "routes": [{"path": "/b"}], "response_types": ["json (200)"]},
         {"name": "c", "routes": [{"path": "/c"}], "response_types": ["render (200)"]},
         {"name": "d", "routes": [{"path": "/d"}], "response_types": ["json (200)"]},
     ], []) == "mixed")

# ═══════════════════════════════════════════════════════════════════════════
# 8. NON-RÉGRESSION
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Non-régression ──")

test("balanced block",
     _extract_balanced_block("{ if ($x) { return $x; } }", 0) == "{ if ($x) { return $x; } }")

# ── Tests du fix commentaires ──
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

# ═══════════════════════════════════════════════════════════════════════════
# RÉSULTAT
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"RÉSULTAT :  {PASSED} passés,  {FAILED} échoués")
if ERRORS:
    print(f"Tests en échec :")
    for e in ERRORS:
        print(f"  - {e}")
print(f"{'='*60}")
sys.exit(1 if FAILED else 0)
