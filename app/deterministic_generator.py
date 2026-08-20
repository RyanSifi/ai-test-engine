"""
Générateur déterministe de WebTestCase PHP — bypass LLM, génère directement
depuis les chunks indexés (élimine toute hallucination structurelle).

Contient aussi le moteur de scénarios (builders) qui consomme les métadonnées
parsées par _parse_chunk_metadata(). Les deux sont gardés dans le même module
car fortement couplés : le dict `ctx` produit par _parse_chunk_metadata() est
consommé directement par chaque builder — les séparer déplacerait le couplage
sans le réduire.
"""
import re
from functools import lru_cache
from typing import Dict, List, Optional

from config import settings
from chunk_format import (
    _RE_METHOD_NAME, _RE_ROUTE, _RE_RESPONSE_TYPE, _RE_H1, _RE_HIDDEN_IDS,
    _RE_CONSTRUCTOR, _RE_ROUTE_PARAM, _RE_HTTP_VERBS, _RE_AJAX_ONLY,
    _RE_FORM_TYPE, _RE_VOTER, _RE_CLASS_ROLES, _RE_METHOD_ROLE, _RE_PROFILE,
    _RE_INTERNAL, _RE_BODY_INFERRED_VERB, _RE_PARAMS_LINE,
)

# Identifiant volontairement hors de portée, utilisé pour construire les tests
# d'erreur (« ressource inexistante → 404 », « voter refuse l'accès »).
# Assez grand pour qu'aucune entité réelle ne porte cet ID, tout en restant un
# entier valide côté Symfony — un ID non numérique déclencherait une erreur de
# routage (404 aussi, mais pour la mauvaise raison : le test ne prouverait rien).
BOGUS_ENTITY_ID = "99999999"

# Nombre d'éléments repris dans les assertions, parmi ceux extraits du template.
# Un seul suffit : l'objectif est de vérifier que la page est bien celle attendue,
# pas d'en re-tester tout le DOM — chaque assertion supplémentaire est un point de
# rupture de plus quand le template évolue.
MAX_ASSERTED_HIDDEN_IDS = 1

# Nombre de rôles secondaires testés en plus du rôle principal. Au-delà, on
# multiplie les tests quasi identiques sans gagner en couverture réelle.
MAX_SECONDARY_ROLES_TESTED = 1

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


@lru_cache(maxsize=1)
def _role_key_map() -> Dict[str, str]:
    """
    Parse settings.auth_role_key_map ("ROLE_A:KEY_A,ROLE_B:KEY_B") une seule fois.
    Le cache est invalidé naturellement si le process redémarre (settings est figé).
    """
    raw = settings.auth_role_key_map or ""
    mapping: Dict[str, str] = {}
    for pair in raw.split(","):
        if ":" not in pair:
            continue
        src, dst = pair.split(":", 1)
        mapping[src.strip()] = dst.strip()
    return mapping


def _role_to_factory_key(role: str) -> str:
    """
    Mappe un rôle Symfony (ex: ROLE_PARCOURS) vers la clé attendue par TestUserFactory.
    Sans mapping, le rôle est retourné tel quel.
    """
    return _role_key_map().get(role, role)


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


# Heuristiques de détection des verbes HTTP par nom de méthode et de route.
# Utilisé en fallback quand #[Route(methods: [...])] n'est pas explicite.
_VERB_NAME_HINTS = {
    "DELETE": ("delete", "remove", "destroy", "trash", "erase", "unlink"),
    "PUT":    ("update", "edit", "modify", "replace", "put"),
    "PATCH":  ("patch", "toggle", "enable", "disable", "activate", "deactivate"),
    "POST":   ("create", "add", "new", "save", "store", "submit", "send",
               "post", "register", "ajout", "ajouter", "creer", "creation",
               "valider", "soumettre", "publier", "envoyer"),
}


def _infer_http_verb_from_name(method_name: str, route_path: str) -> Optional[str]:
    """Infère un verbe HTTP depuis le nom de méthode ou la route, en français/anglais."""
    name_l = method_name.lower()
    route_l = route_path.lower() if route_path else ""

    for verb, hints in _VERB_NAME_HINTS.items():
        for hint in hints:
            # Préfixe de méthode (deleteFoo, updateBar) ou suffixe de route (/delete, /update)
            if name_l.startswith(hint) or f"/{hint}" in route_l or route_l.endswith(hint):
                return verb
    return None


def _infer_http_verb_from_body(content: str) -> Optional[str]:
    """
    Récupère le verbe HTTP inféré par l'analyse profonde du body au moment de
    l'indexation (cf. code_parser._analyze_method_body : isMethod, $request->request,
    upload de fichiers, formulaire détecté).
    """
    m = _RE_BODY_INFERRED_VERB.search(content)
    if m:
        return m.group(1).strip()
    # Fallback sur le marqueur Formulaire (compat ancien index)
    if "Formulaire:" in content or "→ Formulaire" in content:
        return "POST"
    return None


def _extract_http_verb(content: str, method_name: str = "", route_path: str = "") -> str:
    """
    Détermine le verbe HTTP à utiliser pour un test, par ordre de priorité :
      1. Verbe explicite déclaré dans #[Route(methods: [...])]
      2. Indices dans le body de la méthode (formulaire, etc.)
      3. Heuristique sur le nom de méthode (delete*, update*, etc.) ou la route
      4. GET par défaut
    """
    # Verbe explicite
    m = _RE_HTTP_VERBS.search(content)
    if m:
        verbs = [v.strip() for v in m.group(1).split(",") if v.strip()]
        # Multi-verbes : prioriser POST > PUT > DELETE > PATCH > GET
        for preferred in ("POST", "PUT", "DELETE", "PATCH", "GET"):
            if preferred in verbs:
                return preferred
        if verbs:
            return verbs[0]

    # Indices dans le body
    inferred = _infer_http_verb_from_body(content)
    if inferred:
        return inferred

    # Heuristique nom + route
    inferred = _infer_http_verb_from_name(method_name, route_path)
    if inferred:
        return inferred

    return "GET"


def _render_test_class_skeleton(test_class_name: str, private_comment: str = ""):
    """
    Rend le squelette DÉTERMINISTE d'une classe de test WebTestCase (namespace,
    imports, propriétés, setUp, getTestUser) et retourne un couple (header, footer)
    tel que `header + corps_des_méthodes + footer` produit un fichier PHP complet.

    Utilisé à la fois par le générateur déterministe (_generate_php_test_from_chunks)
    et par le générateur par route (le corps des méthodes vient alors du LLM, route
    par route, mais le squelette reste garanti sans hallucination).
    """
    factory       = settings.auth_test_class
    sso_user      = settings.auth_sso_user_class
    factory_short = factory.split("\\")[-1]
    sso_short     = sso_user.split("\\")[-1]

    header = (
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
        "\n"
    )
    footer = "\n}\n"
    return header, footer


def _generate_php_test_from_chunks(
    chunks: List[Dict],
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
    # Dédoublonnage sur le COUPLE (méthode, route) et non sur le seul nom de
    # méthode : learn_from_code() produit UN CHUNK PAR ROUTE, donc une méthode
    # portant plusieurs #[Route] apparaît plusieurs fois sous le même nom.
    # Dédoublonner par nom faisait perdre silencieusement toutes ses routes
    # sauf la première — exactement le type de disparition muette que le
    # passage à un parseur AST visait à éliminer en amont.
    routes_seen: set     = set()   # couples (méthode, route) déjà traités
    route_rank: Dict[str, int] = {}   # méthode -> nb de routes déjà rencontrées
    private_seen: set    = set()   # méthodes sans route (déduplication du commentaire)
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
        if method_name == "__construct":
            continue

        route_m = _RE_ROUTE.search(content)
        if not route_m:
            # Méthode interne : une seule mention suffit dans le commentaire final
            if method_name not in private_seen:
                private_seen.add(method_name)
                skipped_private.append(method_name)
            continue

        raw_route = route_m.group(1)
        if (method_name, raw_route) in routes_seen:
            continue
        routes_seen.add((method_name, raw_route))

        # 1re route : noms de test inchangés (rétro-compatibilité).
        # Routes suivantes : suffixe Route2, Route3… pour éviter la collision.
        rank = route_rank.get(method_name, 0)
        route_rank[method_name] = rank + 1
        cap_suffix = "" if rank == 0 else f"Route{rank + 1}"

        # Extraire les métadonnées du chunk
        ctx = _parse_chunk_metadata(content, raw_route, class_role, cap_suffix)

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

    # Assemblage final : squelette déterministe (header/footer) + corps des méthodes.
    body = "\n\n".join(test_methods)
    header, footer = _render_test_class_skeleton(test_class_name, private_comment)
    return header + body + footer


# Squelette de DataFixtures

# Types à exclure (frameworks, services, etc. — pas des entités métier à mocker)
_NON_ENTITY_TYPES = {
    "Request", "Response", "Session", "ContainerInterface",
    "EntityManagerInterface", "EntityManager", "ManagerRegistry",
    "FormFactoryInterface", "RouterInterface", "TranslatorInterface",
    "LoggerInterface", "EventDispatcherInterface", "ParameterBagInterface",
    "DataTableQuery", "Security", "TokenStorageInterface",
    "AuthorizationCheckerInterface", "FormView", "FormBuilderInterface",
    "string", "int", "float", "bool", "array", "object", "mixed",
    "Yaml", "AbstractController",
}


def _extract_entity_types_from_chunks(chunks: List[Dict]) -> List[str]:
    """
    Récupère les types d'entités potentiels apparaissant dans les paramètres
    de méthode des chunks (ex : 'Creance', 'CreanceRegroupee').
    Filtre les types techniques (Request, EntityManager, etc.).
    """
    found: Dict[str, int] = {}
    for c in chunks:
        content = c.get("content", "")
        for params_match in _RE_PARAMS_LINE.finditer(content):
            params_blob = params_match.group(1)
            # Chaque paramètre : "Type $name"
            for part in params_blob.split(","):
                tokens = part.strip().split()
                if len(tokens) < 2:
                    continue
                # Premier token = type (peut être "?Type")
                t = tokens[0].lstrip("?\\")
                if not t or not t[0].isupper():
                    continue
                if t in _NON_ENTITY_TYPES:
                    continue
                if t.endswith("Service") or t.endswith("Manager") or t.endswith("Repository"):
                    continue
                if t.endswith("Type"):  # FormType
                    continue
                found[t] = found.get(t, 0) + 1
    # Trier par fréquence décroissante
    return sorted(found.keys(), key=lambda k: (-found[k], k))


def _generate_fixtures_skeleton(
    entity_types: List[str],
    test_class_name: str,
) -> str:
    """
    Génère un squelette PHP DataFixtures à compléter manuellement par la MOA.
    Crée une entité minimale par type détecté, à id=1 (valeur par défaut des tests).
    """
    if not entity_types:
        return ""

    use_lines = "\n".join(f"// use App\\Entity\\{t};" for t in entity_types)
    todos = []
    for t in entity_types:
        var = t[0].lower() + t[1:]
        todos.append(
            f"        // TODO: créer une instance de {t} pour les tests\n"
            f"        // ${var} = new {t}();\n"
            f"        // $manager->persist(${var});\n"
            f"        // $this->addReference('{t.lower()}_test_1', ${var});"
        )
    todos_block = "\n\n".join(todos)

    fixtures_class_name = test_class_name + "Fixtures"
    php = f"""<?php

namespace App\\Tests\\Functional\\Fixtures;

{use_lines}
use Doctrine\\Bundle\\FixturesBundle\\Fixture;
use Doctrine\\Persistence\\ObjectManager;

/**
 * Fixtures squelette auto-générées pour {test_class_name}.
 *
 * À COMPLÉTER : décommente les stubs ci-dessous et ajuste les setters
 * aux contraintes de ton domaine. Les tests utilisent par défaut id=1
 * pour les paramètres de route.
 *
 * Activation dans config/packages/test/doctrine.yaml :
 *   doctrine:
 *     dbal:
 *       url: 'sqlite:///:memory:'
 *
 * Chargement avant chaque test :
 *   php bin/console doctrine:fixtures:load --env=test --no-interaction
 */
final class {fixtures_class_name} extends Fixture
{{
    public function load(ObjectManager $manager): void
    {{
{todos_block}

        $manager->flush();
    }}
}}
"""
    return php


# MOTEUR DE SCÉNARIOS

def _parse_chunk_metadata(content: str, raw_route: str, class_role: str,
                          cap_suffix: str = "") -> Dict:
    """
    Parse toutes les métadonnées d'un chunk de méthode en un dict plat
    réutilisable par tous les builders de scénarios.

    `cap_suffix` désambiguïse les noms de méthodes de test quand une même
    méthode PHP porte PLUSIEURS #[Route] : chaque route produit son propre jeu
    de tests, et sans suffixe les deux jeux s'appelleraient `testXxx...` à
    l'identique — donc du PHP invalide (duplicate method).
    """
    method_m   = _RE_METHOD_NAME.search(content)
    method_name = method_m.group(1) if method_m else "unknown"
    cap         = method_name[0].upper() + method_name[1:] + cap_suffix

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
        "http_verb":      _extract_http_verb(content, method_name, raw_route),
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


# BUILDERS DE SCÉNARIOS — chaque builder retourne 0..N scénarios
#
# Pour ajouter un nouveau pattern de test :
#   Écrire une fonction (ctx, fw, redirect, code, sec_roles) -> List[Dict]
#   L'ajouter à SCENARIO_BUILDERS en bas de cette section

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
        for hid in ctx["hidden_ids"][:MAX_ASSERTED_HIDDEN_IDS]:
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
    """
    Voter sur entité — tester l'accès refusé avec un ID inexistant.
    Si la route n'a pas de paramètre, on teste juste l'accès courant.
    """
    if not ctx["has_voter"]:
        return []
    fk = ctx["factory_key"]
    rl = ctx["role_label"]
    voter_attr = ctx.get("voter_attr") or "?"

    # Si la route est paramétrée, on cible une entité inexistante
    # → le voter ne peut pas accorder l'accès (entité=null) ou throw 404.
    if "{" in ctx["raw_route"]:
        bogus = re.sub(r"\{[^}]+\}", BOGUS_ENTITY_ID, ctx["raw_route"])
        return [{
            "comment":   f"{ctx['raw_route']} — voter '{voter_attr}' avec entité inexistante",
            "func_name": f"test{ctx['cap']}VoterDeniesAccessWith{rl}Role",
            "body": [
                f"$this->client->loginUser($this->getTestUser('{fk}'), '{fw}');",
                f"// Voter '{voter_attr}' attendu sur cette route — entité {BOGUS_ENTITY_ID} inexistante.",
                f"$this->client->request('{ctx['http_verb']}', '{bogus}');",
                "self::assertContains(",
                "    $this->client->getResponse()->getStatusCode(),",
                "    [403, 404, 500],",
                f"    \"Le voter '{voter_attr}' devrait refuser ou l'entité ne devrait pas exister\"",
                ");",
            ],
        }]
    # Pas de paramètre dans la route → on teste l'accès brut
    return [{
        "comment":   f"{ctx['raw_route']} — voter '{voter_attr}' — accès courant",
        "func_name": f"test{ctx['cap']}VoterCheckWith{rl}Role",
        "body": [
            f"$this->client->loginUser($this->getTestUser('{fk}'), '{fw}');",
            f"// Voter '{voter_attr}' attendu — adapter selon les fixtures",
            f"$this->client->request('{ctx['http_verb']}', '{ctx['route']}');",
            "self::assertContains(",
            "    $this->client->getResponse()->getStatusCode(),",
            "    [200, 302, 403],",
            f"    \"Voter '{voter_attr}' à valider\"",
            ");",
        ],
    }]


def _scenario_not_found(ctx, fw, _rp, _rc, _sec_roles):
    """
    Pour les routes paramétrées ({id}, {slug}, …) : ID inexistant → 404.
    Cas d'erreur classique que la MOA doit valider à chaque release.
    Skippé pour les voters (déjà traité ailleurs) et les routes sans param.
    """
    if "{" not in ctx["raw_route"]:
        return []
    if ctx["has_voter"]:
        return []
    fk = ctx["factory_key"]
    rl = ctx["role_label"]
    # ID inexistant (cf. BOGUS_ENTITY_ID)
    bogus_route = re.sub(r"\{[^}]+\}", BOGUS_ENTITY_ID, ctx["raw_route"])
    return [{
        "comment":   f"{ctx['raw_route']} — ressource inexistante → 404",
        "func_name": f"test{ctx['cap']}NotFoundWith{rl}Role",
        "body": [
            f"$this->client->loginUser($this->getTestUser('{fk}'), '{fw}');",
            f"$this->client->request('{ctx['http_verb']}', '{bogus_route}');",
            "self::assertContains(",
            "    $this->client->getResponse()->getStatusCode(),",
            "    [404, 302, 500],",
            "    'Une ressource inexistante doit produire 404 (ou redirect/erreur applicative)'",
            ");",
        ],
    }]


def _scenario_secondary_role(ctx, fw, _rp, _rc, sec_roles):
    """
    Vérifie qu'un rôle secondaire a aussi accès. Réplique les assertions
    sémantiques (assertJson, assertSelectorTextContains) du test ADMIN pour
    éviter une asymétrie qui ferait passer un test à tort.
    """
    if not sec_roles:
        return []
    if ctx["is_ajax"] or ctx["has_form"] or ctx["has_voter"]:
        return []
    if ctx["method_role"] != ctx["class_role"]:
        return []

    results = []
    for sr in sec_roles[:MAX_SECONDARY_ROLES_TESTED]:
        sr_fk    = _role_to_factory_key(sr)
        sr_label = sr_fk.replace("ROLE_", "").title()
        body = [
            f"$this->client->loginUser($this->getTestUser('{sr_fk}'), '{fw}');",
            f"$this->client->request('{ctx['http_verb']}', '{ctx['route']}');",
            "self::assertResponseIsSuccessful();",
        ]
        # Propage les assertions sémantiques du test principal :
        if ctx.get("has_json"):
            body.append("$this->assertJson($this->client->getResponse()->getContent());")
        if ctx.get("h1"):
            body.append(f"$this->assertSelectorTextContains('h1', '{ctx['h1']}');")

        results.append({
            "comment":   f"{ctx['raw_route']} — {sr_fk} (rôle secondaire)",
            "func_name": f"test{ctx['cap']}IsReachedWith{sr_label}Role",
            "body":      body,
        })
    return results


# Registre des builders
# L'ordre n'a PAS d'importance fonctionnelle (il détermine juste l'ordre
# des tests dans le fichier PHP). Ajouter un builder ici = nouveau pattern.

SCENARIO_BUILDERS = [
    _scenario_noauth,
    _scenario_auth_form,
    _scenario_auth_simple,
    _scenario_ajax_no_xhr,
    _scenario_role_insufficient,
    _scenario_voter,
    _scenario_not_found,
    _scenario_secondary_role,
]
