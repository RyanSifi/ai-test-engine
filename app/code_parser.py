import os
import re
from typing import List, Dict, Optional
import logging

# ---------------------------------------------------------------------------
# CONSTANTES REGEX
# ---------------------------------------------------------------------------

CLASS_NAME_REGEX = re.compile(
    r"(?:class|interface|trait)\s+([a-zA-Z_\x7f-\xff][a-zA-Z0-9_\x7f-\xff]*)",
    re.IGNORECASE
)
ROUTE_REGEX = re.compile(r"Route\(\s*(?:path\s*:\s*)?['\"]([^'\"]+)['\"]")
PHPDOC_ROUTE_REGEX = re.compile(r"@Route\(\s*['\"]([^'\"]+)['\"]")
PHPDOC_SUMMARY_REGEX = re.compile(
    r"/\*\*"
    r"(?:"
    r"\s*\n\s*\*\s*(.*?)\n"   # Multi-ligne : /** \n * Résumé \n */
    r"|"
    r"\s+(.*?)\s*\*/"         # Mono-ligne  : /** Résumé */
    r")",
    re.DOTALL,
)
TWIG_LINK_REGEX    = re.compile(r"path\(['\"]([^'\"]+)['\"]")
TWIG_H1_REGEX      = re.compile(r"<h[123][^>]*>(.*?)</h[123]>", re.DOTALL | re.IGNORECASE)  # h1, h2, h3
TWIG_BUTTON_REGEX  = re.compile(r"<(?:button|a)[^>]*>(.*?)</(?:button|a)>", re.DOTALL | re.IGNORECASE)
TWIG_INPUT_NAME_REGEX = re.compile(r'<(?:input|select|textarea)\b[^>]*\bname=[\'"]([^\'"]+)[\'"]', re.IGNORECASE)
TWIG_HIDDEN_ID_REGEX  = re.compile(r'<input\b[^>]*\btype=[\'"]hidden[\'"][^>]*\bid=[\'"]([^\'"]+)[\'"]|<input\b[^>]*\bid=[\'"]([^\'"]+)[\'"][^>]*\btype=[\'"]hidden[\'"]', re.IGNORECASE)
TWIG_PAGE_TITLE_REGEX = re.compile(r"\{%[-\s]*set\s+page_title\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
TWIG_NAV_LINK_REGEX   = re.compile(r'class="[^"]*nav-link[^"]*"[^>]*>\s*([\w\s\-/()\'.àâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]+)', re.IGNORECASE)
TWIG_INCLUDE_REGEX    = re.compile(r"include\(['\"]([^'\"]+)['\"]")
TWIG_TABLE_ID_REGEX   = re.compile(r'<table\b[^>]*\bid=[\'"]([^\'"]+)[\'"]', re.IGNORECASE)  # IDs de tables (DataTable)
RENDER_REGEX          = re.compile(r"\->render\(['\"]([^'\"]+)['\"]")

# Nouvelles regex

# Extraction des méthodes HTTP depuis les attributs #[Route] et @Route
# Supporte les deux syntaxes :
#   - PHP 8 nommé   : methods: ['GET', 'POST']
#   - PHP 8 posit.  : methods: ["GET"]
#   - PHPDoc legacy  : methods={"GET","POST"}
ROUTE_METHODS_REGEX = re.compile(
    r"methods\s*[:=]\s*[\[{]\s*(['\"][A-Z]+['\"](?:\s*,\s*['\"][A-Z]+['\"])*)\s*[\]}]",
    re.IGNORECASE,
)

# Extraction de #[IsGranted('ROLE_XXX')] — attribut PHP 8
# Capture le rôle entre quotes et l'éventuel message
ISGRANTED_ATTR_REGEX = re.compile(
    r"#\[IsGranted\(\s*['\"]([^'\"]+)['\"]"
    r"(?:\s*,\s*message\s*:\s*['\"]([^'\"]+)['\"])?"
    r"\s*\)\]",
    re.IGNORECASE,
)

# Extraction de @IsGranted / @Security dans PHPDoc (Symfony < 6)
ISGRANTED_PHPDOC_REGEX = re.compile(
    r"@IsGranted\(\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)

# Détection de denyAccessUnlessGranted() dans le corps de méthode
DENY_ACCESS_REGEX = re.compile(
    r"\$this->denyAccessUnlessGranted\(\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)

# Détection d'appel AJAX-only : $request->isXmlHttpRequest()
AJAX_CHECK_REGEX = re.compile(
    r"\bisXmlHttpRequest\(\)",
    re.IGNORECASE,
)

# Détection de formulaire : createForm + handleRequest + isSubmitted
FORM_CREATE_REGEX    = re.compile(r"\$this->createForm\(\s*([A-Za-z0-9_\\]+)::class", re.IGNORECASE)
FORM_HANDLE_REGEX    = re.compile(r"->handleRequest\(", re.IGNORECASE)
FORM_SUBMITTED_REGEX = re.compile(r"->isSubmitted\(\)", re.IGNORECASE)

# Types de réponse supplémentaires (au-delà de render/redirect/json)
BINARY_RESPONSE_RE   = re.compile(r"BinaryFileResponse|StreamedResponse", re.IGNORECASE)
FILE_DOWNLOAD_RE     = re.compile(r"Content-Disposition.*attachment|file_get_contents|readfile\(", re.IGNORECASE)
EXPORT_RESPONSE_RE   = re.compile(r"ExportResponse|CsvResponse|XlsResponse", re.IGNORECASE)

# Extraction de toutes les routes d'un bloc d'attributs (support multi-route)
ALL_ROUTES_REGEX = re.compile(
    r"#\[Route\(\s*(?:path\s*:\s*)?['\"]([^'\"]+)['\"]"
    r"((?:\s*,\s*[a-zA-Z_]+\s*:\s*(?:['\"][^'\"]*['\"]|[\[{][^\]]*[\]}]|\w+))*)"
    r"\s*\)\]",
    re.DOTALL,
)

# Extraction du name: d'une route
ROUTE_NAME_REGEX = re.compile(
    r"name\s*[:=]\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _extract_balanced_block(content: str, open_pos: int) -> Optional[str]:
    """
    Extrait un bloc délimité par des accolades équilibrées { ... }
    à partir de la position open_pos (qui doit pointer sur '{').
    Retourne le bloc complet (accolades comprises) ou None.

    Gère correctement :
    - les séquences d'échappement PHP (\\' et \\")
    - les commentaires ligne (//, #) et bloc (/* ... */)
    - les chaînes de caractères (' et ")
    pour ne pas compter d'accolades à l'intérieur de ces éléments.
    """
    if open_pos >= len(content) or content[open_pos] != '{':
        return None
    length = len(content)
    depth = 0
    in_single_quote = False
    in_double_quote = False
    i = open_pos
    while i < length:
        ch = content[i]

        # Dans une chaîne : gérer les échappements
        if in_single_quote or in_double_quote:
            if ch == '\\':
                # Séquence d'échappement : sauter le caractère suivant
                i += 2
                continue
            if ch == "'" and in_single_quote:
                in_single_quote = False
            elif ch == '"' and in_double_quote:
                in_double_quote = False
            i += 1
            continue

        # Hors chaîne : détecter commentaires, chaînes, accolades
        next_ch = content[i + 1] if i + 1 < length else ''

        # Commentaire ligne : // ou #
        if (ch == '/' and next_ch == '/') or ch == '#':
            # Sauter jusqu'à la fin de la ligne
            nl = content.find('\n', i)
            i = nl + 1 if nl != -1 else length
            continue

        # Commentaire bloc : /* ... */
        if ch == '/' and next_ch == '*':
            end = content.find('*/', i + 2)
            i = end + 2 if end != -1 else length
            continue

        # Ouverture de chaîne
        if ch == "'":
            in_single_quote = True
            i += 1
            continue
        if ch == '"':
            in_double_quote = True
            i += 1
            continue

        # Accolades
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return content[open_pos:i + 1]

        i += 1
    return None  # Accolade fermante manquante


def _find_class_block(content: str, class_name: str) -> Optional[str]:
    """
    Localise la définition de la classe et retourne son bloc complet
    (accolades équilibrées), en ignorant l'entête de namespace/use.
    """
    # Cherche la ligne "class ClassName" (ou abstract class, interface, trait)
    pattern = re.compile(
        rf"(?:abstract\s+)?(?:class|interface|trait)\s+{re.escape(class_name)}"
        r"(?:\s+(?:extends|implements)\s+[^{]+)?\s*\{",
        re.IGNORECASE
    )
    m = pattern.search(content)
    if not m:
        return None
    # L'accolade ouvrante est le dernier caractère du match
    open_pos = m.end() - 1
    block = _extract_balanced_block(content, open_pos)
    if block is None:
        return None
    # Inclut l'entête de classe
    return content[m.start():m.start() + (m.end() - m.start()) + len(block) - 1]


def _find_method_block(class_content: str, method_name: str) -> Optional[str]:
    """
    Localise une méthode dans le contenu d'une classe et retourne
    son bloc complet (accolades équilibrées).
    """
    pattern = re.compile(
        r"(?:public|protected|private|final|static|abstract)?\s*"
        rf"function\s+{re.escape(method_name)}\s*\(.*?\)\s*(?::\s*\S+\s*)?\{{",
        re.DOTALL
    )
    m = pattern.search(class_content)
    if not m:
        return None
    open_pos = m.end() - 1
    block = _extract_balanced_block(class_content, open_pos)
    if block is None:
        return None
    return class_content[m.start():m.start() + (m.end() - m.start()) + len(block) - 1]


# Nouveaux helpers

def _extract_http_methods(attr_text: str) -> List[str]:
    """
    Extrait les verbes HTTP depuis un bloc d'attributs #[Route] ou @Route.
    Ex : methods: ['GET', 'POST']  →  ['GET', 'POST']
    Ex : methods={"POST"}          →  ['POST']
    Retourne ['GET'] par défaut si rien n'est spécifié.
    """
    m = ROUTE_METHODS_REGEX.search(attr_text)
    if not m:
        return []  # vide = non spécifié (le caller décidera du défaut)
    raw = m.group(1)
    return [v.strip().strip("'\"").upper() for v in raw.split(",")]


def _extract_isgranted_from_attrs(attr_text: str) -> List[Dict]:
    """
    Extrait tous les #[IsGranted] ou @IsGranted d'un bloc d'attributs.
    Retourne une liste de {"role": "ROLE_XXX", "message": "..."|None}.
    """
    results = []
    for m in ISGRANTED_ATTR_REGEX.finditer(attr_text):
        results.append({
            "role":    m.group(1),
            "message": m.group(2) if m.group(2) else None,
        })
    for m in ISGRANTED_PHPDOC_REGEX.finditer(attr_text):
        results.append({
            "role":    m.group(1),
            "message": None,
        })
    return results


def _extract_all_routes(attr_text: str, class_route_prefix: str) -> List[Dict]:
    """
    Extrait TOUTES les routes d'un bloc d'attributs (supporte les multi-Route).
    Retourne une liste de {"path": "/foo", "name": "foo_index", "http_methods": ["GET"]}.
    """
    routes = []
    for m in ALL_ROUTES_REGEX.finditer(attr_text):
        raw_path = m.group(1)
        full_match = m.group(0)

        # Concaténation avec le préfixe de classe
        if class_route_prefix:
            path = (class_route_prefix.rstrip('/') + '/' + raw_path.lstrip('/')).replace('//', '/')
        else:
            path = raw_path

        # Nom de la route
        name_m = ROUTE_NAME_REGEX.search(full_match)
        name = name_m.group(1) if name_m else None

        # Verbes HTTP
        http_methods = _extract_http_methods(full_match)

        routes.append({
            "path":         path,
            "name":         name,
            "http_methods": http_methods,
        })
    return routes


def _analyze_method_body(method_body: str) -> Dict:
    """
    Analyse le corps d'une méthode pour détecter des patterns fonctionnels :
    - type(s) de réponse (render, redirect, json, file_download, export, binary)
    - vérification AJAX (isXmlHttpRequest)
    - gestion de formulaire (createForm, handleRequest, isSubmitted)
    - contrôle d'accès dynamique (denyAccessUnlessGranted + sujet)
    - template rendu

    Retourne un dict de métadonnées.
    """
    result = {
        "rendered_template": None,
        "response_types":    [],
        "is_ajax_only":      False,
        "has_form":          False,
        "form_type":         None,
        "voter_checks":      [],
    }

    # Template rendu
    rm_tpl = RENDER_REGEX.search(method_body)
    if rm_tpl:
        result["rendered_template"] = rm_tpl.group(1)

    # Types de réponse
    REDIRECT_RE      = re.compile(r'\$this->redirect\(|RedirectResponse|\->redirectToRoute\(', re.IGNORECASE)
    JSON_RESPONSE_RE = re.compile(r'JsonResponse|->json\(', re.IGNORECASE)

    if result["rendered_template"]:
        result["response_types"].append("render (200)")
    if REDIRECT_RE.search(method_body):
        result["response_types"].append("redirect (302)")
    if JSON_RESPONSE_RE.search(method_body):
        result["response_types"].append("json (200)")
    if EXPORT_RESPONSE_RE.search(method_body):
        result["response_types"].append("export (200)")
    if FILE_DOWNLOAD_RE.search(method_body):
        result["response_types"].append("file_download (200)")
    if BINARY_RESPONSE_RE.search(method_body):
        result["response_types"].append("binary (200)")

    # Si aucune détection, on ne met PAS de défaut
    # c'est le caller (main.py) qui décidera.

    # AJAX-only
    if AJAX_CHECK_REGEX.search(method_body):
        result["is_ajax_only"] = True

    # Formulaire
    form_m = FORM_CREATE_REGEX.search(method_body)
    if form_m:
        result["has_form"] = True
        form_class = form_m.group(1)
        # Extraire le nom court (sans namespace)
        result["form_type"] = form_class.split("\\")[-1]
    elif FORM_HANDLE_REGEX.search(method_body) or FORM_SUBMITTED_REGEX.search(method_body):
        result["has_form"] = True

    for deny_m in DENY_ACCESS_REGEX.finditer(method_body):
        attribute = deny_m.group(1)
        # Essayer de trouver le sujet (2e argument)
        # Pattern : denyAccessUnlessGranted('view', $creance, ...)
        after = method_body[deny_m.end():]
        subject_m = re.match(r"\s*,\s*\$([a-zA-Z_]+)", after)
        result["voter_checks"].append({
            "attribute": attribute,
            "subject":   subject_m.group(1) if subject_m else None,
        })

    return result


def _classify_controller(
    class_name: str,
    methods: List[Dict],
    class_grants: List[Dict],
) -> str:
    """
    Détermine le « profil » d'un contrôleur pour guider la stratégie
    de génération de tests en aval.

    Profils possibles :
    - "web_crud"     : routes HTTP classiques avec render/redirect (ex: CreanceController)
    - "api"          : retourne majoritairement du JSON (JsonResponse, ->json())
    - "internal"     : pas de route HTTP, appelé en interne (ex: EtatImportController)
    - "mixed"        : mélange de web et d'API (routes render + routes json)
    - "batch"        : contrôleur de traitement batch/command sans route
    """
    routed   = [m for m in methods if m.get("routes")]
    unrouted = [m for m in methods if not m.get("routes") and m["name"] != "__construct"]

    if not routed:
        return "internal"

    render_count = 0
    json_count   = 0
    for m in routed:
        rtypes = m.get("response_types", [])
        if any("render" in r for r in rtypes):
            render_count += 1
        if any("json" in r for r in rtypes):
            json_count += 1

    total_routed = len(routed)
    if total_routed == 0:
        return "internal"

    json_ratio   = json_count / total_routed
    render_ratio = render_count / total_routed

    if json_ratio > 0.7:
        return "api"
    if render_ratio > 0.7:
        return "web_crud"
    if json_count > 0 and render_count > 0:
        return "mixed"

    return "web_crud"


# ---------------------------------------------------------------------------
# PARSING D'UN FICHIER PHP
# ---------------------------------------------------------------------------

_PARAM_MODIFIERS = {"private", "public", "protected", "readonly", "abstract", "final", "static"}

# Regex hissées en module pour éviter la recompilation à chaque méthode
_PHPDOC_BACK_RE     = re.compile(r"/\*\*.*?\*/", re.DOTALL)
_ATTR_BACK_RE       = re.compile(r"(?:#\[(?:[^\[\]]|\[[^\]]*\])*\]\s*)+", re.DOTALL)
_CLOSE_BRACE_RE     = re.compile(r'(?:\n[ \t]*\}|\}[ \t]*\n)')
_OPEN_CLASS_BRACE_RE = re.compile(r'\n[ \t]*\{')
_LOOKBACK_CHARS     = 600


def _extract_params(params_str: str) -> List[Dict]:
    """
    Transforme la chaîne des paramètres d'une méthode en liste structurée.
    Gère les valeurs par défaut (= 1, = null, = []), les modifiers de promotion
    de propriétés (private readonly), les types nullables (?Foo) et namespacés.
    """
    params = []
    for raw in params_str.split(','):
        raw = raw.strip()
        if not raw:
            continue
        # Retire la valeur par défaut éventuelle (= ...)
        if '=' in raw:
            raw = raw.split('=', 1)[0].strip()
        parts = raw.split()
        if not parts:
            continue
        # Le nom du paramètre est le dernier token qui commence par '$'
        name_idx = next(
            (i for i in range(len(parts) - 1, -1, -1) if parts[i].startswith('$')),
            -1,
        )
        if name_idx == -1:
            continue
        param_name = parts[name_idx].lstrip('$')
        # Le type, s'il y en a un, est le dernier token avant le nom (hors modifiers)
        type_tokens = [p for p in parts[:name_idx] if p not in _PARAM_MODIFIERS]
        param_type = type_tokens[-1].lstrip('?\\') if type_tokens else None
        params.append({"type": param_type, "name": param_name})
    return params


def _parse_file_content(content: str) -> Dict:
    """
    Analyse le contenu d'un fichier PHP et retourne :
    - class_name           : nom de la classe
    - methods              : liste des méthodes avec routes, descriptions, params,
                             response_types, is_ajax_only, has_form, voter_checks, etc.
    - properties           : propriétés de la classe
    - constructor_params   : paramètres du constructeur (injection de dépendances)
    - class_grants         : #[IsGranted] au niveau de la classe
    - class_route_prefix   : préfixe de route de la classe
    - controller_profile   : profil du contrôleur (web_crud, api, internal, mixed)

    Stratégie en deux passes pour éviter les problèmes de regex greediness :
    Passe 1 — localiser les positions des mots-clés "function".
    Passe 2 — remonter dans le texte (fenêtre limitée) pour trouver PHPDoc/attributs.
    """
    class_name_match = CLASS_NAME_REGEX.search(content)
    class_name = class_name_match.group(1) if class_name_match else None
    if not class_name:
        return {}

    # --- Route de classe (préfixe) ---
    class_route_prefix = ""
    class_def_start = content.find(f"class {class_name}")
    if class_def_start != -1:
        class_header = content[:class_def_start]
        m = ROUTE_REGEX.search(class_header) or PHPDOC_ROUTE_REGEX.search(class_header)
        if m:
            class_route_prefix = m.group(1)

    # #[IsGranted] au niveau classe
    class_grants: List[Dict] = []
    if class_def_start != -1:
        class_header = content[:class_def_start]
        class_grants = _extract_isgranted_from_attrs(class_header)

    # On trouve les méthodes (function + paramètres).
    # Accepte n'importe quel ordre de modificateurs (final/abstract/static/visibilité)
    # et `function` seul (visibilité publique implicite). Ne capture QUE le nom et les params.
    FUNC_RE = re.compile(
        r"(?:(?:public|protected|private|final|abstract|static)\s+)*"
        r"function\s+"
        r"([a-zA-Z_\x7f-\xff][a-zA-Z0-9_\x7f-\xff]*)"  # nom
        r"\s*\((.*?)\)"                                   # params (non-greedy, DOTALL)
        r"(?:\s*:\s*(\??[a-zA-Z0-9_|\\\[\]&]+))?",       # retour optionnel (nullable / union / intersection)
        re.DOTALL
    )

    methods = []
    constructor_params: List[Dict] = []

    for func_m in FUNC_RE.finditer(content):
        method_name = func_m.group(1)
        params_str  = func_m.group(2) or ""
        return_type = func_m.group(3)
        func_start  = func_m.start()

        # Fenêtre d'analyse arrière :
        # On cherche la dernière "accolade fermante de corps de méthode" avant func_start.
        # On détecte deux formes :
        #   - Accolade en début de ligne (méthodes multi-lignes) :  \n[ \t]*}
        #   - Accolade en fin de ligne   (méthodes inline)        :  }[ \t]*\n
        # Cela évite de confondre avec des { } dans des strings (ex. "/{id}").
        window_start = max(0, func_start - _LOOKBACK_CHARS)
        raw_window   = content[window_start:func_start]

        close_brace_matches = list(_CLOSE_BRACE_RE.finditer(raw_window))
        if close_brace_matches:
            # Couper juste après la dernière accolade fermante détectée
            cut = close_brace_matches[-1].end()
            window = raw_window[cut:]
        else:
            # Pas de méthode précédente : délimiter au début du corps de classe
            open_matches = list(_OPEN_CLASS_BRACE_RE.finditer(raw_window))
            if open_matches:
                window = raw_window[open_matches[-1].end():]
            else:
                window = raw_window

        # PHPDoc : on prend le DERNIER bloc /** ... */ dans la fenêtre
        phpdoc = ""
        phpdoc_matches = list(_PHPDOC_BACK_RE.finditer(window))
        if phpdoc_matches:
            phpdoc = phpdoc_matches[-1].group(0)

        # Attributs PHP 8 : on prend le DERNIER bloc #[...] dans la fenêtre
        route_attr = ""
        attr_matches = list(_ATTR_BACK_RE.finditer(window))
        if attr_matches:
            route_attr = attr_matches[-1].group(0)

        # Extraction des paramètres
        params = _extract_params(params_str)
        if method_name == "__construct":
            constructor_params = params

        # Extraction MULTI-ROUTES
        # Au lieu d'une seule route, on extrait toutes les #[Route] du bloc
        routes = _extract_all_routes(route_attr, class_route_prefix)

        # Fallback PHPDoc @Route
        if not routes:
            rm = ROUTE_REGEX.search(phpdoc) or PHPDOC_ROUTE_REGEX.search(phpdoc)
            if rm:
                raw_path = rm.group(1)
                if class_route_prefix:
                    path = (class_route_prefix.rstrip('/') + '/' + raw_path.lstrip('/')).replace('//', '/')
                else:
                    path = raw_path
                routes = [{"path": path, "name": None, "http_methods": _extract_http_methods(phpdoc)}]

        # Fallback convention : si pas de route mais préfixe de classe et méthode spéciale
        if not routes and class_route_prefix and method_name != "__construct":
            if method_name.endswith('Action') or method_name == 'index':
                routes = [{"path": class_route_prefix, "name": None, "http_methods": []}]

        # Champ legacy « route » : première route pour rétro-compatibilité
        route = routes[0]["path"] if routes else None

        # #[IsGranted] au niveau méthode
        method_grants = _extract_isgranted_from_attrs(route_attr)
        # Chercher aussi dans le PHPDoc
        method_grants += _extract_isgranted_from_attrs(phpdoc)

        # Résumé PHPDoc
        summary = None
        sm = PHPDOC_SUMMARY_REGEX.search(phpdoc)
        if sm:
            raw = sm.group(1) or sm.group(2) or ""
            summary = raw.strip("* ").strip() or None

        # Analyse enrichie du corps de méthode
        body_search_start = func_m.end()
        open_brace_pos    = content.find('{', body_search_start)
        if open_brace_pos != -1:
            method_body = _extract_balanced_block(content, open_brace_pos) or ""
        else:
            method_body = ""

        body_analysis = _analyze_method_body(method_body)

        # Champs legacy pour rétro-compatibilité
        rendered_template = body_analysis["rendered_template"]
        response_types    = body_analysis["response_types"]

        # Legacy response_type (premier type pour compat)
        response_type = response_types[0] if response_types else None

        methods.append({
            "name":          method_name,
            # ── Legacy (rétro-compat) ──
            "route":         route,
            "renders":       rendered_template,
            "response_type": response_type,
            # ── AXE 1 : champs enrichis ──
            "routes":          routes,         # TOUTES les routes de la méthode
            "response_types":  response_types, # TOUS les types de réponse détectés
            "method_grants":   method_grants,  # #[IsGranted] sur la méthode
            "is_ajax_only":    body_analysis["is_ajax_only"],
            "has_form":        body_analysis["has_form"],
            "form_type":       body_analysis["form_type"],
            "voter_checks":    body_analysis["voter_checks"],
            # ── Inchangés ──
            "description":   summary,
            "params":        params,
            "return_type":   return_type,
        })

    # Extraction des propriétés
    properties = []
    PROP_RE = re.compile(
        r"(?:private|protected|public)\s+(?:\?|)"
        r"(?P<type>[a-zA-Z0-9\\]+)\s+\$(?P<n>[a-zA-Z_\x7f-\xff][a-zA-Z0-9_\x7f-\xff]*)"
    )
    for m in PROP_RE.finditer(content):
        properties.append({"type": m.group("type"), "name": m.group("n")})

    # Classification du contrôleur ─────────────────────────────
    controller_profile = _classify_controller(class_name, methods, class_grants)

    return {
        "class_name":        class_name,
        "methods":           methods,
        "properties":        properties,
        "constructor_params": constructor_params,
        # Nouvelles métadonnées de classe
        "class_grants":       class_grants,
        "class_route_prefix": class_route_prefix,
        "controller_profile": controller_profile,
    }


# ---------------------------------------------------------------------------
# ANALYSE D'UN PROJET SYMFONY
# ---------------------------------------------------------------------------

def analyze_project_code(project_base_path: str) -> Dict:
    """
    Scanne les répertoires clés d'un projet Symfony et retourne un dictionnaire
    structuré avec contrôleurs, entités, services, repositories, etc.
    """
    folders_to_scan = [
        "Controller", "Entity", "Service", "Repository",
        "Command", "Form", "Security",
    ]

    analysis: Dict[str, List] = {f.lower() + "s": [] for f in folders_to_scan}
    analysis["templates"] = []

    for folder in folders_to_scan:
        path = os.path.join(project_base_path, "src", folder)
        key  = folder.lower() + "s"
        logging.info(f"Analyse du dossier : {path}")

        if not os.path.isdir(path):
            logging.debug(f"Dossier inexistant (ignoré) : {path}")
            continue

        for root, _, files in os.walk(path):
            for filename in files:
                if not filename.endswith(".php"):
                    continue
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    parsed = _parse_file_content(content)
                    if parsed and parsed.get("class_name"):
                        analysis[key].append({
                            "file":              os.path.relpath(file_path, os.path.join(project_base_path, "src")),
                            "class":             parsed["class_name"],
                            "methods":           parsed["methods"],
                            "properties":        parsed["properties"],
                            "constructor_params": parsed["constructor_params"],
                            # Nouvelles métadonnées
                            "class_grants":       parsed.get("class_grants", []),
                            "class_route_prefix": parsed.get("class_route_prefix", ""),
                            "controller_profile": parsed.get("controller_profile", "web_crud"),
                        })
                except Exception as e:
                    logging.warning(f"Impossible d'analyser {file_path}: {e}")

    # Templates Twig
    templates_path = os.path.join(project_base_path, "templates")
    if os.path.isdir(templates_path):
        for root, _, files in os.walk(templates_path):
            for filename in files:
                if not filename.endswith(".twig"):
                    continue
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Titre de page ({% set page_title = '...' %})
                    page_titles = TWIG_PAGE_TITLE_REGEX.findall(content)

                    # H1 et H2
                    h_list = [
                        re.sub(r'\{%.*?%\}|\{\{.*?\}\}|<.*?>', '', m.group(1)).strip()
                        for m in TWIG_H1_REGEX.finditer(content)
                    ]
                    h_list = [h for h in h_list if h]

                    # Onglets de navigation (nav-link)
                    nav_tabs = [t.strip() for t in TWIG_NAV_LINK_REGEX.findall(content)]
                    nav_tabs = [t for t in nav_tabs if len(t) > 2]
                    nav_tabs_set = set(nav_tabs)

                    # Routes Twig path('nom_route')
                    links = list(set(TWIG_LINK_REGEX.findall(content)))

                    # Boutons et liens (on exclut les textes déjà capturés comme onglets)
                    buttons = [
                        re.sub(r'<.*?>', '', m.group(1)).strip()
                        for m in TWIG_BUTTON_REGEX.finditer(content)
                    ]
                    buttons = [
                        b for b in buttons
                        if b and not b.startswith('{') and len(b) > 1 and b not in nav_tabs_set
                    ]

                    # Champs de formulaire (name= sur <input>, <select>, <textarea> uniquement)
                    inputs = [m.group(1) for m in TWIG_INPUT_NAME_REGEX.finditer(content) if m.group(1)]

                    # IDs des inputs cachés (marqueurs DOM stables pour assertions)
                    hidden_ids = [
                        m.group(1) or m.group(2)
                        for m in TWIG_HIDDEN_ID_REGEX.finditer(content)
                        if m.group(1) or m.group(2)
                    ]

                    # IDs des tables (DataTable, tableaux de données)
                    table_ids = TWIG_TABLE_ID_REGEX.findall(content)

                    # Sous-templates inclus
                    includes = TWIG_INCLUDE_REGEX.findall(content)

                    # Construction du chunk texte pour l'embedding
                    parts = [f"Le template {os.path.relpath(file_path, templates_path)}."]
                    if page_titles:
                        parts.append(f"Titre de page : {', '.join(page_titles)}.")
                    if h_list:
                        parts.append(f"Sections : {', '.join(h_list)}.")
                    if nav_tabs:
                        parts.append(f"Onglets : {', '.join(nav_tabs[:10])}.")
                    if buttons:
                        parts.append(f"Boutons : {', '.join(list(dict.fromkeys(buttons))[:10])}.")
                    if hidden_ids:
                        parts.append(f"IDs cachés : {', '.join(hidden_ids[:10])}.")
                    if table_ids:
                        parts.append(f"IDs tables : {', '.join(table_ids[:5])}.")
                    if links:
                        parts.append(f"Routes Twig : {', '.join(links[:10])}.")
                    if inputs:
                        parts.append(f"Champs formulaire : {', '.join(inputs[:10])}.")
                    if includes:
                        parts.append(f"Inclut : {', '.join(includes[:5])}.")

                    analysis["templates"].append({
                        "file":       os.path.relpath(file_path, templates_path),
                        "links":      links,
                        "h1":         page_titles + h_list,
                        "buttons":    list(dict.fromkeys(buttons)),
                        "inputs":     inputs,
                        "hidden_ids": hidden_ids,
                        "table_ids":  table_ids,
                        "content":    " ".join(parts),
                    })
                except Exception as e:
                    logging.warning(f"Impossible d'analyser le template {file_path}: {e}")

    return analysis


# ---------------------------------------------------------------------------
# EXTRACTION DU CODE SOURCE D'UN SYMBOLE
# ---------------------------------------------------------------------------

def extract_code_for_symbol(
    file_path: str,
    class_name: str,
    method_name: Optional[str] = None,
) -> Optional[List[str]]:
    """
    Extrait le code source complet d'une classe ou d'une méthode spécifique
    à partir d'un fichier PHP, en utilisant un compteur d'accolades équilibrées
    (plus robuste qu'une regex naïve).
    Retourne une liste de lignes ou None si l'élément est introuvable.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        logging.error(f"Fichier non trouvé : {file_path}")
        return None
    except Exception as e:
        logging.error(f"Erreur lecture {file_path}: {e}")
        return None

    class_block = _find_class_block(content, class_name)
    if class_block is None:
        logging.warning(f"Classe '{class_name}' non trouvée dans {file_path}")
        return None

    if method_name:
        method_block = _find_method_block(class_block, method_name)
        if method_block is None:
            logging.warning(f"Méthode '{method_name}' introuvable dans '{class_name}' ({file_path})")
            return None
        logging.info(f"Méthode '{method_name}' extraite de '{class_name}'.")
        return method_block.splitlines()

    logging.info(f"Classe '{class_name}' extraite de {file_path}.")
    return class_block.splitlines()
