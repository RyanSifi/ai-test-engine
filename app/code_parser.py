import json
import os
import re
import subprocess
from typing import List, Dict, Optional
import logging

from chunk_format import (
    MAX_TWIG_ITEMS, MAX_TWIG_SECONDARY,
)


def _read_php_file(file_path: str) -> str:
    """
    Lit un fichier PHP/Twig en tolérant les encodages non-UTF8 fréquents
    (Latin-1 dans les vieux projets Symfony français). Évite que des
    contrôleurs entiers soient silencieusement ignorés sur UnicodeDecodeError.
    """
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    # Dernier recours : on remplace les bytes invalides
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

# ---------------------------------------------------------------------------
# CONSTANTES REGEX
# ---------------------------------------------------------------------------

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

# Extraction de @IsGranted / @Security dans PHPDoc (Symfony < 6, annotation en
# commentaire — pas de la vraie syntaxe PHP, l'AST ne peut pas aider ici).
ISGRANTED_PHPDOC_REGEX = re.compile(
    r"@IsGranted\(\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)

# Extraction du methods={...} d'une annotation PHPDoc @Route (Symfony < 6).
# Idem : annotation en commentaire, pas de la vraie syntaxe PHP.
PHPDOC_ROUTE_METHODS_REGEX = re.compile(
    r"methods\s*[:=]\s*[\[{]\s*(['\"][A-Z]+['\"](?:\s*,\s*['\"][A-Z]+['\"])*)\s*[\]}]",
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


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _extract_balanced(content: str, open_pos: int, opening: str, closing: str) -> Optional[str]:
    """
    Extrait un bloc délimité par un jeu de délimiteurs équilibrés (ex: "{" / "}",
    ou "([{" / ")]}" pour une liste d'arguments PHP pouvant mélanger parenthèses,
    crochets et accolades). La profondeur est comptée globalement (peu importe le
    type exact de délimiteur ouvrant/fermant) : du PHP valide ne mélange jamais
    ces délimiteurs de façon incohérente, donc ça suffit à gérer un nombre
    arbitraire de niveaux d'imbrication — contrairement à une regex à profondeur
    fixe (type [^\\]]*), qui casse silencieusement dès 2 niveaux.

    open_pos doit pointer sur un caractère de `opening`.
    Retourne le bloc complet (délimiteurs compris) ou None.

    Gère correctement :
    - les séquences d'échappement PHP (\\' et \\")
    - les commentaires ligne (//, #) et bloc (/* ... */)
    - les chaînes de caractères (' et ")
    pour ne pas compter de délimiteurs à l'intérieur de ces éléments.
    """
    if open_pos >= len(content) or content[open_pos] not in opening:
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

        # Hors chaîne : détecter commentaires, chaînes, délimiteurs
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

        # Délimiteurs
        if ch in opening:
            depth += 1
        elif ch in closing:
            depth -= 1
            if depth == 0:
                return content[open_pos:i + 1]

        i += 1
    return None  # Délimiteur fermant manquant


def _extract_balanced_block(content: str, open_pos: int) -> Optional[str]:
    """
    Extrait un bloc délimité par des accolades équilibrées { ... }
    à partir de la position open_pos (qui doit pointer sur '{').
    Cas particulier de _extract_balanced() pour les corps de code PHP.
    """
    return _extract_balanced(content, open_pos, "{", "}")


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


def _analyze_method_body(method_body: str) -> Dict:
    """
    Analyse le corps d'une méthode pour détecter des patterns fonctionnels :
    - type(s) de réponse (render, redirect, json, file_download, export, binary)
    - vérification AJAX (isXmlHttpRequest)
    - gestion de formulaire (createForm, handleRequest, isSubmitted)
    - contrôle d'accès dynamique (denyAccessUnlessGranted + sujet)
    - template rendu
    - verbe HTTP inféré du body (lectures de POST data, isMethod, etc.)

    Retourne un dict de métadonnées.
    """
    result = {
        "rendered_template":  None,
        "response_types":     [],
        "is_ajax_only":       False,
        "has_form":           False,
        "form_type":          None,
        "voter_checks":       [],
        "body_inferred_verb": None,
        "body_reads": [],
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

    # Inférence du verbe HTTP par analyse du body
    # Indices fortes (par priorité décroissante)
    if re.search(r"\$request->isMethod\(\s*['\"]POST['\"]\s*\)", method_body, re.I):
        result["body_inferred_verb"] = "POST"
    elif re.search(r"\$request->isMethod\(\s*['\"]DELETE['\"]\s*\)", method_body, re.I):
        result["body_inferred_verb"] = "DELETE"
    elif re.search(r"\$request->isMethod\(\s*['\"]PUT['\"]\s*\)", method_body, re.I):
        result["body_inferred_verb"] = "PUT"
    elif re.search(r"\$request->request->(?:get|all|has)\(", method_body):
        # Lit explicitement le body POST
        result["body_inferred_verb"] = "POST"
        result["body_reads"].append("POST data")
    elif result["has_form"]:
        # createForm + handleRequest → typiquement GET (affichage) + POST (soumission)
        result["body_inferred_verb"] = "POST"

    if re.search(r"\$request->query->(?:get|all|has)\(", method_body):
        result["body_reads"].append("query string")
    if re.search(r"\$request->files->(?:get|all|has)\(", method_body):
        result["body_reads"].append("uploaded files")
        # Upload de fichiers → quasi-toujours POST
        if not result["body_inferred_verb"]:
            result["body_inferred_verb"] = "POST"

    return result


# Proportion de routes d'un même type au-delà de laquelle le contrôleur est dit
# « dominé » par ce type. À 0,7, un contrôleur dont 7 routes sur 10 renvoient du
# JSON est classé "api" ; en dessous, il devient "mixed" et reçoit les règles de
# génération des deux familles.
#
# ⚠️ Ce seuil pilote la stratégie de test appliquée en aval (règles de prompt,
# assertions attendues) — le baisser rend les profils plus tranchés mais fait
# passer des contrôleurs hybrides pour des API pures, et inversement.
PROFILE_DOMINANCE_RATIO = 0.7


def _classify_controller(
    methods: List[Dict],
    class_grants: List[Dict],
) -> str:
    """
    Détermine le « profil » d'un contrôleur pour guider la stratégie
    de génération de tests en aval.

    Profils possibles :
    - "web_crud"     : routes HTTP classiques avec render/redirect (ex: CreanceController)
    - "api"          : retourne majoritairement du JSON (JsonResponse, ->json())
    - "internal"     : pas de route HTTP, appelé en interne (ex: EtatImportController,
                       ou contrôleur batch/command — non distingué d'"internal" pour
                       l'instant, faute de signal fiable pour les séparer)
    - "mixed"        : mélange de web et d'API (routes render + routes json)

    NOTE : un profil "batch" distinct était documenté mais n'est jamais produit ici —
    retiré pour ne pas laisser un consommateur (ex: _RE_PROFILE côté main.py) chercher
    une valeur qui n'existe pas. Si un signal de détection batch/command apparaît un jour
    (ex: présence dans Command/, ou suffixe de nom de classe), l'ajouter ici explicitement.
    """
    routed = [m for m in methods if m.get("routes")]
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
    json_ratio   = json_count / total_routed
    render_ratio = render_count / total_routed

    if json_ratio > PROFILE_DOMINANCE_RATIO:
        return "api"
    if render_ratio > PROFILE_DOMINANCE_RATIO:
        return "web_crud"
    if json_count > 0 and render_count > 0:
        return "mixed"

    return "web_crud"


# ---------------------------------------------------------------------------
# PARSING D'UN FICHIER PHP — via AST (nikic/php-parser, pont app/php_ast/)
# ---------------------------------------------------------------------------
#
# Historique : ce module parsait auparavant les routes, attributs #[Route]/
# #[IsGranted] et paramètres par regex + comptage d'accolades maison. Un bug
# concret (01/07/2026) a montré la limite de cette approche : une regex à
# profondeur fixe échouait silencieusement sur un #[Route(...)] contenant un
# tableau imbriqué à 2 niveaux dans `requirements`. Plutôt que de durcir
# indéfiniment les regex au fur et à mesure des cas piège découverts, la
# structure (classes, méthodes, attributs, params) est maintenant extraite
# par un vrai parseur PHP (nikic/php-parser) via un petit script pont
# (php_ast/parse.php), appelé en sous-processus. Le contenu du corps des
# méthodes reste ensuite analysé par regex simples (_analyze_method_body) :
# ce sont des recherches de sous-chaînes/patterns, pas du comptage de
# profondeur, donc pas la même classe de risque.

_PHP_AST_SCRIPT = os.path.join(os.path.dirname(__file__), "php_ast", "parse.php")


def _parse_via_ast(content: str) -> Optional[Dict]:
    """
    Appelle le pont PHP (php_ast/parse.php) pour parser `content` via un vrai
    AST. Retourne le dict JSON qu'il produit (class_name, class_attributes,
    class_docblock, methods), ou None si le parsing échoue (PHP absent,
    syntaxe PHP invalide, sortie JSON invalide) — dans ce cas l'appelant
    traite le fichier comme non parsable, même comportement qu'avant quand
    aucune classe n'était trouvée.
    """
    try:
        proc = subprocess.Popen(
            ["php", _PHP_AST_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        stdout, stderr = proc.communicate(input=content)
    except FileNotFoundError:
        logging.warning("PHP non disponible — parsing AST impossible.")
        return None
    if proc.returncode != 0:
        logging.warning(f"Erreur de parsing AST : {stderr.strip()}")
        return None
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as e:
        logging.warning(f"Sortie JSON invalide du pont AST : {e}")
        return None


def _route_from_attr(attr: Dict, class_route_prefix: str) -> Optional[Dict]:
    """
    Construit {"path", "name", "http_methods"} depuis un attribut #[Route(...)]
    déjà résolu en dict par le pont AST (voir php_ast/parse.php). Les
    arguments imbriqués (ex: requirements avec sous-tableau) sont déjà
    résolus sans ambiguïté par l'AST — pas de re-parsing texte ici.
    """
    args = attr.get("args", [])
    positional = [a["value"] for a in args if a.get("name") is None]
    named = {a["name"]: a["value"] for a in args if a.get("name")}

    raw_path = named.get("path")
    if raw_path is None and positional:
        raw_path = positional[0]
    if not isinstance(raw_path, str):
        return None  # attribut malformé ou path non littéral — ignoré (best effort)

    if class_route_prefix:
        path = (class_route_prefix.rstrip('/') + '/' + raw_path.lstrip('/')).replace('//', '/')
    else:
        path = raw_path

    methods_val = named.get("methods")
    http_methods = [str(m).upper() for m in methods_val] if isinstance(methods_val, list) else []

    name_val = named.get("name")
    return {
        "path":         path,
        "name":         name_val if isinstance(name_val, str) else None,
        "http_methods": http_methods,
    }


def _isgranted_from_attr(attr: Dict) -> Dict:
    """
    Construit {"role", "message"} depuis un attribut #[IsGranted(...)]
    déjà résolu en dict par le pont AST.
    """
    args = attr.get("args", [])
    positional = [a["value"] for a in args if a.get("name") is None]
    named = {a["name"]: a["value"] for a in args if a.get("name")}
    role = positional[0] if positional else named.get("attribute")
    message = named.get("message")
    return {
        "role":    role if isinstance(role, str) else None,
        "message": message if isinstance(message, str) else None,
    }


def _isgranted_from_phpdoc(docblock: Optional[str]) -> List[Dict]:
    """Fallback @IsGranted en PHPDoc (Symfony < 6) — pas de vraie syntaxe PHP, reste en regex."""
    if not docblock:
        return []
    return [{"role": m.group(1), "message": None} for m in ISGRANTED_PHPDOC_REGEX.finditer(docblock)]


def _http_methods_from_phpdoc(docblock: str) -> List[str]:
    """Fallback methods={...} d'une annotation PHPDoc @Route (Symfony < 6)."""
    m = PHPDOC_ROUTE_METHODS_REGEX.search(docblock)
    if not m:
        return []
    return [v.strip().strip("'\"").upper() for v in m.group(1).split(",")]


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

    La structure (classes/méthodes/attributs/params) vient du pont AST
    (_parse_via_ast). Le contenu du corps de chaque méthode continue d'être
    analysé par _analyze_method_body (regex simples, pas de profondeur).
    """
    ast_result = _parse_via_ast(content)
    if not ast_result or not ast_result.get("class_name"):
        return {}

    class_name     = ast_result["class_name"]
    class_attrs    = ast_result.get("class_attributes", [])
    class_docblock = ast_result.get("class_docblock")

    # --- Préfixe de route de classe ---
    class_route_prefix = ""
    for attr in class_attrs:
        if attr.get("name") == "Route":
            route = _route_from_attr(attr, "")
            if route:
                class_route_prefix = route["path"]
                break
    if not class_route_prefix and class_docblock:
        m = PHPDOC_ROUTE_REGEX.search(class_docblock)
        if m:
            class_route_prefix = m.group(1)

    # --- #[IsGranted] au niveau classe ---
    class_grants = [_isgranted_from_attr(a) for a in class_attrs if a.get("name") == "IsGranted"]
    class_grants += _isgranted_from_phpdoc(class_docblock)

    methods: List[Dict] = []
    constructor_params: List[Dict] = []

    for m_data in ast_result.get("methods", []):
        method_name = m_data.get("name", "")
        attrs       = m_data.get("attributes", [])
        docblock    = m_data.get("docblock")

        # Paramètres (déjà typés/nommés par l'AST — nettoyage du type cohérent
        # avec l'ancien comportement : on retire seulement les '?'/'\' en tête)
        params = []
        for p in m_data.get("params", []):
            if not p.get("name"):
                continue
            ptype = p.get("type")
            if ptype:
                ptype = ptype.lstrip("?\\")
            params.append({"type": ptype, "name": p["name"]})
        if method_name == "__construct":
            constructor_params = params

        # Routes depuis les attributs #[Route(...)] (gère nativement les
        # arguments imbriqués à N niveaux, contrairement à l'ancienne regex)
        routes = []
        for attr in attrs:
            if attr.get("name") == "Route":
                route = _route_from_attr(attr, class_route_prefix)
                if route:
                    routes.append(route)

        # Fallback PHPDoc @Route (Symfony < 6)
        if not routes and docblock:
            rm = ROUTE_REGEX.search(docblock) or PHPDOC_ROUTE_REGEX.search(docblock)
            if rm:
                raw_path = rm.group(1)
                if class_route_prefix:
                    path = (class_route_prefix.rstrip('/') + '/' + raw_path.lstrip('/')).replace('//', '/')
                else:
                    path = raw_path
                routes = [{"path": path, "name": None, "http_methods": _http_methods_from_phpdoc(docblock)}]

        # Fallback convention : si pas de route mais préfixe de classe et méthode spéciale
        if not routes and class_route_prefix and method_name != "__construct":
            if method_name.endswith('Action') or method_name == 'index':
                routes = [{"path": class_route_prefix, "name": None, "http_methods": []}]

        # Champ legacy « route » : première route pour rétro-compatibilité
        route = routes[0]["path"] if routes else None

        # #[IsGranted] au niveau méthode + fallback PHPDoc
        method_grants = [_isgranted_from_attr(a) for a in attrs if a.get("name") == "IsGranted"]
        method_grants += _isgranted_from_phpdoc(docblock)

        # Résumé PHPDoc
        summary = None
        if docblock:
            sm = PHPDOC_SUMMARY_REGEX.search(docblock)
            if sm:
                raw = sm.group(1) or sm.group(2) or ""
                summary = raw.strip("* ").strip() or None

        # Analyse du corps de méthode (texte exact extrait par l'AST)
        body_text = m_data.get("body") or ""
        body_analysis = _analyze_method_body(body_text)

        rendered_template = body_analysis["rendered_template"]
        response_types    = body_analysis["response_types"]
        response_type     = response_types[0] if response_types else None

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
            "return_type":   m_data.get("return_type"),
        })

    # Extraction des propriétés (inchangé, regex sur le contenu brut — hors
    # périmètre de ce remplacement, pas de risque de profondeur imbriquée ici)
    properties = []
    PROP_RE = re.compile(
        r"(?:private|protected|public)\s+(?:\?|)"
        r"(?P<type>[a-zA-Z0-9\\]+)\s+\$(?P<n>[a-zA-Z_\x7f-\xff][a-zA-Z0-9_\x7f-\xff]*)"
    )
    for m in PROP_RE.finditer(content):
        properties.append({"type": m.group("type"), "name": m.group("n")})

    # Classification du contrôleur ─────────────────────────────
    controller_profile = _classify_controller(methods, class_grants)

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
    # Conventions Symfony. « Service » et « Services » coexistent selon les
    # projets : on scanne les deux plutôt que d'en imposer une.
    folders_to_scan = [
        "Controller", "Entity", "Service", "Services", "Repository",
        "Command", "Form", "Security",
    ]

    # Les deux orthographes alimentent la MÊME clé. Sans cela « Services »
    # produirait « servicess », qui remonte jusque dans le type annoncé au
    # modèle (« La classe PHP X (type: servicess) »).
    cles = {"Services": "services"}
    cle = lambda f: cles.get(f, f.lower() + "s")

    analysis: Dict[str, List] = {cle(f): [] for f in folders_to_scan}
    analysis["templates"] = []

    absents: List[str] = []

    for folder in folders_to_scan:
        path = os.path.join(project_base_path, "src", folder)
        key  = cle(folder)
        logging.info(f"Analyse du dossier : {path}")

        if not os.path.isdir(path):
            absents.append(folder)
            continue

        for root, _, files in os.walk(path):
            for filename in files:
                if not filename.endswith(".php"):
                    continue
                file_path = os.path.join(root, filename)
                try:
                    content = _read_php_file(file_path)
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
                    content = _read_php_file(file_path)

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
                        parts.append(f"Onglets : {', '.join(nav_tabs[:MAX_TWIG_ITEMS])}.")
                    if buttons:
                        parts.append(f"Boutons : {', '.join(list(dict.fromkeys(buttons))[:MAX_TWIG_ITEMS])}.")
                    if hidden_ids:
                        parts.append(f"IDs cachés : {', '.join(hidden_ids[:MAX_TWIG_ITEMS])}.")
                    if table_ids:
                        parts.append(f"IDs tables : {', '.join(table_ids[:MAX_TWIG_SECONDARY])}.")
                    if links:
                        parts.append(f"Routes Twig : {', '.join(links[:MAX_TWIG_ITEMS])}.")
                    if inputs:
                        parts.append(f"Champs formulaire : {', '.join(inputs[:MAX_TWIG_ITEMS])}.")
                    if includes:
                        parts.append(f"Inclut : {', '.join(includes[:MAX_TWIG_SECONDARY])}.")

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

    # Bilan explicite : un dossier absent ne doit jamais passer inaperçu.
    # Sans cela, un projet dont l'arborescence diffère est analysé à moitié
    # en silence, et rien ne le signale.
    trouves = ", ".join(f"{k} ({len(v)})" for k, v in analysis.items() if v)
    logging.info(f"Analyse terminée — trouvés : {trouves or 'aucun'}")
    # On ne signale que les catégories restées VIDES : inutile d'alerter sur
    # « Service » quand le projet écrit « Services » et que la clé est remplie.
    muets = [f for f in absents if not analysis.get(cle(f))]
    if muets:
        logging.warning(
            f"Dossiers absents de src/, aucune donnée pour : {', '.join(muets)}. "
            f"Si le projet les nomme autrement, compléter folders_to_scan."
        )

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
    if not os.path.isfile(file_path):
        logging.error(f"Fichier non trouvé : {file_path}")
        return None
    try:
        content = _read_php_file(file_path)
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
