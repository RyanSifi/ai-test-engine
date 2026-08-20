"""
Regex de parsing du texte des chunks de méthode enrichis produits à
l'indexation (voir learn_from_code() dans main.py).

⚠️  COUPLAGE FRAGILE (dette connue) : ces regex parsent le TEXTE des chunks
construits dynamiquement dans main.py (cf. la boucle qui produit
`profile_info`, `info`, etc. autour de "→ Profil:", "→ Rôles classe:", ...).
Ce texte est lui-même dérivé des champs structurés renvoyés par
code_parser.py (controller_profile, class_grants, method_grants, ...).

Il n'y a AUCUNE vérification automatique que ces deux côtés restent
synchronisés : si le format du chunk change dans main.py (libellé,
ponctuation, casse) sans que les regex ci-dessous soient mises à jour à
l'identique, la détection échoue silencieusement et retombe sur les valeurs
par défaut (ex: profil "web_crud") sans erreur visible.
Si le format de chunk évolue, mettre à jour ces regex dans le même commit,
et idéalement ajouter un test qui génère un chunk puis vérifie que chaque
regex le retrouve.
"""
import re

# ─────────────────────────────────────────────────────────────────────────────
# LIMITES DE TRONCATURE DES CHUNKS
#
# Un template Twig peut contenir des dizaines de boutons, de champs ou de liens.
# Tout embarquer gonflerait le chunk sans rien apporter : le modèle d'embedding
# dilue le signal, et le prompt consomme des tokens pour des éléments que le test
# n'utilisera jamais. On ne garde donc que les premiers de chaque famille.
#
# Ces constantes étaient auparavant écrites en dur — et en DOUBLE, dans
# code_parser.py (construction du chunk de template) et dans main.py (reprise de
# ces mêmes listes dans le chunk de méthode), avec des valeurs différentes.
# Elles sont ici pour rester cohérentes entre les deux.
# ─────────────────────────────────────────────────────────────────────────────

# Chunk de TEMPLATE (code_parser.py) — vue d'ensemble du fichier Twig.
MAX_TWIG_ITEMS   = 10   # onglets, boutons, IDs cachés, routes, champs
MAX_TWIG_SECONDARY = 5  # IDs de tables, sous-templates inclus

# Chunk de MÉTHODE (main.py) — plus resserré : seuls les éléments directement
# exploitables pour écrire une assertion sont utiles.
MAX_METHOD_H1      = 3  # titres H1 → assertSelectorTextContains
MAX_METHOD_FIELDS  = 8  # champs de formulaire et IDs cachés

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

_RE_BODY_INFERRED_VERB = re.compile(r"→ Verbe HTTP inféré \(body\):\s*([A-Z]+)")

_RE_PARAMS_LINE = re.compile(r"Params:\s*\(([^)]*)\)")
