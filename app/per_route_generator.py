"""
Génération de test fonctionnel PAR ROUTE.

Au lieu d'un seul appel LLM qui demande le fichier complet (source des
troncatures done_reason:length et des timeouts), on fait UN appel LLM par
route, chacun ne produisant que 2 à 3 méthodes de test. Le squelette de la
classe reste DÉTERMINISTE (_render_test_class_skeleton) — le LLM ne génère
jamais le namespace/les imports/le setUp, seulement des corps de méthodes.

Garde-fous :
- noms de méthodes imposés et uniques (anti-collision « duplicate method »)
- retry borné par route (length / timeout / bloc invalide) puis stub markTestIncomplete
- chaque bloc de méthode borné par accolades équilibrées avant insertion
- fichier réassemblé validé par `php -l`
- log par route : route, tentatives, durée, tokens, done_reason
"""
import logging
import re
import time
from typing import Dict, List

import requests

from config import settings
from chunk_format import _RE_METHOD_NAME, _RE_ROUTE
from code_parser import _extract_balanced_block
from php_writer import validate_php_syntax
from llm_client import _call_llm_meta, _build_prompt
from prompt_safety import wrap_untrusted_context
from deterministic_generator import (
    _render_test_class_skeleton, _detect_class_role, _parse_chunk_metadata,
)

_PER_ROUTE_NUM_PREDICT = 1024   # 2-3 méthodes tiennent largement là-dedans
_PER_ROUTE_TIMEOUT     = 120    # secondes par appel de route
_MAX_ATTEMPTS          = 3      # 1 essai initial + 2 retries

# ── Budget de temps GLOBAL ───────────────────────────────────────────────────
# Sans ce plafond, la durée totale n'était bornée par rien : chaque route pouvait
# consommer _MAX_ATTEMPTS × _PER_ROUTE_TIMEOUT = 360 s, soit 60 minutes sur un
# contrôleur à 10 routes en échec.
#
# Or le client Streamlit abandonne au bout de 900 s. Passé ce délai il affichait
# « erreur réseau » — mais le serveur, lui, continuait, et finissait par ÉCRIRE
# le fichier. D'où le symptôme observé : une erreur à l'écran, puis un fichier
# qui apparaît tout seul plusieurs dizaines de minutes plus tard.
#
# 840 s laisse une marge d'une minute sous le timeout client : la réponse a le
# temps de partir avant que Streamlit ne raccroche.
_TOTAL_BUDGET_SEC = 840

# En dessous de ce reliquat, on ne lance pas un nouvel appel : il n'aurait pas le
# temps d'aboutir et consommerait le budget pour rien.
_MIN_REMAINING_SEC = 20

_RE_TEST_METHOD_DEF = re.compile(r"public\s+function\s+(test\w+)\s*\(", re.IGNORECASE)


def _expected_assertion(ctx: Dict) -> str:
    """Résume, pour le prompt, l'assertion attendue après authentification."""
    if ctx.get("has_json"):
        return "assertResponseIsSuccessful() puis assertJson sur le contenu de la réponse"
    if ctx.get("has_redirect") and not ctx.get("has_render"):
        return "assertResponseRedirects() SANS argument"
    if ctx.get("has_file") or ctx.get("has_export"):
        return "assertResponseIsSuccessful()"
    # render (200) ou défaut
    base = "assertResponseIsSuccessful()"
    if ctx.get("h1"):
        base += f" ; puis assertSelectorTextContains('h1', '{ctx['h1']}')"
    return base


def _plan_methods(ctx: Dict, used_names: set) -> List[Dict]:
    """
    Construit la liste des méthodes à générer pour une route : 2 de base
    (sans auth / avec auth) + au plus 1 conditionnelle selon le profil de la
    route (AJAX / formulaire / voter). Les noms sont rendus uniques dans le
    fichier pour éviter toute collision PHP.
    """
    cap = ctx["cap"]

    def uniq(base: str) -> str:
        name = base
        i = 2
        while name in used_names:
            name = f"{base}{i}"
            i += 1
        used_names.add(name)
        return name

    plan = [
        {
            "name": uniq(f"test{cap}RedirectsWhenNotAuthenticated"),
            "spec": (
                f"NON authentifié (pas de loginUser) : "
                f"$this->client->request('{ctx['http_verb']}', '{ctx['route']}') "
                f"puis assertResponseStatusCodeSame({settings.auth_redirect_status}) "
                f"et assertResponseRedirects('{settings.auth_redirect_path}')."
            ),
        },
        {
            "name": uniq(f"test{cap}AuthenticatedWith{ctx['role_label']}Role"),
            "spec": (
                f"AUTHENTIFIÉ : "
                f"$this->client->loginUser($this->getTestUser('{ctx['factory_key']}'), "
                f"'{settings.auth_firewall_name}') puis "
                f"$this->client->request('{ctx['http_verb']}', '{ctx['route']}') "
                f"puis {_expected_assertion(ctx)}."
            ),
        },
    ]

    # Une seule conditionnelle, par ordre de priorité AJAX > Form > Voter.
    if ctx.get("is_ajax"):
        plan.append({
            "name": uniq(f"test{cap}WithoutXhrReturns404"),
            "spec": (
                f"Route AJAX appelée SANS header XHR : loginUser('{ctx['factory_key']}') puis "
                f"$this->client->request('{ctx['http_verb']}', '{ctx['route']}') "
                f"(sans HTTP_X-Requested-With) puis assertResponseStatusCodeSame(404)."
            ),
        })
    elif ctx.get("has_form"):
        plan.append({
            "name": uniq(f"test{cap}FormSubmit"),
            "spec": (
                f"Soumission du formulaire : loginUser('{ctx['factory_key']}') puis "
                f"$this->client->request('POST', '{ctx['route']}') puis "
                f"{'assertResponseRedirects()' if ctx.get('has_redirect') else 'assertResponseIsSuccessful()'}."
            ),
        })
    elif ctx.get("has_voter"):
        plan.append({
            "name": uniq(f"test{cap}VoterDeniesAccess"),
            "spec": (
                f"Contrôle d'accès par voter '{ctx.get('voter_attr') or '?'}' : "
                f"loginUser('{ctx['factory_key']}') puis requête sur la route avec une "
                f"entité inexistante, et assertContains du status dans [403, 404, 500]."
            ),
        })

    return plan


def _build_route_prompt(ctx: Dict, method_plan: List[Dict], route_context: str) -> str:
    """Prompt réduit au contexte d'UNE route, imposant les noms de méthodes."""
    fw       = settings.auth_firewall_name
    factory  = settings.auth_test_class.split("\\")[-1]

    system = (
        "Tu es un expert Symfony spécialisé dans les tests fonctionnels PHPUnit / WebTestCase.\n"
        "Tu génères UNIQUEMENT des méthodes de test (corps PHP complets) — "
        "PAS de <?php, PAS de namespace, PAS de use, PAS de déclaration de classe, "
        "PAS de setUp ni getTestUser (ils existent déjà). Commence DIRECTEMENT par "
        "'public function'.\n"
        "Le client est $this->client (KernelBrowser). "
        f"Authentification : $this->client->loginUser($this->getTestUser('ROLE_XXX'), '{fw}').\n"
        "N'invente aucune URL, aucun texte de page, aucun sélecteur CSS non fourni.\n\n"
        "CONTEXTE RÉEL DE LA ROUTE (ne pas inventer au-delà) :\n"
        f"{route_context}"
    )

    lines = [
        f"Route : {ctx['http_verb']} {ctx['route']}",
        "",
        "Génère EXACTEMENT ces méthodes (utilise ces noms précis, rien d'autre) :",
    ]
    for i, m in enumerate(method_plan, 1):
        lines.append(f"{i}. public function {m['name']}(): void — {m['spec']}")
    lines.append("")
    lines.append("Retourne uniquement les corps de méthodes, sans texte autour.")
    user = "\n".join(lines)

    return _build_prompt(system, [], user)


def _extract_method_blocks(llm_text: str) -> List[str]:
    """
    Extrait chaque bloc `public function testXxx(...) { ... }` du texte LLM, en
    ignorant tout wrapper parasite (<?php, classe, use) que le modèle aurait pu
    ajouter. Le corps est borné par _extract_balanced_block → un bloc extrait
    est garanti à accolades équilibrées (sinon il est ignoré : sortie tronquée).
    """
    blocks = []
    for m in _RE_TEST_METHOD_DEF.finditer(llm_text):
        brace = llm_text.find("{", m.end())
        if brace == -1:
            continue
        body = _extract_balanced_block(llm_text, brace)
        if body is None:
            continue
        # Méthode complète telle que le LLM l'a écrite (signature + { ... }),
        # puis ré-indentation uniforme de 4 espaces pour l'insérer dans la classe
        # (préserve l'indentation RELATIVE du corps quelle que soit celle du LLM).
        method_src = (llm_text[m.start():brace] + body).strip()
        indented = "\n".join(
            ("    " + ln) if ln.strip() else ln
            for ln in method_src.splitlines()
        )
        blocks.append(indented)
    return blocks


def _stub_methods(method_plan: List[Dict], ctx: Dict, done_reason) -> List[str]:
    """Méthodes de repli (markTestIncomplete) quand la génération d'une route échoue."""
    stubs = []
    for m in method_plan:
        stubs.append(
            f"    public function {m['name']}(): void\n"
            f"    {{\n"
            f"        // TODO: génération échouée pour la route "
            f"{ctx['http_verb']} {ctx['raw_route']} (done_reason={done_reason}).\n"
            f"        $this->markTestIncomplete("
            f"'Génération automatique échouée — à écrire à la main.');\n"
            f"    }}"
        )
    return stubs


def _dedupe_block_names(blocks: List[str]) -> List[str]:
    """Filet de sécurité : renomme tout nom de méthode dupliqué (LLM désobéissant)."""
    seen = set()
    out = []
    for b in blocks:
        m = _RE_TEST_METHOD_DEF.search(b)
        if not m:
            out.append(b)
            continue
        name = m.group(1)
        if name in seen:
            new = name
            i = 2
            while new in seen:
                new = f"{name}{i}"
                i += 1
            b = b[:m.start(1)] + new + b[m.end(1):]
            name = new
        seen.add(name)
        out.append(b)
    return out


def generate_functional_test_per_route(
    chunks: List[Dict],
    class_name: str,
    test_class_name: str,
) -> Dict:
    """
    Génère un fichier de test fonctionnel en faisant UN appel LLM par route.

    Retourne :
        {
          "code":             str,        # fichier PHP complet réassemblé
          "php_valid":        bool,       # `php -l` OK ?
          "php_error":        str|None,
          "routes_total":     int,
          "routes_generated": int,        # routes réussies (≥ 1 méthode extraite)
          "routes_failed":    [str],      # routes retombées sur stub
          "per_route":        [dict],     # diagnostics par route (logs)
        }
    """
    class_role = _detect_class_role(chunks)
    used_names: set = set()
    all_blocks: List[str] = []
    per_route: List[Dict] = []
    routes_failed: List[str] = []
    routes_generated = 0
    # Dédoublonnage sur le COUPLE (méthode, route) : learn_from_code() produit
    # UN CHUNK PAR ROUTE, donc une méthode portant plusieurs #[Route] revient
    # plusieurs fois sous le même nom. Dédoublonner par nom faisait perdre
    # silencieusement toutes ses routes sauf la première.
    routes_seen: set = set()
    route_rank: Dict[str, int] = {}
    debut = time.monotonic()          # référence du budget global
    routes_abandonnees: List[str] = []  # non traitées faute de temps

    # Chunk de classe (profil, rôles) — contexte partagé, ajouté à chaque route.
    # On filtre sur chunk_type (donnée STRUCTURÉE posée par learn_from_code) et non
    # sur le texte : la version précédente testait startswith("Classe") alors que
    # main.py produit « La classe PHP … », donc class_ctx était TOUJOURS vide et le
    # profil/les rôles de la classe n'atteignaient jamais le prompt. Le repli texte
    # couvre les chunks d'un index antérieur qui n'auraient pas de chunk_type.
    class_ctx = next(
        (c.get("content", "") for c in chunks
         if str(c.get("chunk_type", "")).endswith("_class")
         or c.get("content", "").startswith("La classe PHP")), ""
    )

    for chunk in chunks:
        content = chunk.get("content", "")
        if not content.startswith("Méthode '"):
            continue
        mm = _RE_METHOD_NAME.search(content)
        if not mm:
            continue
        method_name = mm.group(1)
        if method_name == "__construct":
            continue
        rm = _RE_ROUTE.search(content)
        if not rm:
            continue  # méthode sans route HTTP → pas de test fonctionnel

        raw_route = rm.group(1)
        if (method_name, raw_route) in routes_seen:
            continue
        routes_seen.add((method_name, raw_route))

        # Budget global épuisé → on arrête d'appeler le LLM. Les routes restantes
        # reçoivent un stub, et le fichier est quand même produit : mieux vaut un
        # squelette complet et honnête qu'une réponse qui n'arrive jamais.
        reste = _TOTAL_BUDGET_SEC - (time.monotonic() - debut)
        budget_epuise = reste < _MIN_REMAINING_SEC

        # 1re route : noms de test inchangés. Suivantes : suffixe Route2, Route3…
        # (uniq() dans _plan_methods reste le filet de sécurité final).
        rank = route_rank.get(method_name, 0)
        route_rank[method_name] = rank + 1
        cap_suffix = "" if rank == 0 else f"Route{rank + 1}"

        ctx = _parse_chunk_metadata(content, raw_route, class_role, cap_suffix)
        method_plan = _plan_methods(ctx, used_names)

        blocks: List[str] = []
        last_done_reason = None
        attempts = 0
        total_tokens = 0
        total_elapsed = 0.0

        if budget_epuise:
            last_done_reason = "budget_epuise"
            routes_abandonnees.append(f"{ctx['http_verb']} {ctx['raw_route']}")
            logging.warning(
                f"[per-route] budget global de {_TOTAL_BUDGET_SEC}s épuisé — "
                f"route {ctx['raw_route']} non générée (stub)"
            )
        else:
            # Contexte issu du code analysé : neutralisé et délimité avant le
            # prompt — un docblock peut contenir des jetons de contrôle ChatML
            # ou des consignes déguisées (cf. prompt_safety.py).
            route_context = wrap_untrusted_context(
                (class_ctx + "\n" + content).strip()
            )
            prompt = _build_route_prompt(ctx, method_plan, route_context)

            for attempt in range(1, _MAX_ATTEMPTS + 1):
                attempts = attempt
                reste = _TOTAL_BUDGET_SEC - (time.monotonic() - debut)
                if reste < _MIN_REMAINING_SEC:
                    last_done_reason = "budget_epuise"
                    break
                # Le timeout de l'appel ne peut jamais dépasser le reliquat global.
                timeout_appel = int(min(_PER_ROUTE_TIMEOUT, reste))

                # Une troncature (`length`) ne se corrige pas en rejouant le MÊME
                # appel : à température 0,1, la réponse repart identique et
                # retronque au même endroit. On élargit donc le budget de sortie
                # à chaque tentative.
                n_predict = _PER_ROUTE_NUM_PREDICT * attempt

                try:
                    res = _call_llm_meta(
                        prompt,
                        timeout=timeout_appel,
                        num_predict=n_predict,
                        _skip_health=(attempt > 1),
                    )
                except (requests.RequestException, RuntimeError) as e:
                    last_done_reason = f"exception:{type(e).__name__}"
                    logging.warning(f"[per-route] {ctx['raw_route']} tentative {attempt} — {e}")
                    continue

                last_done_reason = res["done_reason"]
                total_tokens += res["eval_count"]
                total_elapsed += res["elapsed"]
                nouveaux = _extract_method_blocks(res["text"])

                # Succès franc : des méthodes extraites, sans troncature.
                if nouveaux and res["done_reason"] != "length":
                    blocks = nouveaux
                    break
                # Tronqué mais exploitable : on GARDE le meilleur résultat obtenu
                # jusqu'ici au lieu de le jeter. Si les tentatives suivantes
                # échouent, on écrira ces méthodes plutôt que des stubs.
                if len(nouveaux) > len(blocks):
                    blocks = nouveaux

        if blocks:
            routes_generated += 1
            all_blocks.extend(blocks)
            ok = True
        else:
            routes_failed.append(f"{ctx['http_verb']} {ctx['raw_route']}")
            all_blocks.extend(_stub_methods(method_plan, ctx, last_done_reason))
            ok = False

        diag = {
            "route":       ctx["raw_route"],
            "method":      method_name,
            "attempts":    attempts,
            "done_reason": last_done_reason,
            "tokens":      total_tokens,
            "elapsed":     round(total_elapsed, 1),
            "ok":          ok,
        }
        per_route.append(diag)
        logging.info(
            f"[per-route] route={diag['route']} method={diag['method']} "
            f"attempts={diag['attempts']} done_reason={diag['done_reason']} "
            f"tokens={diag['tokens']} elapsed={diag['elapsed']}s ok={diag['ok']}"
        )

    all_blocks = _dedupe_block_names(all_blocks)
    header, footer = _render_test_class_skeleton(test_class_name)
    body = "\n\n".join(all_blocks)
    code = header + body + footer

    php_error = validate_php_syntax(code)
    if php_error:
        logging.warning(f"[per-route] fichier réassemblé invalide (php -l) : {php_error}")

    duree = round(time.monotonic() - debut, 1)
    logging.info(
        f"[per-route] terminé en {duree}s — {routes_generated}/{len(per_route)} routes générées"
        + (f", {len(routes_abandonnees)} abandonnées (budget)" if routes_abandonnees else "")
    )

    return {
        "code":             code,
        "php_valid":        php_error is None,
        "php_error":        php_error,
        "routes_total":     len(per_route),
        "routes_generated": routes_generated,
        "routes_failed":    routes_failed,
        # Routes non tentées faute de temps — à distinguer des routes tentées et
        # échouées : ici le LLM n'a même pas été appelé.
        "routes_skipped":   routes_abandonnees,
        "budget_exceeded":  bool(routes_abandonnees),
        "elapsed_sec":      duree,
        "per_route":        per_route,
    }
