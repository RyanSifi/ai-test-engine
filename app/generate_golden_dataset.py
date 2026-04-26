#!/usr/bin/env python3
"""
Génère golden_dataset_functional.json à partir des valeurs de config.
À relancer chaque fois que config.py / .env change (auth, firewall, rôles...).

Usage :
    python generate_golden_dataset.py
    python generate_golden_dataset.py --output /app/golden_dataset_functional.json
"""

import json
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import settings


def _role_key(role: str) -> str:
    """Applique auth_role_key_map si défini."""
    if settings.auth_role_key_map:
        for pair in settings.auth_role_key_map.split(","):
            if ":" in pair:
                src, dst = pair.strip().split(":", 1)
                if src.strip() == role:
                    return dst.strip()
    return role


def build_dataset() -> list:
    fw       = settings.auth_firewall_name
    redirect = settings.auth_redirect_path
    code     = settings.auth_redirect_status
    factory  = settings.auth_test_class
    sso      = settings.auth_sso_user_class
    roles    = [r.strip() for r in settings.auth_test_roles.split(",")]
    primary  = roles[0]
    secondary = roles[1] if len(roles) > 1 else primary

    factory_short = factory.split("\\")[-1]
    sso_short     = sso.split("\\")[-1]

    # ── Boilerplate setUp commun ────────────────────────────────────────────
    def boilerplate(class_name: str) -> str:
        return f"""<?php

namespace App\\Tests\\Functional\\Controller;

use {factory};
use {sso};
use Symfony\\Bundle\\FrameworkBundle\\KernelBrowser;
use Symfony\\Bundle\\FrameworkBundle\\Test\\WebTestCase;
use Symfony\\Component\\HttpFoundation\\Response;

final class {class_name} extends WebTestCase
{{
    protected KernelBrowser $client;
    private {factory_short} $testUserFactory;

    protected function setUp(): void
    {{
        $this->client = self::createClient();
        $this->testUserFactory = $this->client->getContainer()->get({factory_short}::class);
    }}

    private function getTestUser(string $key): {sso_short}
    {{
        return $this->testUserFactory->create($key);
    }}"""

    # ══════════════════════════════════════════════════════════════════════
    # Exemple 1 — CRUD simple : render + redirect + json
    # ══════════════════════════════════════════════════════════════════════
    context_1 = [
        f"Méthode 'index' (CreanceController) — Route: /creance/ — Template: creance/index.html.twig\n  → Type de réponse: render (200)\n  → H1: Liste des créances",
        f"Méthode 'show' (CreanceController) — Route: /creance/{{id}} — Template: creance/show.html.twig\n  → Type de réponse: render (200)",
        f"Méthode 'new' (CreanceController) — Route: /creance/new — Params: (Request $request)\n  → Type de réponse: redirect (302)",
        f"Méthode 'export' (CreanceController) — Route: /creance/export\n  → Type de réponse: json (200)",
        f"La classe PHP CreanceController (type: controllers) dans Controller/CreanceController.php. Son constructeur injecte : (CreanceRepository $repo, ManagerRegistry $em). → Profil: web_crud → Rôles classe: {primary}",
    ]

    test_1 = boilerplate("CreanceControllerTest") + f"""

    /** /creance/ — non authentifié */
    public function testIndexRedirectsWhenNotAuthenticated(): void
    {{
        $this->client->request('GET', '/creance/');
        self::assertResponseStatusCodeSame({code});
        self::assertResponseRedirects('{redirect}');
    }}

    /** /creance/ — authentifié {primary} */
    public function testIndexIsReachedWith{primary.title()}Role(): void
    {{
        $this->client->loginUser($this->getTestUser('{primary}'), '{fw}');
        $this->client->request('GET', '/creance/');
        self::assertResponseIsSuccessful();
        $this->assertSelectorTextContains('h1', 'Liste des créances');
    }}

    /** /creance/{{id}} — non authentifié */
    public function testShowRedirectsWhenNotAuthenticated(): void
    {{
        $this->client->request('GET', '/creance/1');
        self::assertResponseStatusCodeSame({code});
        self::assertResponseRedirects('{redirect}');
    }}

    /** /creance/{{id}} — authentifié {primary} */
    public function testShowIsReachedWith{primary.title()}Role(): void
    {{
        $this->client->loginUser($this->getTestUser('{primary}'), '{fw}');
        $this->client->request('GET', '/creance/1');
        self::assertResponseIsSuccessful();
    }}

    /** /creance/new — non authentifié */
    public function testNewRedirectsWhenNotAuthenticated(): void
    {{
        $this->client->request('POST', '/creance/new');
        self::assertResponseStatusCodeSame({code});
        self::assertResponseRedirects('{redirect}');
    }}

    /** /creance/new — authentifié {primary} → redirect après succès */
    public function testNewRedirectsWith{primary.title()}Role(): void
    {{
        $this->client->loginUser($this->getTestUser('{primary}'), '{fw}');
        $this->client->request('POST', '/creance/new');
        self::assertResponseRedirects();
    }}

    /** /creance/export — non authentifié */
    public function testExportRedirectsWhenNotAuthenticated(): void
    {{
        $this->client->request('GET', '/creance/export');
        self::assertResponseStatusCodeSame({code});
        self::assertResponseRedirects('{redirect}');
    }}

    /** /creance/export — authentifié {primary} */
    public function testExportReturnsJsonWith{primary.title()}Role(): void
    {{
        $this->client->loginUser($this->getTestUser('{primary}'), '{fw}');
        $this->client->request('GET', '/creance/export');
        self::assertResponseIsSuccessful();
        $this->assertJson($this->client->getResponse()->getContent());
    }}
}}"""

    # ══════════════════════════════════════════════════════════════════════
    # Exemple 2 — Route AJAX : test avec et sans header XHR
    # ══════════════════════════════════════════════════════════════════════
    context_2 = [
        f"Méthode 'getDetails' (DossierController) — Route: /dossier/{{id}}/details — Params: (int $id, Request $request)\n  → Type de réponse: json (200)\n  → AJAX uniquement\n  → Verbes HTTP: GET",
        f"La classe PHP DossierController (type: controllers) dans Controller/DossierController.php. → Profil: api → Rôles classe: {primary}",
    ]

    test_2 = boilerplate("DossierControllerTest") + f"""

    /** /dossier/{{id}}/details — non authentifié */
    public function testGetDetailsRedirectsWhenNotAuthenticated(): void
    {{
        $this->client->request('GET', '/dossier/1/details');
        self::assertResponseStatusCodeSame({code});
        self::assertResponseRedirects('{redirect}');
    }}

    /** /dossier/{{id}}/details — authentifié {primary} avec header XHR */
    public function testGetDetailsReturnsJsonWith{primary.title()}Role(): void
    {{
        $this->client->loginUser($this->getTestUser('{primary}'), '{fw}');
        $this->client->request('GET', '/dossier/1/details', [], [], ['HTTP_X-Requested-With' => 'XMLHttpRequest']);
        self::assertResponseIsSuccessful();
        $this->assertJson($this->client->getResponse()->getContent());
    }}

    /** /dossier/{{id}}/details — authentifié {primary} SANS header XHR → 404 */
    public function testGetDetailsWithoutXhrReturns404(): void
    {{
        $this->client->loginUser($this->getTestUser('{primary}'), '{fw}');
        $this->client->request('GET', '/dossier/1/details');
        self::assertResponseStatusCodeSame(404);
    }}
}}"""

    # ══════════════════════════════════════════════════════════════════════
    # Exemple 3 — Formulaire : GET affiche, POST soumet et redirige
    # ══════════════════════════════════════════════════════════════════════
    context_3 = [
        f"Méthode 'edit' (ParcoursController) — Route: /parcours/{{id}}/edit — Params: (int $id, Request $request) — Template: parcours/edit.html.twig\n  → Type de réponse: render (200), redirect (302)\n  → Formulaire: ParcoursType\n  → Verbes HTTP: GET, POST\n  → H1: Modifier le parcours",
        f"La classe PHP ParcoursController (type: controllers) dans Controller/ParcoursController.php. → Profil: web_crud → Rôles classe: {primary}",
    ]

    test_3 = boilerplate("ParcoursControllerTest") + f"""

    /** /parcours/{{id}}/edit — non authentifié */
    public function testEditRedirectsWhenNotAuthenticated(): void
    {{
        $this->client->request('GET', '/parcours/1/edit');
        self::assertResponseStatusCodeSame({code});
        self::assertResponseRedirects('{redirect}');
    }}

    /** /parcours/{{id}}/edit — authentifié {primary} — affichage formulaire */
    public function testEditDisplaysFormWith{primary.title()}Role(): void
    {{
        $this->client->loginUser($this->getTestUser('{primary}'), '{fw}');
        $this->client->request('GET', '/parcours/1/edit');
        self::assertResponseIsSuccessful();
        $this->assertSelectorTextContains('h1', 'Modifier le parcours');
    }}

    /** /parcours/{{id}}/edit — authentifié {primary} — soumission formulaire */
    public function testEditSubmitWith{primary.title()}Role(): void
    {{
        $this->client->loginUser($this->getTestUser('{primary}'), '{fw}');
        $this->client->request('POST', '/parcours/1/edit');
        self::assertResponseRedirects();
    }}
}}"""

    # ══════════════════════════════════════════════════════════════════════
    # Exemple 4 — Rôle méthode spécifique : test 403 rôle insuffisant
    # ══════════════════════════════════════════════════════════════════════
    context_4 = [
        f"Méthode 'index' (ParametrageController) — Route: /parametrage/ — Template: parametrage/index.html.twig\n  → Type de réponse: render (200)\n  → H1: Paramétrage",
        f"Méthode 'delete' (ParametrageController) — Route: /parametrage/{{id}}/delete — Params: (int $id)\n  → Type de réponse: redirect (302)\n  → Rôle requis (méthode): {primary}\n  → Verbes HTTP: DELETE",
        f"La classe PHP ParametrageController (type: controllers) dans Controller/ParametrageController.php. → Profil: web_crud → Rôles classe: {secondary}",
    ]

    test_4 = boilerplate("ParametrageControllerTest") + f"""

    /** /parametrage/ — non authentifié */
    public function testIndexRedirectsWhenNotAuthenticated(): void
    {{
        $this->client->request('GET', '/parametrage/');
        self::assertResponseStatusCodeSame({code});
        self::assertResponseRedirects('{redirect}');
    }}

    /** /parametrage/ — authentifié {secondary} */
    public function testIndexIsReachedWith{secondary.title()}Role(): void
    {{
        $this->client->loginUser($this->getTestUser('{secondary}'), '{fw}');
        $this->client->request('GET', '/parametrage/');
        self::assertResponseIsSuccessful();
        $this->assertSelectorTextContains('h1', 'Paramétrage');
    }}

    /** /parametrage/{{id}}/delete — non authentifié */
    public function testDeleteRedirectsWhenNotAuthenticated(): void
    {{
        $this->client->request('DELETE', '/parametrage/1/delete');
        self::assertResponseStatusCodeSame({code});
        self::assertResponseRedirects('{redirect}');
    }}

    /** /parametrage/{{id}}/delete — authentifié {primary} (rôle requis) */
    public function testDeleteRedirectsWith{primary.title()}Role(): void
    {{
        $this->client->loginUser($this->getTestUser('{primary}'), '{fw}');
        $this->client->request('DELETE', '/parametrage/1/delete');
        self::assertResponseRedirects();
    }}

    /** /parametrage/{{id}}/delete — authentifié {secondary} (rôle insuffisant) → 403 */
    public function testDeleteForbiddenWith{secondary.title()}Role(): void
    {{
        $this->client->loginUser($this->getTestUser('{secondary}'), '{fw}');
        $this->client->request('DELETE', '/parametrage/1/delete');
        self::assertResponseStatusCodeSame(403);
    }}
}}"""

    # ══════════════════════════════════════════════════════════════════════
    # Exemple 5 — Contrôleur mixte : render + json dans le même contrôleur
    # ══════════════════════════════════════════════════════════════════════
    context_5 = [
        f"Méthode 'index' (TableauBordController) — Route: /tableau-bord/ — Template: tableau-bord/index.html.twig\n  → Type de réponse: render (200)\n  → H1: Tableau de bord",
        f"Méthode 'stats' (TableauBordController) — Route: /tableau-bord/stats\n  → Type de réponse: json (200)",
        f"La classe PHP TableauBordController (type: controllers) dans Controller/TableauBordController.php. → Profil: mixed → Rôles classe: {primary}",
    ]

    test_5 = boilerplate("TableauBordControllerTest") + f"""

    /** /tableau-bord/ — non authentifié */
    public function testIndexRedirectsWhenNotAuthenticated(): void
    {{
        $this->client->request('GET', '/tableau-bord/');
        self::assertResponseStatusCodeSame({code});
        self::assertResponseRedirects('{redirect}');
    }}

    /** /tableau-bord/ — authentifié {primary} */
    public function testIndexIsReachedWith{primary.title()}Role(): void
    {{
        $this->client->loginUser($this->getTestUser('{primary}'), '{fw}');
        $this->client->request('GET', '/tableau-bord/');
        self::assertResponseIsSuccessful();
        $this->assertSelectorTextContains('h1', 'Tableau de bord');
    }}

    /** /tableau-bord/stats — non authentifié */
    public function testStatsRedirectsWhenNotAuthenticated(): void
    {{
        $this->client->request('GET', '/tableau-bord/stats');
        self::assertResponseStatusCodeSame({code});
        self::assertResponseRedirects('{redirect}');
    }}

    /** /tableau-bord/stats — authentifié {primary} */
    public function testStatsReturnsJsonWith{primary.title()}Role(): void
    {{
        $this->client->loginUser($this->getTestUser('{primary}'), '{fw}');
        $this->client->request('GET', '/tableau-bord/stats');
        self::assertResponseIsSuccessful();
        $this->assertJson($this->client->getResponse()->getContent());
    }}
}}"""

    # ══════════════════════════════════════════════════════════════════════
    # Exemple 6 — Route sans type de réponse détecté → assertResponseIsSuccessful
    # ══════════════════════════════════════════════════════════════════════
    context_6 = [
        f"Méthode 'clearCache' (CacheController) — Route: /cache/clear\n  → response (200)",
        f"La classe PHP CacheController (type: controllers) dans Controller/CacheController.php. → Profil: web_crud → Rôles classe: {primary}",
    ]

    test_6 = boilerplate("CacheControllerTest") + f"""

    /** /cache/clear — non authentifié */
    public function testClearCacheRedirectsWhenNotAuthenticated(): void
    {{
        $this->client->request('GET', '/cache/clear');
        self::assertResponseStatusCodeSame({code});
        self::assertResponseRedirects('{redirect}');
    }}

    /** /cache/clear — authentifié {primary} — type non détecté → assertResponseIsSuccessful uniquement */
    public function testClearCacheIsReachedWith{primary.title()}Role(): void
    {{
        $this->client->loginUser($this->getTestUser('{primary}'), '{fw}');
        $this->client->request('GET', '/cache/clear');
        self::assertResponseIsSuccessful();
    }}
}}"""

    return [
        {
            "demande_utilisateur": (
                "Tests fonctionnels pour CreanceController. "
                "Routes : GET /creance/ (liste, render), GET /creance/{id} (détail, render), "
                "POST /creance/new (création, redirect), GET /creance/export (json)."
            ),
            "contexte_pertinent": context_1,
            "test_ideal": test_1,
        },
        {
            "demande_utilisateur": (
                "Tests fonctionnels pour DossierController. "
                "Route AJAX : GET /dossier/{id}/details — json, uniquement via XHR."
            ),
            "contexte_pertinent": context_2,
            "test_ideal": test_2,
        },
        {
            "demande_utilisateur": (
                "Tests fonctionnels pour ParcoursController. "
                "Route formulaire : GET/POST /parcours/{id}/edit — formulaire ParcoursType, redirect après soumission."
            ),
            "contexte_pertinent": context_3,
            "test_ideal": test_3,
        },
        {
            "demande_utilisateur": (
                "Tests fonctionnels pour ParametrageController. "
                "Route /parametrage/ accessible au rôle classe, route DELETE /parametrage/{id}/delete réservée au rôle supérieur."
            ),
            "contexte_pertinent": context_4,
            "test_ideal": test_4,
        },
        {
            "demande_utilisateur": (
                "Tests fonctionnels pour TableauBordController (profil mixte). "
                "Route render /tableau-bord/ et route json /tableau-bord/stats."
            ),
            "contexte_pertinent": context_5,
            "test_ideal": test_5,
        },
        {
            "demande_utilisateur": (
                "Tests fonctionnels pour CacheController. "
                "Route /cache/clear sans type de réponse détecté dans le contexte."
            ),
            "contexte_pertinent": context_6,
            "test_ideal": test_6,
        },
    ]


def main():
    parser = argparse.ArgumentParser(description="Génère golden_dataset_functional.json")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "golden_dataset_functional.json"),
        help="Chemin de sortie (défaut : même dossier que ce script)",
    )
    args = parser.parse_args()

    dataset = build_dataset()
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"✓ {len(dataset)} exemple(s) généré(s) → {args.output}")
    print(f"  firewall={settings.auth_firewall_name}")
    print(f"  redirect={settings.auth_redirect_path} ({settings.auth_redirect_status})")
    print(f"  rôles={settings.auth_test_roles}")
    print(f"  factory={settings.auth_test_class}")


if __name__ == "__main__":
    main()
