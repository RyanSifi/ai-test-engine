# TODO — points techniques en attente

> 👉 **Pour l'état à jour du reste à faire avant la soutenance (mémoire, présentation,
> code), voir [`docs/reste-a-faire.md`](docs/reste-a-faire.md)** — vérification en
> 4 passes du 18/08/2026. Le présent fichier reste l'historique des correctifs
> techniques appliqués.

> Relevés lors de la revue de conformité au référentiel (20/07/2026).
> Rien ici n'empêche le projet de fonctionner — il tourne. Ce sont des points de
> **rigueur méthodologique** et des **livrables attendus par le référentiel**.

---

## 0. Traité le 26/07/2026 — audit complet du code

Un audit intégral du code a été mené puis appliqué. Détail et vérifications :
[`docs/audit-code.md`](docs/audit-code.md).

**Défauts fonctionnels corrigés** (tous reproduits par exécution avant correction) :

| Correctif | Effet |
|---|---|
| Contexte de classe jamais transmis au LLM (`startswith("Classe")` vs `"La classe PHP"`) | Profil et rôles atteignent enfin le prompt par route |
| Méthodes à plusieurs `#[Route]` : toutes les routes sauf la première étaient perdues **en silence**, dans les **deux** générateurs | Dédoublonnage sur le couple (méthode, route) ; noms de test suffixés `Route2`, `Route3` |
| Mocks de tests visant `main._check_ollama_alive` au lieu de `llm_client.*` | Plus aucun appel réseau réel ; 2 tests exerçaient le repli sur stub au lieu du chemin annoncé |
| `sys.exit()` au niveau module dans un fichier `test_*` | `pytest` plantait à la collecte — **0 test ne tournait**. Fichier renommé `check_*` + adaptateur pytest |
| Seuil du dashboard (0,7) ≠ seuil des métriques publiées (0,5) | Unifié à 0,5, constantes hissées au niveau module ; +5,5 points de rappel |
| `cross_val_report()` recevant le pipeline LogReg **non** optimisé | Comparaison loyale entre modèles — écart mesuré 0,0003, conclusion inchangée |
| Interface proposant encore seuil 1, features v2, ancien dataset | Alignée sur le code |
| Code mort (`save_code_context`, `CLASS_NAME_REGEX`, params inutilisés, branche inatteignable…) | Supprimé |

**Ajouté** : évaluation **agrégée par contrôleur** (`file_level_report()`), le chiffre honnête
pour l'usage réel — MCC 0,529 contre 0,758 par ligne.

**Résultat** : `pytest /app` passe de *0 test collecté* à **104 tests verts** (38 à l'issue de
l'audit, portés à 104 par les tests de sécurité et d'injection ajoutés ensuite), vérifié aussi
dans le conteneur Docker.

---

## 1. ~~Incohérences méthodologiques dans `ml/train.py`~~ — RÉSOLU le 29/07/2026

> Les trois points de ce chapitre sont désormais corrigés. Conservé pour mémoire :
> le raisonnement reste utile en soutenance.

### 1.1 — ✅ Les hyperparamètres étaient optimisés sur le F1, pas sur le MCC

`GridSearchCV(..., scoring="f1")` alors que toute la documentation explique que le
F1 est trompeur sur données déséquilibrées et que la **sélection du modèle se fait
sur le MCC**. On optimisait donc sur une métrique déclarée non fiable.

**Corrigé** : `TUNING_SCORING = "matthews_corrcoef"` (constante en tête de `train.py`).

### 1.2 — ✅ Le GridSearch n'était pas groupé par fichier

`cv=5` produisait un `StratifiedKFold` **non groupé** : pendant la recherche
d'hyperparamètres, des méthodes d'un même contrôleur se retrouvaient à la fois en
apprentissage et en validation.

**Corrigé** : `_cv_tuning()` renvoie un `StratifiedGroupKFold`, et `.fit()` reçoit
`groups=groups[train_idx]`.

### 1.3 — ✅ LogisticRegression cross-validée sans ses hyperparamètres

`cross_val_report` recevait `lr_pipe` (non optimisé) tandis que RandomForest
recevait `best_rf` (optimisé). **Corrigé** — écart mesuré : 0,0003 de MCC.

### ✅ Vérification : les chiffres publiés ne changent pas

C'était le risque principal de ces correctifs. Comparaison des hyperparamètres
sélectionnés, ancien tuning contre nouveau :

| Modèle | Avant (`cv=5`, F1) | Après (groupé, MCC) | Identiques ? |
|---|---|---|---|
| LogisticRegression | `C = 10.0` | `C = 10.0` | ✅ |
| RandomForest | `max_depth=4, min_samples_leaf=3, n_estimators=200` | idem | ✅ |

Les scores de tuning diffèrent (0,8712 → 0,5781 pour LogReg) — normal, ce sont
deux métriques distinctes. Mais **le choix est le même**, donc après
réentraînement les métriques sont **strictement identiques** :
MCC 0,758 · MCC fichier 0,529 · bal-acc 0,861 · AUC 0,920 · RandomForest retenu.

**Ce que cela permet d'affirmer en soutenance** : *« Le MCC guide l'intégralité du
pipeline — réglage, sélection et évaluation — et aucune étape ne mélange les
fichiers. Nous l'avons vérifié : la fuite de groupe pendant le réglage n'avait pas
faussé le choix des hyperparamètres, les résultats publiés sont donc inchangés. »*

---

## 2. Livrables techniques attendus par le référentiel

Détail complet et priorisation dans [`docs/conformite-examen.md`](docs/conformite-examen.md).

| # | Point | État |
|---|---|---|
| P1 | Benchmark requêtes optimisées / non-optimisées | ✅ **fait** — [`docs/benchmark-bdd.md`](docs/benchmark-bdd.md) |
| P2 | Deep learning | ✅ **présent** — voir §4 ci-dessous (correction d'une analyse initiale erronée) |
| P3 | Tableau de suivi des problématiques techniques (index, date, libellé, résolution, solution) | ✅ **fait le 26/07** — [`docs/complements-memoire.md`](docs/complements-memoire.md) §P3 : **16 entrées** au format exigé |
| P4 | Dictionnaire de données | ✅ **fait le 26/07** — [`docs/dictionnaire-donnees.md`](docs/dictionnaire-donnees.md) |
| P5 | Compte-rendu valeurs manquantes / incohérences | ✅ **fait le 26/07** — [`docs/rapport-qualite-donnees.md`](docs/rapport-qualite-donnees.md) : 0 valeur manquante, 0 doublon, 10/10 règles de cohérence |
| P6 | README → PDR complet (identifiants de test, connexion SQL, accès admin, multi-navigateur) | ✅ **fait le 26/07** — sections « Accès et identifiants » + « Compatibilité navigateurs » du [README](README.md), et [`docs/installation.md`](docs/installation.md) |
| P7 | Dump SQL pour le ZIP | ✅ **fait le 26/07** — [`sql/schema.sql`](sql/schema.sql) + [README §6](README.md#6-base-de-données) (structure seule : les données contiennent du code source CNAM) |
| P8 | RGPD / données personnelles | ✅ **rédigé le 26/07** — [`docs/complements-memoire.md`](docs/complements-memoire.md) §P8. ✅ **Correctif appliqué le 29/07** : l'import Google Fonts a été retiré, aucune police n'est plus chargée depuis un service externe |
| P9 | Accessibilité (handicap) | ✅ **audité et rédigé le 26/07** — même document §P9. ✅ **Correctifs appliqués le 29/07** : `#0084B2` → `#00769F` (4,25:1 → 5,13:1, il échouait comme fond de bouton ET comme couleur de lien) et bordures de champs `#CED4DA` → `#8A9199` (1,49:1 → 3,19:1). Les bordures décoratives `#DEE2E6` sont volontairement conservées — le critère 1.4.11 ne vise que les éléments nécessaires pour IDENTIFIER un composant |

**Décisions à prendre** :
- **URL publique** exigée dans le ZIP → déployer, ou assumer l'hébergement local pour raison de
  confidentialité (défendable ici).
- **Streamlit** n'est pas cité par le référentiel (« flask, dash ou shiny ») → prévoir une phrase
  de justification (Streamlit ≈ Dash).

---

## 4. Deep learning — mise au point (correction du 20/07/2026)

> Une première analyse avait conclu à tort à l'absence de deep learning. C'était **faux** : la
> recherche ne couvrait que les marqueurs explicites (`tensorflow`, `keras`, `MLPClassifier`) en
> surface de `ml/` et `app/`. Le deep learning du projet est **encapsulé** dans
> `sentence-transformers` et **déporté** dans Ollama.

**Vérifié dans le conteneur `brain-api` en fonctionnement :**

```
PyTorch                : 2.12.0+cu130
sentence-transformers  : 2.6.1
Modèle d'embedding     : BertModel
  couches   : 12 couches Transformer
  attention : 12 têtes
  dim cachée: 384
  paramètres: 117 653 760
```

**Deux réseaux de neurones profonds, centraux dans l'architecture :**

| Réseau | Architecture | Paramètres | Rôle |
|---|---|---|---|
| **MiniLM** (`paraphrase-multilingual-MiniLM-L12-v2`) | BERT, 12 couches Transformer | **117,7 M** | Vectorisation du code (embeddings) — socle de la recherche sémantique |
| **qwen2.5-coder:3b** (via Ollama) | LLM Transformer | **3 Md** | Génération du code des tests PHPUnit |

Sans ces deux réseaux, le module RAG n'existe pas. Le deep learning n'est pas un accessoire du
projet : il en constitue le moteur.

**Nuance résiduelle** : la **tâche supervisée** (prédiction de risque) utilise, elle, du ML
classique (LogReg / RandomForest / XGBoost). Si le référentiel est lu strictement comme
« la tâche supervisée doit employer du DL », il subsiste un angle mort.

**Option « ceinture et bretelles »** (~1 h, facultative) : ajouter un `MLPClassifier` à la
comparaison des modèles dans `train.py`. Il sous-performera très probablement — et c'est
justement l'argument : *« un réseau de neurones a été testé sur la tâche supervisée ; avec
54 échantillons indépendants, il sur-apprend, ce qui justifie le choix du ML classique »*.
Montrer qu'on l'a testé et écarté avec méthode vaut mieux que de ne pas l'avoir testé.

---

## 3. Correctif d'infrastructure appliqué (pour mémoire)

✅ **Fait le 20/07/2026** — le volume PostgreSQL utilisait un *bind mount* Windows, qui corrompait
le cluster (`could not read block ...`). Basculé sur un **volume Docker nommé**. Vérifié :
écriture massive OK, persistance après redémarrage OK, recréation du schéma OK.

L'ancien dossier corrompu est conservé sous `data/pg_data.corrompu-20260720` (78 Mo) — supprimable
une fois le correctif confirmé en usage réel.
