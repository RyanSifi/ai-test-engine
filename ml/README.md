# Module ML — Risk Predictor

Module de prédiction supervisée du risque de régression par méthode de
controller Symfony, intégré au AI Test Engine pour prioriser la génération
de tests fonctionnels.

## Architecture

```
git history + code source PHP
         │
         ▼
  extract_dataset.py   ──►  CSV (features + label_risk)
         │
         ▼
     train.py          ──►  best_model.pkl + métriques
         │
         ▼
  streamlit_app.py     ──►  Dashboard MOA
         │
         ▼
  AI Test Engine API   ──►  Tests PHP générés (mode déterministe)
```

## Méthode

### Label : bugfix proxy depuis git
Une méthode est **à risque** si son fichier a été modifié par au moins
N commits de bugfix dans la fenêtre temporelle. Méthodologie issue de la
littérature en software defect prediction (Hassan 2009, Kamei et al. 2013).

### Features (22)

| Catégorie | Features |
|-----------|----------|
| Routes    | `nb_routes_method`, `has_route_attr` |
| Structure | `nb_params`, `has_form`, `is_ajax_only`, `nb_voter_checks`, `nb_response_types`, `has_render`, `has_redirect`, `has_json`, `has_file_download`, `nb_method_grants`, `nb_class_grants`, `nb_constructor_deps`, `nb_methods_in_class`, `is_invoke` |
| Complexité| `file_loc`, `method_loc`, `cyclomatic_complexity` |
| Git       | `git_total_commits`, `git_nb_authors`, `git_days_since_change` |

`git_bugfix_count` est **exclu des features** car il sert à calculer le label
(prévention de data leakage).

### Modèles comparés

- **LogisticRegression** : baseline interprétable, GridSearch sur `C`
- **RandomForest** : robuste, GridSearch sur `n_estimators`, `max_depth`, `min_samples_leaf`
- **XGBoost** : `scale_pos_weight` pour gérer le déséquilibre

## Installation

```bash
pip install -r ml/requirements.txt
```

## Pipeline complet

### 1. Extraire un dataset
```bash
PYTHONIOENCODING=utf-8 python ml/extract_dataset.py \
    _dataset/sylius _dataset/dataset.csv \
    --since "36 months ago" --bugfix-threshold 1
```

### 2. Entraîner les modèles
```bash
PYTHONIOENCODING=utf-8 python ml/train.py _dataset/dataset.csv
```

Sortie : `ml/models/best_model.pkl`, `feature_names.json`, `metrics.json`,
`feature_importance.csv`, matrices de confusion PNG.

### 3. Lancer le dashboard
```bash
streamlit run ml/streamlit_app.py
```
→ http://localhost:8501

Le dashboard se connecte à `http://localhost:8000` (AI Test Engine) pour
la génération de tests. Configurer via :
```bash
export TEST_ENGINE_URL=http://localhost:8000
export TEST_ENGINE_API_KEY=xxx   # si l'auth est activée
```

## Résultats actuels (Sylius, 36m, threshold=1)

| Modèle | Accuracy | F1 | AUC |
|--------|----------|------|------|
| LogReg     | 0.873 | 0.696 | 0.958 |
| RF         | 0.945 | 0.823 | 0.994 |
| **XGBoost**| **0.945** | **0.857** | 0.989 |

**Top features (RandomForest)** :
1. `git_total_commits`     (25.7%)
2. `git_nb_authors`        (23.0%)
3. `file_loc`              (11.8%)
4. `git_days_since_change` (8.8%)
5. `nb_methods_in_class`   (8.5%)

→ Cohérent avec la littérature : un fichier modifié par beaucoup d'auteurs
récemment et de grande taille est un meilleur prédicteur que les features
structurelles seules.

## Datasets supportés

| Projet | Cloner avec | Notes |
|--------|-------------|-------|
| Sylius | `git clone --depth=5000 https://github.com/Sylius/Sylius.git` | Dataset principal |
| Akeneo PIM | `git clone --depth=3000 https://github.com/akeneo/pim-community-dev.git` | History plat (peu de signal) |
| Symfony Demo | `git clone --depth=2000 https://github.com/symfony/demo.git` | Trop petit |

## Limites

- **Petit dataset** (~183 lignes Sylius) — résultats à généraliser avec prudence
- **Label proxy** — un commit "fix:" n'est pas un vrai test failure
- **Pas de séparation temporelle** (train/test mélangé) — pour un vrai usage
  prod, faire un split temporel (anciennes versions = train, dernière = test)
- **Features git au niveau fichier** — pas de granularité méthode
