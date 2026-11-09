# P5 — Pipeline MLOps complet avec DVC et MLflow

Prévision de la demande d'un distributeur : un pipeline reproductible qui va des
données brutes jusqu'à une API de prévision, avec versionnage des données (DVC),
suivi des expériences et registre de modèles (MLflow), conteneurs (Docker) et
intégration continue (GitHub Actions + CML).

Projet de fin de module DevOps et MLOps — ENSET, 2026-2027.

## Relancer le projet

Prérequis : Git et Docker (avec Docker Compose). Rien d'autre.

```bash
git clone <URL-DU-DEPOT> && cd <DEPOT>
docker compose up --build
```

Cette commande unique :

1. construit l'image ;
2. démarre le serveur MLflow (suivi + registre) ;
3. récupère les données brutes depuis le remote DVC (`dvc pull`), puis exécute tout le pipeline (`dvc repro`) ;
4. démarre l'API avec le modèle qui porte l'alias `production`.

| Service | Adresse |
|---|---|
| MLflow (expériences, registre) | http://localhost:5000 |
| API — documentation interactive | http://localhost:8000/docs |
| API — métriques Prometheus | http://localhost:8000/metrics |

Exemple d'appel :

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"store":1,"item":3,"date":"2018-01-15","history":[30,28,35,31,29,40,44,30,27,33,32,30,41,45,31,29,34,30,28,42,46,32,30,35,31,29,43,47]}'
```

Tout arrêter et repartir de zéro (efface aussi le registre MLflow) :

```bash
docker compose down -v
```

## Architecture

```
params.yaml ──┐
              ▼
 train.csv → prepare → featurize → train → evaluate → promote       (dvc.yaml)
 (Kaggle,      │          │         │         │          │
  via DVC) data/processed data/features       │          └─ alias « production »
                                    │         │             dans le registre MLflow
                                    └── run MLflow : paramètres, métriques,
                                        modèle, commit Git, empreinte des données
                                                                     │
                                              API FastAPI ◄──────────┘
                                   /predict  /health  /ready  /metrics  /reload
```

| Étape DVC | Rôle | Sortie |
|---|---|---|
| — | Données brutes Kaggle, suivies par `data/raw/train.csv.dvc` | `data/raw/train.csv` |
| `prepare` | Dédoublonnage, imputation des ventes manquantes ou négatives | `data/processed/sales_clean.csv` |
| `featurize` | Retards, moyennes glissantes, calendrier ; découpage chronologique | `data/features/{train,test}.csv` |
| `train` | Entraînement, journalisation MLflow, enregistrement d'une version dans le registre | `models/model.pkl`, `models/run.json` |
| `evaluate` | RMSE, MAE, WAPE sur la période de test, comparaison au modèle naïf | `metrics/metrics.json`, figure |
| `promote` | Applique la règle de promotion | `metrics/promotion.json` |

**Règle de promotion** (`params.yaml > promote`) : une version reçoit l'alias
`production` seulement si son RMSE de test est sous le seuil `max_rmse` **et**
meilleur que celui du modèle actuellement en production. Retour arrière :
déplacer l'alias vers une version précédente dans MLflow, puis `POST /reload`.

**Modèles disponibles** (`params.yaml > train.model`) : `naive` (ventes de la
semaine précédente, sert de référence), `ridge`, `random_forest`, `gbr`.

## Structure du dépôt

```
src/data/prepare.py                nettoyage des données brutes
src/data/sample.py                 jeu synthétique de substitution (tests, essais sans le fichier Kaggle)
src/features.py                    variables (partagées par le pipeline et l'API)
src/models.py, train.py, evaluate.py, promote.py
src/api/main.py                    API FastAPI
tests/unit/                        tests du code de préparation, des variables, de l'API
tests/model/                       tests de validation du modèle entraîné
dvc.yaml, params.yaml              pipeline et paramètres
requirements.txt, constraints.txt  dépendances directes et versions figées de tout l'arbre
Dockerfile, docker-compose.yml     image multi-étapes et services
.github/workflows/ci.yml           intégration continue
```

## Travailler en local (sans Docker)

Python 3.12.

```bash
python -m venv .venv
```

```bash
.venv/Scripts/pip install -r requirements-dev.txt -c constraints.txt
```

(sous Linux/macOS : `.venv/bin/pip`, et activer l'environnement avant la suite.)

```bash
dvc repro
```

```bash
pytest
```

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Sans serveur, MLflow écrit dans `mlflow.db` et `mlruns/` (ignorés par Git).
Lancer une expérience : modifier `params.yaml`, puis `dvc repro` ; seules les
étapes concernées sont rejouées. `dvc metrics diff` compare avec le dernier commit.

## Variables d'environnement

Aucune n'est obligatoire et le projet n'utilise aucun secret. Voir `.env.example`.

| Variable | Défaut | Rôle |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` (dans Compose : `http://mlflow:5000`) | Serveur de suivi et registre |
| `MODEL_NAME` | `demand-forecast` | Nom du modèle dans le registre |
| `MLFLOW_PORT` | `5000` | Port publié du serveur MLflow |
| `API_PORT` | `8000` | Port publié de l'API |

## Intégration continue

À chaque push sur `main` et à chaque pull request (`.github/workflows/ci.yml`) :

1. `ruff` et tests unitaires ;
2. `dvc repro`, tests de validation du modèle, puis rapport CML (métriques,
   figure, écart avec `main`) publié en commentaire de la pull request ;
3. construction de l'image Docker.
