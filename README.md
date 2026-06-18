# Guide de lancement du projet

Ce projet simule une chaine pharmaceutique francaise avec trois agents:

- le fabricant
- le grossiste
- la pharmacie

La simulation permet:

- de generer des donnees temporelles sur plusieurs semaines
- de detecter des situations de rupture
- d'entrainer un modele de prediction de rupture a 4 semaines
- d'explorer les resultats dans un dashboard Streamlit

## Prerequis

- Python 3.10 ou plus
- `pip`

## Installation

1. Creer un environnement virtuel:

```bash
python -m venv ./.venv
```

2. Activer l'environnement virtuel:

```bash
.\.venv\Scripts\Activate.ps1
```

3. Installer les dependances:

```bash
pip install -r ./requirements.txt
```

## Lancer la simulation

Executer le point d'entree principal:

```bash
python main.py
```

Par defaut, cette commande:

- lance une simulation sur 104 periodes
- affiche un resume dans le terminal
- genere un graphique dans `images/resultats_simulation_v3.png`
- exporte un dataset de simulation dans `data/dataset_simulation_ml.csv`

### Parametres utiles

```bash
python main.py --n-periodes 156 --d0 100 --sigma 15 --seed 42
```

- `--n-periodes` definit la duree de la simulation
- `--d0` definit la demande moyenne
- `--sigma` definit la variabilite de la demande
- `--seed` fixe la graine aleatoire pour reproduire les resultats

## Generer un dataset Monte Carlo

Pour produire un dataset plus large pour le machine learning:

```bash
python main.py --generate-dataset --mc-simulations 1000
```

Le fichier est sauvegarde dans:

```bash
data/dataset_xgboost.csv
```

## Entrainer le modele

Le dashboard peut entrainer et sauvegarder un modele de prediction de rupture a partir du dataset genere.

Fichier modele:

```bash
models/rupture_model.pkl
```

Si `xgboost` est disponible, il est utilise en priorite. Sinon, le projet bascule sur un modele logistique simple de secours.

## Lancer le dashboard

L'application Streamlit peut etre lancee avec:

```bash
streamlit run app.py
```

Ou directement:

```bash
streamlit run dashboard/app.py
```

Le dashboard permet:

- de generer le dataset depuis l'interface
- d'entrainer le modele
- de charger un modele deja sauvegarde
- de visualiser les predictions sur une simulation
- de tester une prediction manuelle

## Fichiers produits

- `data/dataset_simulation_ml.csv` : dataset issu d'une simulation unique
- `data/dataset_xgboost.csv` : dataset Monte Carlo pour l'entrainement
- `images/resultats_simulation_v3.png` : graphique de synthese
- `models/rupture_model.pkl` : modele sauvegarde
- `logs/dashboard.log` : journal du dashboard

## Arborescence utile

```text
.
|-- app.py
|-- main.py
|-- ml_utils.py
|-- dashboard/
|   `-- app.py
|-- data/
|-- images/
|-- logs/
`-- models/
```

## Notes

- Les donnees sont generees de facon aleatoire, mais la graine `--seed` permet de reproduire une simulation.
- Le projet est concu pour fonctionner meme sans `xgboost`, grace au modele de secours.
- Si le graphique ne se genere pas, verifier que `matplotlib` est bien installe.
