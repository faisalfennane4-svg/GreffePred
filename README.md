# GreffePred — version corrigée

Cette refonte corrige les problèmes principaux du projet initial :

- l'interface demandait des variables redondantes ou incohérentes ;
- le modèle utilisait des **variables de fuite** (`survival_time`, `ANCrecovery`, etc.) ;
- la prédiction pouvait sembler presque constante parce que les mauvaises variables dominaient ;
- aucune vraie base de données locale n'enregistrait les dossiers saisis.

## Ce qui a été corrigé

### 1. Variables de fuite retirées du modèle
Les colonnes suivantes sont **exclues** de l'entraînement :

- `IIIV`
- `Relapse`
- `aGvHDIIIIV`
- `extcGvHD`
- `ANCrecovery`
- `PLTrecovery`
- `time_to_aGvHD_III_IV`
- `survival_time`

### 2. Interface métier plus logique
Le médecin saisit uniquement des informations cohérentes :

- données du receveur ;
- données du donneur ;
- compatibilité immunologique ;
- paramètres de la greffe.

Les variables dérivées sont calculées automatiquement :

- `Donorage35`
- `Recipientage10`
- `Recipientageint`
- `ABOmatch`
- `CMVstatus`
- `Diseasegroup`
- `HLAmismatch`
- `Gendermatch`

### 3. Connexion médecin
Une fenêtre de connexion a été ajoutée.

**Compte de démonstration**
- utilisateur : `medecin`
- mot de passe : `Greffe2026!`

### 4. Historique Excel
Chaque prédiction est enregistrée dans :

`data/user_predictions.xlsx`

Le classeur contient :

- une feuille `predictions` avec toutes les saisies et le résultat ;
- une feuille `legend` qui explique les variables dérivées.

### 5. Tests ajoutés
Quatre familles de tests sont incluses :

- **tests des données** ;
- **tests des valeurs entrantes utilisateur / aberrations** ;
- **tests de sélection du modèle** ;
- **tests de journalisation Excel**.

## Structure

```text
greffepred_refonte/
├── app/
│   └── app.py
├── src/
│   ├── data_processing.py
│   ├── input_validation.py
│   ├── prediction_logger.py
│   └── train_model.py
├── tests/
│   ├── test_data_processing.py
│   ├── test_input_validation.py
│   ├── test_model_selection.py
│   └── test_prediction_logger.py
├── data/
│   ├── bone_marrow.csv
│   ├── bone-marrow.arff
│   ├── best_model.joblib
│   ├── model_comparison.csv
│   ├── global_feature_importance.csv
│   ├── background_reference.csv
│   ├── training_summary.json
│   └── user_predictions.xlsx
├── requirements.txt
├── Dockerfile
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
python src/train_model.py
streamlit run app/app.py
```

## Lancer les tests

```bash
pytest tests -v
```

## Ce que l'interface montre en plus maintenant

- une **connexion médecin** ;
- une **prédiction enregistrée automatiquement** dans Excel ;
- une **explication locale** des variables qui poussent la décision ;
- une **courbe de sensibilité à l'âge du receveur** pour vérifier que la probabilité n'est plus figée ;
- un **tableau de comparaison des modèles** ;
- un **historique des dossiers saisis**.

## Remarque importante
Le dataset reste petit et le signal prédictif reste modeste après suppression des fuites.  
Cette version privilégie donc la **cohérence clinique**, la **traçabilité** et la **qualité de l'interface** plutôt qu'un score artificiellement gonflé par des variables post-greffe.
