# GreffePred — Aide à la Décision Médicale 
   
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

**Prédiction du succès des greffes de moelle osseuse pédiatriques**

Machine Learning · SHAP · Interface Médicale Streamlit

Ecole Centrale Casablanca — Coding Week — Mars 2026
 
---
   
## Equipe

| Nom | Prénom |
|-----|--------|
| FENNANE | Faisal |
| JANNEN | Ahlem |
| MAAYZOU | Ikram |
| EL AMRANI | Aberrazak |
| SAME | Abel |

---

## Description

GreffePred est une application d'aide à la décision clinique destinée aux médecins spécialisés en greffe de moelle osseuse pédiatrique. Elle prédit la probabilité de succès d'une greffe à partir des données pré-opératoires du patient et du donneur, en s'appuyant sur des algorithmes de Machine Learning entraînés sur le dataset [Bone Marrow Transplant Children (UCI, 187 patients)](https://archive.ics.uci.edu/dataset/565/bone+marrow+transplant+children).

### Fonctionnalités principales

- Connexion médecin sécurisée avec authentification
- Formulaire clinique complet avec validation des saisies en temps réel
- Jauge interactive Plotly — probabilité de survie en temps réel
- Explication SHAP locale — facteurs qui influencent chaque prédiction
- Courbe de sensibilité à l'âge — analyse de l'impact de l'âge du receveur
- Comparaison des modèles ML — RandomForest, ExtraTrees, GradientBoosting
- Historique Excel — chaque prédiction est sauvegardée et téléchargeable

---

## Installation et Lancement

### Prérequis
- Python 3.9+

### Etape 1 — Installer les dépendances
```bash
pip install -r requirements.txt
```

### Etape 2 — Entraîner le modèle
```bash
python src/train_model.py
```

### Etape 3 — Lancer l'application
```bash
streamlit run app/app.py
```

### Connexion
| Champ | Valeur |
|-------|--------|
| Utilisateur | `medecin` |
| Mot de passe | `Greffe2026!` |

---

## Structure du projet

```
GreffePred/
├── .github/
│   └── workflows/
│       └── ci.yml
├── app/
│   └── app.py
├── src/
│   ├── data_processing.py
│   ├── input_validation.py
│   ├── prediction_logger.py
│   └── train_model.py
├── data/
│   ├── bone_marrow.csv
│   ├── bone-marrow.arff
│   ├── best_model.joblib
│   ├── model_comparison.csv
│   ├── global_feature_importance.csv
│   ├── background_reference.csv
│   ├── training_summary.json
│   └── user_predictions.xlsx
├── tests/
│   ├── test_data_processing.py
│   ├── test_input_validation.py
│   ├── test_model_selection.py
│   └── test_prediction_logger.py
├── __pycache__/
├── .pytest_cache/
├── conftest.py
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Modèles Machine Learning

| Modèle | Description |
|--------|-------------|
| RandomForest | Forêt aléatoire avec class_weight=balanced |
| ExtraTrees | Arbres extrêmement aléatoires |
| GradientBoosting | Boosting par gradient |

Le meilleur modèle est sélectionné automatiquement par validation croisée (5-fold, métrique : ROC-AUC).

### Variables utilisées

Les variables post-greffe sont volontairement exclues pour éviter la fuite d'information :

Variables exclues : `survival_time`, `ANCrecovery`, `PLTrecovery`, `Relapse`, `aGvHDIIIIV`

Variables utilisées : `Recipientage`, `Donorage`, `HLAmatch`, `CD34kgx10d6`, `Disease`, `CMVstatus`

---

## Tests

```bash
pytest tests -v
```

4 familles de tests :
- Traitement des données
- Validation des saisies utilisateur
- Sélection du modèle
- Journalisation Excel

---

## Docker

```bash
docker build -t greffepred .
docker run -p 8501:8501 greffepred
```

---

## Dataset

| Attribut | Valeur |
|----------|--------|
| Source | UCI Machine Learning Repository |
| Patients | 187 |
| Variables | 37 |
| Cible | `survival_status` (0/1) |
| Lien | [UCI Dataset #565](https://archive.ics.uci.edu/dataset/565/bone+marrow+transplant+children) |

---

## Avertissement clinique

Cette application est un outil d'aide à la décision destiné à un usage académique et de recherche. Elle ne remplace en aucun cas le jugement clinique d'un médecin spécialisé. Toute décision médicale doit être prise par des professionnels de santé qualifiés.

---

Ecole Centrale Casablanca — Coding Week — Mars 2026 — Projet 4

---

## Guide d'utilisation

### Etape 1 — Connexion
Lancez l'application et connectez-vous avec les identifiants medecin / demo1234.

### Etape 2 — Saisie du dossier clinique
Dans la barre laterale gauche, remplissez les informations du patient reparties en 5 sections :

- Donnees du receveur : age, poids, sexe, groupe sanguin, statut CMV
- Donnees du donneur : age, sexe, groupe sanguin, statut CMV
- Compatibilite immunologique : compatibilite HLA, differences antigeniques et alleliques
- Parametres de la greffe : source des cellules souches, dose CD34+, dose CD3+
- Informations administratives : identifiant patient, note clinique libre

Les variables derivees (ABOmatch, CMVstatus, HLAmismatch, etc.) sont calculees automatiquement.

### Etape 3 — Lancer l'analyse
Cliquez sur le bouton **Analyser le dossier** dans l'onglet **Nouveau dossier**.

### Etape 4 — Interpreter le resultat
Rendez-vous dans l'onglet **Resultat clinique** :

- La jauge affiche la probabilite de survie de 0 a 100%
- Le niveau de risque est indique : Favorable (>= 70%), Intermediaire (45-70%), Eleve (< 45%)
- Le graphique SHAP montre les 10 variables qui ont le plus influence la prediction
- La courbe de sensibilite montre comment la probabilite evolue selon l'age du receveur

### Etape 5 — Consulter les analyses globales
L'onglet **Analyses du modele** presente :

- Le tableau comparatif des modeles ML avec leur AUC respectif
- L'importance globale des variables sur l'ensemble du dataset

### Etape 6 — Historique
L'onglet **Historique Excel** affiche toutes les predictions enregistrees. Vous pouvez telecharger le fichier Excel complet via le bouton de telechargement.
