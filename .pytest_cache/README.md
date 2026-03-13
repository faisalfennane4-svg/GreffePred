# 🩺 GreffePred — Prédiction du succès des greffes de moelle osseuse pédiatriques

> **Coding Week 09-15 Mars 2026 — Centrale Casablanca**
> Application d'aide à la décision médicale avec ML explicable (SHAP)

---

## 🚀 Démarrage rapide

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Entraîner le modèle (crée data/best_model.pkl)
python src/train_model.py

# 3. (Optionnel) Générer le SHAP summary plot
python src/evaluate_model.py

# 4. Lancer l'application
streamlit run app/app.py
```

> ⚠️ **Important** : toujours lancer les commandes depuis la **racine du projet** (là où se trouve ce README).

---

## 🗂️ Architecture

```
project/
├── data/
│   ├── bone-marrow.arff       # Dataset brut UCI
│   ├── bone_marrow.csv        # Dataset converti en CSV
│   ├── best_model.pkl         # Modèle entraîné (généré)
│   ├── feature_names.pkl      # Noms des features (généré)
│   └── shap_summary.png       # SHAP plot global (généré)
├── notebooks/
│   └── eda.ipynb              # Analyse exploratoire complète
├── src/
│   ├── data_processing.py     # Pipeline prétraitement + optimize_memory()
│   ├── train_model.py         # Comparaison et entraînement des modèles
│   └── evaluate_model.py      # Métriques + SHAP
├── app/
│   └── app.py                 # Interface Streamlit
├── tests/
│   └── test_data_processing.py
├── .github/workflows/
│   └── ci.yml                 # Pipeline CI/CD GitHub Actions
├── conftest.py                # Configuration pytest
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 📊 Réponses aux questions critiques

### 1. Le dataset était-il équilibré ?

**Non**, légèrement déséquilibré :
- Classe 0 (décès) : **102 patients (54.5%)**
- Classe 1 (survie) : **85 patients (45.5%)**
- Ratio : **1.2×**

**Points notables :**
- **81 valeurs manquantes** sur 12 colonnes (`extcGvHD` : 31 NaN, `CMVstatus` : 16 NaN)
- **Valeurs 1000000** = données censurées dans les colonnes temporelles → remplacées par NaN

**Stratégie** : SMOTE + `class_weight='balanced'` dans les modèles  
**Impact** : Meilleur rappel sur la classe minoritaire, ROC-AUC amélioré de ~5%

---

### 2. Quel modèle a obtenu les meilleures performances ?

| Modèle        | ROC-AUC | Accuracy | Précision | Rappel | F1-score |
|---------------|---------|----------|-----------|--------|----------|
| Random Forest | 0.78    | 0.76     | 0.77      | 0.79   | 0.78     |
| XGBoost       | 0.80    | 0.78     | 0.79      | 0.80   | 0.79     |
| SVM           | 0.75    | 0.73     | 0.74      | 0.75   | 0.74     |
| **LightGBM**  | **0.82**| **0.80** | **0.81**  | **0.82**| **0.81**|

✅ **LightGBM** sélectionné pour son meilleur ROC-AUC en validation croisée.

---

### 3. Quelles features influencent le plus les prédictions (SHAP) ?

1. **CD34kgx10d6** — dose cellules CD34+ par kg (impact fort positif)
2. **Rbodymass** — masse corporelle du receveur
3. **survival_time** — temps de survie
4. **HLAmatch** — compatibilité HLA donneur/receveur
5. **Donorage** — âge du donneur
6. **Disease** — type de maladie (ALL, AML, etc.)
7. **CMVstatus** — statut CMV combiné

---

### 4. Prompt Engineering — Tâche documentée

**Tâche choisie** : Fonction `optimize_memory(df)`

**Prompt initial :**
```
Écris une fonction Python `optimize_memory(df)` qui prend un DataFrame pandas
et réduit son empreinte mémoire en convertissant float64 en float32 et int64 en int32.
Retourne une copie sans modifier l'original. Ajoute une docstring claire.
```

**Problème identifié** : La première version modifiait aussi les colonnes object/category,
provoquant des erreurs de type.

**Prompt amélioré :**
```
Même fonction, mais en ajoutant ces contraintes :
- Opérer UNIQUEMENT sur les colonnes numériques (float64, int64)
- Ignorer complètement les colonnes de type object, category, bool
- Ne jamais modifier le DataFrame original (utiliser .copy())
- Ajouter un paramètre optionnel exclude_cols=[] pour exclure des colonnes spécifiques
```

**Résultat** : Code directement utilisable, sans modification.
**Enseignement** : Spécifier explicitement ce que la fonction ne doit PAS faire
est aussi important que ce qu'elle doit faire.

---

## 🧪 Tests

```bash
# Depuis la racine du projet
pytest tests/ -v
```

Tests inclus :
- Gestion des valeurs manquantes (imputation)
- `optimize_memory()` — réduction mémoire, types, conservation des valeurs
- Encodage des variables catégorielles
- Gestion des outliers (IQR)
- Chargement et prédiction du modèle

---

## 🐳 Docker

```bash
docker build -t greffepred .
docker run -p 8501:8501 greffepred
# → http://localhost:8501
```

---

## ✅ Checklist des livrables

- [x] Code structuré et professionnel
- [x] EDA documentée (notebooks/eda.ipynb)
- [x] Gestion déséquilibre (SMOTE + class_weight)
- [x] 4 modèles comparés (RandomForest, XGBoost, SVM, LightGBM)
- [x] SHAP intégré (summary plot + explications locales)
- [x] Interface Streamlit intuitive
- [x] CI/CD GitHub Actions (tests + Docker)
- [x] `optimize_memory()` implémentée et testée
- [x] Tests automatisés pytest (13 tests)
- [x] Dockerfile
- [x] Prompt engineering documenté
- [x] Reproductible avec 3 commandes

---

*Centrale Casablanca — Coding Week Mars 2026*
