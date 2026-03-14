# Explication détaillée du projet corrigé
 
Ce document explique **fichier par fichier** la logique du projet refait.  
Pour rester lisible, l'explication est faite par **petits blocs de lignes** plutôt que par une phrase isolée pour chacune des 1700+ lignes.  
Chaque bloc correspond à quelques lignes contiguës qui travaillent ensemble.

---

## 1. `src/data_processing.py`

### Lignes 1 à 8
Le bloc d'ouverture annonce l'objectif du module : préparer les données et, surtout, **séparer proprement** les variables de saisie, les variables dérivées et les variables de fuite. Cette idée est la base de toute la correction du projet.

### Lignes 10 à 19
Ces lignes importent les bibliothèques nécessaires :
- `Path` sert à construire des chemins robustes ;
- `Any` permet de typer des valeurs génériques ;
- `numpy` et `pandas` servent à manipuler les données ;
- `ColumnTransformer`, `SimpleImputer`, `Pipeline` et `OneHotEncoder` servent à construire le pipeline sklearn.

Autrement dit, ces lignes installent le “socle technique” du prétraitement.

### Lignes 22 à 24
`ROOT_DIR` et `DEFAULT_DATA_PATH` calculent automatiquement le chemin du projet et du CSV principal.  
Le but est simple : éviter les chemins écrits en dur comme `C:/...` qui cassent dès qu'on change de machine.

### Lignes 25 à 32
`TARGET_COLUMN` fixe la cible à prédire : `survival_status`.  
`MISSING_TOKENS` regroupe les marqueurs de données manquantes qui peuvent être présents dans les fichiers.  
Dans la pratique, ces constantes évitent d'avoir des comparaisons répétées dans tout le code.

### Lignes 34 à 43
`POST_TRANSPLANT_COLUMNS` est une liste **très importante**.  
Elle contient toutes les colonnes qui ne doivent **pas** entrer dans le modèle parce qu'elles décrivent l'après-greffe ou l'issue clinique :
- `IIIV`
- `Relapse`
- `aGvHDIIIIV`
- `extcGvHD`
- `ANCrecovery`
- `PLTrecovery`
- `time_to_aGvHD_III_IV`
- `survival_time`

C'est précisément le cœur de la correction : on retire ce qui faisait “tricher” le modèle.

### Lignes 45 à 46
`FORM_ONLY_COLUMNS` recense les champs utilisés par l'interface mais **pas** par le modèle :
- `PatientID`
- `ClinicalNote`
- `DonorGender`

`PatientID` et `ClinicalNote` servent à l'historique Excel.  
`DonorGender` sert seulement à calculer `Gendermatch`.

### Lignes 48 à 77
`MODEL_FEATURE_COLUMNS` fixe l'ordre exact des variables utilisées dans le modèle.  
Cette liste joue trois rôles :

1. elle documente noir sur blanc les variables conservées ;
2. elle garantit le même ordre entre entraînement et prédiction ;
3. elle évite qu'une colonne parasite entre dans le pipeline.

On retrouve ici :
- les vraies variables cliniques de départ ;
- les variables dérivées utiles (`Donorage35`, `ABOmatch`, `CMVstatus`, etc.) ;
- mais **aucune** variable post-greffe.

### Lignes 79 à 101
`FRIENDLY_FEATURE_NAMES` transforme les noms techniques en libellés lisibles.  
C'est utile pour deux usages :
- l'explication locale affichée dans l'application ;
- la lecture des importances de variables.

Sans ce dictionnaire, l'utilisateur verrait des noms bruts comme `Recipientage10` ou `CD3dkgx10d8`, ce qui alourdirait l'interface.

### Lignes 104 à 118 — `optimize_memory`
Cette fonction prend un DataFrame et retourne une copie plus légère :
- les `float` sont réduits vers des types plus petits ;
- les `int` aussi ;
- les colonnes éventuellement exclues restent intactes.

L'idée n'est pas d'optimiser à l'extrême, mais d'avoir une fonction propre, testable, et conforme à ce que demandait le projet.

### Lignes 121 à 131 — `_recipient_age_bucket`
Cette fonction transforme l'âge du receveur en classe d'âge :
- `0` si l'âge est inférieur ou égal à 5 ;
- `1` s'il est entre 5 et 10 ;
- `2` au-delà.

Cette logique recrée `Recipientageint` sans demander au médecin de la saisir à la main.  
C'est une vraie amélioration métier : la variable est dérivée automatiquement, donc elle ne peut plus être contradictoire avec `Recipientage`.

### Lignes 134 à 155 — `_cmv_status`
Cette fonction reconstruit le code `CMVstatus` à partir de deux entrées simples :
- `DonorCMV`
- `RecipientCMV`

Le mapping est clairement explicitement codé :
- D-/R- → 0
- D+/R- → 1
- D-/R+ → 2
- D+/R+ → 3

Ce bloc est essentiel car il élimine un autre problème de l'ancien projet : demander à l'utilisateur **et** les données sources **et** la variable combinée.

### Lignes 158 à 166 — `_disease_group`
Cette fonction convertit la maladie détaillée en groupe binaire :
- `1` si la maladie est maligne ;
- `0` si elle est non maligne.

L'intérêt est double :
- le code garde la granularité `Disease` ;
- le modèle bénéficie aussi de la vue agrégée `Diseasegroup`.

### Lignes 168 à 175 — `load_data`
Ce bloc charge le CSV, traite les marqueurs de valeurs manquantes et nettoie les noms de colonnes.  
Le nettoyage des colonnes est utile car certains CSV contiennent des espaces parasites en tête ou en fin.

### Lignes 178 à 219 — `derive_clinical_features`
C'est l'une des fonctions les plus importantes du projet.

Elle :
- recalcule `Donorage35` depuis `Donorage` ;
- recalcule `Recipientage10` et `Recipientageint` depuis `Recipientage` ;
- recalcule `ABOmatch` depuis `DonorABO` et `RecipientABO` ;
- recalcule `CMVstatus` depuis `DonorCMV` et `RecipientCMV` ;
- recalcule `Diseasegroup` depuis `Disease` ;
- recalcule `HLAmismatch` depuis `HLAmatch` ;
- recalcule `Gendermatch` si `DonorGender` est disponible.

La logique métier du projet corrigé est là :
on ne demande plus au médecin des colonnes déjà déductibles.

### Lignes 222 à 232 — `validate_dataset_schema`
Cette fonction vérifie la structure du dataset :
- toutes les features attendues existent ;
- la cible est bien binaire.

C'est un garde-fou de qualité.  
Si le fichier change, le projet lève une erreur claire au lieu de produire des résultats silencieusement faux.

### Lignes 235 à 253 — `prepare_model_dataframe`
Cette fonction applique la vraie politique de nettoyage métier :
1. elle dérive les variables calculables ;
2. elle supprime les colonnes post-greffe ;
3. elle vérifie que les features finales sont toutes présentes ;
4. elle remet les colonnes dans l'ordre prévu.

C'est le “sas” entre les données brutes et le modèle.

### Lignes 256 à 268 — `build_training_data`
Ici, on enchaîne les étapes nécessaires à l'entraînement :
- charger ;
- préparer ;
- valider ;
- séparer `X` et `y` ;
- optimiser la mémoire.

Le résultat final est un couple `(feature_frame, target)` prêt à être envoyé dans sklearn.

### Lignes 271 à 297 — `build_preprocessor`
Cette fonction construit le préprocesseur sklearn :
- colonnes numériques → imputation par médiane ;
- colonnes catégorielles → imputation par mode + one-hot encoding.

Le choix est cohérent pour ce dataset :
- on garde les variables numériques telles quelles ;
- on ne transforme `Disease` qu'au moment du pipeline, proprement.

### Lignes 300 à 335 — `make_prediction_frame`
Cette fonction est le miroir “temps réel” de l'entraînement.  
Elle prend les valeurs du formulaire Streamlit et les transforme en DataFrame **strictement compatible** avec le modèle.

Le point clé :
- le médecin ne saisit que les champs cliniquement logiques ;
- le code recrée ensuite les colonnes dérivées ;
- la sortie finale respecte `MODEL_FEATURE_COLUMNS`.

C'est cette symétrie qui évite les prédictions absurdes.

### Lignes 338 à 371 — `dataset_quality_summary`
Cette fonction produit un résumé lisible du dataset :
- nombre de lignes ;
- nombre de colonnes ;
- distribution de la cible ;
- colonnes avec valeurs manquantes ;
- variables de fuite retirées ;
- liste des features utilisées.

Elle alimente directement l'onglet “Analyses du modèle” de l'application.

### Lignes 374 à 385 — `humanize_transformed_feature_name`
Quand sklearn passe par `ColumnTransformer`, les noms deviennent techniques :
- `num__Recipientage`
- `cat__Disease_ALL`

Cette fonction retransforme ces noms en libellés utilisables dans l'interface.  
C'est un détail important pour que l'explication locale soit compréhensible par un humain.

---

## 2. `src/input_validation.py`

### Lignes 1 à 7
Le docstring fixe la philosophie du module : empêcher les saisies impossibles et repérer les valeurs aberrantes.  
Cette couche répond directement à votre demande de “test pour les valeurs entrantes du user”.

### Lignes 10 à 23
Les dictionnaires `ABO_LABELS` et `BINARY_LABELS` servent de référentiel d'affichage :
- les codes internes restent compacts ;
- l'interface montre des mots compréhensibles.

### Lignes 25 à 214 — `FORM_SCHEMA`
`FORM_SCHEMA` est la “description officielle” du formulaire.  
Chaque champ y possède :
- un libellé ;
- un groupe ;
- un type de widget ;
- une valeur par défaut ;
- parfois des bornes ;
- parfois des options de sélection.

Ce bloc centralise la logique de l'interface.  
Le bénéfice est majeur : l'application, la validation et l'explication reposent sur la **même définition des champs**.

### Lignes 28 à 40
Ces lignes décrivent les champs administratifs :
- `PatientID`
- `ClinicalNote`

Ils ne servent pas à la prédiction, mais à la traçabilité et à l'historique Excel.

### Lignes 41 à 86
Ce segment définit les champs “patient” :
- sexe du receveur ;
- sexe du donneur ;
- âge du receveur ;
- poids ;
- maladie ;
- groupe de risque ;
- deuxième greffe après rechute.

C'est ici que l'interface devient plus réaliste pour un médecin :
les libellés sont lisibles et les bornes sont posées dès la définition du champ.

### Lignes 87 à 104
Ici, on décrit les données du donneur :
- source des cellules souches ;
- âge du donneur.

Le projet initial traitait ces informations de façon plus brute.  
La version corrigée les regroupe proprement.

### Lignes 105 à 170
Ce bloc couvre toute la compatibilité immunologique :
- ABO donneur ;
- ABO receveur ;
- Rhésus ;
- CMV donneur/receveur ;
- niveau de match HLA ;
- écarts antigéniques et alléliques ;
- type de différence HLA.

Cela correspond à une vue plus clinique du problème.

### Lignes 171 à 213
Ce bloc finit la description du formulaire avec les paramètres de greffe :
- dose CD34+ ;
- ratio CD3+/CD34+ ;
- dose CD3+.

Ces champs étaient déjà présents dans le dataset, mais ici ils sont encadrés par des bornes cohérentes.

### Lignes 216 à 223 — `schema_by_group`
Cette fonction regroupe les champs par famille (`patient`, `donneur`, `immunologie`, etc.).  
Elle permet à Streamlit d'afficher le formulaire par blocs logiques sans recopier les noms de champs à la main.

### Lignes 226 à 229 — `_range_error`
Petit helper, mais utile :
il fabrique un message d'erreur uniforme pour les nombres hors plage.
Le code reste alors plus propre dans la validation principale.

### Lignes 231 à 287 — `validate_input_record`
C'est la fonction de validation métier.

#### Lignes 237 à 265
Premier passage :
- vérifier que tous les champs existent ;
- vérifier que les `select` contiennent une valeur autorisée ;
- vérifier que les champs numériques sont bien des nombres ;
- vérifier que chaque nombre reste dans les bornes prévues.

Si une erreur est détectée, la fonction renvoie immédiatement un bloc d'erreurs.

#### Lignes 267 à 287
Deuxième passage :
on n'est plus dans l'erreur bloquante mais dans l'**alerte clinique**.

Exemples :
- receveur très âgé pour un dataset pédiatrique ;
- donneur très âgé ;
- poids extrêmement bas ;
- dose CD34+ très haute ;
- ratio CD3/CD34 élevé ;
- incohérence entre le `HLAmatch` et les détails HLA.

Ce sont précisément les “aberrations” que vous vouliez voir apparaître.

---

## 3. `src/prediction_logger.py`

### Lignes 1 à 6
Le docstring annonce clairement la mission du module : enregistrer les prédictions dans Excel.

### Lignes 8 à 14
Les imports préparent :
- la date/heure ;
- les chemins ;
- les types ;
- `pandas` pour la lecture ;
- `openpyxl` pour écrire réellement dans le classeur.

### Lignes 15 à 18
`ROOT_DIR` et `DEFAULT_LOG_PATH` définissent le chemin du classeur :
`data/user_predictions.xlsx`

### Lignes 20 à 57 — `LOG_COLUMNS`
Cette liste définit toutes les colonnes du journal :
- métadonnées (`timestamp`, `username`, `patient_id`) ;
- résultat du modèle ;
- champs saisis ;
- variables dérivées recalculées ;
- note clinique.

Le fait de loguer **aussi** les variables dérivées est très utile :
on peut revoir après coup exactement ce que le modèle a réellement vu.

### Lignes 59 à 64 — `_style_header_row`
Cette petite fonction applique un style visuel à la première ligne :
- fond bleu ;
- texte blanc ;
- gras.

Ce n'est pas seulement esthétique : le classeur devient immédiatement exploitable.

### Lignes 67 à 105 — `initialize_workbook`
Cette fonction crée le classeur s'il n'existe pas encore.

#### Lignes 74 à 82
Elle crée la feuille `predictions`, écrit les en-têtes, fige la ligne 1 et active un filtre.

#### Lignes 84 à 101
Elle crée ensuite une feuille `legend` qui explique les variables dérivées.  
C'est un vrai plus pour la lisibilité du fichier Excel.

### Lignes 108 à 113 — `_risk_label`
Cette fonction convertit une probabilité en niveau de risque :
- favorable ;
- intermédiaire ;
- élevé.

Le but est d'avoir un vocabulaire lisible à côté de la valeur numérique.

### Lignes 116 à 175 — `append_prediction`
C'est la fonction qui enregistre concrètement une prédiction.

#### Lignes 126 à 133
Elle prépare le classeur si nécessaire.

#### Lignes 135 à 172
Elle construit une ligne complète avec :
- les métadonnées ;
- la probabilité ;
- le niveau de risque ;
- les alertes ;
- tous les champs cliniques utiles ;
- les variables dérivées ;
- la note libre.

Le projet dispose donc d'une vraie trace persistante.

### Lignes 178 à 185 — `read_history`
Cette fonction recharge l'historique pour l'application Streamlit.
Elle trie les prédictions par date décroissante et permet d'afficher les plus récentes en premier.

---

## 4. `src/train_model.py`

### Lignes 1 à 6
Le docstring précise que ce module gère l'entraînement, la comparaison et la sauvegarde des artefacts.

### Lignes 8 à 20
Les imports couvrent :
- JSON pour les résumés ;
- `joblib` pour sauvegarder le pipeline ;
- `pandas` ;
- les estimateurs sklearn ;
- les fonctions utilitaires venant de `data_processing`.

### Lignes 24 à 31
Ces constantes définissent les chemins de sortie :
- modèle final ;
- tableau de comparaison ;
- résumé JSON ;
- importance globale ;
- échantillon de référence.

C'est ce qui alimente ensuite l'application.

### Lignes 34 à 55 — `make_candidate_models`
Cette fonction crée le dictionnaire des modèles candidats :
- `RandomForest`
- `ExtraTrees`
- `GradientBoosting`

Le paramètre `fast_mode` sert aux tests :
il réduit le nombre d'arbres pour accélérer la validation automatique.

### Lignes 58 à 68 — `build_model_pipeline`
Cette fonction assemble :
- le préprocesseur ;
- le modèle.

Autrement dit, elle crée un pipeline complet sklearn, réutilisable autant pour la comparaison que pour l'entraînement final.

### Lignes 71 à 112 — `compare_models`
C'est la partie qui répond à votre demande de “test pour choisir le modèle d'analyse”.

#### Lignes 78 à 82
Le code prépare la validation croisée stratifiée.

#### Lignes 84 à 107
Pour chaque modèle :
- il construit le pipeline ;
- il lance `cross_validate` ;
- il calcule les moyennes de ROC-AUC, accuracy, balanced accuracy et F1 ;
- il stocke une ligne de résultats.

#### Lignes 109 à 112
Le tableau est ensuite trié par :
1. `roc_auc_mean`
2. `balanced_accuracy_mean`
3. `accuracy_mean`

Donc le “meilleur modèle” n'est pas choisi au hasard : il est classé de manière reproductible.

### Lignes 115 à 121 — `select_best_model`
Cette fonction lit simplement la première ligne du tableau trié et retourne son nom.  
Elle isole la logique de sélection pour pouvoir la tester indépendamment.

### Lignes 124 à 145 — `export_feature_importance`
Cette fonction sauvegarde l'importance globale des variables si le modèle la fournit.

#### Lignes 131 à 134
Si le modèle n'expose pas `feature_importances_`, la fonction écrit un CSV vide mais propre.

#### Lignes 136 à 145
Sinon :
- elle récupère les noms transformés via le préprocesseur ;
- elle récupère les importances du modèle ;
- elle construit un DataFrame trié ;
- elle l'écrit dans `global_feature_importance.csv`.

### Lignes 148 à 189 — `train_and_persist`
C'est le pipeline d'entraînement complet.

#### Lignes 155 à 158
Chargement des données préparées.

#### Lignes 159 à 163
Création des candidats, comparaison des modèles, choix du meilleur.

#### Lignes 164 à 166
Réentraînement du meilleur pipeline sur toutes les données disponibles.

#### Lignes 168 à 170
Sauvegarde du modèle et du tableau comparatif.

#### Lignes 171 à 174
Export de l'importance globale et d'un échantillon de référence pour les explications locales.

#### Lignes 176 à 184
Création du résumé JSON avec :
- les infos du dataset ;
- le meilleur modèle ;
- ses métriques ;
- le nombre de features.

#### Lignes 186 à 189
Retour des chemins des artefacts générés.

### Lignes 192 à 193
Ce petit bloc permet d'exécuter le fichier directement en script.

---

## 5. `app/app.py`

### Lignes 1 à 5
Le docstring annonce qu'il s'agit de l'application Streamlit principale.

### Lignes 7 à 18
Imports généraux :
- hashage pour la connexion ;
- JSON, chemins et types ;
- `joblib`, `numpy`, `pandas` ;
- `plotly` pour les graphiques ;
- `streamlit` pour l'interface.

### Lignes 20 à 28
Ce bloc calcule les chemins applicatifs puis ajoute `src/` dans `sys.path`.  
Cela permet d'importer les modules du projet même quand Streamlit lance le fichier depuis le dossier `app/`.

### Lignes 30 à 41
Imports métiers depuis les modules du projet :
- préparation des données ;
- validation ;
- journalisation ;
- chemins des artefacts de modèle.

### Lignes 43 à 48 — `DEFAULT_USERS`
Ici, on définit un compte de démonstration.  
Le mot de passe n'est pas stocké en clair mais sous forme de hash SHA-256.  
C'est un compromis simple, propre et suffisant pour un projet académique.

### Lignes 50 à 56 — `GROUP_TITLES`
Ce dictionnaire sert à afficher des titres humains dans le formulaire :
- administratif ;
- patient ;
- donneur ;
- immunologie ;
- greffe.

### Lignes 58 à 63 — `RISK_TEXT`
Ce dictionnaire associe chaque niveau de risque à un commentaire clinique court.  
Cela rend la sortie plus utile qu'un simple pourcentage.

### Lignes 66 à 96 — fonctions `load_*`
Ces fonctions chargent :
- le pipeline ;
- la comparaison des modèles ;
- les importances ;
- le résumé JSON ;
- l'échantillon de référence.

Le décorateur `@st.cache_*` évite de relire les mêmes fichiers à chaque interaction, ce qui rend l'app plus fluide.

### Lignes 99 à 105 — `check_password`
Cette fonction vérifie les identifiants :
- on cherche l'utilisateur ;
- on recalcule le hash du mot de passe saisi ;
- on compare avec le hash stocké.

### Lignes 107 à 113 — `risk_level_from_probability`
Cette fonction convertit la probabilité de succès en trois niveaux de risque.  
Le découpage est simple, lisible, et réutilisé à plusieurs endroits.

### Lignes 115 à 131 — `render_login`
Cette fonction dessine l'écran de connexion :
- titre ;
- explication courte ;
- formulaire login/mot de passe.

Au clic :
- si l'identifiant est bon, on enregistre l'état dans `session_state` ;
- sinon, on affiche une erreur.

C'est la “fenêtre médecin” que vous aviez signalée comme manquante.

### Lignes 134 à 145 — `_metric_card`
Cette fonction fabrique une carte visuelle réutilisable.  
Le HTML est volontairement simple :
- un titre ;
- une grande valeur ;
- un sous-texte ;
- une couleur de bordure.

On l'utilise ensuite pour rendre le résultat plus propre.

### Lignes 148 à 160 — `render_sidebar`
La barre latérale affiche :
- l'utilisateur connecté ;
- un bouton de déconnexion ;
- le chemin du fichier Excel ;
- la liste des colonnes de fuite retirées.

C'est un rappel très utile de la logique de la version corrigée.

### Lignes 163 à 195 — `render_field`
Cette fonction est le moteur générique du formulaire.  
À partir de `FORM_SCHEMA`, elle décide quel widget afficher :
- `text_input`
- `text_area`
- `selectbox`
- `number_input`

Le très gros avantage est qu'on n'écrit pas 25 fois du code répétitif.

### Lignes 198 à 217 — `render_form`
Cette fonction construit le formulaire complet.

#### Lignes 201 à 214
Elle parcourt les groupes de champs et les affiche par sections.  
Les colonnes Streamlit permettent de garder un écran équilibré visuellement.

#### Lignes 216 à 217
Le booléen `_submitted` est ajouté au dictionnaire retourné pour que `main()` sache si le bouton a été pressé.

### Lignes 220 à 239 — `probability_gauge`
Cette fonction construit la jauge Plotly :
- zone rouge ;
- zone orange ;
- zone verte ;
- aiguille numérique.

Elle améliore fortement la lecture du résultat.

### Lignes 242 à 286 — `local_explanation`
C'est l'une des pièces les plus utiles de l'interface.

#### Lignes 243 à 252
Le code récupère le préprocesseur, le modèle, l'échantillon de référence, puis transforme la ligne à expliquer et le fond de référence.

#### Lignes 254 à 270
On essaie de calculer de vraies contributions locales avec SHAP.

#### Lignes 272 à 275
Si SHAP échoue, on passe sur un fallback plus simple à base d'importances globales pondérées par l'écart au fond moyen.

#### Lignes 277 à 286
Le résultat est mis dans un DataFrame trié par contribution absolue.  
C'est ce DataFrame qui alimente le graphique “top facteurs”.

### Lignes 289 à 297 — `age_sensitivity_curve`
Cette fonction répond directement à votre remarque sur la probabilité “constante”.

Elle :
- fait varier `Recipientage` de 1 à 20 ans ;
- reconstruit le DataFrame de prédiction à chaque fois ;
- recalcule la probabilité ;
- stocke les résultats.

On obtient alors une courbe qui montre si le modèle réagit à l'âge du receveur.

### Lignes 300 à 301 — `save_prediction_to_session`
Cette fonction met juste le dernier résultat dans `st.session_state`.  
Elle est courte, mais elle clarifie le code principal.

### Lignes 304 à 359 — `render_result`
Cette fonction dessine tout l'onglet de résultat.

#### Lignes 305 à 309
On récupère la probabilité et le niveau de risque.

#### Lignes 311 à 327
Colonne de gauche :
- jauge ;
- carte de décision ;
- carte de niveau de risque ;
- affichage éventuel des warnings.

#### Lignes 329 à 341
Toujours à gauche, on affiche les variables dérivées calculées automatiquement.  
C'est très utile pour montrer que l'application ne dépend plus d'une saisie manuelle incohérente.

#### Lignes 343 à 359
Colonne de droite :
- graphique des contributions locales ;
- graphique de sensibilité à l'âge.

C'est précisément la partie “choix ou manière de bien étudier les analyses” que vous demandiez.

### Lignes 362 à 391 — `render_model_analysis`
Cet onglet montre la partie “science du modèle” :
- tableau de comparaison des modèles ;
- top des importances globales ;
- résumé du dataset ;
- note explicite sur la suppression des variables de fuite.

Cette séparation est importante : l'utilisateur ne mélange pas la vue dossier patient et la vue analytique.

### Lignes 394 à 409 — `render_history`
Cette fonction affiche l'historique stocké dans Excel.  
Si l'historique est vide, elle le dit clairement.  
Sinon, elle :
- affiche la table ;
- propose un bouton de téléchargement du classeur.

### Lignes 412 à 492 — `main`
C'est l'orchestrateur de toute l'application.

#### Lignes 413 à 417
Configuration générale Streamlit.

#### Lignes 419 à 422
Vérification que le modèle existe.  
Si l'entraînement n'a pas été lancé, l'app s'arrête proprement avec un message clair.

#### Lignes 424 à 427
Gestion de l'authentification : tant qu'on n'est pas connecté, on affiche seulement l'écran de login.

#### Lignes 429 à 437
Chargement du pipeline, du résumé, de la sidebar, du titre et de la description.

#### Lignes 439 à 441
Création des quatre onglets :
- nouveau dossier ;
- résultat clinique ;
- analyses du modèle ;
- historique Excel.

#### Lignes 443 à 477 — onglet formulaire
- récupération des valeurs ;
- validation ;
- message d'erreur si besoin ;
- sinon calcul de la probabilité ;
- génération de l'explication locale ;
- génération de la courbe de sensibilité ;
- sauvegarde dans Excel ;
- sauvegarde en session.

#### Lignes 479 à 485 — onglet résultat
On affiche soit un message d'attente, soit le rendu complet du dernier résultat.

#### Lignes 487 à 489 — onglet analyses
Affichage de la vue globale du modèle.

#### Lignes 491 à 492 — onglet historique
Affichage de l'historique Excel.

### Lignes 494 et fin
Le garde-fou `if __name__ == "__main__":` permet le lancement direct du fichier.

---

## 6. `tests/test_data_processing.py`

### Lignes 1 à 12
Les imports récupèrent les fonctions critiques du pipeline de données.

### Lignes 15 à 24
Le premier test vérifie que `prepare_model_dataframe` supprime bien toutes les colonnes de fuite.  
C'est une garantie directe contre le problème principal du projet d'origine.

### Lignes 27 à 47
Le test suivant reconstruit un petit exemple synthétique et vérifie que les variables dérivées sont bien recalculées automatiquement :
- âge binaire ;
- bucket d'âge ;
- ABO match ;
- CMV status ;
- disease group ;
- HLA mismatch ;
- gender match.

### Lignes 50 à 82
Ce test vérifie que `make_prediction_frame` renvoie exactement les colonnes attendues par le modèle, dans le bon ordre.

### Lignes 85 à 90
Ce test contrôle que `build_training_data` ne met plus la cible dans les features.

### Lignes 93 à 103
Le dernier test de ce fichier vérifie que le résumé du dataset mentionne bien les colonnes de fuite supprimées.

---

## 7. `tests/test_input_validation.py`

### Lignes 1 à 25
La fonction `valid_record()` construit un dossier patient cohérent servant de base aux tests.  
Le but est d'éviter de répéter le même dictionnaire dans chaque test.

### Lignes 28 à 31
On vérifie qu'un dossier normal ne produit pas d'erreur.

### Lignes 34 à 39
On force un âge hors plage et on vérifie qu'une erreur bloquante est bien produite.

### Lignes 42 à 49
On teste le cas des valeurs aberrantes mais encore “possibles” :
- dose CD34 très élevée ;
- âge du donneur très haut.

Le système doit produire des warnings, pas des plantages.

### Lignes 52 à 60
On vérifie que l'incohérence HLA est bien signalée par un warning.

---

## 8. `tests/test_model_selection.py`

### Lignes 1 à 4
Imports de la partie sélection de modèle.

### Lignes 7 à 20
Le premier test vérifie que `compare_models` renvoie bien les colonnes attendues du tableau comparatif.

### Lignes 23 à 28
Le deuxième test s'assure que le tableau est trié dans le bon ordre, c'est-à-dire par ROC-AUC décroissant.

### Lignes 31 à 39
Le dernier test isole la fonction `select_best_model` pour vérifier qu'elle prend bien la première ligne du tableau trié.

---

## 9. `tests/test_prediction_logger.py`

### Lignes 1 à 6
Imports des fonctions liées à l'Excel.

### Lignes 9 à 30
`sample_raw_inputs()` fabrique un exemple de saisie utilisateur.

### Lignes 33 à 43
`sample_engineered_inputs()` fabrique les variables dérivées correspondantes.

### Lignes 46 à 50
Premier test : vérifier que `initialize_workbook` crée bien un classeur.

### Lignes 53 à 67
Deuxième test :
- on écrit une prédiction ;
- on relit l'historique ;
- on vérifie que la ligne attendue a bien été stockée.

---

## 10. Lecture d'ensemble : pourquoi cette version est meilleure

### 1. La logique métier est enfin propre
Le médecin ne saisit plus des colonnes redondantes ou contradictoires.  
Les variables calculées sont reconstruites automatiquement.

### 2. Le modèle ne triche plus
Les colonnes post-greffe et de survie sont explicitement supprimées du pipeline.

### 3. Les entrées sont contrôlées
Les erreurs bloquantes et les valeurs aberrantes sont traitées avant la prédiction.

### 4. L'interface permet une vraie lecture de l'analyse
On dispose maintenant :
- d'une probabilité claire ;
- d'une explication locale ;
- d'une courbe de sensibilité à l'âge ;
- d'un onglet d'analyse globale des modèles.

### 5. Les dossiers utilisateurs sont sauvegardés
Le fichier `user_predictions.xlsx` joue le rôle de base locale légère mais exploitable immédiatement.

### 6. Le projet est testable
Les tests couvrent :
- la préparation des données ;
- la validation des entrées ;
- la sélection du modèle ;
- la journalisation Excel.

---

## 11. Conseil d'utilisation pratique

Ordre conseillé pour exécuter le projet :

1. `pip install -r requirements.txt`
2. `python src/train_model.py`
3. `streamlit run app/app.py`

Puis :
- se connecter avec `medecin / Greffe2026!`
- saisir un dossier
- analyser
- vérifier le résultat
- consulter l'historique Excel

---
