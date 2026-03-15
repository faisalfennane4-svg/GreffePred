"""
src/data_processing.py
Pipeline de prétraitement — Greffe de moelle osseuse pédiatrique
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

TARGET_COL = "survival_status"

# Valeur sentinelle = événement non observé (censuré)
SENTINEL_VALUE = 1_000_000
CENSORED_COLS = ["ANCrecovery", "PLTrecovery", "time_to_aGvHD_III_IV", "survival_time"]

# Chemin par défaut vers les données (relatif à la racine du projet)
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "bone_marrow.csv"


def load_data(path=None) -> pd.DataFrame:
    """
    Charge le dataset CSV.
    Remplace automatiquement les valeurs sentinelles (1000000)
    par NaN dans les colonnes temporelles censurées.
    """
    if path is None:
        path = DEFAULT_DATA_PATH
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for col in CENSORED_COLS:
        if col in df.columns:
            df[col] = df[col].replace(SENTINEL_VALUE, np.nan)
    return df


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Réduit l'empreinte mémoire en convertissant :
      float64 → float32
      int64   → int32
    Retourne une copie sans modifier le DataFrame original.
    """
    df_opt = df.copy()
    for col in df_opt.select_dtypes(include=[np.float64]).columns:
        df_opt[col] = df_opt[col].astype(np.float32)
    for col in df_opt.select_dtypes(include=[np.int64]).columns:
        df_opt[col] = df_opt[col].astype(np.int32)
    return df_opt


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute les valeurs manquantes :
      - Colonnes numériques   → médiane
      - Colonnes catégorielles → mode
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if num_cols:
        imp = SimpleImputer(strategy="median")
        df[num_cols] = imp.fit_transform(df[num_cols])
    if cat_cols:
        imp = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = imp.fit_transform(df[cat_cols])
    return df


def encode_categoricals(df: pd.DataFrame):
    """
    Encode les colonnes catégorielles avec LabelEncoder.
    Retourne (df_encodé, dict_encoders).
    """
    df = df.copy()
    encoders = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def handle_outliers(df: pd.DataFrame, exclude_cols=None) -> pd.DataFrame:
    """
    Plafonne les outliers par la méthode IQR (Winsorizing 1.5×IQR)
    sur toutes les colonnes numériques sauf celles exclues.
    """
    df = df.copy()
    if exclude_cols is None:
        exclude_cols = [TARGET_COL]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in exclude_cols]
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
    return df


def apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """
    Applique SMOTE pour rééquilibrer les classes.
    Fallback sur class_weight='balanced' si imblearn n'est pas installé.
    """
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=random_state)
        return sm.fit_resample(X, y)
    except ImportError:
        print("⚠  imblearn non disponible — SMOTE ignoré (utilisez class_weight='balanced')")
        return X, y


def preprocess_pipeline(path=None, use_smote: bool = True):
    """
    Pipeline complet :
    1. Chargement (+ remplacement valeurs sentinelles)
    2. Imputation des valeurs manquantes
    3. Encodage des variables catégorielles
    4. Gestion des outliers (IQR)
    5. Optimisation mémoire
    6. Séparation X / y
    7. SMOTE (optionnel)

    Retourne : X, y, feature_names, encoders
    """
    df = load_data(path)
    df = handle_missing_values(df)
    df, encoders = encode_categoricals(df)
    df = handle_outliers(df)
    df = optimize_memory(df)

    feature_names = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_names].values
    y = df[TARGET_COL].values.astype(int)

    if use_smote:
        X, y = apply_smote(X, y)

    return X, y, feature_names, encoders


MODEL_FEATURE_COLUMNS = [
    'Recipientgender', 'Stemcellsource', 'Donorage', 'Donorage35',
    'IIIV', 'Gendermatch', 'DonorABO', 'RecipientABO', 'RecipientRh',
    'ABOmatch', 'CMVstatus', 'DonorCMV', 'RecipientCMV', 'Disease',
    'Riskgroup', 'Txpostrelapse', 'Diseasegroup', 'HLAmatch',
    'HLAmismatch', 'Antigen', 'Alel', 'HLAgrI', 'Recipientage',
    'Recipientage10', 'Recipientageint', 'Relapse', 'aGvHDIIIIV',
    'extcGvHD', 'CD34kgx10d6', 'CD3dkgx10d8', 'CD3dCD34',
    'Rbodymass', 'ANCrecovery', 'PLTrecovery',
    'time_to_aGvHD_III_IV', 'survival_time'
]
