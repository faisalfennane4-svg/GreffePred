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

SENTINEL_VALUE = 1_000_000
CENSORED_COLS = ["ANCrecovery", "PLTrecovery", "time_to_aGvHD_III_IV", "survival_time"]

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "bone_marrow.csv"


def load_data(path=None) -> pd.DataFrame:
    if path is None:
        path = DEFAULT_DATA_PATH
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for col in CENSORED_COLS:
        if col in df.columns:
            df[col] = df[col].replace(SENTINEL_VALUE, np.nan)
    return df


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    df_opt = df.copy()
    for col in df_opt.select_dtypes(include=[np.float64]).columns:
        df_opt[col] = df_opt[col].astype(np.float32)
    for col in df_opt.select_dtypes(include=[np.int64]).columns:
        df_opt[col] = df_opt[col].astype(np.int32)
    return df_opt


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
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
    df = df.copy()
    encoders = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def handle_outliers(df: pd.DataFrame, exclude_cols=None) -> pd.DataFrame:
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
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=random_state)
        return sm.fit_resample(X, y)
    except ImportError:
        print("imblearn non disponible — SMOTE ignoré")
        return X, y


def preprocess_pipeline(path=None, use_smote: bool = True):
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


def build_preprocessor(feature_frame=None):
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])


def build_training_data(path=None):
    df = load_data(path)
    df = handle_missing_values(df)
    df, _ = encode_categoricals(df)
    df = handle_outliers(df)
    df = optimize_memory(df)
    feature_names = [c for c in df.columns if c != TARGET_COL]
    feature_frame = pd.DataFrame(df[feature_names])
    target = pd.Series(df[TARGET_COL].values.astype(int), name=TARGET_COL)
    return feature_frame, target


def dataset_quality_summary(path=None):
    df = load_data(path)
    return {
        "n_samples": len(df),
        "n_features": len(df.columns) - 1,
        "missing_values": int(df.isnull().sum().sum()),
        "class_balance": df[TARGET_COL].value_counts().to_dict()
    }


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

FRIENDLY_FEATURE_NAMES = {
    "Recipientgender": "Recipient Gender",
    "Stemcellsource": "Stem Cell Source",
    "Donorage": "Donor Age (years)",
    "Donorage35": "Donor Age > 35",
    "IIIV": "CMV Serostatus",
    "Gendermatch": "Gender Match",
    "DonorABO": "Donor ABO Blood Group",
    "RecipientABO": "Recipient ABO Blood Group",
    "RecipientRh": "Recipient Rh Factor",
    "ABOmatch": "ABO Compatibility",
    "CMVstatus": "CMV Status",
    "DonorCMV": "Donor CMV",
    "RecipientCMV": "Recipient CMV",
    "Disease": "Underlying Disease",
    "Riskgroup": "Risk Group",
    "Txpostrelapse": "Transplant Post-Relapse",
    "Diseasegroup": "Disease Group",
    "HLAmatch": "HLA Match Level",
    "HLAmismatch": "HLA Mismatch",
    "Antigen": "Antigen Mismatch",
    "Alel": "Allele Mismatch",
    "HLAgrI": "HLA Group I",
    "Recipientage": "Recipient Age (years)",
    "Recipientage10": "Recipient Age > 10",
    "Recipientageint": "Recipient Age Interval",
    "Relapse": "Relapse",
    "aGvHDIIIIV": "Acute GvHD Grade III-IV",
    "extcGvHD": "Extensive Chronic GvHD",
    "CD34kgx10d6": "CD34+ Cells (x10^6/kg)",
    "CD3dkgx10d8": "CD3+ Cells (x10^8/kg)",
    "CD3dCD34": "CD3/CD34 Ratio",
    "Rbodymass": "Recipient Body Mass (kg)",
    "ANCrecovery": "ANC Recovery Day",
    "PLTrecovery": "Platelet Recovery Day",
    "time_to_aGvHD_III_IV": "Time to Acute GvHD (days)",
    "survival_time": "Survival Time (days)",
}


def humanize_transformed_feature_name(name: str) -> str:
    return FRIENDLY_FEATURE_NAMES.get(name, name.replace("_", " ").title())
