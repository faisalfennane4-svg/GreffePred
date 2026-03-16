"""
Préparation des données pour GreffePred. 
 
Le point important de cette version est de séparer clairement :
1) les variables que le médecin peut saisir avant la greffe,
2) les variables dérivées calculées automatiquement,
3) les variables de fuite d'information qui doivent être exclues du modèle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "bone_marrow.csv"

TARGET_COLUMN = "survival_status"
MISSING_TOKENS = ["?", "NA", "N/A", "nan", "None", ""]

POST_TRANSPLANT_COLUMNS = [
    "IIIV",
    "Relapse",
    "aGvHDIIIIV",
    "extcGvHD",
    "ANCrecovery",
    "PLTrecovery",
    "time_to_aGvHD_III_IV",
    "survival_time",
]

FORM_ONLY_COLUMNS = ["PatientID", "ClinicalNote", "DonorGender"]

MODEL_FEATURE_COLUMNS = [
    "Recipientgender",
    "Stemcellsource",
    "Donorage",
    "Donorage35",
    "Gendermatch",
    "DonorABO",
    "RecipientABO",
    "RecipientRh",
    "ABOmatch",
    "CMVstatus",
    "DonorCMV",
    "RecipientCMV",
    "Disease",
    "Riskgroup",
    "Txpostrelapse",
    "Diseasegroup",
    "HLAmatch",
    "HLAmismatch",
    "Antigen",
    "Alel",
    "HLAgrI",
    "Recipientage",
    "Recipientage10",
    "Recipientageint",
    "CD34kgx10d6",
    "CD3dCD34",
    "CD3dkgx10d8",
    "Rbodymass",
]

FRIENDLY_FEATURE_NAMES = {
    "Recipientgender": "Sexe du receveur",
    "Stemcellsource": "Source des cellules souches",
    "Donorage": "Âge du donneur",
    "Donorage35": "Donneur ≥ 35 ans",
    "Gendermatch": "Donneuse vers receveur homme",
    "DonorABO": "Groupe ABO du donneur",
    "RecipientABO": "Groupe ABO du receveur",
    "RecipientRh": "Rhésus du receveur",
    "ABOmatch": "Incompatibilité ABO",
    "CMVstatus": "Compatibilité CMV",
    "DonorCMV": "CMV du donneur",
    "RecipientCMV": "CMV du receveur",
    "Disease": "Maladie",
    "Riskgroup": "Groupe de risque",
    "Txpostrelapse": "Deuxième greffe après rechute",
    "Diseasegroup": "Maladie maligne",
    "HLAmatch": "Compatibilité HLA",
    "HLAmismatch": "HLA non parfaitement compatible",
    "Antigen": "Différences antigéniques",
    "Alel": "Différences alléliques",
    "HLAgrI": "Type de différence HLA",
    "Recipientage": "Âge du receveur",
    "Recipientage10": "Receveur ≥ 10 ans",
    "Recipientageint": "Classe d'âge du receveur",
    "CD34kgx10d6": "Dose CD34+ / kg",
    "CD3dCD34": "Ratio CD3+/CD34+",
    "CD3dkgx10d8": "Dose CD3+ / kg",
    "Rbodymass": "Poids du receveur",
}


def optimize_memory(df: pd.DataFrame, exclude_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Réduit l'empreinte mémoire du DataFrame sans modifier l'original.
    """
    optimized = df.copy()
    excluded = set(exclude_cols or [])

    for column in optimized.columns:
        if column in excluded:
            continue
        if pd.api.types.is_float_dtype(optimized[column]):
            optimized[column] = pd.to_numeric(optimized[column], downcast="float")
        elif pd.api.types.is_integer_dtype(optimized[column]):
            optimized[column] = pd.to_numeric(optimized[column], downcast="integer")
    return optimized


def _recipient_age_bucket(age: float | int | None) -> float:
    """
    Convertit l'âge du receveur en classe 0, 1 ou 2.
    """
    if pd.isna(age):
        return np.nan
    if age <= 5:
        return 0.0
    if age <= 10:
        return 1.0
    return 2.0


def _cmv_status(donor_cmv: float | int | None, recipient_cmv: float | int | None) -> float:
    """
    Recrée le codage CMV du dataset original.

    0 = D-/R-
    1 = D+/R-
    2 = D-/R+
    3 = D+/R+
    """
    if pd.isna(donor_cmv) or pd.isna(recipient_cmv):
        return np.nan

    donor = int(donor_cmv)
    recipient = int(recipient_cmv)

    mapping = {
        (0, 0): 0.0,
        (1, 0): 1.0,
        (0, 1): 2.0,
        (1, 1): 3.0,
    }
    return mapping.get((donor, recipient), np.nan)


def _disease_group(disease: Any) -> float:
    """
    Regroupe la maladie en binaire : maligne / non maligne.
    """
    if pd.isna(disease):
        return np.nan
    malignant_diseases = {"ALL", "AML", "chronic", "lymphoma"}
    return 1.0 if str(disease) in malignant_diseases else 0.0


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """
    Charge le CSV et nettoie les marqueurs de valeurs manquantes.
    """
    source_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    df = pd.read_csv(source_path, na_values=MISSING_TOKENS)
    df.columns = [column.strip() for column in df.columns]
    return df


def derive_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalcule toutes les variables dérivées pour garantir la cohérence métier.
    """
    enriched = df.copy()

    if "Donorage" in enriched.columns:
        enriched["Donorage35"] = np.where(enriched["Donorage"].isna(), np.nan, (enriched["Donorage"] >= 35).astype(float))

    if "Recipientage" in enriched.columns:
        enriched["Recipientage10"] = np.where(enriched["Recipientage"].isna(), np.nan, (enriched["Recipientage"] >= 10).astype(float))
        enriched["Recipientageint"] = enriched["Recipientage"].apply(_recipient_age_bucket)

    if "DonorABO" in enriched.columns and "RecipientABO" in enriched.columns:
        enriched["ABOmatch"] = np.where(
            enriched["DonorABO"].isna() | enriched["RecipientABO"].isna(),
            np.nan,
            (enriched["DonorABO"] != enriched["RecipientABO"]).astype(float),
        )

    if "DonorCMV" in enriched.columns and "RecipientCMV" in enriched.columns:
        enriched["CMVstatus"] = [
            _cmv_status(donor_cmv, recipient_cmv)
            for donor_cmv, recipient_cmv in zip(enriched["DonorCMV"], enriched["RecipientCMV"])
        ]

    if "Disease" in enriched.columns:
        enriched["Diseasegroup"] = enriched["Disease"].apply(_disease_group)

    if "HLAmatch" in enriched.columns:
        enriched["HLAmismatch"] = np.where(enriched["HLAmatch"].isna(), np.nan, (enriched["HLAmatch"] != 0).astype(float))

    if "DonorGender" in enriched.columns and "Recipientgender" in enriched.columns:
        recipient_is_male = enriched["Recipientgender"] == 1
        donor_is_female = enriched["DonorGender"].astype(str).str.upper().str.strip() == "F"
        enriched["Gendermatch"] = np.where(
            enriched["Recipientgender"].isna(),
            np.nan,
            (recipient_is_male & donor_is_female).astype(float),
        )

    return enriched


def validate_dataset_schema(df: pd.DataFrame) -> None:
    """
    Vérifie que le dataset contient les colonnes indispensables.
    """
    missing_columns = [column for column in MODEL_FEATURE_COLUMNS + [TARGET_COLUMN] if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Colonnes manquantes dans le dataset : {missing_columns}")

    target_values = set(df[TARGET_COLUMN].dropna().astype(int).unique().tolist())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"La cible doit être binaire. Valeurs trouvées : {sorted(target_values)}")


def prepare_model_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enlève les variables de fuite et remet les colonnes dans l'ordre attendu.
    """
    enriched = derive_clinical_features(df)

    for column in POST_TRANSPLANT_COLUMNS:
        if column in enriched.columns:
            enriched = enriched.drop(columns=column)

    missing_features = [column for column in MODEL_FEATURE_COLUMNS if column not in enriched.columns]
    if missing_features:
        raise ValueError(f"Features indisponibles après préparation : {missing_features}")

    ordered_columns = MODEL_FEATURE_COLUMNS.copy()
    if TARGET_COLUMN in enriched.columns:
        ordered_columns.append(TARGET_COLUMN)

    return enriched[ordered_columns].copy()


def build_training_data(path: str | Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Retourne X et y à partir du fichier CSV.
    """
    raw_df = load_data(path)
    prepared_df = prepare_model_dataframe(raw_df)
    validate_dataset_schema(prepared_df)

    feature_frame = prepared_df[MODEL_FEATURE_COLUMNS].copy()
    target = prepared_df[TARGET_COLUMN].astype(int).copy()

    feature_frame = optimize_memory(feature_frame)
    return feature_frame, target


def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    """
    Construit le préprocesseur sklearn.
    """
    numeric_columns = feature_frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in feature_frame.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def make_prediction_frame(user_inputs: dict[str, Any]) -> pd.DataFrame:
    """
    Transforme les valeurs du formulaire en DataFrame prêt pour la prédiction.
    """
    base_record = {
        "Recipientgender": user_inputs["Recipientgender"],
        "Stemcellsource": user_inputs["Stemcellsource"],
        "Donorage": user_inputs["Donorage"],
        "DonorABO": user_inputs["DonorABO"],
        "RecipientABO": user_inputs["RecipientABO"],
        "RecipientRh": user_inputs["RecipientRh"],
        "DonorCMV": user_inputs["DonorCMV"],
        "RecipientCMV": user_inputs["RecipientCMV"],
        "Disease": user_inputs["Disease"],
        "Riskgroup": user_inputs["Riskgroup"],
        "Txpostrelapse": user_inputs["Txpostrelapse"],
        "HLAmatch": user_inputs["HLAmatch"],
        "Antigen": user_inputs["Antigen"],
        "Alel": user_inputs["Alel"],
        "HLAgrI": user_inputs["HLAgrI"],
        "Recipientage": user_inputs["Recipientage"],
        "CD34kgx10d6": user_inputs["CD34kgx10d6"],
        "CD3dCD34": user_inputs["CD3dCD34"],
        "CD3dkgx10d8": user_inputs["CD3dkgx10d8"],
        "Rbodymass": user_inputs["Rbodymass"],
        "DonorGender": user_inputs["DonorGender"],
    }

    prediction_df = pd.DataFrame([base_record])
    prediction_df = derive_clinical_features(prediction_df)

    for column in MODEL_FEATURE_COLUMNS:
        if column not in prediction_df.columns:
            prediction_df[column] = np.nan

    return prediction_df[MODEL_FEATURE_COLUMNS].copy()


def dataset_quality_summary(path: str | Path | None = None) -> dict[str, Any]:
    """
    Produit un résumé lisible de la qualité du dataset.
    """
    raw_df = load_data(path)
    prepared_df = prepare_model_dataframe(raw_df)

    target_distribution = (
        prepared_df[TARGET_COLUMN]
        .astype(int)
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
        .to_dict()
    )

    missing_counts = raw_df.isna().sum()
    missing_columns = {
        column: int(count)
        for column, count in missing_counts.sort_values(ascending=False).items()
        if int(count) > 0
    }

    summary = {
        "rows": int(raw_df.shape[0]),
        "columns_before_cleaning": int(raw_df.shape[1]),
        "columns_used_for_model": len(MODEL_FEATURE_COLUMNS),
        "leakage_columns_removed": POST_TRANSPLANT_COLUMNS,
        "target_distribution_percent": target_distribution,
        "columns_with_missing_values": missing_columns,
        "features_used": MODEL_FEATURE_COLUMNS,
    }
    return summary


def humanize_transformed_feature_name(feature_name: str) -> str:
    """
    Transforme un nom de feature sklearn en libellé lisible.
    """
    if feature_name.startswith("num__"):
        raw_name = feature_name.replace("num__", "", 1)
        return FRIENDLY_FEATURE_NAMES.get(raw_name, raw_name)

    if feature_name.startswith("cat__Disease_"):
        disease_value = feature_name.replace("cat__Disease_", "", 1)
        return f"Maladie = {disease_value}"

    return feature_name
