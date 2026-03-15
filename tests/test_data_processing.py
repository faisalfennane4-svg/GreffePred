from pathlib import Path
 
import pandas as pd
  
from src.data_processing import (
    MODEL_FEATURE_COLUMNS,
    POST_TRANSPLANT_COLUMNS,
    TARGET_COLUMN,
    build_training_data,
    dataset_quality_summary,
    derive_clinical_features,
    make_prediction_frame,
    prepare_model_dataframe,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "bone_marrow.csv"



def test_prepare_model_dataframe_removes_leakage_columns():
    raw_df = pd.read_csv(DATA_PATH, na_values=["?"])
    prepared_df = prepare_model_dataframe(raw_df)

    for column in POST_TRANSPLANT_COLUMNS:
        assert column not in prepared_df.columns

    assert TARGET_COLUMN in prepared_df.columns
    assert set(MODEL_FEATURE_COLUMNS).issubset(set(prepared_df.columns))


def test_derived_features_are_recomputed():
    sample = pd.DataFrame(
        [
            {
                "Recipientgender": 1,
                "DonorGender": "F",
                "Donorage": 40,
                "DonorABO": 1,
                "RecipientABO": 0,
                "DonorCMV": 1,
                "RecipientCMV": 0,
                "Disease": "ALL",
                "HLAmatch": 2,
                "Recipientage": 11,
            }
        ]
    )

    enriched = derive_clinical_features(sample)
    assert enriched.loc[0, "Donorage35"] == 1
    assert enriched.loc[0, "Recipientage10"] == 1
    assert enriched.loc[0, "Recipientageint"] == 2
    assert enriched.loc[0, "ABOmatch"] == 1
    assert enriched.loc[0, "CMVstatus"] == 1
    assert enriched.loc[0, "Diseasegroup"] == 1
    assert enriched.loc[0, "HLAmismatch"] == 1
    assert enriched.loc[0, "Gendermatch"] == 1


def test_make_prediction_frame_has_expected_columns():
    raw_inputs = {
        "PatientID": "TEST-001",
        "ClinicalNote": "",
        "Recipientgender": 1,
        "DonorGender": "F",
        "Stemcellsource": 1,
        "Donorage": 30.0,
        "DonorABO": 0,
        "RecipientABO": 1,
        "RecipientRh": 1,
        "DonorCMV": 0,
        "RecipientCMV": 1,
        "Disease": "ALL",
        "Riskgroup": 1,
        "Txpostrelapse": 0,
        "HLAmatch": 1,
        "Antigen": 1,
        "Alel": 0,
        "HLAgrI": 1,
        "Recipientage": 9.0,
        "CD34kgx10d6": 8.0,
        "CD3dCD34": 4.0,
        "CD3dkgx10d8": 5.0,
        "Rbodymass": 28.0,
    }

    frame = make_prediction_frame(raw_inputs)
    assert list(frame.columns) == MODEL_FEATURE_COLUMNS
    assert frame.shape == (1, len(MODEL_FEATURE_COLUMNS))
    assert frame.loc[0, "ABOmatch"] == 1
    assert frame.loc[0, "CMVstatus"] == 2


def test_build_training_data_has_no_target_in_features():
    feature_frame, target = build_training_data()
    assert TARGET_COLUMN not in feature_frame.columns
    assert feature_frame.shape[1] == len(MODEL_FEATURE_COLUMNS)
    assert target.isin([0, 1]).all()


def test_dataset_quality_summary_mentions_removed_columns():
    summary = dataset_quality_summary()
    assert "leakage_columns_removed" in summary
    assert len(summary["leakage_columns_removed"]) == len(POST_TRANSPLANT_COLUMNS)
