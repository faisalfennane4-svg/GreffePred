from pathlib import Path

from src.prediction_logger import append_prediction, initialize_workbook, read_history
    
def sample_raw_inputs(): 
    return {
        "PatientID": "DOS-XYZ",
        "ClinicalNote": "note de test",
        "Recipientgender": 1,
        "Stemcellsource": 1,
        "Donorage": 28.0,
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
        "Rbodymass": 30.0,
    }


def sample_engineered_inputs():
    return {
        "Gendermatch": 1,
        "ABOmatch": 1,
        "CMVstatus": 2,
        "Diseasegroup": 1,
        "HLAmismatch": 1,
        "Recipientage10": 0,
        "Recipientageint": 1,
        "Donorage35": 0,
    }


def test_initialize_workbook_creates_file(tmp_path):
    workbook_path = tmp_path / "history.xlsx"
    initialize_workbook(workbook_path)
    assert workbook_path.exists()


def test_append_prediction_adds_row(tmp_path):
    workbook_path = tmp_path / "history.xlsx"
    append_prediction(
        username="medecin",
        raw_inputs=sample_raw_inputs(),
        engineered_inputs=sample_engineered_inputs(),
        predicted_label="Survie probable",
        success_probability=0.61,
        warnings=["warning 1"],
        path=workbook_path,
    )

    history = read_history(workbook_path, limit=10)
    assert not history.empty
    assert history.loc[0, "patient_id"] == "DOS-XYZ"
    assert float(history.loc[0, "success_probability"]) == 0.61
