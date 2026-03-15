from src.input_validation import validate_input_record

 
def valid_record():
    return {
        "PatientID": "DOS-01",
        "ClinicalNote": "",
        "Recipientgender": 1,
        "DonorGender": "F",
        "Recipientage": 8.0,
        "Rbodymass": 26.0,
        "Disease": "ALL",
        "Riskgroup": 0,
        "Txpostrelapse": 0,
        "Stemcellsource": 1,
        "Donorage": 28.0,
        "DonorABO": 0,
        "RecipientABO": 1,
        "RecipientRh": 1,
        "DonorCMV": 0,
        "RecipientCMV": 1,
        "HLAmatch": 1,
        "Antigen": 1,
        "Alel": 0,
        "HLAgrI": 1,
        "CD34kgx10d6": 8.0,
        "CD3dCD34": 4.0,
        "CD3dkgx10d8": 5.0,
    }


def test_valid_record_has_no_error():
    result = validate_input_record(valid_record())
    assert result["errors"] == []


def test_out_of_range_age_is_an_error():
    record = valid_record()
    record["Recipientage"] = 25.0
    result = validate_input_record(record)
    assert result["errors"]
    assert "Âge du receveur" in result["errors"][0]


def test_aberrant_but_possible_values_generate_warning():
    record = valid_record()
    record["CD34kgx10d6"] = 45.0
    record["Donorage"] = 56.0
    result = validate_input_record(record)
    assert not result["errors"]
    assert len(result["warnings"]) >= 2


def test_hla_incoherence_generates_warning():
    record = valid_record()
    record["HLAmatch"] = 0
    record["Antigen"] = 1
    result = validate_input_record(record)
    assert not result["errors"]
    assert any("Compatibilité HLA indiquée à 10/10" in warning for warning in result["warnings"])
