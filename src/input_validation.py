"""
Validation des entrées utilisateur. 

Cette couche évite deux problèmes :
- les valeurs impossibles,
- les valeurs très aberrantes qui rendraient la prédiction peu crédible.
"""

from __future__ import annotations

from typing import Any

ABO_LABELS = {
    -1: "B",
    0: "O",
    1: "A",
    2: "AB",
}

BINARY_LABELS = {
    0: "Non",
    1: "Oui",
}

FORM_SCHEMA = {
    "PatientID": {
        "label": "Identifiant patient",
        "group": "administratif",
        "kind": "text",
        "default": "",
        "help": "Code dossier interne.",
    },
    "ClinicalNote": {
        "label": "Note clinique libre",
        "group": "administratif",
        "kind": "text_area",
        "default": "",
        "help": "Champ facultatif enregistré dans le fichier Excel.",
    },
    "Recipientgender": {
        "label": "Sexe du receveur",
        "group": "patient",
        "kind": "select",
        "options": {1: "Homme", 0: "Femme"},
        "default": 1,
    },
    "DonorGender": {
        "label": "Sexe du donneur",
        "group": "patient",
        "kind": "select",
        "options": {"M": "Homme", "F": "Femme"},
        "default": "M",
    },
    "Recipientage": {
        "label": "Âge du receveur (ans)",
        "group": "patient",
        "kind": "number",
        "min": 0.1,
        "max": 20.0,
        "step": 0.1,
        "default": 8.0,
    },
    "Rbodymass": {
        "label": "Poids du receveur (kg)",
        "group": "patient",
        "kind": "number",
        "min": 2.0,
        "max": 110.0,
        "step": 0.1,
        "default": 25.0,
    },
    "Disease": {
        "label": "Maladie",
        "group": "patient",
        "kind": "select",
        "options": {
            "ALL": "ALL",
            "AML": "AML",
            "chronic": "Chronic",
            "lymphoma": "Lymphoma",
            "nonmalignant": "Non malignant",
        },
        "default": "ALL",
    },
    "Riskgroup": {
        "label": "Groupe de risque",
        "group": "patient",
        "kind": "select",
        "options": {0: "Faible", 1: "Élevé"},
        "default": 0,
    },
    "Txpostrelapse": {
        "label": "Deuxième greffe après rechute",
        "group": "patient",
        "kind": "select",
        "options": BINARY_LABELS,
        "default": 0,
    },
    "Stemcellsource": {
        "label": "Source des cellules souches",
        "group": "donneur",
        "kind": "select",
        "options": {0: "Moelle osseuse", 1: "Sang périphérique"},
        "default": 1,
    },
    "Donorage": {
        "label": "Âge du donneur (ans)",
        "group": "donneur",
        "kind": "number",
        "min": 18.0,
        "max": 65.0,
        "step": 0.1,
        "default": 28.0,
    },
    "DonorABO": {
        "label": "ABO du donneur",
        "group": "immunologie",
        "kind": "select",
        "options": ABO_LABELS,
        "default": 0,
    },
    "RecipientABO": {
        "label": "ABO du receveur",
        "group": "immunologie",
        "kind": "select",
        "options": ABO_LABELS,
        "default": 0,
    },
    "RecipientRh": {
        "label": "Rhésus du receveur",
        "group": "immunologie",
        "kind": "select",
        "options": {1: "Rh+", 0: "Rh-"},
        "default": 1,
    },
    "DonorCMV": {
        "label": "CMV du donneur",
        "group": "immunologie",
        "kind": "select",
        "options": {0: "Négatif", 1: "Positif"},
        "default": 0,
    },
    "RecipientCMV": {
        "label": "CMV du receveur",
        "group": "immunologie",
        "kind": "select",
        "options": {0: "Négatif", 1: "Positif"},
        "default": 0,
    },
    "HLAmatch": {
        "label": "Compatibilité HLA",
        "group": "immunologie",
        "kind": "select",
        "options": {0: "10/10", 1: "9/10", 2: "8/10", 3: "7/10"},
        "default": 0,
    },
    "Antigen": {
        "label": "Différences antigéniques",
        "group": "immunologie",
        "kind": "select",
        "options": {-1: "Aucune", 0: "Une", 1: "Deux", 2: "Trois"},
        "default": -1,
    },
    "Alel": {
        "label": "Différences alléliques",
        "group": "immunologie",
        "kind": "select",
        "options": {-1: "Aucune", 0: "Une", 1: "Deux", 2: "Trois", 3: "Quatre"},
        "default": -1,
    },
    "HLAgrI": {
        "label": "Type de différence HLA",
        "group": "immunologie",
        "kind": "select",
        "options": {
            0: "Aucune",
            1: "Une différence antigénique",
            2: "Une différence allélique",
            3: "Différence DRB1",
            4: "Deux différences",
            5: "Deux différences (variante)",
            7: "Différences multiples",
        },
        "default": 0,
    },
    "CD34kgx10d6": {
        "label": "Dose CD34+ / kg (10^6/kg)",
        "group": "greffe",
        "kind": "number",
        "min": 0.1,
        "max": 60.0,
        "step": 0.01,
        "default": 8.0,
    },
    "CD3dCD34": {
        "label": "Ratio CD3+/CD34+",
        "group": "greffe",
        "kind": "number",
        "min": 0.1,
        "max": 60.0,
        "step": 0.01,
        "default": 4.0,
    },
    "CD3dkgx10d8": {
        "label": "Dose CD3+ / kg (10^8/kg)",
        "group": "greffe",
        "kind": "number",
        "min": 0.05,
        "max": 20.0,
        "step": 0.01,
        "default": 4.0,
    },
}


def schema_by_group() -> dict[str, list[str]]:
    """
    Retourne les champs regroupés pour l'interface.
    """
    groups: dict[str, list[str]] = {}
    for field_name, config in FORM_SCHEMA.items():
        groups.setdefault(config["group"], []).append(field_name)
    return groups


def _range_error(field_name: str, value: float, minimum: float, maximum: float) -> str:
    label = FORM_SCHEMA[field_name]["label"]
    return f"{label} doit rester entre {minimum} et {maximum}. Valeur reçue : {value}."


def validate_input_record(record: dict[str, Any]) -> dict[str, list[str]]:
    """
    Retourne une structure {'errors': [...], 'warnings': [...]}.
    """
    result = {"errors": [], "warnings": []}

    for field_name, config in FORM_SCHEMA.items():
        if field_name not in record:
            result["errors"].append(f"Champ manquant : {config['label']}.")
            continue

        value = record[field_name]
        if config["kind"] in {"text", "text_area"}:
            continue

        if config["kind"] == "select":
            valid_choices = set(config["options"].keys())
            if value not in valid_choices:
                result["errors"].append(f"Valeur invalide pour {config['label']} : {value}.")
            continue

        if config["kind"] == "number":
            minimum = float(config["min"])
            maximum = float(config["max"])
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                result["errors"].append(f"{config['label']} doit être numérique.")
                continue

            if numeric_value < minimum or numeric_value > maximum:
                result["errors"].append(_range_error(field_name, numeric_value, minimum, maximum))

    if result["errors"]:
        return result

    if float(record["Recipientage"]) > 18.0:
        result["warnings"].append("Âge du receveur très élevé par rapport au caractère pédiatrique du dataset.")

    if float(record["Donorage"]) > 55.0:
        result["warnings"].append("Âge du donneur dans la zone haute observée du dataset.")

    if float(record["Rbodymass"]) < 5.0:
        result["warnings"].append("Poids du receveur très faible : vérifier qu'il n'y a pas d'erreur de saisie.")

    if float(record["CD34kgx10d6"]) > 40.0:
        result["warnings"].append("Dose CD34+ très élevée : valeur possible mais inhabituelle.")

    if float(record["CD3dCD34"]) > 30.0:
        result["warnings"].append("Ratio CD3+/CD34+ élevé : vérifier l'unité de saisie.")

    if int(record["HLAmatch"]) == 0 and (int(record["Antigen"]) != -1 or int(record["Alel"]) != -1 or int(record["HLAgrI"]) != 0):
        result["warnings"].append("Compatibilité HLA indiquée à 10/10 mais détails HLA non cohérents.")

    if int(record["HLAmatch"]) > 0 and int(record["Antigen"]) == -1 and int(record["Alel"]) == -1 and int(record["HLAgrI"]) == 0:
        result["warnings"].append("Compatibilité HLA non parfaite mais détails HLA encore vides ou trop optimistes.")

    return result
