"""
Application Streamlit principale.
""" 
 
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
SRC_DIR = ROOT_DIR / "src"

import sys
sys.path.insert(0, str(SRC_DIR))

from data_processing import (
    FRIENDLY_FEATURE_NAMES,
    MODEL_FEATURE_COLUMNS,
    build_training_data,
    humanize_transformed_feature_name,
    make_prediction_frame,
)
from input_validation import FORM_SCHEMA, schema_by_group, validate_input_record
from prediction_logger import DEFAULT_LOG_PATH, append_prediction, read_history
from train_model import (
    BACKGROUND_PATH,
    COMPARISON_PATH,
    FEATURE_IMPORTANCE_PATH,
    MODEL_PATH,
    SUMMARY_PATH,
)

DEFAULT_USERS = {
    "medecin": {
        "password_hash": "5a87e350cc1ce75b6d4c12cb826d5e0d298ec39c9df48231a90a03adc38cebc4",
        "display_name": "Médecin de démonstration",
    }
}

GROUP_TITLES = {
    "administratif": "Informations administratives",
    "patient": "Données du receveur",
    "donneur": "Données du donneur",
    "immunologie": "Compatibilité immunologique",
    "greffe": "Paramètres de la greffe",
}

RISK_TEXT = {
    "Favorable": "Profil plutôt favorable au vu des variables saisies.",
    "Intermédiaire": "Profil intermédiaire : interprétation prudente recommandée.",
    "Élevé": "Profil de risque élevé : revue clinique renforcée recommandée.",
}


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_model_comparison() -> pd.DataFrame:
    if COMPARISON_PATH.exists():
        return pd.read_csv(COMPARISON_PATH)
    return pd.DataFrame()


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    if FEATURE_IMPORTANCE_PATH.exists():
        return pd.read_csv(FEATURE_IMPORTANCE_PATH)
    return pd.DataFrame(columns=["feature", "importance"])


@st.cache_data
def load_training_summary() -> dict[str, Any]:
    if SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    return {}


@st.cache_data
def load_background_reference() -> pd.DataFrame:
    if BACKGROUND_PATH.exists():
        return pd.read_csv(BACKGROUND_PATH)
    feature_frame, _ = build_training_data()
    return feature_frame.head(60)


def check_password(username: str, password: str) -> bool:
    user = DEFAULT_USERS.get(username.strip().lower())
    if user is None:
        return False
    password_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return user["password_hash"] == password_hash


def risk_level_from_probability(probability: float) -> str:
    if probability >= 0.70:
        return "Favorable"
    if probability >= 0.45:
        return "Intermédiaire"
    return "Élevé"


def render_login() -> None:
    st.title("GreffePred — Connexion médecin")
    st.caption("Version corrigée avec contrôle de saisie, comparaison de modèles et historique Excel.")

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter")

    if submitted:
        if check_password(username, password):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username.strip().lower()
            st.success("Connexion réussie.")
            st.rerun()
        else:
            st.error("Identifiants incorrects.")


def _metric_card(title: str, value: str, subtitle: str, color: str) -> None:
    st.markdown(
        f"""
        <div style="padding:1rem 1.1rem;border-radius:16px;background:white;border-left:8px solid {color};
                    box-shadow:0 8px 20px rgba(0,0,0,0.05);margin-bottom:0.8rem;">
            <div style="font-size:0.8rem;color:#6b7280;font-weight:600;">{title}</div>
            <div style="font-size:1.9rem;font-weight:800;color:#111827;margin:0.2rem 0;">{value}</div>
            <div style="font-size:0.9rem;color:#374151;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(training_summary: dict[str, Any]) -> None:
    st.sidebar.title("GreffePred")
    st.sidebar.write(f"Utilisateur connecté : **{st.session_state['username']}**")
    if st.sidebar.button("Se déconnecter"):
        st.session_state.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.write("**Base Excel**")
    st.sidebar.code(str(DEFAULT_LOG_PATH.relative_to(ROOT_DIR)))
    st.sidebar.write("**Variables de fuite exclues**")
    for column in training_summary.get("leakage_columns_removed", []):
        st.sidebar.caption(f"• {column}")


def render_field(field_name: str):
    config = FORM_SCHEMA[field_name]
    label = config["label"]
    kind = config["kind"]

    if kind == "text":
        return st.text_input(label, value=config.get("default", ""), help=config.get("help"))

    if kind == "text_area":
        return st.text_area(label, value=config.get("default", ""), help=config.get("help"))

    if kind == "select":
        options = list(config["options"].keys())
        labels = config["options"]
        default_value = config["default"]
        default_index = options.index(default_value)
        return st.selectbox(
            label,
            options=options,
            index=default_index,
            format_func=lambda value: labels[value],
        )

    if kind == "number":
        return st.number_input(
            label,
            min_value=float(config["min"]),
            max_value=float(config["max"]),
            value=float(config["default"]),
            step=float(config["step"]),
        )

    raise ValueError(f"Type de widget inconnu : {kind}")


def render_form() -> dict[str, Any]:
    values: dict[str, Any] = {}
    groups = schema_by_group()

    with st.form("clinical_form"):
        for group_name, field_names in groups.items():
            st.subheader(GROUP_TITLES[group_name])
            if group_name == "administratif":
                first_col, second_col = st.columns([1, 1.5])
            else:
                first_col, second_col = st.columns(2)

            for index, field_name in enumerate(field_names):
                with (first_col if index % 2 == 0 else second_col):
                    values[field_name] = render_field(field_name)

        submitted = st.form_submit_button("Analyser le dossier")

    values["_submitted"] = submitted
    return values


def probability_gauge(probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(probability * 100, 1),
            number={"suffix": "%"},
            title={"text": "Probabilité de succès"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 45], "color": "#fee2e2"},
                    {"range": [45, 70], "color": "#fef3c7"},
                    {"range": [70, 100], "color": "#dcfce7"},
                ],
                "bar": {"color": "#2563eb"},
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def local_explanation(pipeline, row_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    background_df = load_background_reference()

    transformed_row = preprocessor.transform(row_df)
    transformed_background = preprocessor.transform(background_df)

    if hasattr(transformed_row, "toarray"):
        transformed_row = transformed_row.toarray()
    if hasattr(transformed_background, "toarray"):
        transformed_background = transformed_background.toarray()

    transformed_names = preprocessor.get_feature_names_out().tolist()

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed_row)

        if isinstance(shap_values, list):
            contributions = shap_values[1][0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            contributions = shap_values[0, :, 1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            contributions = shap_values[0]
        else:
            contributions = np.zeros(len(transformed_names))

    except Exception:
        baseline = np.nan_to_num(transformed_background.mean(axis=0))
        importances = getattr(model, "feature_importances_", np.ones(len(transformed_names)))
        contributions = importances * (transformed_row[0] - baseline)

    explanation = pd.DataFrame(
        {
            "feature": transformed_names,
            "contribution": contributions,
        }
    )
    explanation["abs_contribution"] = explanation["contribution"].abs()
    explanation["feature_label"] = explanation["feature"].apply(humanize_transformed_feature_name)
    explanation = explanation.sort_values("abs_contribution", ascending=False).head(top_n).reset_index(drop=True)
    return explanation


def age_sensitivity_curve(pipeline, raw_inputs: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for age in np.arange(1.0, 20.1, 1.0):
        probe_inputs = raw_inputs.copy()
        probe_inputs["Recipientage"] = float(age)
        probe_df = make_prediction_frame(probe_inputs)
        probability = float(pipeline.predict_proba(probe_df)[0, 1])
        rows.append({"age": float(age), "success_probability": probability})
    return pd.DataFrame(rows)


def save_prediction_to_session(result: dict[str, Any]) -> None:
    st.session_state["last_prediction"] = result


def render_result(result: dict[str, Any]) -> None:
    probability = result["success_probability"]
    risk_level = result["risk_level"]

    left_col, right_col = st.columns([1, 1.2], gap="large")

    with left_col:
        st.plotly_chart(probability_gauge(probability), use_container_width=True)
        _metric_card("Décision du modèle", result["predicted_label"], RISK_TEXT[risk_level], "#2563eb")
        _metric_card("Niveau de risque", risk_level, f"Probabilité = {probability:.1%}", "#0f766e")

        if result["warnings"]:
            st.warning("Alertes de saisie détectées :")
            for warning in result["warnings"]:
                st.write(f"- {warning}")

        st.write("**Variables dérivées calculées automatiquement**")
        engineered = result["engineered_inputs"]
        derived_rows = [
            ("Donorage35", engineered["Donorage35"]),
            ("Recipientage10", engineered["Recipientage10"]),
            ("Recipientageint", engineered["Recipientageint"]),
            ("ABOmatch", engineered["ABOmatch"]),
            ("CMVstatus", engineered["CMVstatus"]),
            ("Diseasegroup", engineered["Diseasegroup"]),
            ("HLAmismatch", engineered["HLAmismatch"]),
            ("Gendermatch", engineered["Gendermatch"]),
        ]
        st.dataframe(pd.DataFrame(derived_rows, columns=["variable", "value"]), use_container_width=True, hide_index=True)

    with right_col:
        st.write("**Explication locale de la prédiction**")
        explanation = result["explanation"].copy()
        chart = px.bar(
            explanation.sort_values("contribution"),
            x="contribution",
            y="feature_label",
            orientation="h",
            title="Top facteurs qui poussent la prédiction",
            labels={"contribution": "Contribution locale", "feature_label": "Variable"},
        )
        chart.update_layout(height=430, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(chart, use_container_width=True)

        st.write("**Sensibilité à l'âge du receveur**")
        sensitivity = result["age_sensitivity"]
        sensitivity_chart = px.line(
            sensitivity,
            x="age",
            y="success_probability",
            markers=True,
            title="Comment la probabilité change quand l'âge du receveur change",
            labels={"age": "Âge du receveur", "success_probability": "Probabilité de succès"},
        )
        sensitivity_chart.update_layout(height=330, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(sensitivity_chart, use_container_width=True)


def render_model_analysis(training_summary: dict[str, Any]) -> None:
    st.subheader("Analyse globale du modèle")

    comparison_df = load_model_comparison()
    if not comparison_df.empty:
        st.write("**Comparaison des modèles candidats**")
        st.dataframe(comparison_df.round(4), use_container_width=True, hide_index=True)

    importance_df = load_feature_importance()
    if not importance_df.empty:
        top_importance = importance_df.head(12).copy()
        top_importance["feature_label"] = top_importance["feature"].apply(humanize_transformed_feature_name)
        chart = px.bar(
            top_importance.sort_values("importance"),
            x="importance",
            y="feature_label",
            orientation="h",
            title="Importance globale des variables",
            labels={"importance": "Importance", "feature_label": "Variable"},
        )
        chart.update_layout(height=460, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(chart, use_container_width=True)

    st.write("**Résumé des données**")
    st.json(training_summary, expanded=False)

    st.info(
        "Cette version retire volontairement les variables post-greffe "
        "et les variables de survie qui faussaient la prédiction."
    )


def render_history() -> None:
    history = read_history()
    st.subheader("Historique enregistré dans Excel")
    if history.empty:
        st.info("Aucune prédiction enregistrée pour le moment.")
        return

    st.dataframe(history, use_container_width=True, hide_index=True)

    workbook_bytes = DEFAULT_LOG_PATH.read_bytes()
    st.download_button(
        label="Télécharger le fichier Excel",
        data=workbook_bytes,
        file_name=DEFAULT_LOG_PATH.name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def main() -> None:
    st.set_page_config(page_title="GreffePred", page_icon="🩺", layout="wide")

    if not MODEL_PATH.exists():
        st.error("Modèle introuvable. Lancez d'abord : python src/train_model.py")
        st.stop()

    if not st.session_state.get("authenticated"):
        render_login()
        return

    pipeline = load_pipeline()
    training_summary = load_training_summary()
    render_sidebar(training_summary)

    st.title("GreffePred — aide à la décision pour la greffe pédiatrique")
    st.caption(
        "Interface corrigée : connexion médecin, contrôles de saisie, "
        "comparaison de modèles, explication locale et historique Excel."
    )

    tab_form, tab_result, tab_analysis, tab_history = st.tabs(
        ["Nouveau dossier", "Résultat clinique", "Analyses du modèle", "Historique Excel"]
    )

    with tab_form:
        input_values = render_form()

        if input_values["_submitted"]:
            validation = validate_input_record(input_values)

            if validation["errors"]:
                st.error("Le formulaire contient des erreurs.")
                for error in validation["errors"]:
                    st.write(f"- {error}")
            else:
                prediction_frame = make_prediction_frame(input_values)
                probability = float(pipeline.predict_proba(prediction_frame)[0, 1])
                predicted_label = "Survie probable" if probability >= 0.5 else "Risque de non-survie"
                risk_level = risk_level_from_probability(probability)

                explanation_df = local_explanation(pipeline, prediction_frame)
                age_sensitivity_df = age_sensitivity_curve(pipeline, input_values)

                engineered_inputs = prediction_frame.iloc[0].to_dict()
                append_prediction(
                    username=st.session_state["username"],
                    raw_inputs=input_values,
                    engineered_inputs=engineered_inputs,
                    predicted_label=predicted_label,
                    success_probability=probability,
                    warnings=validation["warnings"],
                )

                result = {
                    "raw_inputs": input_values,
                    "engineered_inputs": engineered_inputs,
                    "success_probability": probability,
                    "predicted_label": predicted_label,
                    "risk_level": risk_level,
                    "warnings": validation["warnings"],
                    "explanation": explanation_df,
                    "age_sensitivity": age_sensitivity_df,
                }
                save_prediction_to_session(result)
                st.success("Prédiction calculée et enregistrée dans le fichier Excel.")

    with tab_result:
        result = st.session_state.get("last_prediction")
        if result is None:
            st.info("Aucune prédiction récente. Remplissez le formulaire dans l'onglet précédent.")
        else:
            render_result(result)

    with tab_analysis:
        render_model_analysis(training_summary)

    with tab_history:
        render_history()


if __name__ == "__main__":
    main()
