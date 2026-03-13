"""
Entraînement, comparaison et sauvegarde du modèle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from data_processing import (
    MODEL_FEATURE_COLUMNS,
    build_preprocessor,
    build_training_data,
    dataset_quality_summary,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

MODEL_PATH = DATA_DIR / "best_model.joblib"
COMPARISON_PATH = DATA_DIR / "model_comparison.csv"
SUMMARY_PATH = DATA_DIR / "training_summary.json"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "global_feature_importance.csv"
BACKGROUND_PATH = DATA_DIR / "background_reference.csv"


def make_candidate_models(random_state: int = 42, fast_mode: bool = False) -> dict[str, Any]:
    """
    Déclare les modèles candidats.
    """
    rf_estimators = 50 if fast_mode else 220
    et_estimators = 70 if fast_mode else 260

    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=rf_estimators,
            random_state=random_state,
            class_weight="balanced",
            min_samples_leaf=2,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=et_estimators,
            random_state=random_state,
            class_weight="balanced",
            min_samples_leaf=2,
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
    }


def build_model_pipeline(feature_frame: pd.DataFrame, estimator: Any) -> Pipeline:
    """
    Assemble préprocesseur et estimateur.
    """
    preprocessor = build_preprocessor(feature_frame)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def compare_models(
    feature_frame: pd.DataFrame,
    target: pd.Series,
    models: dict[str, Any] | None = None,
    cv_splits: int = 5,
) -> pd.DataFrame:
    """
    Évalue tous les modèles et renvoie un tableau trié.
    """
    candidate_models = models or make_candidate_models()
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    rows: list[dict[str, Any]] = []

    for name, estimator in candidate_models.items():
        pipeline = build_model_pipeline(feature_frame, estimator)
        scores = cross_validate(
            pipeline,
            feature_frame,
            target,
            cv=cv,
            scoring=["roc_auc", "accuracy", "balanced_accuracy", "f1"],
            n_jobs=1,
        )

        rows.append(
            {
                "model_name": name,
                "roc_auc_mean": float(scores["test_roc_auc"].mean()),
                "roc_auc_std": float(scores["test_roc_auc"].std()),
                "accuracy_mean": float(scores["test_accuracy"].mean()),
                "balanced_accuracy_mean": float(scores["test_balanced_accuracy"].mean()),
                "f1_mean": float(scores["test_f1"].mean()),
            }
        )

    results = pd.DataFrame(rows)
    results = results.sort_values(
        by=["roc_auc_mean", "balanced_accuracy_mean", "accuracy_mean"],
        ascending=False,
    ).reset_index(drop=True)
    return results


def select_best_model(results: pd.DataFrame) -> str:
    """
    Sélectionne le meilleur modèle à partir du tableau comparatif.
    """
    if results.empty:
        raise ValueError("Aucun résultat de comparaison disponible.")
    return str(results.loc[0, "model_name"])


def export_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """
    Sauvegarde l'importance globale des features si le modèle le permet.
    """
    estimator = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    if not hasattr(estimator, "feature_importances_"):
        empty_report = pd.DataFrame(columns=["feature", "importance"])
        empty_report.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
        return empty_report

    transformed_names = preprocessor.get_feature_names_out().tolist()
    importance_df = pd.DataFrame(
        {
            "feature": transformed_names,
            "importance": estimator.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    return importance_df


def train_and_persist(fast_mode: bool = False) -> dict[str, Path]:
    """
    Lance l'entraînement complet et écrit tous les artefacts utiles.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    feature_frame, target = build_training_data()
    candidate_models = make_candidate_models(fast_mode=fast_mode)
    comparison = compare_models(feature_frame, target, models=candidate_models, cv_splits=5 if not fast_mode else 3)
    best_model_name = select_best_model(comparison)
    best_pipeline = build_model_pipeline(feature_frame, candidate_models[best_model_name])
    best_pipeline.fit(feature_frame, target)

    joblib.dump(best_pipeline, MODEL_PATH)
    comparison.to_csv(COMPARISON_PATH, index=False)
    export_feature_importance(best_pipeline)

    background_reference = feature_frame.sample(min(60, len(feature_frame)), random_state=42)
    background_reference.to_csv(BACKGROUND_PATH, index=False)

    summary = dataset_quality_summary()
    summary.update(
        {
            "best_model": best_model_name,
            "best_model_metrics": comparison.iloc[0].to_dict(),
            "model_artifact": MODEL_PATH.name,
            "comparison_artifact": COMPARISON_PATH.name,
            "feature_count": len(MODEL_FEATURE_COLUMNS),
        }
    )
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "model": MODEL_PATH,
        "comparison": COMPARISON_PATH,
        "summary": SUMMARY_PATH,
        "feature_importance": FEATURE_IMPORTANCE_PATH,
        "background_reference": BACKGROUND_PATH,
    }


if __name__ == "__main__":
    artifacts = train_and_persist()
    print("Artefacts générés :")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")
