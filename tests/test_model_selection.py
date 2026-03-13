import pandas as pd

from src.data_processing import build_training_data
from src.train_model import compare_models, make_candidate_models, select_best_model


def test_compare_models_returns_expected_columns():
    feature_frame, target = build_training_data()
    fast_models = make_candidate_models(fast_mode=True)
    results = compare_models(feature_frame, target, models=fast_models, cv_splits=3)

    expected_columns = {
        "model_name",
        "roc_auc_mean",
        "roc_auc_std",
        "accuracy_mean",
        "balanced_accuracy_mean",
        "f1_mean",
    }
    assert expected_columns.issubset(results.columns)
    assert len(results) == len(fast_models)


def test_compare_models_is_sorted_by_roc_auc():
    feature_frame, target = build_training_data()
    fast_models = make_candidate_models(fast_mode=True)
    results = compare_models(feature_frame, target, models=fast_models, cv_splits=3)
    roc_auc_values = results["roc_auc_mean"].tolist()
    assert roc_auc_values == sorted(roc_auc_values, reverse=True)


def test_select_best_model_returns_first_row_name():
    sample_results = pd.DataFrame(
        [
            {"model_name": "A", "roc_auc_mean": 0.7, "balanced_accuracy_mean": 0.6, "accuracy_mean": 0.6},
            {"model_name": "B", "roc_auc_mean": 0.6, "balanced_accuracy_mean": 0.7, "accuracy_mean": 0.7},
        ]
    )
    assert select_best_model(sample_results) == "A"
