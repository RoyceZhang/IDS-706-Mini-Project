import pandas as pd
import pytest

from insurance_project import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    evaluate_predictions,
    filter_smokers,
    get_smoker_summary,
    load_data,
    prepare_features,
    preprocess_data,
    train_model,
)


def test_load_data():
    df = load_data("insurance.csv")

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "charges" in df.columns


def test_preprocess_data():
    df = load_data("insurance.csv")

    cleaned_df = preprocess_data(df)

    assert cleaned_df.duplicated().sum() == 0
    assert len(cleaned_df) <= len(df)


def test_preprocess_data_removes_duplicate_rows():
    duplicate_row = {
        "age": 30,
        "sex": "female",
        "bmi": 25.0,
        "children": 0,
        "smoker": "no",
        "region": "northeast",
        "charges": 5000.0,
    }
    df = pd.DataFrame([duplicate_row, duplicate_row])

    cleaned_df = preprocess_data(df)

    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0].to_dict() == duplicate_row


def test_filter_smokers_returns_empty_dataframe_when_there_are_no_smokers():
    df = pd.DataFrame(
        {
            "smoker": ["no", "no"],
            "charges": [3000.0, 4500.0],
        }
    )

    smokers = filter_smokers(df)

    assert smokers.empty
    assert list(smokers.columns) == list(df.columns)


def test_filter_and_group():
    df = load_data("insurance.csv")

    smokers = filter_smokers(df)

    assert len(smokers) > 0
    assert (smokers["smoker"] == "yes").all()

    summary = get_smoker_summary(df)

    assert "yes" in summary.index
    assert "no" in summary.index
    assert "mean" in summary.columns
    assert "count" in summary.columns


def test_prepare_features_selects_expected_columns():
    df = load_data("insurance.csv")

    features, target = prepare_features(df)

    assert list(features.columns) == FEATURE_COLUMNS
    assert target.name == TARGET_COLUMN
    assert len(features) == len(target) == len(df)


def test_evaluate_predictions_with_perfect_predictions():
    y_true = [1000.0, 2000.0, 3000.0]

    mse, r2 = evaluate_predictions(y_true, y_true)

    assert mse == pytest.approx(0.0)
    assert r2 == pytest.approx(1.0)


def test_full_machine_learning_workflow():
    df = load_data("insurance.csv")
    df = preprocess_data(df)

    model, predictions, mse, r2 = train_model(df)

    assert len(predictions) > 0
    assert mse >= 0
    assert r2 <= 1
    assert hasattr(model, "predict")
