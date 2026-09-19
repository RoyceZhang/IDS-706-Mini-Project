import pandas as pd

from insurance_project import (
    load_data,
    preprocess_data,
    filter_smokers,
    get_smoker_summary,
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


def test_full_machine_learning_workflow():
    df = load_data("insurance.csv")
    df = preprocess_data(df)

    model, predictions, mse, r2 = train_model(df)

    assert len(predictions) > 0
    assert mse >= 0
    assert -1 <= r2 <= 1
    assert hasattr(model, "predict")