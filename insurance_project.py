import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["age", "bmi", "children"]
TARGET_COLUMN = "charges"


def load_data(file_path):
    """Load the insurance dataset."""
    return pd.read_csv(file_path)


def preprocess_data(df):
    """Remove duplicate rows from the dataset."""
    return df.drop_duplicates()


def filter_smokers(df):
    """Return only rows where the person is a smoker."""
    return df[df["smoker"] == "yes"]


def get_smoker_summary(df):
    """Calculate average charges and count by smoking status."""
    return df.groupby("smoker")["charges"].agg(["mean", "count"])


def prepare_features(df):
    """Select the model features and target from the dataset."""
    features = df[FEATURE_COLUMNS]
    target = df[TARGET_COLUMN]
    return features, target


def evaluate_predictions(y_true, predictions):
    """Calculate mean squared error and R-squared for predictions."""
    mse = mean_squared_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    return mse, r2


def train_model(df):
    """Train a linear regression model using numerical features."""
    features, target = prepare_features(df)

    (
        features_train,
        features_test,
        target_train,
        target_test,
    ) = train_test_split(features, target, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(features_train, target_train)

    predictions = model.predict(features_test)

    mse, r2 = evaluate_predictions(target_test, predictions)

    return model, predictions, mse, r2
