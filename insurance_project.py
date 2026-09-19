import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


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


def train_model(df):
    """Train a linear regression model using numerical features."""
    X = df[["age", "bmi", "children"]]
    y = df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, predictions, mse, r2