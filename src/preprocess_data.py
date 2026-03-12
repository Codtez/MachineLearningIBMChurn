from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd


def load_data(filepath):

    df = pd.read_csv(filepath)

    # Fix TotalCharges column
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop rows with missing TotalCharges
    df = df.dropna(subset=["TotalCharges"])

    # Drop customerID column
    df = df.drop(columns=["customerID"])

    return df


def create_preprocessing_pipeline():

    # Numeric columns
    numeric_features = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    # Categorical columns
    categorical_features = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ]

    # Numeric preprocessing
    numeric_transformer = StandardScaler()

    # Categorical preprocessing
    # EX: Automatically converts 'Contract' 3 string answers (month-month etc) to 3 yes/no's (is 2 year contract?)
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")  # if new categories appear in test data, ignore

    # Combine transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor