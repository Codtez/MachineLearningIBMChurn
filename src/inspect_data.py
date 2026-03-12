import pandas as pd


def inspect_dataset(path):

    print("\n=============================")
    print(f"Inspecting: {path}")
    print("=============================")

    df = pd.read_csv(path)

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns)

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values Per Column:")
    print(df.isnull().sum())

    print("\nChurn Distribution:")
    print(df["Churn"].value_counts())

    print("\nFirst 5 Rows:")
    print(df.head())


#inspect_dataset("../data/Churn4500.csv")
inspect_dataset("../data/Churn2500.csv")