import pandas as pd
import matplotlib.pyplot as plt
from train_model import train_model_simple

def get_importances(pipeline):

    # Extract model from pipeline
    model = pipeline.named_steps["model"]

    # Get feature names after preprocessing
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    # Get importance scores
    importances = model.feature_importances_

    # Build dataframe
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    # Sort by importance
    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    return importance_df


def plot_importances(importance_df, top_n=20):

    top_features = importance_df.head(top_n)

    plt.figure(figsize=(10,6))
    plt.barh(
        top_features["feature"],
        top_features["importance"]
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Importance-Influence/1.0")
    plt.title("Top Features Driving Churn")

    plt.show()

# Run


pipeline = train_model_simple("../data/Churn4500.csv")

importance_df = get_importances(pipeline)

print(importance_df.head(20))

plot_importances(importance_df)

