from train_model import train_full_model
from sklearn.metrics import roc_auc_score, classification_report
from preprocess_data import load_data
import numpy as np

# Train final model on ALL original data
pipeline = train_full_model("../data/Churn4500.csv")

# Load new dataset
df_new = load_data("../data/Churn2500.csv")

X_new = df_new.drop(columns=["Churn"])
y_new = df_new["Churn"]

# Predict
y_probs = pipeline.predict_proba(X_new)[:, 1]

threshold = 0.42
y_pred = np.where(y_probs >= threshold, "Yes", "No")

# Evaluate
print("ROC-AUC:", roc_auc_score(y_new, y_probs))
print(classification_report(y_new, y_pred))