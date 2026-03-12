import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocess_data import load_data, create_preprocessing_pipeline


def train_model(filepath):

    # Load dataset
    df = load_data(filepath)

    # Separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=1111
    )

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline()

    # Create model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=1111
    )

    # Combine preprocessing + model
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )
    #####################################################
    # Cross-validation for reviewing consistency of model
    cv_strategy = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=1111
    )

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv_strategy,
        scoring="accuracy"
    )

    print("Cross-validation scores:", cv_scores)
    print("Mean c-v score:", cv_scores.mean())
    ##########################################

    # Train model on full training set (Churn4500.csv)
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Evaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return pipeline
