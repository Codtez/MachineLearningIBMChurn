from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from preprocess_data import load_data, create_preprocessing_pipeline


def optimize_hyperparameters(filepath):

    df = load_data(filepath)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=1111
    )

    preprocessor = create_preprocessing_pipeline()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=1111))
    ])

    param_dist = {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
        "model__class_weight": [None, "balanced"]
    }

    cv_strategy = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=1111
    )

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=30,
        cv=cv_strategy,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        random_state=1111
    )

    search.fit(X_train, y_train)

    print("Best Params:", search.best_params_)
    print("Best ROC-AUC:", search.best_score_)

    return search.best_estimator_


optimize_hyperparameters("../data/Churn4500.csv")
