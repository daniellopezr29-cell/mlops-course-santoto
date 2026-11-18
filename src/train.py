import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

os.makedirs("models", exist_ok=True)
mlflow.set_experiment("Proyecto_Final_Grado")

with mlflow.start_run(run_name="Pipeline_Semana2_Optimo"):
    n_estimators = 50
    max_depth = 3
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    train = pd.read_csv("data/processed/train.csv")
    X_train = train.drop("species", axis=1)
    y_train = train["species"]

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    test = pd.read_csv("data/processed/test.csv")
    X_test = test.drop("species", axis=1)
    y_test = test["species"]

    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(
        model,
        "iris_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    joblib.dump(model, "models/model.pkl")
    print(f"[OK] Modelo entrenado con accuracy: {accuracy:.4f} y guardado en models/model.pkl")
