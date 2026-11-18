import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from src.validate_data import validar_archivo_csv, validar_payload_json

def test_data_loading():
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    assert not train.empty and not test.empty
    assert "species" in train.columns

def test_native_data_validation():
    ok_train, _ = validar_archivo_csv("data/processed/train.csv")
    ok_test, _ = validar_archivo_csv("data/processed/test.csv")
    assert ok_train and ok_test

def test_data_contract():
    with open("data_contract.json", "r", encoding="utf-8") as f:
        contract = json.load(f)
    ok_v, _ = validar_payload_json(contract["examples"]["valid_input"]["payload"])
    ok_inv, errs = validar_payload_json(contract["examples"]["invalid_input"]["payload"])
    assert ok_v is True
    assert ok_inv is False and len(errs) == 4

def test_model_exists():
    model = joblib.load("models/model.pkl")
    assert model is not None

def test_model_accuracy():
    model = joblib.load("models/model.pkl")
    test = pd.read_csv("data/processed/test.csv")
    X_test = test.drop("species", axis=1)
    y_test = test["species"]
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy > 0.8
