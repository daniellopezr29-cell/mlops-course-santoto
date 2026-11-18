import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

os.makedirs("data/processed", exist_ok=True)
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data["species"] = iris.target

train, test = train_test_split(data, test_size=0.2, random_state=42)
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)
print("[OK] Datos procesados guardados en data/processed/train.csv y test.csv")
