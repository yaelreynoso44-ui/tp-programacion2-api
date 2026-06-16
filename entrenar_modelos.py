
import joblib
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# ==========================
# MODELO 1 - TITANIC
# ==========================

print("Entrenando modelo Titanic...")

titanic = fetch_openml("titanic", version=1, as_frame=True)

df = titanic.frame

df = df[["pclass", "sex", "age", "fare", "survived"]]

df["age"] = df["age"].fillna(df["age"].median())
df["fare"] = df["fare"].fillna(df["fare"].median())

df["sex"] = df["sex"].map({
    "male": 0,
    "female": 1
})

X = df[["pclass", "sex", "age", "fare"]]
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

modelo_titanic = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

modelo_titanic.fit(X_train, y_train)

pred = modelo_titanic.predict(X_test)

acc = accuracy_score(y_test, pred)

print(f"Accuracy Titanic: {acc:.4f}")

joblib.dump(
    modelo_titanic,
    "saved_models/titanic_model.pkl"
)

# ==========================
# MODELO 2 - CALIFORNIA
# ==========================

print("Entrenando modelo Housing...")

housing = fetch_california_housing(as_frame=True)

X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

modelo_housing = RandomForestRegressor(
    n_estimators=10,
    random_state=42
)

modelo_housing.fit(X_train, y_train)

pred = modelo_housing.predict(X_test)

mse = mean_squared_error(y_test, pred)

print(f"MSE Housing: {mse:.4f}")

joblib.dump(
    modelo_housing,
    "saved_models/housing_model.pkl"
)

print("Modelos guardados correctamente.")