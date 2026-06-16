from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="API de Predicción con Machine Learning",
    version="1.0"
)

# ==========================
# CARGAR MODELOS
# ==========================

modelo_titanic = joblib.load("saved_models/titanic_model.pkl")
modelo_housing = joblib.load("saved_models/housing_model.pkl")

# ==========================
# MODELOS DE ENTRADA
# ==========================

class TitanicInput(BaseModel):
    pclass: int
    sex: str
    age: float
    fare: float

class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# ==========================
# ENDPOINT STATUS
# ==========================

@app.get("/status")
def status():
    return {
        "status": "ok",
        "mensaje": "API funcionando correctamente"
    }

# ==========================
# ENDPOINT INFO
# ==========================

@app.get("/info")
def info():
    return {
        "materia": "Programación II",
        "proyecto": "API de Predicción con Machine Learning",
        "descripcion": "API REST desarrollada con FastAPI para clasificación y regresión.",

        "integrantes": [
            {
                "nombre": "Yael Reynoso",
                "dni": "43513562",
                "email": "Yaelreynoso44@gmail.com"
            },
            {
                "nombre": "Mauricio Antonilli",
                "dni": "37336018",
                "email": "m.antonilli@hotmail.com.ar"
            },
            {   "nombre": "Cesar Ezquer",
                "dni": "37310202",
                "email": "cesar.ezquer23@gmail.com"
            }
        ],

        "modelo_clasificacion": {
            "nombre": "Titanic Survival Predictor",
            "tipo": "Clasificación",
            "dataset": "Titanic",
            "endpoint": "/modelo1"
        },

        "modelo_regresion": {
            "nombre": "California Housing Predictor",
            "tipo": "Regresión",
            "dataset": "California Housing",
            "endpoint": "/modelo2"
        }
    }

# ==========================
# MODELO 1 - TITANIC
# ==========================

@app.post("/modelo1")
def predecir_titanic(data: TitanicInput):

    sexo = 1 if data.sex.lower() == "female" else 0

    entrada = pd.DataFrame([{
        "pclass": data.pclass,
        "sex": sexo,
        "age": data.age,
        "fare": data.fare
    }])

    pred = modelo_titanic.predict(entrada)[0]

    return {
        "prediccion": "Sobrevive" if pred == "1" or pred == 1 else "No sobrevive"
    }

# ==========================
# MODELO 2 - HOUSING
# ==========================

@app.post("/modelo2")
def predecir_housing(data: HousingInput):

    entrada = pd.DataFrame([{
        "MedInc": data.MedInc,
        "HouseAge": data.HouseAge,
        "AveRooms": data.AveRooms,
        "AveBedrms": data.AveBedrms,
        "Population": data.Population,
        "AveOccup": data.AveOccup,
        "Latitude": data.Latitude,
        "Longitude": data.Longitude
    }])

    pred = modelo_housing.predict(entrada)[0]

    return {
        "precio_estimado": round(float(pred), 2)
    }