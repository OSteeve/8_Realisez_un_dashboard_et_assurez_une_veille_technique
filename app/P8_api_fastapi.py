from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import os
import numpy as np
from typing import Optional

app = FastAPI()

# Base directory
BASE_DIR = os.path.dirname(__file__)

# Chargement du modèle et des données utiles
pipe_path = os.path.join(BASE_DIR, "pipe_lgbm.joblib")
pipe = joblib.load(pipe_path)

threshold_path = os.path.join(BASE_DIR, "threshold_lgbm.joblib")
threshold = joblib.load(threshold_path)

data_path = os.path.join(BASE_DIR, "app_data.joblib")
data = joblib.load(data_path)

global_importance_path = os.path.join(BASE_DIR, "global_importance.joblib")
global_importance_data = joblib.load(global_importance_path)

# Extraction du modèle et imputer du pipeline
model = pipe.named_steps["model"]
imputer = pipe.named_steps["imputer"]

# Suppression des colonnes non explicatives
# features utilisées par le modèle
feats = [
    f for f in data.columns
    if f not in ["TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"]
]

# Suppression de la variable sensible (RGPD) "CODE_GENDER"
feats_safe = [
    f for f in data.columns
    if f not in ["TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index","CODE_GENDER"]
]

# Dataset réduit aux features explicatives
X_data = data[feats].copy()
X_data_transformed = imputer.transform(X_data)

# Scores de la population
population_scores = pipe.predict_proba(X_data)[:, 1]
data["MODEL_SCORE"] = population_scores

population_prediction = (data["MODEL_SCORE"] >= threshold).astype(int)
data["MODEL_PREDICTION"] = population_prediction

# définition de shap
explainer = shap.TreeExplainer(model)

feature_names = global_importance_data["feature_names"]
global_importances = global_importance_data["shap_values"]

#shap_values_global = explainer(X_data_transformed)
#global_importances = np.abs(shap_values_global.values).mean(axis=0).tolist()
#feature_names = [str(col) for col in feats]


class ClientRequest(BaseModel):
    SK_ID_CURR: int

class FeatureRequest(BaseModel):
    SK_ID_CURR: int
    feature: str
    filter_feature: Optional[str] = None

class ModifyFeatureRequest(BaseModel):
    SK_ID_CURR: int
    modified_features: dict


def get_client_features(client_id: int) -> pd.DataFrame:
    # selection du client
    client_data = data[data["SK_ID_CURR"] == client_id].copy()

    # Erreur si introuvable
    if client_data.empty:
        raise ValueError(f"Client {client_id} introuvable")
    
    # Récupération des data du clients
    X_client = client_data[feats].copy()
   
    return X_client

# ENDPOINT CLIENT
@app.get("/clients")
def get_clients():
    return data["SK_ID_CURR"].tolist()

# ENDPOINT PREDICTION
@app.post("/predict")
def predict(request: ClientRequest):
    # features du client selectionné
    X_client = get_client_features(request.SK_ID_CURR)
    
    proba_client = pipe.predict_proba(X_client)[0, 1]
    prediction = int(proba_client >= threshold)

    return {
        "SK_ID_CURR": request.SK_ID_CURR,
        "proba_client": float(proba_client),
        "prediction": prediction,
        "threshold": float(threshold)
    }

# ENDPOINT NOUVELLE VALEUR CLIENT
@app.post("/new_predict")
def new_predict(request: ModifyFeatureRequest) :
    # récupérer les données client originales
    X_client = get_client_features(request.SK_ID_CURR)

    # appliquer les modifications
    for feature, value in request.modified_features.items():
        X_client.loc[:, feature] = value

    # prédiction sur client modifié
    proba = pipe.predict_proba(X_client)[0, 1]
    prediction = int(proba >= threshold)

    return {
        "SK_ID_CURR": request.SK_ID_CURR,
        "proba": float(proba),
        "prediction": prediction,
        "threshold": float(threshold),
        "modified_features": request.modified_features
    }  


# ENDPOINT IMPORTANCE GLOBALE
@app.get("/global_importance")
def get_global_importance():
    return {
        "feature_names": list(feature_names),
        "shap_values": list(global_importances)
    }

# ENDPOINT IMPORTANCE LOCALE
@app.post("/local_importance")
def importance(request: ClientRequest):
    # features du client selectionné
    X_client = get_client_features(request.SK_ID_CURR)

    # Calcul des valeurs shap client
    X_client_transformed = imputer.transform(X_client)
    shap_values = explainer(X_client_transformed)

    # Extraction des noms des features
    feature_names_locale = [str(col) for col in X_client.columns.tolist()]
    
    # Récupération des valeurs des chaque features
    feature_values_locale = X_client_transformed[0].tolist()
    base_value = float(shap_values.base_values[0])

    return {
        "SK_ID_CURR": request.SK_ID_CURR,
        "shap_values": shap_values.values[0].tolist(),
        "feature_names": feature_names_locale,
        "feature_values": feature_values_locale,
        "base_value":base_value
        }

# ENDPOINT COMPARAISON DES FEATURES
@app.post("/feature_comparison")
def feature_comparison(request: FeatureRequest):
    client_row = data.set_index("SK_ID_CURR").loc[request.SK_ID_CURR]

    # Valeur client
    client_value = client_row[request.feature]
    client_value = None if pd.isna(client_value) else float(client_value)

    # Population value
    population_values = (
        data[request.feature]
        .dropna()
        .astype(float)
        .tolist()
    )

    # Dataframe des valeurs d'un feature et le score 
    df_feature = data[[request.feature, "MODEL_SCORE"]].dropna()

    # Mise sous forme de dictionnaire exploitable par l'API
    # et modification des noms de colonnes
    population_data = df_feature.rename(
        columns={request.feature: "value", "MODEL_SCORE": "score"}
        ).to_dict("records")


    # Population value
    population_values = (
        data[request.feature]
        .dropna()
        .astype(float)
        .tolist()
    )



    '''population_data = [
        {"value": v, "score": s}
        for v, s in zip(population_values, population_scores)
    ]'''

    return {
        "feature": request.feature,
        "client_value": client_value,
        "population_values": population_values,
        }




