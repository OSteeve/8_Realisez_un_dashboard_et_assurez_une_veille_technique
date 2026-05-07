import pandas as pd
import streamlit as st
import requests
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import numpy as np
import os

# API_URL = "http://localhost:8000"
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Scoring client")



# Charger les clients via API
clients = requests.get(f"{API_URL}/clients").json()
# sélection du client
client_id = st.selectbox(
    "Selectionner ou taper un ID client",
    clients
)


#########################################################################################
# AFFICHAGE DU SCORE
#########################################################################################
# appel API predict
response = requests.post(
    f"{API_URL}/predict",
    json={"SK_ID_CURR": int(client_id)}
)
result = response.json()

# Valeurs
proba = result["proba_client"]
threshold = result["threshold"]
prediction = result["prediction"]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Probabilité client", f"{proba:.3f}")
with col2:
    st.metric("Seuil de décision", f"{threshold:.3f}")
with col3:
    st.metric("Écart au seuil", f"{(proba - threshold):.3f}")


#########################################################################################
# Prédiction
#########################################################################################
def gauge (proba, threshold, prediction, title="Score client" ) :
    
    st.caption(f"Seuil de décision : {threshold:.3f}")
    # Jauge
    bar_color = "red" if proba >= threshold else "green"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=proba,
            number={"valueformat": ".3f"},
            title={"text": title},
            gauge={
                "bar": {"color": bar_color},  # aiguille / valeur
                "axis": {"range": [0, 1]},
                "steps": [
                {"range": [0, threshold], "color": "#9ce7a6"},
                {"range": [threshold, 1], "color": "#e77a77"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 2},
                    "value": threshold
                }
            }
        )
    )
    st.plotly_chart(fig)

    # Décision
    st.subheader("Décision")
    if prediction == 1:
        st.error("## REFUS")
    else:
        st.success("## ACCORD")

gauge(proba=proba,
      threshold=threshold,
      prediction=prediction
      )

#########################################################################################
# FEATURE IMPORTANCE LOCALE
#########################################################################################


def local_importance () :

    proba = result["proba"]
    threshold = result["threshold"]

    # shap, envoi de la requête
    response_shap = requests.post(
        f"{API_URL}/local_importance",
        json={"SK_ID_CURR": int(client_id)}
    )

    # Récupération du dictionnaire en reponse
    shap_result = response_shap.json()


    st.subheader("Influence des variables")

    col1, col2 = st.columns(2)

    with col1:
        st.badge("Diminue le *risque* → ACCORD", color="blue")

    with col2:
        st.badge("Augmente le *risque* → REFUS", color="red")

    # Créeation de l'objet SHAP
    explanation = shap.Explanation(
        values=np.array(shap_result["shap_values"]),
        base_values=shap_result["base_value"],
        data=np.array(shap_result["feature_values"]),
        feature_names=shap_result["feature_names"]
        )
    

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=20, show=False)
    st.pyplot(fig)

if st.button("Influence des variables client"):  
    local_importance ()

#########################################################################################
# ANALYSE DES VARIABLES
#########################################################################################
st.subheader("Analyse des variables")

analysis_type = st.selectbox(
    "",
    ["- Choix de l'analyse -","Contribution des variables au score", "Valeur de la variable"]
    )


# 1. Chargement des données importance
if analysis_type != "- Choix de l'analyse -":

    response_shap = requests.post(
        f"{API_URL}/local_importance",
        json={"SK_ID_CURR": int(client_id)}
    ).json()

    response_global = requests.get(
        f"{API_URL}/global_importance"
    ).json()

    # Importance locale
    local_df = pd.DataFrame({
        "feature": response_shap["feature_names"],
        "shap_value": response_shap["shap_values"]
    })
    local_df["abs_shap"] = local_df["shap_value"].abs()

    # Importance globale
    global_df = pd.DataFrame({
        "feature": response_global["feature_names"],
        "global_importance": response_global["importance_values"]
    })

    # Top 30 global
    top_feat_global = (
        global_df
        .sort_values("global_importance", ascending=False)
        .head(30)["feature"]
        .tolist()
    )

    # Top 10 local
    top_local_importance = (
        local_df
        .sort_values("abs_shap", ascending=False)
        .head(10)
        .copy()
    )

    top_feat_local = top_local_importance["feature"].tolist()

    # Union top local + top global
    feat_local_global = (
        pd.Series(top_feat_local + top_feat_global)
        .drop_duplicates()
        .tolist()
    )

    # Comparaison locale / globale sur top 10 local
    compare_df = top_local_importance.merge(
        global_df,
        on="feature",
        how="left"
    )

    compare_df = compare_df.sort_values("abs_shap", ascending=True)



# 2. IMPORTANCE LOCAL VS GLOBAL
if analysis_type == "Contribution des variables au score":

    col1, col2 = st.columns(2)
    with col1:
        #st.subheader("Importance locale")
        fig_local = go.Figure()
        fig_local.add_trace(go.Bar(
            x=compare_df["abs_shap"],
            y=compare_df["feature"],
            orientation="h",
            marker_color=[
                "#d9534f" if x > 0 else "green"
                for x in compare_df["shap_value"]
            ]
        ))
        fig_local.update_layout(
            height=500,
            title={
            "text": "TOP variables client",
            "x": 0.5,
            "xanchor": "center"}
        )
        st.plotly_chart(fig_local, use_container_width=True)

    with col2:
        #st.subheader("Importance globale")
        fig_global = go.Figure()
        fig_global.add_trace(go.Bar(
            x=compare_df["global_importance"],
            y=compare_df["feature"],
            orientation="h",
            marker_color="lightgray"
        ))
        fig_global.update_layout(
            xaxis_title="",
            yaxis_title="",
            height=500,
            title={
            "text": "généralité (valeur absolue)",
            "x": 0.5,
            "xanchor": "center"}
        )
        st.plotly_chart(fig_global, use_container_width=True)
    # st.write(compare_df.sort_values("abs_shap", ascending=False))


# 3. VALEUR DES PRINCIPALES VARIABLES
if analysis_type == "Valeur de la variable":

    st.caption("Choisissez une variable pour voir où se situe le client par rapport aux autres.")

    selected_feature = st.selectbox(
    "Choisir une variable",
    feat_local_global
    )


    response_requests_feature = requests.post(
        f"{API_URL}/feature_comparison",
        json={
            "SK_ID_CURR": int(client_id),
            "feature": selected_feature,
            }
            )
    
    response_feature = response_requests_feature.json()

    client_value = response_feature["client_value"]
    population_values = response_feature["population_values"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=population_values,
        xbins=dict(
            start=min(population_values),
            end=max(population_values),
            size=(max(population_values) - min(population_values)) / 30
            ),
        name="Population"
    ))

    if client_value is not None:
        fig.add_vline(
            x=client_value,
            line_color="red",
            line_width=3,
            annotation_text="Client"
        )

    fig.update_layout(
        title=f"Valeur client vs population : {selected_feature}",
        xaxis_title=selected_feature,
        yaxis_title="Nombre de clients"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.metric(
        f"Valeur client - {selected_feature}",f"{client_value:.3f}"
    )



    #########################################################################################
    # 3.1 Simulation de modification client
    #########################################################################################
    st.subheader("Simulation de modification")


    # Récupérer la valeur actuelle de la variable
    response_feature_modif = requests.post(
        f"{API_URL}/feature_comparison",
        json={
            "SK_ID_CURR": int(client_id),
            "feature": selected_feature
        }
    ).json()


    new_value = st.number_input(
    "Nouvelle valeur",
    value=float(response_feature_modif["client_value"]) if response_feature_modif.get("client_value") is not None else 0.0
    )


    if st.button("Simuler le nouveau score"):

        modified_result = requests.post(
            f"{API_URL}/new_predict",
            json={
                "SK_ID_CURR": int(client_id),
                "modified_features": {selected_feature: new_value}
            }
        ).json()

        new_proba = modified_result["proba"]

        # tracer la nouvelle jauge
        gauge(proba=new_proba,
              threshold=threshold,
              prediction=modified_result["prediction"],
              title=f"Score après simulation par modification de :{selected_feature}")


        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Score initial")
            st.metric("Probabilité", f"{proba:.3f}")
            st.metric("Écart au seuil", f"{proba - threshold:.3f}")

        with col2:
            st.subheader("Score simulé")
            st.metric(
                "Nouvelle probabilité",
                f"{new_proba:.3f}",
                delta=f"{new_proba - proba:.3f}"
                )
            st.metric("Nouvel écart au seuil",
                    f"{new_proba - threshold:.3f}"
                    )

        if modified_result["prediction"] == 1:
            st.error("Décision simulée : REFUS")
        else:
            st.success("Décision simulée : ACCORD")


