import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
#import lightgbm
from matplotlib.image import imread
import altair as alt
import requests
import json
from sklearn.neighbors import NearestNeighbors


def main():
    seuil = 0.62
    #Title display
    html_temp = """
    <div style="background-color: tomato; padding:10px; border-radius:10px">
    <h1 style="color: white; text-align:center">PRET A DEPENSER</h1>
    <h1 style="color: white; text-align:center">Tableau de bord</h1>
    </div>
    <p style="font-size: 20px; font-weight: bold; text-align:center">Aide à la décision d'attribution d'un crédit</p>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    with st.empty():
        st.write("Chargement des données...", )
        [
            pipeline, data, data_display, data_display_target, liste_quali,
            shap_values, shap_exp, admin, data_shap
        ] = load_data()

        st.write("")

    PAGES = ["Score du client", "Modèle de scoring"]
    with st.sidebar:
        st.write('')
        st.write('')

        st.title('Menu')
        selection = st.radio("", PAGES)

    # client_ID = ID du client dans la base de données
    # client_id_tab = index du client dans la base de données
    client_ID, client_id_tab = get_client(data)
    score, result = prediction(client_ID, seuil)

    if selection == "Score du client":
        #Display Customer ID from Sidebar

        st.write("**ID client sélectionné :**", client_ID)
        infos_client(admin, client_ID)
        visu_score_client(score, result, seuil, client_id_tab, client_ID,
                          shap_exp, shap_values, data_display)
        if st.checkbox("Affichage des données détaillées du client"):
            st.write(identite_client(data_display, client_ID))
        if st.checkbox(
                "Positionnement du client par rapport à des clients similaires"
        ):
            comparaison(data, data_display_target, client_ID, admin)

    if selection == "Modèle de scoring":
        st.title("Modèle de scoring")
        st.write(
            "Comportement général du modèle : quelles informations sont principalement utilisées pour prévoir la solvabilité du client"
        )
        feature_importance(shap_value=shap_values,
                           df=data_shap,
                           max_display=15)


@st.cache
def load_data():
    '''
    chargement des données
    '''
    job_dir = './JOBLIB'
    # chargement du modèle
    pipeline = joblib.load(job_dir + '/pipeline.joblib')

    # jeu de données utilisé pour la modélisation (déjà encodé)
    data = joblib.load(job_dir + '/data_small.joblib')
    # st.write(data.shape)

    # jeu de données non encodé + target
    data_display_target = joblib.load(job_dir + '/data_no_encod.joblib')
    data_display = data_display_target.drop('TARGET', axis=1)

    # liste des données qualitatives
    liste_quali = joblib.load(job_dir + './liste_quali.joblib')

    # Données administratives
    admin = joblib.load(job_dir + './admin.joblib')
    admin['YEARS_BIRTH'] = admin['DAYS_BIRTH'].apply(lambda x: -int(x / 365))

    # preparation données pour shap
    data_shap = data.copy()
    data_shap[liste_quali] = data_shap[liste_quali].astype('int')
    explainer = shap.Explainer(pipeline[1], data_shap)
    shap_values = explainer(data_shap)
    shap_exp = explainer.expected_value

    return [
        pipeline, data, data_display, data_display_target, liste_quali,
        shap_values, shap_exp, admin, data_shap
    ]


def get_client(df):
    """
    Sélection d'un client via une selectbox
    """
    df = df.reset_index()
    client = st.sidebar.selectbox('Client', df['SK_ID_CURR'])
    idx_client = df.index[df['SK_ID_CURR'] == client][0]
    return client, idx_client


def prediction(id, seuil):
    '''Fonction permettant de prédire la capacité du client à rembourser son emprunt.
    les paramètres sont le modèle, le dataframe et l'ID du client'''

    # api-endpoint
    URL = "https://app-p7.herokuapp.com/" + str(id)

    # sending get request and saving the response as response object
    r = requests.get(url=URL)  #, params = PARAMS)

    result = json.loads(r.json())
    y_pred = result[0]
    decision = np.where(1 - y_pred < 1 - seuil, "Prêt Rejeté", "Prêt Accepté")

    return y_pred, decision


def get_client(df):
    #"""Sélection d'un client via une selectbox"""
    df = df.reset_index()
    client = st.sidebar.selectbox('Client', df['SK_ID_CURR'])
    idx_client = df.index[df['SK_ID_CURR'] == client][0]
    return client, idx_client


def color(pred):
    '''Définition de la couleur selon la prédiction'''
    if pred == "Prêt Accepté":
        col = 'Green'
    else:
        col = 'Red'
    return col


def st_shap(plot, height=None):
    """Fonction permettant l'affichage de graphique shap values"""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def infos_client(df, client_ID):
    """Affichage des infos du client sélectionné dans la barre latérale"""
    with st.sidebar:
        st.write("**_ID client :_**", str(client_ID))
        st.write("**_Sexe :_**", df.loc[client_ID, 'CODE_GENDER'])
        st.write("**_Statut familial :_**", df.loc[client_ID,
                                                   'NAME_FAMILY_STATUS'])
        st.write("**_Nombre d'enfants :_**",
                 str(int(df.loc[client_ID, 'CNT_CHILDREN'])))
        st.write("**_Age :_**", str(df.loc[client_ID, 'YEARS_BIRTH']), "ans")
        st.write("**_Statut professionnel :_**", df.loc[client_ID,
                                                        'NAME_INCOME_TYPE'])
        st.write("**_Niveau d'études:_**", df.loc[client_ID,
                                                  'NAME_EDUCATION_TYPE'])


def visu_score_client(score, result, seuil, client_id_tab, client_ID, shap_exp,
                      shap_values, data_display):
    '''
    visualisation du score et des variables principales ayant contribuées à ce score
    (explicabilité du modèle)
    '''
    st.title('Score du client')
    fig = go.Figure(
        go.Indicator(mode="gauge+number",
                     value=1 - score,
                     number={'font': {
                         'size': 48
                     }},
                     domain={
                         'x': [0, 1],
                         'y': [0, 1]
                     },
                     title={
                         'text': result.tolist(),
                         'font': {
                             'size': 28,
                             'color': color(result)
                         }
                     },
                     gauge={
                         'axis': {
                             'range': [0, 1],
                             'tickcolor': color(result)
                         },
                         'bar': {
                             'color': color(result)
                         },
                         'steps': [{
                             'range': [0, 1 - seuil],
                             'color': 'lightcoral'
                         }, {
                             'range': [1 - seuil, 1],
                             'color': 'lightgreen'
                         }],
                         'threshold': {
                             'line': {
                                 'color': "black",
                                 'width': 5
                             },
                             'thickness': 1,
                             'value': 1 - seuil
                         }
                     }))

    st.plotly_chart(fig)

    if st.checkbox(
            "Quelles variables ont joué un rôle décisif dans l'attribution de ce score ?"
    ):
        # st.write(
        #     "Quelles variables ont joué un rôle décisif dans l'attribution de ce score ?"
        # )
        shap.initjs()
        st.subheader("Diagramme SHAP - Waterfall")
        fig2, ax2 = plt.subplots(figsize=(20, 20))
        shap.plots.waterfall(shap_values[client_id_tab])
        st.pyplot(fig2)

        st.subheader("Diagramme SHAP - Force Plot")
        st_shap(
            shap.force_plot(shap_exp, shap_values[client_id_tab, :].values,
                            data_display.loc[client_ID, :]))


def identite_client(data, id):
    data_client = data[data.index == int(id)]
    return data_client


def comparaison(df, data_target, client_ID, admin):
    # df_train2, df_train, idx_client
    """Fonction principale de l'onglet 'Comparaison clientèle' """

    st.subheader('Comparaison clientèle')
    idx_neigh, total = get_neigh(df, client_ID)

    liste_var = np.sort(df.columns.to_list() +
                        [x for x in admin.columns if x not in df.columns])

    variable = st.selectbox("Variable à comparer :", liste_var)

    if variable in df.columns.to_list():
        db_neigh = data_target.iloc[idx_neigh, :].copy()
    else:
        db_neigh = admin.iloc[idx_neigh, :].copy()

    db_neigh['AMT_INCOME_TOTAL'] = db_neigh['AMT_INCOME_TOTAL'].apply(
        lambda x: int(x))
    db_neigh['AMT_CREDIT'] = db_neigh['AMT_CREDIT'].apply(lambda x: int(x))
    db_neigh['AMT_ANNUITY'] = db_neigh['AMT_ANNUITY'].apply(
        lambda x: x if pd.isna(x) else int(x))

    if total:
        if variable in df.columns:
            st.write("Valeur du client :", data_target.loc[client_ID,
                                                           variable])
            display_charts(data_target, variable, client_ID)
        else:
            st.write("Valeur du client :", admin.loc[client_ID, variable])
            display_charts(admin, variable, client_ID)

    else:
        st.write("Valeur du client :", db_neigh.loc[client_ID, variable])
        display_charts(db_neigh, variable, client_ID)


def get_neigh(df, idx_client):
    """Calcul des voisins les plus proches du client sélectionné
    Sélection du nombre de voisins par un slider.
    Retourne les proches voisins et un booléen indiquant la clientèle globale ou non"""
    row1, row_spacer1, row2, row_spacer2 = st.columns([5, .5, .5, .5])
    size = row1.slider("Taille du groupe de comparaison",
                       min_value=10,
                       max_value=500,
                       value=50)
    row2.write('')
    total = row2.button(label="Clientèle globale")
    neigh = NearestNeighbors(n_neighbors=size)
    neigh.fit(df)
    k_neigh = neigh.kneighbors(df.loc[idx_client].values.reshape(1, -1),
                               return_distance=False)[0]
    k_neigh = np.sort(k_neigh)
    return k_neigh, total


def display_charts(df, variable, client_ID):
    """Affichage du graphe de comparaison selectionné"""

    if df[variable].dtype == 'float':
        chart_kde(df, variable, client_ID)
    elif df[variable].dtype == 'int64':
        chart_kde(df, variable, client_ID)
    elif df[variable].dtype == 'object':
        chart_bar(df, variable, client_ID)


def chart_kde(df, col, client):
    """Définition des graphes KDE avec une ligne verticale indiquant la position du client"""

    fig, ax = plt.subplots()
    sns.kdeplot(df.loc[df['TARGET'] == 0, col],
                color='green',
                label='Prêt accepté')
    sns.kdeplot(df.loc[df['TARGET'] == 1, col],
                color='red',
                label='Prêt refusé')
    plt.axvline(x=df.loc[client, col], ymax=1, color='black')
    plt.legend()
    st.pyplot(fig)


def chart_bar(df, col, client):
    """Définition des graphes barres avec une ligne horizontale indiquant la position du client"""

    fig, ax = plt.subplots()
    data = df[['TARGET', col]]
    if data[col].dtypes != 'object':
        data[col] = data[col].astype('str')

        data1 = round(
            data[col].loc[data['TARGET'] == 1].value_counts() /
            data[col].loc[data['TARGET'] == 1].value_counts().sum() * 100, 2)
        data0 = round(
            data[col].loc[data['TARGET'] == 0].value_counts() /
            data[col].loc[data['TARGET'] == 0].value_counts().sum() * 100, 2)
        data = pd.concat([
            pd.DataFrame({
                "Pourcentage": data0,
                'TARGET': "Prêt accepté"
            }),
            pd.DataFrame({
                'Pourcentage': data1,
                'TARGET': "Prêt refusé"
            })
        ]).reset_index().rename(columns={'index': col})
        sns.barplot(data=data,
                    x='Pourcentage',
                    y=col,
                    hue='TARGET',
                    palette=['green', 'red'],
                    order=sorted(data[col].unique()))

        data[col] = data[col].astype('int64')

        plt.axhline(y=sorted(data[col].unique()).index(df.loc[client, col]),
                    xmax=1,
                    color='black',
                    linewidth=4)
        st.pyplot(fig)
    else:

        data1 = round(
            data[col].loc[data['TARGET'] == 1].value_counts() /
            data[col].loc[data['TARGET'] == 1].value_counts().sum() * 100, 2)
        data0 = round(
            data[col].loc[data['TARGET'] == 0].value_counts() /
            data[col].loc[data['TARGET'] == 0].value_counts().sum() * 100, 2)
        data = pd.concat([
            pd.DataFrame({
                "Pourcentage": data0,
                'TARGET': "Prêt accepté"
            }),
            pd.DataFrame({
                'Pourcentage': data1,
                'TARGET': "Prêt refusé"
            })
        ]).reset_index().rename(columns={'index': col})
        sns.barplot(data=data,
                    x='Pourcentage',
                    y=col,
                    hue='TARGET',
                    palette=['green', 'red'],
                    order=sorted(data[col].unique()))

        plt.axhline(y=sorted(data[col].unique()).index(df.loc[client, col]),
                    xmax=1,
                    color='black',
                    linewidth=4)
        st.pyplot(fig)


def feature_importance(shap_value, df, max_display):
    # st.write("Diagramme SHAP - Summary plot")
    shap.initjs()
    st.subheader("Diagramme SHAP - Summary plot")
    st.write("affichage des valeurs moyennes de shapley par variable")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_value, df, max_display=max_display, plot_type='bar')
    st.pyplot(fig)

    st.subheader("Diagramme SHAP - Summary plot")
    st.write(
        "Visualisation de la repartition de chaque client dans l'évaluation du score"
    )
    fig, ax = plt.subplots()
    shap.summary_plot(shap_value, df, max_display=max_display)
    st.pyplot(fig)


if __name__ == '__main__':
    main()
