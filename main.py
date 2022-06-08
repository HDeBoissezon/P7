# uvicorn main:app --reload
# http://127.0.0.1:8000

from fastapi import FastAPI
from pydantic import BaseModel, create_model, conlist
import joblib
import pandas as pd
from typing import List
import json

app = FastAPI(
    title="Bank Credit attribution API",
    description="API pour récupérer les prédictions d'attribution de crédits")

# charger les données
job_dir = './JOBLIB'
data_dict = joblib.load(job_dir + '/dataModel.joblib')
pipeline = joblib.load(job_dir + '/pipeline.joblib')
# data = joblib.load(job_dir + '/data.joblib')
data = joblib.load(job_dir + '/data_small.joblib')

Info_client = create_model('loan_data', **data_dict, __base__=BaseModel)

# test configuration
# @app.get("/{client_id}")
# def read_client_id(client_id: int):
#     return {"ID a rechercher": "{}".format(client_id)}


# option 1 : on envoie juste l'id client par l'url et on recupere le predict
@app.get("/{client_id}")
def prediction_get(client_id: int):
    if client_id not in data.index:
        decision = "ID client inconnue"

    else:
        data_user = pd.DataFrame(data.loc[client_id, :]).T
        y_proba_user = pipeline.predict_proba(data_user)[:, 1]

        if y_proba_user > 0.62:
            decision = 1
        else:
            decision = 0

    # return {"Client {} : {}".format(client_id, decision)}
    return json.dumps(y_proba_user.tolist())


# option 2 : on envois toutes les info ==> post + utilise le request body
@app.post("/prediction/")
def prediction_post(data_client: Info_client):
    data_client = data_client.dict()
    df = pd.DataFrame(data=[data_client.values()], columns=data_client.keys())
    # print(type(df))
    # print(df.shape)
    y_proba_user = pipeline.predict_proba(df)[:, 1]
    print(y_proba_user)
    if y_proba_user > 0.62:
        decision = 1
    else:
        decision = 0
    print(decision)
    # return {"decision": decision,
    #         "probabilite": y_proba_user}
    return {"decision : {}".format(decision)}


# @app.post("/client/")
# async def create_client(item: Info_client):
#     return item

# uvicorn main:app --reload
