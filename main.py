# uvicorn main:app --reload
# http://127.0.0.1:8000

from fastapi import FastAPI
import joblib
import pandas as pd
import json

app = FastAPI(
    title="Bank Credit attribution API",
    description="API pour récupérer les prédictions d'attribution de crédits")

# charger les données
job_dir = './JOBLIB'
# data_dict = joblib.load(job_dir + '/dataModel.joblib')
pipeline = joblib.load(job_dir + '/pipeline.joblib')
data = joblib.load(job_dir + '/data_small.joblib')


@app.get("/{client_id}")
def prediction_get(client_id: int):
    if client_id not in data.index:
        decision = "ID client inconnue"

    else:
        data_user = pd.DataFrame(data.loc[client_id, :]).T
        y_proba_user = pipeline.predict_proba(data_user)[:, 1]

    return json.dumps(y_proba_user.tolist())
