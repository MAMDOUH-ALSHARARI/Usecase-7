import joblib

from pydantic import BaseModel
from fastapi import FastAPI

##################################################################################

DBSCAN_model = joblib.load('DBSCAN_model.joblib')
DBSCAN_scaler = joblib.load('DBSCAN_scaler.joblib')

##################################################################################

class InputFeatures(BaseModel):
    minutes_played: int
    current_value: int
    award: int
   

##################################################################################

def preprocessing(input_features: InputFeatures):
    dict_f = {
        'minutes played': input_features.minutes_played,
        'current_value':input_features.current_value,
        'award': input_features.award,
    }

    # Convert dictionary values to a list in the correct order
    features_list = [dict_f[key] for key in (dict_f)]

    # Scale the input features
    scaled_features = DBSCAN_scaler.transform([list(dict_f.values())])

    return scaled_features

##################################################################################

app = FastAPI()

# GET request
@app.get("/")
def read_root():
    return {"message": "Welcome to Tuwaiq Academy"}

@app.post("/DBSCAN_Football_Players/predict")
async def Football_Players(input_features: InputFeatures):
    data = preprocessing(input_features)
    print('done')
    y_pred = DBSCAN_model.predict(data)
    print('done')
    return {"pred": y_pred.tolist()[0]}
