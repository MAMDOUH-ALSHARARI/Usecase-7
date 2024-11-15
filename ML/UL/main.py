import joblib

from pydantic import BaseModel
from fastapi import FastAPI

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

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
        'award': input_features.award
    }
    new_data = pd.DataFrame([dict_f])
    # Get the feature names the scaler was trained on
    #feature_names = DBSCAN_scaler.get_feature_names_out() 

    # Ensure new_data has the same columns and order as the original data
    #new_data = dict_f.reindex(columns=feature_names) # Reorder or add missing columns

    # 3. Scale the new data using the loaded scaler
    new_data_scaled = DBSCAN_scaler.transform(new_data)
    

    # Convert dictionary values to a list in the correct order
    #features_list = [dict_f[key] for key in (dict_f)]

    # Scale the input features
    #scaled_features = DBSCAN_scaler.transform([list(dict_f.values())])

    return new_data_scaled

##################################################################################

app = FastAPI()

# GET request
@app.get("/")
def read_root():
    return {"message": "Welcome to Tuwaiq Academy"}

@app.post("/DBSCAN_Football_Players/predict")
async def Football_Players(input_features: InputFeatures):
    #data = preprocessing(input_features)
    #print('done')
    #y_pred = DBSCAN_model.predict(data)
    #print('done')
    #return {"pred": y_pred.tolist()[0]}

    data = preprocessing(input_features)
    y_pred = DBSCAN_model.fit_predict(data)  # Use fit_predict instead of predict
    return {"pred": int(y_pred[0])}
