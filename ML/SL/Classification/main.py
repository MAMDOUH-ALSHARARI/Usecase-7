from fastapi import FastAPI, HTTPException
app = FastAPI()
# GET request
@app.get("/")
def read_root():
    return {"message": "Welcome to Tuwaiq Academy"}
# get request
@app.get("/items/")
def create_item(item: dict):
    return {"item": item}

import joblib
model = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')


from pydantic import BaseModel
# Define a Pydantic model for input data validation
class InputFeatures(BaseModel):
    goals: float
    appearance: int
    minutes_played: int
    highest_value: int
    award: int
    

def preprocessing(input_features: InputFeatures):
    dict_f = {
        'goals': input_features.goals,
        'appearance': input_features.appearance,
        'minutes_played': input_features.minutes_played,
        'highest_value': input_features.highest_value,
        'award': input_features.award
}
    return dict_f


@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model.predict(data)
    return {"pred": y_pred.tolist()[0]}