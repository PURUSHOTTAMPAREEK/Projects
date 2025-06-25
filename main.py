from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("covid_model.pkl")

class covid_Input(BaseModel):
    age : int
    gender : int
    fever : float
    cough : int
    city : int

@app.get("/")
def welcome():
    return "Welcome to predication page"

@app.post("/predict")
def covid_predict(data : covid_Input):
    input_data = [[data.age , data.gender , data.fever , data.cough , data.city]]
    predication = model.predict(input_data)
    result = "positive" if predication[0]==1 else "negative"
    return {"predication" : result}