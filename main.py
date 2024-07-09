from fastapi import FastAPI
from utils.models import InputData
from database.train import predict_salary

app = FastAPI()


@app.post("/predict")
def predict(personal_data: InputData):
    data = personal_data.dict()
    prediction = predict_salary(data)
    response = {"message": "salary predicted",
                "prediction": prediction}
    return response


# run me with uvicorn main:app --reload
