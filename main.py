from fastapi import FastAPI
from utils.models import InputData
from database.Neural_Salary import predict_salary

app = FastAPI()


@app.post("/predict")
def predict(personal_data: InputData):
    print(personal_data.dict())
    data = personal_data.dict()
    prediction = predict_salary(data)
    print(prediction)
    print(type(prediction)) # from tensor to int
    response = {"message": "salary predicted",
                "prediction": prediction}
    print(response)
    return response


# run me with uvicorn
