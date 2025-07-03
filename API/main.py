from fastapi import FastAPI
from pydantic import BaseModel
import h2o
from h2o.frame import H2OFrame
import pandas as pd

app = FastAPI()

h2o.init()
model = h2o.load_model('models/StackedEnsemble_BestOfFamily_1_AutoML_1_20250629_111259')

class FinIn(BaseModel):
    total_assets: float
    total_liabilities: float
    operating_cash_flow: float
    net_income: float
    FCF_margin: float
    OCF_margin: float

@app.post("/predict")
def predict(input: FinIn):
    data = pd.DataFrame([input.dict()])
    h2o_data = H2OFrame(data)

    prediction = model.predict(h2o_data).as_data_frame().values[0][0]
    return {"burn_cash_prediction": prediction}

