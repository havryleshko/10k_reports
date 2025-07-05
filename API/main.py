from fastapi import FastAPI
from pydantic import BaseModel
import h2o
from h2o.frame import H2OFrame
import pandas as pd
import gradio as gr
from fastapi.responses import RedirectResponse

app = FastAPI()

h2o.init(log_level='ERROR')
model = h2o.load_model('models/StackedEnsemble_BestOfFamily_1_AutoML_1_20250629_111259')

class FinIn(BaseModel):
    total_assets: float
    total_liabilities: float
    operating_cash_flow: float
    net_income: float
    FCF_margin: float
    OCF_margin: float
    free_cash_flow: float

@app.post("/predict")
def predict(input: FinIn):
    data = pd.DataFrame([input.dict()])
    h2o_data = H2OFrame(data)

    prediction = model.predict(h2o_data).as_data_frame().values[0][0]
    return {"burn_cash_prediction": prediction}

def predict_cash_burn(total_assets, total_liabilities, operating_cash_flow, free_cash_flow, net_income, OCF_margin, FCF_margin):
    input_dict = {
        'total_assets': total_assets,
        'total_liabilities': total_liabilities,
        'operating_cash_flow': operating_cash_flow,
        'free_cash_flow': free_cash_flow,
        'net_income': net_income,
        'OCF_margin': OCF_margin,
        'FCF_margin': FCF_margin
    }
    df = pd.DataFrame([input_dict])
    h2o_df = h2o.H2OFrame(df)
    prediction = model.predict(h2o_df).as_data_frame().values[0][0]
    return f'Burn cash prediction for your instance: {prediction}'


iface = gr.Interface(
    fn=predict_cash_burn,
    inputs=[
        gr.Number(label='Total Assets'),
        gr.Number(label='Total Liabilities'),
        gr.Number(label='Operating Cash Flow'),
        gr.Number(label='Free Cash Flow'),
        gr.Number(label='Net Income'),
        gr.Number(label='OCF Margin'),
        gr.Number(label='FCF Margin')
    ],
    outputs=gr.Text(label='Prediction'),
    title='Cash Burn Predictor',
    description='Enter requested financial metrics to predict whether the company is likely to burn cash next year or not.'
)

app.mount('/gradio', iface.app)

@app.get('/')
def root():
    return RedirectResponse(url='/gradio')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)