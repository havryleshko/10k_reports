import gradio as gr
import h2o
import pandas as pd

h2o.init()
model = h2o.load_model('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/models/StackedEnsemble_BestOfFamily_1_AutoML_1_20250629_111259')



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

if __name__ == '__main__': # this means 'only run this when I run it manually myself'
    iface.launch(share=True)