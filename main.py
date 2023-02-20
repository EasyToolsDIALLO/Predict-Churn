from fastapi import FastAPI
from joblib import load
from pycaret.classification import load_model, predict_model
import pandas as pd

import uvicorn

# create the model API
app = FastAPI()
# chargement du modèle
saved_model = load_model('data/Final_Model_convert_churn_type')

# Define predict function
@app.post('/predict')
def predict(number_vmail_messages,total_day_minutes,total_day_calls,total_day_charge,total_eve_minutes,total_eve_calls,
         total_eve_charge,total_night_minutes,total_night_calls,total_night_charge,total_intl_minutes,total_intl_calls,
         total_intl_charge,	customer_service_calls):

    data = pd.DataFrame([[number_vmail_messages,total_day_minutes,total_day_calls,total_day_charge,total_eve_minutes,total_eve_calls,
         total_eve_charge,total_night_minutes,total_night_calls,total_night_charge,total_intl_minutes,total_intl_calls,
         total_intl_charge,	customer_service_calls]])

    data.columns=['number_vmail_messages','total_day_minutes','total_day_calls','total_day_charge','total_eve_minutes','total_eve_calls',
         'total_eve_charge','total_night_minutes','total_night_calls','total_night_charge','total_intl_minutes','total_intl_calls',
         'total_intl_charge','customer_service_calls']

    predictions = predict_model(saved_model, data=data)
    return {'prediction': predictions['Label'][0]}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)


