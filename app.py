import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pandas as pd
import pickle

app = FastAPI()
classifier=pickle.load(open('classifier.pkl','rb'))

@app.get('/')
def index():
    return {'message':'Hello everyone'}

@app.get('/{name}')
def get_name(name:str):
    return {f'Welcome {name}. Thanks for using our service'}


@app.post('/predict')
def predict_banknote(data:BankNote):
    data = data.dict()
    variance=data['variance']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy=data['entropy']
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    if(prediction[0]>0.5):
        prediction="Its a Fake note"
    else:
        prediction="Its a legal Note"
    return {
        'prediction': prediction
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)














