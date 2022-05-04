# 1. Library imports
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from Well_Data_Validation import Well_data
from utils import *
import numpy as np
from tensorflow import keras
from tensorflow.python.keras.models import load_model
import tensorflow as tf
import starlette.responses as _responses

# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras! 
# Otherwise, their weights will be unavailable in the threads after the session there has been set



# 2. Create the app object
app = FastAPI()

@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)},
    )

lithology_map = {0:'Sandstone',
                 1:'Sandstone/Shale',
                 2:'Shale',3:'Marl',
                 4:'Dolomite',5:'Limestone',
                 6:'Chalk',7:'Halite',
                 8:'Anhydrite',9:'Tuff',
                 10:'Coal',11:'Basement'}


enc_loaded_G, enc_loaded_F, enc_loaded_W  = load_encoders()
scaler = load_scaler()

# Index route, opens automatically on redocs
@app.get("/")
async def root():
    return _responses.RedirectResponse("/redoc")

# Expose the prediction functionality, make a prediction from the passed
# JSON data and return the predicted lithology with the confidence level.

@app.post('/predict')
async def predict_lithology(data:Well_data):
    data = data.dict()
    CALI = data['CALI']
    DEPTH_MD = data['DEPTH_MD']
    DRHO = data['DRHO']
    DTC = data['DTC']
    GR = data['GR']
    NPHI = data['NPHI']
    PEF = data['PEF']
    RDEP = data['RDEP']
    RHOB = data['RHOB']
    RMED = data['RMED']
    SP =  data['SP']

    ## encoding well, formation and formation with their repsective encoder pickles
    GROUP_encoded = enc_loaded_G.transform([data['GROUP']])[0]
    FORMATION_encoded= enc_loaded_F.transform([data['FORMATION']])[0]
    WELL_encoded = enc_loaded_W.transform([data['WELL']])[0]
    input_features = np.array([[DEPTH_MD, CALI, RMED, RDEP, RHOB, GR, NPHI, PEF, DTC,
                        SP, DRHO, GROUP_encoded, FORMATION_encoded, WELL_encoded]])
    ##scale and augument input parameters
    input_aug = augment_features(input_features,depth=data['DEPTH_MD'], N_neig=1)
    scaled_features = scaler.transform(np.array(input_aug))
    print(scaled_features)
    #predict with SNN
    
    #return prediction
    model = load_model("model/Lithology_Model", custom_objects={"f1_weighted":  f1_weighted })
    prediction = model.predict(scaled_features)
    
    index= np.argmax(prediction)

    return {
        'prediction': lithology_map[index]
    }

# Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload

