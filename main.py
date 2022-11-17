# fastAPI for resnet50-1D
from fastapi import FastAPI, UploadFile, File
from starlette.responses import RedirectResponse
from fastapi.responses import FileResponse
from uvicorn import run
import os
import time
SEEDS = 46946
import random as rn
rn.seed(SEEDS)
import numpy as np
np.random.seed(SEEDS)
from pathlib import Path
from predictions import Predictions
# https://testdriven.io/blog/fastapi-streamlit/
app = FastAPI(title="RESNET50-1D for CRPS Prediction!",
              description="This web interface allows file uploading for a TensorFlow container running a ResNET501D pre-trained model for CRPS subtype",
              version="1.0.0",)


# home dir
@app.get("/")
def home_screen():
    return RedirectResponse(url='/docs')


@app.post("/CRPS.Predict")
async def get_prediction_files(file: UploadFile=File(...)):
    """
    Submit a file to be analyzed by the RESNET50-1D
    """
    start_time = time.time()
    tmp_path = Path(os.path.join("temp", file.filename))
    with open(tmp_path, 'wb') as f:
        while contents := file.file.read(1024):
            f.write(contents)
    PREDICTRES = Predictions(os.path.join("temp", file.filename))
    filename = PREDICTRES.predicition_results_file()
    process_time = time.time() - start_time
    headers = {
        'message': 'submit ' + file.filename,
        'processTime': str(process_time),
        'Content-Disposition': 'attachment; filename="predictions.xlsx"',
        'predictions': PREDICTRES.get_predictions_summary().to_json(),
    }
    return FileResponse(filename, media_type='application/octet-stream', headers=headers)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    run(app, host="127.0.0.1", port=port)


