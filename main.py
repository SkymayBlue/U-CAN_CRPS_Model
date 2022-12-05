# fastAPI for UCAN CRPS Model
from fastapi import FastAPI, UploadFile, File
from starlette.responses import RedirectResponse
from fastapi.responses import FileResponse
from uvicorn import run
import os
import shutil
import time
SEEDS = 46946
import random as rn
rn.seed(SEEDS)
import numpy as np
np.random.seed(SEEDS)
from pathlib import Path
from predictions import Predictions
# https://testdriven.io/blog/fastapi-streamlit/
app = FastAPI(title="RESNET50 for CRPS Prediction!",
              description="This web interface allows file uploading for a TensorFlow container running a ResNET50 pre-trained model for CRPS subtype",
              version="1.0.0", redoc_url=None)


# home dir
@app.get("/")
def home_screen():
    return RedirectResponse(url="/docs")


@app.post("/Predict")
async def get_prediction_files(file: UploadFile=File(...), status_code=200):
    """
    Submit a file to be analyzed by the RESNET50-1D
    """
    start_time = time.time()
    # create the tmp dir in current
    shutil.rmtree("tmp")
    os.makedirs("tmp")
    tmp_path = Path(os.path.join("tmp",file.filename))
    with open(tmp_path, 'wb') as f:
        while contents := file.file.read(1024):
            f.write(contents)
    PREDICTRES = Predictions(os.path.join("tmp", file.filename))
    filename = PREDICTRES.predicition_results_file()
    process_time = time.time() - start_time
    shutil.rmtree("tmp")
    headers = {
        'message': 'submit ' + file.filename,
        'processTime': str(process_time),
        'Content-Disposition': 'attachment; filename="predictions.xlsx"',
        'predictions': PREDICTRES.get_predictions_summary().to_json(),
    }
    return FileResponse(filename, media_type='application/octet-stream', headers=headers)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    run("main:app", host="127.0.0.1", port=port)


