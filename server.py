# fastAPI for UCAN CRPS Model
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
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


def ex_input():
    df = pd.read_csv("./testdata/GSE35896_logArray.csv.gz", index_col=0)
    # uploaded_file: UploadFile = File(example=df.head())
    return df.head


def es_input():
    df = pd.read_csv("./testdata/GSE35896_ES_rpy2.csv.gz", index_col=0)
    # uploaded_file: UploadFile = File(example=df.head())
    return df.head


tags_metadata = [{"name": "CRPS EXP",
                  "description": "CRPS for Expression Matrix",
                  "externalDocs": {
                      "description": "download example",
                      "url": "http://localhost:8080/downapi/expression/download",
                  }},
                 {"name": "CRPS ES",
                  "description": "CRPS for Enrich Score",
                  "externalDocs": {
                      "description": "download example",
                      "url": "http://localhost:8080/downapi/enrichment/download",
                  }}]

# https://testdriven.io/blog/fastapi-streamlit/
app = FastAPI(title="RESNET50 for CRPS Prediction!",
              description="This web interface allows file uploading for a TensorFlow container running a ResNET50 pre-trained model for CRPS subtype",
              version="1.0.0",
              redoc_url=None,
              swagger_ui_parameters={'tryItOutEnabled': True, "deepLinking": False},
              openapi_tags=tags_metadata)


# home dir
# @app.get("/")
# def home_screen():
#     return RedirectResponse(url="/docs")
# from contextlib import asynccontextmanager
# main_app_lifespan = app.router.lifespan_context
# @asynccontextmanager
# async def lifespan_wrapper(app):
#     print("sub startup")
#     async with main_app_lifespan(app) as maybe_state:
#         yield maybe_state
#     print("sub shutdown")
#
# app.router.lifespan_context = lifespan_wrapper


def get_predict_res(inputf, dsource):
    PREDICTRES = Predictions(inputf, dsource)
    filename, data_check_log = PREDICTRES.predicition_results_file()
    predict_desc = PREDICTRES.get_predictions_summary().to_json()
    return filename, predict_desc


@app.post("/api/expression/predict", tags=["CRPS EXP"])
async def get_predict(uploaded_file: UploadFile = File(description="row:gene_symbol * column:sample")):  #
    # Submit an expression file to calculate ES and analysis by the RESNET50-1D
    start_time = time.time()
    path_to_save_file = Path.home() / "tmp"
    # create the tmp dir to save file
    if not uploaded_file:
        return {"message": "No upload file sent"}
    if not os.path.exists(path_to_save_file):
        path_to_save_file.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(os.path.join(path_to_save_file, uploaded_file.filename))
    try:
        with open(tmp_path, 'wb') as f:
            while contents := uploaded_file.file.read(1024 * 1024):
                f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=403, detail="There was an error uploading the file, retry it!")
        # return {"message": "There was an error uploading the file, retry it! "}
    finally:
        uploaded_file.file.close()
    filename, predict_desc = get_predict_res(tmp_path, dsource="expression")
    process_time = time.time() - start_time
    headers = {
        'message': 'submit file: ' + uploaded_file.filename + "; size: " + str(uploaded_file.file.__sizeof__()),
        'processTime': str(process_time),
        'Content-Disposition': 'attachment; filename="predictions.xlsx"',
        'prediction_summary': predict_desc,
    }
    return FileResponse(filename, media_type='application/octet-stream', headers=headers)


@app.post("/api/enrichscore/predict", tags=["CRPS ES"])
async def get_ESpredict(uploaded_file: UploadFile = File(...)):
    # Submit an ES file to be analysed by the RESNET50-1D
    start_time = time.time()
    path_to_save_file = Path.home() / "tmp"
    # create the tmp dir to save file
    if not uploaded_file:
        return {"message": "No upload file sent"}
    if not os.path.exists(path_to_save_file):
        path_to_save_file.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(os.path.join(path_to_save_file, uploaded_file.filename))
    try:
        with open(tmp_path, 'wb') as f:
            while contents := uploaded_file.file.read(1024 * 1024):
                f.write(contents)
    except Exception as e:
        return {"message": "There was an error uploading the file"}
    finally:
        uploaded_file.file.close()
    filename, predict_desc = get_predict_res(tmp_path, dsource="es")
    process_time = time.time() - start_time
    headers = {
        'message': 'submit file: ' + uploaded_file.filename + "; size: " + str(uploaded_file.file.__sizeof__()),
        'processTime': str(process_time),
        'Content-Disposition': 'attachment; filename="predictions.xlsx"',
        'prediction_summary': predict_desc,
    }
    return FileResponse(filename, media_type='application/octet-stream', headers=headers)


downapi = FastAPI(swagger_ui_parameters={'tryItOutEnabled': True, "deepLinking": False})


def cleanup(temp_file):
    os.remove(temp_file)


@downapi.get("/expression/download")
def download_exp():
    file_path = "./testdata/subGSE_logArray.csv.gz"
    return FileResponse(file_path, filename="example_expression.csv.gz", background=BackgroundTask(cleanup, file_path))


@downapi.get("/enrichment/download")
def download_es():
    file_path = "./testdata/subGSE_rpy2.csv.gz"
    return FileResponse(file_path, filename="example_enrichscore.csv.gz", background=BackgroundTask(cleanup, file_path))


app.mount("/downapi", downapi)

# if __name__ == "__main__":
#     from uvicorn import run
#     port = int(os.environ.get('PORT', 8080))
#     run("server:app", host="127.0.0.1", port=port, reload=True)
