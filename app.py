import  sys
import os

import certifi
ca=certifi.where()

from dotenv import load_dotenv
load_dotenv()

mongo_db_url=os.getenv("MONGO_DB_URL")
print(mongo_db_url)
import pymongo
import networksecurity.exception.exception as exception
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline


from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI,File,UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd
from networksecurity.utils.ml_utils.model.estimator import NetworkModel


from networksecurity.utils.main_utils.utils import load_object
from networksecurity.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME,DATA_INGESTION_DATABASE_NAME
client=pymongo.MongoClient(mongo_db_url,tlsCAFile=ca)
db=client[DATA_INGESTION_DATABASE_NAME]
collection=db[DATA_INGESTION_COLLECTION_NAME]


app=FastAPI()
origins = [
  "*"
]

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="./templates")

app.get("/",tags=["Authentication"])
async def index():
  return RedirectResponse(url="/docs")

@app.get("/train")
async def train():
  try:
    training_pipeline=TrainingPipeline()
    training_pipeline.run_pipeline()
    return Response("Training successful !!")
  except Exception as e:
    return Response(f"Error Occurred! {e}")
  

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        preprocessor = load_object("final_models/preprocessor.pkl")
        final_model = load_object("final_models/model.pkl")

        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

        y_pred = network_model.predict(df)
        df['prediction'] = y_pred

        df.to_csv("prediction_output/output.csv", index=False)

        table_html = df.to_html(classes='table table-striped')

        return templates.TemplateResponse(
            "table.html",
            {"request": request, "table": table_html}
        )

    except Exception as e:
        return Response(f"Error Occurred! {e}")
  
if __name__=="__main__":
  app_run(app,host="localhost",port=8000)

  

