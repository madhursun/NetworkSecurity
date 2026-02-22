from networksecurity.exception.exception import NetworkSecurityException

from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact

import sys,os

import numpy as np
import pandas as pd
import pymongo
from sklearn.model_selection import train_test_split
from typing import List,Tuple

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")  

class DataIngestion:
  def __init__(self,data_ingestion_config:DataIngestionConfig):
    try:
      self.data_ingestion_config=data_ingestion_config

    except Exception as e:
      raise NetworkSecurityException(e,sys) 
    
  def export_collection_as_dataframe(self):
    try:
     database_name=self.data_ingestion_config.database_name
     collection_name=self.data_ingestion_config.collection_name
    
     self.monogo_client=pymongo.MongoClient(MONGO_DB_URL)
     collection=self.monogo_client[database_name][collection_name]
     df=pd.DataFrame(list(collection.find()))
     if "_id" in df.columns:
       df=df.drop("_id",axis=1)

     df.replace({"na":np.nan},inplace=True)

     return df

    except Exception as e:
      raise NetworkSecurityException(e,sys)
    
  def export_data_into_feature_store(self,dataframe:pd.DataFrame):
    try:
      feature_store__file_path=self.data_ingestion_config.feature_store_file_path
      dir_path=os.path.dirname(feature_store__file_path)
      os.makedirs(dir_path,exist_ok=True)
      dataframe.to_csv(feature_store__file_path,index=False,header=True)

    except Exception as e:
      raise NetworkSecurityException(e,sys)
    
  def spilt_data_as_train_test(self,dataframe:pd.DataFrame):
    try:
      train_set,test_set=train_test_split(dataframe,test_size=self.data_ingestion_config.train_test_split_ratio)
      logging.info("train test perform")
      logging.info("exited spilt_data_as_train_test method of DATA_ingestion class")

      dir_path=os.path.dirname(self.data_ingestion_config.training_file_path)

      os.makedirs(dir_path,exist_ok=True)

      logging.info(f"Exporting train and test file path.")

      train_set.to_csv(
        self.data_ingestion_config.training_file_path,index=False,header=True
      )

      test_set.to_csv(
        self.data_ingestion_config.testing_file_path,index=False,header=True
      )

      logging.info(f"Exported train and test file path.")

      
    except Exception as e:
      raise NetworkSecurityException(e,sys)



  def initiate_data_ingestion(self):
    try:
        dataframe = self.export_collection_as_dataframe()

        self.export_data_into_feature_store(dataframe)

        self.spilt_data_as_train_test(dataframe)

        dataingestionartifact = DataIngestionArtifact(
            trained_file_path=self.data_ingestion_config.training_file_path,
            test_file_path=self.data_ingestion_config.testing_file_path
        )

        return dataingestionartifact

    except Exception as e:
        raise NetworkSecurityException(e,sys)
    