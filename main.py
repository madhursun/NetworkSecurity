from networksecurity.components.data_ingestion import DataIngestion 
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import DataIngestionConfig 
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import TrainingPipelineConfig


if __name__ == "__main__":
    try:
        trainingPipelineConfig=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig(trainingPipelineConfig)
        data_ingestion=DataIngestion(dataingestionconfig)
        logging.info("data ingestion start")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        print(dataingestionartifact)
       
    except Exception as e:
        raise NetworkSecurityException(e, sys)