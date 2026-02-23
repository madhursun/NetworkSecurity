from networksecurity.components.data_ingestion import DataIngestion
import sys
from networksecurity.components.data_validation import DataValidation
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import TrainingPipelineConfig


if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()

        # Data Ingestion
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)

        logging.info("Data ingestion start")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion end")
        print(data_ingestion_artifact)

        # Data Validation
        data_validation_config = DataValidationConfig(training_pipeline_config)

        data_validation = DataValidation(
            data_validation_config=data_validation_config,
            data_ingestion_artifact=data_ingestion_artifact
        )

        logging.info("Data validation start")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation end")
        print(data_validation_artifact)

    except Exception as e:
        raise NetworkSecurityException(e, sys)