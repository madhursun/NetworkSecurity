import sys,os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline


from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from networksecurity.constant.training_pipeline import TARGET_COLUMN
from networksecurity.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from networksecurity.utils.main_utils.utils import save_numpy_array_data,save_object

from networksecurity.entity.config_entity import DataTransformationConfig


class DataTransformation:

  def __init__(self,data_validation_artifact:DataValidationArtifact,data_transformation_config:DataTransformationConfig):
    try:
      self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
      self.data_transformation_config:DataTransformationConfig=data_transformation_config
    except Exception as e:
      raise NetworkSecurityException(e,sys)
    
  @staticmethod
  def read_data(file_path)->pd.DataFrame:
    try:
      return pd.read_csv(file_path)
    except Exception as e:
      raise NetworkSecurityException(e,sys)
    
  def get_data_transformer_object(self)->Pipeline:
    logging.info("Entered get_data_transformer_object method of DataTransformation class")
    try:
      imputer:KNNImputer=KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
      logging.info("Created imputer object")
      processor:Pipeline=Pipeline(steps=[("imputer",imputer)])
      logging.info("Created processor object")
      return processor
      
      
    except Exception as e:
      raise NetworkSecurityException(e,sys)

  def initiate_data_transformation(self) -> DataTransformationArtifact:
    logging.info("Entered initiate_data_transformation method")

    try:
        train_df = DataTransformation.read_data(
            file_path=self.data_validation_artifact.valid_train_file_path
        )
        test_df = DataTransformation.read_data(
            file_path=self.data_validation_artifact.valid_test_file_path
        )

        # Split input and target
        input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
        target_feature_train_df = train_df[TARGET_COLUMN].replace(-1, 0)

        input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
        target_feature_test_df = test_df[TARGET_COLUMN].replace(-1, 0)

        # Get preprocessor
        preprocessor = self.get_data_transformer_object()

        # FIT ONLY ON TRAIN DATA ✅
        preprocessor_object = preprocessor.fit(input_feature_train_df)

        # Transform
        transformed_input_train_features = preprocessor_object.transform(input_feature_train_df)
        transformed_input_test_features = preprocessor_object.transform(input_feature_test_df)

        print("X train shape:", transformed_input_train_features.shape)
        print("y train shape:", np.array(target_feature_train_df).shape)

        # Combine
        train_arr = np.c_[transformed_input_train_features, np.array(target_feature_train_df)]
        test_arr = np.c_[transformed_input_test_features, np.array(target_feature_test_df)]

        # Save
        save_numpy_array_data(
            self.data_transformation_config.transformed_train_file_path,
            array=train_arr
        )

        save_numpy_array_data(
            self.data_transformation_config.transformed_test_file_path,
            array=test_arr
        )

        save_object(
            self.data_transformation_config.transformed_object_file_path,
            preprocessor_object
        )

        save_object('final_models/preprocessor.pkl',preprocessor)

        return DataTransformationArtifact(
            transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
            transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
            transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys)