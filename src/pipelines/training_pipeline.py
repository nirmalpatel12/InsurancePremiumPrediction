import os
import sys
sys.path.append('D:\\ML_projects\\InsurancePremimumPrediction\\src')
from logger import logging
from exception import CustomException
import sys
sys.path.append('D:\\ML_projects\\InsurancePremimumPrediction\\src')
import pandas as pd
from componentes.data_ingestion import DataIngestion

from componentes.data_transformation import DataTransformation
from componentes.model_trainer import ModelTrainer



if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    print(train_data_path,test_data_path)

    data_transformation=DataTransformation()

    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

    model_trainer=ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)

