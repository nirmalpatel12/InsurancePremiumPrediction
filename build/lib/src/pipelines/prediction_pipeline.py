import sys
sys.path.append('D:\\ML_projects\\InsurancePremimumPrediction\\src')
import os
from exception import CustomException
from logger import logging
from utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 age:int,
                 sex:str,
                 bmi:float,
                 children:int,
                 smoker:str,
                 region:str,
                 expenses:float
                 
                ):
        
        self.age=age
        self.sex=sex
        self.bmi=bmi
        self.children=children
        self.smoker=smoker
        self.region=region
        self.expenses = expenses
        

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'age':[self.age],
                'sex':[self.sex],
                'bmi':[self.bmi],
                'children':[self.children],
                'smoker':[self.smoker],
                'region':[self.region],
                'expenses':[self.expenses],
                
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)

