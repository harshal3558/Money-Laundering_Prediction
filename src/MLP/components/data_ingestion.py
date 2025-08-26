import os
import sys
from src.MLP.logger import logging
from src.MLP.exception import CustomException
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# for reading sql data
from src.MLP.utils import read_sql_data
from astrapy import DataAPIClient


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # reading from mysql
            # df=read_sql_data()
            # df=pd.read_csv(os.path.join('notebook/data','diabetes.csv'))
            # logging.info('Reading completed mysql database')

            # reading from datastax

            # Initialize the client
            client = DataAPIClient("YOUR_TOKEN")
            df = client.get_database_by_api_endpoint(
                "https://4de23fe7-6692-48e1-8543-6f1ee32533d3-us-east-2.apps.astra.datastax.com"
            )

            print(f"Connected to Astra DB: {df.list_collection_names()}")


            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=True)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data Ingestion is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
if __name__ == "__main":
    obj = DataIngestion()
    obj.initiate_data_ingestion()

# # app.py
# from src.CCDP.logger import logging
# from src.CCDP.exception import CustomException
# from src.CCDP.components.data_ingestion import DataIngestion
# from src.CCDP.components.data_ingestion import DataIngestionConfig
# from src.CCDP.components.data_transformation import DataTransformation
# from src.CCDP.components.data_transformation import DataTransformationConfig
# from src.CCDP.components.model_tranier import ModelTrainerConfig
# from src.CCDP.components.model_tranier import ModelTrainer

# import sys

# if __name__ == "__main__":
#     logging.info("The execution has started")

#     try:
#         # data_ingestion_config=DataIngestionConfig()
#         data_ingestion=DataIngestion() 
#         # data_ingestion.initiate_data_ingestion()
#         train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

#         # data_transformation_config=DataIngestionConfig()
#         data_transformation=DataTransformation()
#         # data_transformation.initiate_data_transformation(train_data_path,test_data_path)
#         train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
        
#         ## Model Training

#         model_trainer=ModelTrainer()
#         print(model_trainer.initiate_model_trainer(train_arr,test_arr))


#     except Exception as e:
#         logging.info("Custom Exception")
#         raise CustomException(e,sys)