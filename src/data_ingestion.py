import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml
# # ensure the "logs" directory exists
# log_dir='logs'
# os.makedirs(log_dir,exist_ok=True) #makes a folder called logs and true checks if it exists already or not

# #logging configuration
# logger=logging.getLogger('data_ingestion')
# logger.setLevel('DEBUG')



# console_handler = logging.StreamHandler() # streamhandler sends logs to terminal
# console_handler.setLevel("DEBUG")


# log_file_path=os.path.join(log_dir,'data_ingestion.log') #file handler sends logs to a file
# file_handler=logging.FileHandler(log_file_path)
# file_handler.setLevel('DEBUG')


#ensures that "logs" folder already exists if not it will make it
log_dir="logs"
os.makedirs(log_dir,exist_ok=True)


# logging configuration / creating loging file
logger=logging.getLogger('data_ingestion')  # data_ingestion = logger name for seeing which logger like in this case we are doing data ingestion then it shows  data_ingestion - loading dataset 
#model-training - model trained etc so on 
logger.setLevel('DEBUG') 

# console handler --> sends logs to terminal
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')



# creating log file ko path 
log_file_path=os.path.join(log_dir,'data_ingestion.log')
file_handler=logging.FileHandler(log_file_path)  # saving logs in logs /data_ingestion.log

file_handler.setLevel('DEBUG')



# formatter - handles our logs formats
# eg : suppose we have 
# logger.info("data ingestion started")
# 
# and if out formatter is set like this 
# formatter = logging.Formatter(
#     '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# ) 

# our log will be like this:
#  2026-08-10 13:30:15 - data_ingestion - INFO - Data ingestion started


formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# setting the formatter of console_handler
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(file_path:str) ->dict:
      """
      load parameters from a yaml file.

      
      """
      try:
            with open(file_path,'r') as f:
                  params=yaml.safe_load(f)
            logger.debug("parameters loaded successfully from %s",file_path)
            return params
      except FileNotFoundError:
            logger.error("File not found in %s",file_path)
            raise
      except yaml.YAMLError as e:
            logger.error("yaml error : %s",e)
            raise
      except Exception as e:
            logger.error("unexpected error %s ",e)
            raise



def load_data(data_url:str) ->pd.DataFrame:
    """load data from a CSV file.""" # doc string
    try:
          df=pd.read_csv(data_url)
          logger.debug("data loaded from %s",data_url)
          return df
    except pd.errors.ParserError as e:
          logger.error('Failed to parse the csv file: %s',e)
          raise # return back error to function and code doesnot continues forward
    except Exception as e:
          logger.error('Unexpected error occurred while loading  the data: %s',e)
          raise


def preprocess(df:pd.DataFrame)->pd.DataFrame:
      """Preprocess the data"""
      try:
            df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
            df.rename(columns={'v1':'target','v2':'text'},inplace=True)
            logger.debug("Data preprocessing completed ")
            return df
      except KeyError as e:
            logger.error("Missing column in the dataframe: %s",e)
            raise
      except Exception as e:
            logger.error("Unexpected error during preprocessing: %s",e)
            raise

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
      """save the train and test datasets"""
      try:
            raw_data_path=os.path.join(data_path,'raw') # we initiate a datapath and in that data path we will put raw folder
            #eg data_path = './data' ==> data/raw  folder indide folder
            os.makedirs(raw_data_path,exist_ok=True) #this will make the folder and file data/raw
            train_data.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
            test_data.to_csv(os.path.join(raw_data_path,"test.csv"),index=False)
            logger.debug("train and test data are saved to %s",raw_data_path)

      except Exception as e:
            logger.error("unexcepted error occured while saving the data : %s",e)
            raise


def main():
    try:
            params=load_params(file_path='params.yaml')
            test_size=params['data_ingestion']['test_size']
            data_path='https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/spam.csv'
            df=load_data(data_path)
            final_df=preprocess(df)
            train_data,test_data=train_test_split(final_df,test_size=test_size,random_state=2)

            save_data(train_data,test_data,data_path="./data") #./ current directory ko root ma go and make a folder data
    except Exception as e:
            logger.error("Failed to complete the data ingestion process %s ",e)
            print(f"Error:{e}")


if __name__ == '__main__':
      main()




# whenever a python file is created a __name__ is assigned to it 
# so it becomes __name__ = "__main__"

# so when i ran data_ingestion.py we are  checking if __name__ == "__main__" 

            