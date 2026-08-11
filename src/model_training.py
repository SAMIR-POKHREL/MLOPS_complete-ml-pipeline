import os
import numpy as np 
import pandas as pd
import pickle 
import logging
from sklearn.ensemble import RandomForestClassifier


# making sure logging folder exists
dir_path='logs'
os.makedirs(dir_path,exist_ok=True)

 #logging configuration
logger=logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')


# console handler
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

# setting path for filehandler
file_path=os.path.join(dir_path,'model_training.log')

# setting file handler
file_handler=logging.FileHandler(file_path)
file_handler.setLevel('DEBUG')

# setting formater 
formatter=logging.Formatter('%(asctime)s - %(name)s -%(levelname)s - %(message)s')

# applying formatter to console handler and file handler
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


logger.addHandler(console_handler)
logger.addHandler(file_handler)



def load_data(file_path:str)->pd.DataFrame:
    """
    Load data from csv file     
    """

    try:
        df=pd.read_csv(file_path)
        logger.debug("data loaded from %s with shape %d",file_path,df.shape[0])
        return df
    except pd.errors.ParserError as e:
        logger.error("failed to parse CSV file %s",e)
        raise
    except FileNotFoundError as e:
        logger.error("file not found %s",e)
        raise
    except Exception as e:
        logger.error("unecpected error occured while loading data %s",e)
        raise



def train_model(X_train:np.ndarray,y_train:np.ndarray,params:dict) ->RandomForestClassifier:
    """"
    train the randomforest model
    :param X_train:training features
    :param y_train:training features
    :param params:dictionary of hpyerparameters
    :return : trained randomforestclassifier

    
    
    """
    print(X_train.shape)
    print(y_train.shape)

    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError ("the number of samples in X_train and y_train does not match")
        logger.debug("initializing randomforest model with parameter %s",params)

        clf=RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])
        logger.debug("model training started with %d samples",X_train.shape[0])
        clf.fit(X_train,y_train)
        logger.debug("model training completed")

        return clf
    except ValueError as e:
        logger.error("value error during model training %s",e)
        raise
    except Exception as e:
        logger.error("error during model training %s",e)
        raise





def save_model(model,file_path:str)-> None:
    """
    save the trained model to a file
    :param mdoel : Trained model object
    :param file_path : path to save the model file
    
    """
    try:
        # ensure the directory exists
        os.makedirs(os.path.dirname(file_path),exist_ok=True) # models/models.pkl os.path.dirname(models) makedir makes it 

        with open(file_path,'wb') as f:  # if no file called model.pkl it will create it itself
            pickle.dump(model,f)
        logger.debug("model saved to %s",file_path)
    except FileNotFoundError as e:
        logger.error('File path not found %s',e)
        raise
    except Exception as e:
        logger.error("error occured while saving the file %s",e)
        raise



def main():
    try:
        params={'n_estimators':25,'random_state':2}
        train_data=load_data('./data/processed/train_tfidf.csv')
        X_train= train_data.iloc[:,:-1].values
        y_train= train_data.iloc[:,-1].values



        clf=train_model(X_train,y_train,params)

        model_save_path='models/model.pkl'
        save_model(clf,model_save_path)
    except Exception as e:
        logger.error("failed to complete the model building process:%s",e)
        print(f"Error %s{e}")




if __name__ == "__main__":
    main()