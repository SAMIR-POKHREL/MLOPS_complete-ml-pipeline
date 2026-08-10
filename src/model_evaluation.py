import os
import numpy as np 
import pandas as pd
import pickle 
import logging
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
import yaml
# from dvclive import Live
import json

# making sure logging folder exists
dir_path='logs'
os.makedirs(dir_path,exist_ok=True)

 #logging configuration
logger=logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')


# console handler
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

# setting path for filehandler
file_path=os.path.join(dir_path,'model_evaluation.log')

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

def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise


def load_data(file_path:str) ->pd.DataFrame:
    """
    load the data from csv file
    """
    try:
        df=pd.read_csv(file_path)
        logging.debug("data loaded from %s",file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error("failed to parse csv file %s",e)
        raise
    except Exception as e:
        logging.debug("unecprcted error occured while loading the data %s",e)


def evaluate_model(clf,X_test:np.ndarray,y_test:np.ndarray)->dict:
    """
    evaluate the model and return the evaluation metriccs.
    """

    try:
        y_pred=clf.predict(X_test)
        y_pred_proba=clf.predict_proba(X_test)[:,1] # probab of spam 1


        accuracy=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred)
        auc=roc_auc_score(y_test,y_pred_proba)

        metrics_dict={
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'auc':auc
        }

        logger.debug("Model evaluation metrics calculated")
        return metrics_dict
    except Exception as e:
        logger.error("Error during model evaluation: %s",e)
        raise

def save_metrics(metrics:dict,file_path:str) -> None:
    """
    save the evaluation  metrics to a json file.
    """
    try:
        #ensure the directory exists
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,'w') as f:
            json.dump(metrics,f,indent=4)
        logger.debug("metrics saved to %s",file_path)
    except Exception as e:
        logger.error("error occured while saving the metrics:%s",e)
        raise

def main():
    try:
        clf=load_model('./models/model.pkl')
        print("clf",clf)
        test_data=load_data('./data/processed/test_tfidf.csv')

        X_test=test_data.iloc[:,:-1].values
        y_test=test_data.iloc[:,-1].values

        metrics=evaluate_model(clf,X_test,y_test)

        save_metrics(metrics,'reports/metrics.json')
    except Exception as e:
        logger.error("Failed to complete the model evaluation process %s",e)
        print(f"Error{e}")



if __name__ == "__main__":
    main()