import pandas as pd
import logging
import os 
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string 
import nltk
nltk.download('stopwords')
nltk.download('punkt')


# ensure the "logs" directory exists
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)


# setting logging config
logger=logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

# console handler for terminal
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

# file path for file handler to save logs in file
log_path_file=os.path.join(log_dir,'data_preprocessing.log')

# file handler for saving logs in file
file_handler=logging.FileHandler(log_path_file)
file_handler.setLevel('DEBUG')

# formatter to format how hanler logs are shown 
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """
    transform the input text by converting it to  lower case , tokenizing ,  removing stopwords , and stemming
    
    """

    ps=PorterStemmer()

    #convert text to lowercase
    text=text.lower()

    # tokenization
    text=nltk.word_tokenize(text)

    # removing non-alpha numeric char and punctuations
    text=[word for word in text if word.isalnum()]

    # removing stopwords
    text=[word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    # stemming words --> to root form
    text=[ ps.stem(word) for word in text]

    # join the tokens into single string
    return " ".join(text)


def preprocess_df(df,text_column='text',target_column='target'):
    """
    preprocess the dataframe by encoding the target column ,removing duplicates and transforming  the text column


    """
    try:
        encoder=LabelEncoder()
        df[target_column]=encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded sucessfully')

        # removing duplicates
        df=df.drop_duplicates(keep='first')
        logger.debug('Duplicates data removed')

        # apply text transformation to the specified text column
        # df.loc[:,text_column]=df[text_column].apply(transform_text) all rows of text_col
        df[text_column]=df[text_column].apply(transform_text) #simpleer version
        logger.debug("Text column transformed")
        return df

    except KeyError as e:
        logger.error("Column not found : %s",e)
        raise
    except Exception as e:
        logger.error("error during text normalization :%s",e )
        raise

def main(text_column='text',target_column='target'):
    """
    main function to load raw data , preprocess it and  save the processed data

    
    """
    try:
            train_data=pd.read_csv('./data/raw/train.csv')
            test_data=pd.read_csv('./data/raw/test.csv')
            logger.debug("Data loaded sucessfully")


                # transform the data
            train_data_processed=preprocess_df(train_data,text_column,target_column)
            test_data_processed=preprocess_df(test_data,text_column,target_column)


            # store the preprocess data inside data/processed
            data_path=os.path.join('./data','interim')
            os.makedirs(data_path,exist_ok=True)

            train_data_processed.to_csv(os.path.join(data_path,"train_data_processed.csv"),index=False)
            test_data_processed.to_csv(os.path.join(data_path,"test_data_processed.csv"),index=False)

            logger.debug("processed data saved to %s",data_path)

    except FileNotFoundError as e:
        logger.error("file not found: %s",e)
    except pd.errors.EmptyDataError as e:
        logger.error("No data: %s",e)
    except Exception as e:
        logger.error("Failed to complete the data transformation process : %s",e)
        print(f"Error:%s {e} ")


if __name__ =="__main__":
   main()
       
        
