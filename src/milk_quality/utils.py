import os
import dill
import json
import pandas as pd
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.milk_quality.logger import logging
from src.milk_quality.exception import CustomException

# Load environment variables from .env file
load_dotenv()


def get_collection_as_dataframe() -> pd.DataFrame:
    """
    Load data from MongoDB collection using .env config.
    """
    try:
        mongo_uri = os.getenv("MONGO_URI")
        database_name = os.getenv("MONGO_DB")
        collection_name = os.getenv("MONGO_COLLECTION")

        logging.info("Starting to load data from MongoDB.")
        logging.info(f"Connecting to MongoDB at URI: {mongo_uri}")
        client = MongoClient(mongo_uri)
        collection = client[database_name][collection_name]

        logging.info(f"Fetching documents from {database_name}.{collection_name}")
        data = list(collection.find())
        df = pd.DataFrame(data)

        if "_id" in df.columns:
            df.drop(columns=["_id"], inplace=True)
            logging.info("Dropped '_id' column from DataFrame.")

        logging.info(f"Successfully loaded data. DataFrame shape: {df.shape}")
        return df

    except Exception as e:
        logging.error("Failed to load data from MongoDB.", exc_info=True)
        raise CustomException(e)


def upload_dataframe_to_mongodb(df: pd.DataFrame) -> None:
    """
    Upload a pandas DataFrame to MongoDB collection using .env config.
    """
    try:
        mongo_uri = os.getenv("MONGO_URI")
        database_name = os.getenv("MONGO_DB")
        collection_name = os.getenv("MONGO_COLLECTION")

        logging.info("Starting to upload data to MongoDB.")
        logging.info(f"Connecting to MongoDB at URI: {mongo_uri}")
        client = MongoClient(mongo_uri)
        collection = client[database_name][collection_name]

        data = df.to_dict(orient="records")

        if data:
            collection.insert_many(data)
            logging.info(
                f"Successfully inserted {len(data)} records into {database_name}.{collection_name}"
            )
        else:
            logging.warning("Provided DataFrame is empty. No data inserted.")

    except Exception as e:
        logging.error("Failed to upload data to MongoDB.", exc_info=True)
        raise CustomException(e)
