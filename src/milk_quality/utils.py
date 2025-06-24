import os
import dill
import json
import pandas as pd
import pymongo
from pymongo import MongoClient
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from dotenv import load_dotenv
from src.milk_quality.logger import logging
from src.milk_quality.exception import CustomException

load_dotenv()

def save_object(file_path: str, obj) -> None:
    """
    Save any Python object using dill.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved at: {file_path}")
    except Exception as e:
        raise CustomException(e)


def load_object(file_path: str):
    """
    Load any Python object saved with dill.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e)


def read_csv(path: str) -> pd.DataFrame:
    """
    Read a CSV file and return as DataFrame.
    """
    try:
        df = pd.read_csv(path)
        logging.info(f"CSV loaded from: {path} with shape {df.shape}")
        return df
    except Exception as e:
        raise CustomException(e)


def write_csv(df: pd.DataFrame, path: str) -> None:
    """
    Save a DataFrame to CSV.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logging.info(f"CSV written to: {path}")
    except Exception as e:
        raise CustomException(e)


def evaluate_model(X_train, y_train, X_test, y_test, model: object) -> dict:
    """
    Train model and return evaluation metrics.
    """
    try:
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        metrics = {
            "train_r2_score": r2_score(y_train, y_pred_train),
            "test_r2_score": r2_score(y_test, y_pred_test),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "test_rmse": mean_squared_error(y_test, y_pred_test, squared=False),
        }

        logging.info(f"Model Evaluation: {metrics}")
        return metrics
    except Exception as e:
        raise CustomException(e)


def save_json(path: str, data: dict) -> None:
    """
    Save a dictionary as a JSON file.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logging.info(f"JSON saved at: {path}")
    except Exception as e:
        raise CustomException(e)


def get_collection_as_dataframe() -> pd.DataFrame:
    """
    Load data from MongoDB collection using .env config.
    """
    try:
        mongo_uri = os.getenv("MONGO_URI")
        database_name = os.getenv("MONGO_DB")
        collection_name = os.getenv("MONGO_COLLECTION")

        logging.info(f"Connecting to MongoDB at {mongo_uri}")
        client = MongoClient(mongo_uri)
        collection = client[database_name][collection_name]
        data = list(collection.find())
        df = pd.DataFrame(data)

        if "_id" in df.columns:
            df.drop(columns=["_id"], inplace=True)

        logging.info(f"Loaded data from {database_name}.{collection_name}, shape: {df.shape}")
        return df
    except Exception as e:
        raise CustomException(e)


def upload_dataframe_to_mongodb(df: pd.DataFrame) -> None:
    """
    Upload a pandas DataFrame to MongoDB collection using .env config.
    """
    try:
        mongo_uri = os.getenv("MONGO_URI")
        database_name = os.getenv("MONGO_DB")
        collection_name = os.getenv("MONGO_COLLECTION")

        logging.info(f"Uploading DataFrame to MongoDB: {database_name}.{collection_name}")
        client = MongoClient(mongo_uri)
        collection = client[database_name][collection_name]

        data = df.to_dict(orient="records")
        if data:
            collection.insert_many(data)
            logging.info(f"Inserted {len(data)} records.")
        else:
            logging.warning("Empty DataFrame. No data uploaded.")
    except Exception as e:
        raise CustomException(e)