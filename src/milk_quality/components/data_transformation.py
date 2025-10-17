import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.milk_quality.logger import logging
from src.milk_quality.exception import CustomException
from src.milk_quality.utils import save_object


class DataTransformationConfig:
    processed_data_dir = os.path.join("artifacts")
    preprocessor_obj_file_path = os.path.join(processed_data_dir, "label_encoder.pkl")
    train_csv_path = os.path.join(processed_data_dir, "train.csv")
    test_csv_path = os.path.join(processed_data_dir, "test.csv")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def initiate_data_transformation(self, df: pd.DataFrame):
        logging.info("Data Transformation initiated.")
        try:
            # Handle missing values (for future use)
            if df.isnull().sum().sum() > 0:
                logging.warning("Missing values found. Filling with mode for now.")
                df.fillna(df.mode().iloc[0], inplace=True)
            else:
                logging.info("No missing values found.")

            # Separate features and target
            X = df.drop("Grade", axis=1)
            y = df["Grade"]

            # Encode target using LabelEncoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            logging.info(
                f"Target classes after encoding: {list(label_encoder.classes_)}"
            )

            # Split into train-test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.3, random_state=42
            )

            # Create output directory
            os.makedirs(self.config.processed_data_dir, exist_ok=True)

            # Save combined train and test CSVs
            train_df = X_train.copy()
            train_df["Grade"] = y_train
            test_df = X_test.copy()
            test_df["Grade"] = y_test

            train_df.to_csv(self.config.train_csv_path, index=False)
            test_df.to_csv(self.config.test_csv_path, index=False)

            logging.info("train.csv and test.csv saved successfully.")

            # Save label encoder
            save_object(self.config.preprocessor_obj_file_path, label_encoder)
            logging.info("Label encoder saved successfully.")

            logging.info("Data transformation completed successfully.")

            return (
                self.config.train_csv_path,
                self.config.test_csv_path,
                self.config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
