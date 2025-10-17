import os
import pandas as pd
from dataclasses import dataclass
import sys
from src.milk_quality.utils import get_collection_as_dataframe, write_csv
from src.milk_quality.logger import logging
from src.milk_quality.exception import CustomException


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> pd.DataFrame:
        """
        Ingest data from MongoDB and save raw CSV.
        """
        logging.info("Data Ingestion started.")

        try:
            # Load data from MongoDB
            df = get_collection_as_dataframe()
            logging.info("Data loaded successfully from MongoDB.")

            # Create directories if needed
            os.makedirs(
                os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True
            )

            # Save as raw CSV
            write_csv(df, self.ingestion_config.raw_data_path)
            logging.info(f"Raw data saved at: {self.ingestion_config.raw_data_path}")

            return df

        except Exception as e:
            logging.error("Data ingestion failed.", exc_info=True)
            raise CustomException(e, sys)
