import os
import sys
from src.milk_quality.components.model_trainer import ModelTrainer
from src.milk_quality.exception import CustomException
from src.milk_quality.logger import logging


def main():
    try:
        logging.info("Model training script initiated.")

        trainer = ModelTrainer()
        model_path, model_name, f1 = trainer.train_and_evaluate(
            train_csv_path="artifacts/train.csv",
            test_csv_path="artifacts/test.csv",
            encoder_path="artifacts/label_encoder.pkl",
        )

        logging.info("Model training complete.")
        logging.info(f"Best model: {model_name}")
        logging.info(f"F1 Score: {f1}")
        logging.info(f"Model saved at: {model_path}")

        print("Model training complete.")
        print(f"Best model: {model_name}")
        print(f"F1 Score: {f1}")
        print(f"Model saved at: {model_path}")

    except Exception as e:
        logging.error("Model training failed.")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
