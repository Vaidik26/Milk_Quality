import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score
from src.milk_quality.logger import logging
from src.milk_quality.exception import CustomException
from src.milk_quality.utils import save_object


class ModelTrainerConfig:
    model_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def train_and_evaluate(
        self, train_csv_path: str, test_csv_path: str, encoder_path: str
    ):
        logging.info("Model training started.")
        try:
            # Load train/test data
            train_df = pd.read_csv(train_csv_path)
            test_df = pd.read_csv(test_csv_path)

            X_train = train_df.drop("Grade", axis=1)
            y_train = train_df["Grade"]

            X_test = test_df.drop("Grade", axis=1)
            y_test = test_df["Grade"]

            # Models to test
            models = {
                "RandomForest": RandomForestClassifier(),
                "GradientBoosting": GradientBoostingClassifier(),
            }

            best_model = None
            best_score = 0
            best_model_name = ""

            for name, model in models.items():
                logging.info(f"Training model: {name}")
                model.fit(X_train, y_train)

                train_preds = model.predict(X_train)
                test_preds = model.predict(X_test)

                train_f1 = f1_score(y_train, train_preds, average="weighted")
                test_f1 = f1_score(y_test, test_preds, average="weighted")

                logging.info(f"{name} Train F1 Score: {train_f1:.4f}")
                logging.info(f"{name} Test  F1 Score: {test_f1:.4f}")

                # Overfitting check
                f1_gap = train_f1 - test_f1
                if f1_gap > 0.03:
                    logging.warning(
                        f"Potential Overfitting Detected in {name} (Train-Test F1 Gap: {f1_gap:.4f})"
                    )
                else:
                    logging.info(
                        f"No overfitting detected in {name} (Gap: {f1_gap:.4f})"
                    )

                if test_f1 > best_score:
                    best_score = test_f1
                    best_model = model
                    best_model_name = name

            # Save best model
            save_object(self.config.model_path, best_model)
            logging.info(
                f"Best model: {best_model_name} with F1 Score: {best_score:.4f}"
            )
            logging.info(f"Model saved at: {self.config.model_path}")

            return self.config.model_path, best_model_name, best_score

        except Exception as e:
            raise CustomException(e, sys)
