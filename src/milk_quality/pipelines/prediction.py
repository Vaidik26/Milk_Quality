import os
import sys
import pandas as pd
from src.milk_quality.utils import load_object
from src.milk_quality.logger import logging
from src.milk_quality.exception import CustomException


class PredictionPipeline:
    def __init__(self, model_path: str, encoder_path: str):
        self.model_path = model_path
        self.encoder_path = encoder_path

    def predict(
        self, input_csv_path: str, output_csv_path: str = "artifacts/predictions.csv"
    ) -> str:
        logging.info("Prediction started.")
        try:
            # Load input data
            df = pd.read_csv(input_csv_path)
            logging.info(f"Input CSV loaded. Shape: {df.shape}")

            # Drop the target column if it exists
            if "Grade" in df.columns:
                df = df.drop(columns=["Grade"])
                logging.info("Dropped target column 'Grade' from input data.")

            # Load model and encoder
            model = load_object(self.model_path)
            encoder = load_object(self.encoder_path)
            logging.info("Model and LabelEncoder loaded successfully.")

            # Perform prediction
            predictions = model.predict(df)
            decoded_preds = encoder.inverse_transform(predictions)

            # Add predictions to the DataFrame
            df["Predicted_Grade"] = decoded_preds

            # Save the output
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            df.to_csv(output_csv_path, index=False)
            logging.info(f"Predictions saved at: {output_csv_path}")

            return output_csv_path

        except Exception as e:
            raise CustomException(e, sys)


# Example usage:
if __name__ == "__main__":
    pipeline = PredictionPipeline(
        model_path="artifacts/model.pkl", encoder_path="artifacts/label_encoder.pkl"
    )
    output_file = pipeline.predict(input_csv_path="artifacts/raw.csv")
    print(f"Predictions saved at: {output_file}")
