import pandas as pd
from src.milk_quality.components.data_transformation import DataTransformation

if __name__ == "__main__":
    try:
        # Load raw data from ingestion step
        df = pd.read_csv("artifacts/raw.csv")
        print(f"✅ Raw Data Shape: {df.shape}")
        print("📋 First 5 Rows:")
        print(df.head())

        # Transform the data
        transformer = DataTransformation()
        train_csv, test_csv, encoder_path = transformer.initiate_data_transformation(df)

        print("\n✅ Data transformation completed.")
        print(f"🔹 Train CSV saved at: {train_csv}")
        print(f"🔹 Test CSV saved at: {test_csv}")
        print(f"🔹 LabelEncoder saved at: {encoder_path}")

    except Exception as e:
        print("❌ Data transformation failed:", e)
