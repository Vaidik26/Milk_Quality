from src.milk_quality.components.data_ingestion import DataIngestion

if __name__ == "__main__":
    ingestion = DataIngestion()
    df = ingestion.initiate_data_ingestion()
    print(df.head())
