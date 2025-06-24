from src.milk_quality.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    trainer = ModelTrainer()
    model_path, model_name, f1 = trainer.train_and_evaluate(
        train_csv_path="artifacts/train.csv",
        test_csv_path="artifacts/test.csv",
        encoder_path="artifacts/label_encoder.pkl",
    )
    print("✅ Model training complete.")
    print(f"🔹 Best model: {model_name}")
    print(f"🎯 F1 Score: {f1}")
    print(f"📦 Model saved at: {model_path}")
