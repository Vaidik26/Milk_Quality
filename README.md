# 🥛 Milk Quality Prediction

This project is an end-to-end machine learning pipeline that predicts the quality of milk as Low, Medium, or High based on physicochemical features. It demonstrates real-world classification problem-solving — from data preprocessing to model selection and deployment.

## 📌 Project Highlights

✅ Supervised classification problem

📊 Exploratory Data Analysis (EDA)

⚙️ Data preprocessing & label encoding

📉 Model training & evaluation

🧠 Achieved 99% test accuracy using Gradient Boosting Classifier

💾 Model saved for future prediction


## 📂 Project Structure

Milk_Quality_Prediction/
│
├── data/                   # Raw and processed datasets
├── notebooks/              # EDA and development notebooks
├── pipelines/              # Training and prediction pipelines
├── artifacts/              # Saved model and evaluation metrics
├── src/                    # Source code
│   └── milk_quality/
│       ├── components/     # Data ingestion, transformation, model trainer
│       └── pipelines/      # train and predict scripts
├── app.py                  # Streamlit app (optional)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation


## 📊 Features Used

pH

Temperature

Taste

Odor

Fat

Turbidity

Colour

(Target: Grade → encoded as Low, Medium, High)


## 🚀 How to Run

1. Clone the Repository

git clone https://github.com/Vaidik26/Milk_Quality
cd Milk_Quality_Prediction

2. Create & Activate Virtual Environment

python -m venv venv
venv\Scripts\activate   # On Windows

3. Install Requirements

pip install -r requirements.txt

4. Train the Model

python -m src.milk_quality.pipelines.training

5. Run Predictions

python -m src.milk_quality.pipelines.prediction


## 📈 Model Performance

| Model                 | Accuracy     |
| --------------------- | ------------ |
| Logistic Regression   | 83.02%       |
| SVM (RBF Kernel)      | 90.57%       |
| Naive Bayes           | 85.38%       |
| **Gradient Boosting** | **99.53%** ✅ |


## 🛠️ Tools & Libraries

Python

scikit-learn

pandas

numpy

matplotlib / seaborn

joblib

## 🤝 Let's Connect

If you find this useful or have suggestions, feel free to fork the repo or connect with me on [Linkedin](https://www.linkedin.com/in/vaidik-yadav-260a60248/)

## 📄 License
This project is open source under the MIT License.