from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import os
from werkzeug.utils import secure_filename
from src.milk_quality.pipelines.prediction import PredictionPipeline

app = Flask(__name__)
UPLOAD_FOLDER = "artifacts"
PREDICTION_CSV = os.path.join(UPLOAD_FOLDER, "predictions.csv")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("home.html")  # Show Get Started button


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(input_path)

            # Run prediction pipeline
            pipeline = PredictionPipeline(
                model_path="artifacts/model.pkl",
                encoder_path="artifacts/label_encoder.pkl",
            )
            pipeline.predict(input_csv_path=input_path, output_csv_path=PREDICTION_CSV)
            return redirect(url_for("result"))
    return render_template("predict.html")  # Show file upload form


@app.route("/result")
def result():
    if os.path.exists(PREDICTION_CSV):
        df = pd.read_csv(PREDICTION_CSV)
        table_html = df.to_html(classes="table table-bordered", index=False)
        return render_template("result.html", table=table_html)
    return "No prediction file found."


@app.route("/download")
def download():
    if os.path.exists(PREDICTION_CSV):
        return send_file(PREDICTION_CSV, as_attachment=True)
    return "No file to download."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
