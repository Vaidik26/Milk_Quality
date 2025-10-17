from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import os
from src.milk_quality.pipelines.prediction import PredictionPipeline

# Initialize FastAPI app
app = FastAPI()

UPLOAD_FOLDER = "artifacts"
PREDICTION_CSV = os.path.join(UPLOAD_FOLDER, "predictions.csv")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Mount static folder if you have CSS/JS files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Template folder (like Flask’s render_template)
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Show home page with Get Started button"""
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/predict", response_class=HTMLResponse)
async def show_predict_page(request: Request):
    """Show file upload page"""
    return templates.TemplateResponse("predict.html", {"request": request})


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    """Handle file upload and trigger prediction pipeline"""
    filename = file.filename
    input_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Run prediction pipeline
    pipeline = PredictionPipeline(
        model_path="artifacts/model.pkl",
        encoder_path="artifacts/label_encoder.pkl",
    )
    pipeline.predict(input_csv_path=input_path, output_csv_path=PREDICTION_CSV)

    return RedirectResponse(url="/result", status_code=303)


@app.get("/result", response_class=HTMLResponse)
async def result(request: Request):
    """Display prediction results as HTML table"""
    if os.path.exists(PREDICTION_CSV):
        df = pd.read_csv(PREDICTION_CSV)
        table_html = df.to_html(classes="table table-bordered", index=False)
        return templates.TemplateResponse("result.html", {"request": request, "table": table_html})
    return HTMLResponse(content="No prediction file found.", status_code=404)


@app.get("/download")
async def download():
    """Download the predictions CSV"""
    if os.path.exists(PREDICTION_CSV):
        return FileResponse(PREDICTION_CSV, filename="predictions.csv", media_type="text/csv")
    return HTMLResponse(content="No file to download.", status_code=404)


# Run with: uvicorn app:app --host 0.0.0.0 --port 5000 --reload
