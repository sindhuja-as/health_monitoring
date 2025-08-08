from fastapi import FastAPI, File, UploadFile
import pandas as pd
import joblib
import io

# Load the pretrained model
model = joblib.load("BW2_classifier.pkl")  # replace with your model filename

# Create FastAPI app
app = FastAPI()

@app.post("/predict/")
async def predict_from_csv(file: UploadFile = File(...)):
    # Read uploaded CSV into DataFrame
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    # Check that exactly one row is present
    if len(df) != 1:
        return {"error": "CSV must contain exactly one row for a single patient"}

    # Make prediction
    prediction = model.predict(df)

    return {
        "prediction": prediction.tolist(),
        "message": "Prediction generated successfully"
    }
