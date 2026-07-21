from fastapi import FastAPI, UploadFile, File
import shutil
import os
from src.inference import predict_digit

app = FastAPI(title="Burmese Digit OCR API", version="2.0")

@app.get("/")
def home():
    """Health check endpoint to ensure the server is running."""
    return {"status": "Online", "model": "Burmese CNN V2"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Accepts an image file, runs it through the V2 CNN, and returns the digit."""
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        prediction = predict_digit(temp_file_path)
        response = {
            "filename": file.filename, 
            "predicted_digit": prediction,
            "success": True
        }
    except Exception as e:
        response = {"error": str(e), "success": False}
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
    return response