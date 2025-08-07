from fastapi import FastAPI, UploadFile, File # type: ignore
from data_loader import DataLoader
from models import SleepRecord
import asyncio
import pandas as pd
import os
from cli import perform_regression_logic
from cli import run_diagnostics_logic

# FastAPI application instance
# This will serve as the entry point for our API
app = FastAPI()


# get the root endpoint
# This is a simple endpoint to check if the API is running
@app.get("/")
def root():
    return {"message": "Sleep API is up and running ✅"}

# Endpoint to analyze sleep data from an uploaded CSV file
# This endpoint accepts a file upload, processes it, and returns analysis results
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    filename = None # prevent unbound local error
    try:
        filename = "uploaded_" + (file.filename or "file.csv")
        with open(filename, "wb") as f:
            f.write(await file.read())

        loader = DataLoader(filename)
        df = loader.preprocess()
        records = loader.validate_records()

        # Validate records against the Pydantic model
        return {
            "filename": file.filename,
            "valid_records": len(records),
            "columns": list(df.columns),
            "shape": df.shape,
            "column_info": df.dtypes.astype(str).to_dict()
        }
    finally:
        if filename is not None and os.path.exists(filename): # will raise an error if the file does not exist
            os.remove(filename)

# Creating endpoints for each command

# Endpoint to run the regression analysis
# This endpoint triggers the regression analysis function from the CLI module
@app.get("/regression")
async def regression():
    result = await perform_regression_logic("Sleep_health_and_lifestyle_dataset.csv")
    return result

# Endpoint to run the diagnostics
# This endpoint triggers the diagnostics function from the CLI module
@app.get("/diagnostics")
def diagnostics():
    return run_diagnostics_logic("Sleep_health_and_lifestyle_dataset.csv")