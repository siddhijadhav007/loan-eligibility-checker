from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.responses import HTMLResponse

app = FastAPI(title="Loan Approval System")

# =========================
# Load model & scaler
# =========================
model = joblib.load("loan_status_predictor.pkl")
scaler = joblib.load("vector.pkl")

# EXACT numeric columns used during scaler fit
num_cols = [
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Credit_History'
]


# Encoding mappings
ENCODING = {
    "Married": {"Yes": 1, "No": 0},
    "Education": {"Graduate": 1, "Not Graduate": 0},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Credit_History": {"Good": 1, "Bad": 0},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
}

# =========================
# Input schema
# =========================
class LoanApprovalInput(BaseModel):
    Married: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Credit_History: str
    Property_Area: str

# =========================
# Encode input
# =========================
def encode_input(data: LoanApprovalInput):
    try:
        return {
            "Married": ENCODING["Married"][data.Married],
            "Education": ENCODING["Education"][data.Education],
            "Self_Employed": ENCODING["Self_Employed"][data.Self_Employed],
            "ApplicantIncome": data.ApplicantIncome,
            "CoapplicantIncome": data.CoapplicantIncome,
            "LoanAmount": data.LoanAmount,
            "Credit_History": ENCODING["Credit_History"][data.Credit_History],
            "Property_Area": ENCODING["Property_Area"][data.Property_Area],
        }
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input value: {e}")

# =========================
# Prediction API
# =========================
@app.post("/predict")
async def predict_loan_status(application: LoanApprovalInput):
    encoded = encode_input(application)
    input_data = pd.DataFrame([encoded])

    # ✅ enforce SAME column order as training
    input_data = input_data[
        [
            'Married',
            'Education',
            'Self_Employed',
            'ApplicantIncome',
            'CoapplicantIncome',
            'LoanAmount',
            'Credit_History',
            'Property_Area'
        ]
    ]

    # ✅ scale EXACT columns used during fit
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # ✅ probability-based decision
    proba = model.predict_proba(input_data)[0][1]

    return {
        "loan_status": "Approved ✅" if proba >= 0.6 else "Not Approved ❌",
        "approval_probability": round(proba * 100, 2)
    }

# =========================
# Routes
# =========================
@app.get("/")
def home():
    return {"message": "Loan Approval API running 🚀"}

@app.get("/ui", response_class=HTMLResponse)
def load_ui():
    with open("templates/index_new.html", "r", encoding="utf-8") as f:
        return f.read()
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

from flask import Flask
app = Flask(__name__)

# cd "E:\---\supervise_ML\Loan Eligibility Checker\Model_main" 
# py -m uvicorn app:app --reload