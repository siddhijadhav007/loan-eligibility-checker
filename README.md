# Loan Eligibility Checker

A machine learning-powered web application that automates loan approval decisions using predictive analytics.

## Problem & Solution

Financial institutions manually review thousands of loan applications, leading to delays and human error. This project uses a trained ML model to provide instant, data-driven loan eligibility predictions via a simple API and web interface.

## Key Features

- **Real-time Predictions**: Get loan approval status in milliseconds using a trained classification model
- **REST API**: Production-ready endpoints for programmatic integration
- **Web Interface**: User-friendly form for manual verification and testing
- **Scalable Architecture**: Built with FastAPI for async, high-performance request handling
- **ML Pipeline**: Includes data preprocessing, feature scaling, and probability-based decision logic

## Tech Stack

| Category | Tools |
|----------|-------|
| **Backend** |Python, FastAPI, Uvicorn |
| **ML & Data** | scikit-learn, pandas, joblib |
| **Frontend** | HTML, CSS, JavaScript |
| **Model** | Trained classifier (LogisticRegression/RandomForest) |

## How to Run

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

```bash
# Navigate to project folder
cd "Loan Eligibility Checker/Model_main"

# Install dependencies
pip install fastapi uvicorn pydantic pandas scikit-learn joblib

# Start the server
uvicorn app:app --reload
```

### Access

- **Web UI**: http://localhost:8000/ui
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/

### Test API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Married": "Yes",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 1500,
    "LoanAmount": 150,
    "Credit_History": "Good",
    "Property_Area": "Urban"
  }'
```

**Response:**
```json
{
  "loan_status": "Approved ✅",
  "approval_probability": 78.45
}
```

## Project Structure

```
Loan Eligibility Checker/
├── Model_main/
│   ├── app.py                      # FastAPI application & routes
│   ├── loan_status_predictor.pkl   # Trained ML model
│   ├── vector.pkl                  # Feature scaler
│   ├── loan_data_main.csv          # Training dataset
│   ├── model.ipynb                 # Model development notebook
│   ├── test.ipynb                  # Evaluation & testing
│   └── templates/
│       └── index_new.html          # Web UI form
└── README.md
```

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Loan eligibility prediction |
| `GET` | `/ui` | Web form interface |

## Future Enhancements

1. **Model Improvement**: Add feature engineering, ensemble methods, and SHAP explainability
2. **Production Deployment**: Docker containerization, CI/CD pipeline, and cloud hosting
3. **Advanced Features**: Batch predictions, user authentication, and prediction history

## Installation for Development

If you want to train the model yourself:

```bash
# Open Jupyter
jupyter notebook model.ipynb

# Run all cells to:
# - Load and preprocess data
# - Train multiple ML models
# - Evaluate performance
# - Export pkl files
```

## Author

Siddhi Jadhav.

---

**Ready to try it?** Start with `uvicorn app:app --reload` and visit http://localhost:8000/ui

