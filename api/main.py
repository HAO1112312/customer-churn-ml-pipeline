"""
main.py
FastAPI REST Service for Real-Time Churn Inference and Batch Scoring.
Includes Pydantic request validation, risk classification, and SHAP reason codes.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

# Initialize FastAPI App
app = FastAPI(
    title="Customer Churn Prediction & Explainability API",
    description="Production-grade Machine Learning inference service for customer churn risk scoring and SHAP explainability.",
    version="1.0.0"
)

# Global variables for loaded models
PREPROCESSOR = None
MODEL_PAYLOAD = None
BEST_MODEL = None
OPTIMAL_THRESHOLD = 0.50
FEATURE_NAMES = []


class CustomerFeaturesInput(BaseModel):
    """Schema for individual customer prediction payload."""
    age: int = Field(..., ge=18, le=100, example=35)
    has_partner: str = Field(..., example="Yes")
    dependents: str = Field(..., example="No")
    tenure_months: int = Field(..., ge=0, le=120, example=4)
    contract_type: str = Field(..., example="Month-to-month")
    payment_method: str = Field(..., example="Electronic check")
    paperless_billing: str = Field(..., example="Yes")
    internet_service: str = Field(..., example="Fiber optic")
    online_security: str = Field(..., example="No")
    tech_support: str = Field(..., example="No")
    streaming_tv: str = Field(..., example="Yes")
    monthly_charges: float = Field(..., ge=10.0, le=300.0, example=89.50)
    total_charges: float = Field(..., ge=0.0, example=358.0)
    days_since_last_login: int = Field(..., ge=0, le=365, example=28)
    login_frequency: int = Field(..., ge=0, le=100, example=4)
    usage_drop_rate: float = Field(..., example=35.0)
    support_tickets: int = Field(..., ge=0, le=50, example=4)
    satisfaction_score: Optional[float] = Field(2.0, ge=1.0, le=5.0, example=2.0)


class PredictionOutput(BaseModel):
    churn_probability: float
    churn_predicted: bool
    risk_tier: str
    decision_threshold: float
    recommended_action: str


@app.on_event("startup")
def load_artifacts():
    """Load preprocessor, trained model and metadata on API startup."""
    global PREPROCESSOR, MODEL_PAYLOAD, BEST_MODEL, OPTIMAL_THRESHOLD, FEATURE_NAMES
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prep_path = os.path.join(base_dir, "models", "preprocessor.pkl")
    model_path = os.path.join(base_dir, "models", "best_model.pkl")

    if os.path.exists(prep_path) and os.path.exists(model_path):
        from src.preprocessor import ChurnDataPreprocessor
        PREPROCESSOR = ChurnDataPreprocessor.load(prep_path)
        MODEL_PAYLOAD = joblib.load(model_path)
        BEST_MODEL = MODEL_PAYLOAD["model"]
        OPTIMAL_THRESHOLD = float(MODEL_PAYLOAD.get("optimal_threshold", 0.50))
        FEATURE_NAMES = MODEL_PAYLOAD.get("feature_names", [])
        print("[+] Models and Preprocessor successfully loaded into FastAPI.")
    else:
        print("[!] Warning: Model files not found. Run training script first.")


@app.get("/")
def index():
    return {
        "service": "Customer Churn Prediction API",
        "status": "Online",
        "docs_url": "/docs",
        "best_model_loaded": BEST_MODEL is not None
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if BEST_MODEL is not None else "degraded",
        "optimal_threshold": OPTIMAL_THRESHOLD,
        "features_count": len(FEATURE_NAMES)
    }


@app.post("/predict", response_model=PredictionOutput)
def predict_churn(customer: CustomerFeaturesInput):
    if BEST_MODEL is None or PREPROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded. Train models first.")

    # 1. Transform raw input
    input_dict = customer.dict()
    X_proc = PREPROCESSOR.transform_single(input_dict)

    # 2. Score with ML Model
    prob = float(BEST_MODEL.predict_proba(X_proc)[0, 1])
    is_churn = bool(prob >= OPTIMAL_THRESHOLD)

    # 3. Categorize Risk Tier & Business Recommendation
    if prob >= 0.70:
        tier = "CRITICAL_RISK"
        action = "Immediate proactive outreach by Senior VIP retention team. Offer 20% renewal discount or dedicated service manager."
    elif prob >= OPTIMAL_THRESHOLD:
        tier = "HIGH_RISK"
        action = "Send automated re-engagement incentive email and schedule customer health check."
    elif prob >= 0.30:
        tier = "MODERATE_RISK"
        action = "Monitor engagement signals and prompt in-app feature discovery onboarding."
    else:
        tier = "LOW_RISK"
        action = "Customer is healthy. Candidate for loyalty rewards, up-sell, and referral campaigns."

    return PredictionOutput(
        churn_probability=round(prob, 4),
        churn_predicted=is_churn,
        risk_tier=tier,
        decision_threshold=round(OPTIMAL_THRESHOLD, 4),
        recommended_action=action
    )


@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...)):
    """Upload CSV and receive predictions for all customer rows."""
    if BEST_MODEL is None or PREPROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")

    try:
        df_input = pd.read_csv(file.file)
        from src.features import engineer_features
        df_featured = engineer_features(df_input)
        df_ready = df_featured.drop(columns=["churn", "customer_id"], errors="ignore")
        X_trans = PREPROCESSOR.preprocessor_pipeline.transform(df_ready)

        probs = BEST_MODEL.predict_proba(X_trans)[:, 1]
        preds = (probs >= OPTIMAL_THRESHOLD).astype(int)

        df_input["churn_probability"] = np.round(probs, 4)
        df_input["churn_predicted"] = preds

        summary = {
            "total_records": len(df_input),
            "predicted_churners": int(preds.sum()),
            "projected_churn_rate_pct": float(round(preds.mean() * 100, 2))
        }
        return {"summary": summary, "preview": df_input.head(10).to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV file: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
