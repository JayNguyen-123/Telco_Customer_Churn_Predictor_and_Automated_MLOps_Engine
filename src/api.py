import os
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# =====================================================================
# 1. APPLICATION LOGGING & SAFEGUARDS CONFIGURATION
# =====================================================================
# Setup unified output directory paths for storing runtime monitoring log traces
os.makedirs("models", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("models/inference_audit.log"),  # Stores persistent audit log files
        logging.StreamHandler()                              # Echo logs out to the active terminal
    ]
)
logger = logging.getLogger("churn_api")


# =====================================================================
# 2. FASTAPI INITIALIZATION & OPENAPI SCHEMA METADATA
# =====================================================================
app = FastAPI(
    title="Telco Customer Churn Prediction REST API",
    description=(
        "Production machine learning microservice built using FastAPI. "
        "Evaluates real-time customer account metrics to calculate cancellation risks "
        "and driving churn propensities."
    ),
    version="1.0.0"
)


# =====================================================================
# 3. PYDANTIC REQUEST VALIDATION FRAMEWORK
# =====================================================================
class CustomerData(BaseModel):
    gender: str = Field(..., description="Customer gender profile label", example="Female")
    SeniorCitizen: int = Field(..., description="Senior citizen flag (0 = No, 1 = Yes)", example=0, ge=0, le=1)
    Partner: str = Field(..., description="Whether the customer has a partner", example="Yes")
    Dependents: str = Field(..., description="Whether the customer has dependents", example="No")
    tenure: int = Field(..., description="Number of months the customer has stayed with the company", example=12, ge=0)
    PhoneService: str = Field(..., description="Whether the customer has a phone service", example="Yes")
    MultipleLines: str = Field(..., description="Whether the customer has multiple lines", example="No")
    InternetService: str = Field(..., description="Customer's internet service provider type", example="Fiber optic")
    OnlineSecurity: str = Field(..., description="Whether the customer has online security", example="No")
    OnlineBackup: str = Field(..., description="Whether the customer has online backup", example="Yes")
    DeviceProtection: str = Field(..., description="Whether the customer has device protection", example="No")
    TechSupport: str = Field(..., description="Whether the customer has tech support", example="No")
    StreamingTV: str = Field(..., description="Whether the customer has streaming TV", example="No")
    StreamingMovies: str = Field(..., description="Whether the customer has streaming movies", example="No")
    Contract: str = Field(..., description="The contract term of the customer", example="Month-to-month")
    PaperlessBilling: str = Field(..., description="Whether the customer has paperless billing", example="Yes")
    PaymentMethod: str = Field(..., description="The customer's payment method type", example="Electronic check")
    MonthlyCharges: float = Field(..., description="The amount charged to the customer monthly", example=70.35, ge=0)
    TotalCharges: float = Field(..., description="The total amount charged to the customer", example=844.20, ge=0)


# =====================================================================
# 4. LIFECYCLE EVENT HANDLERS
# =====================================================================
# Global memory cache container hosting our compiled Scikit-Learn evaluation object
model_pipeline = None

@app.on_event("startup")
def load_model_pipeline():
    """Loads the pre-compiled Scikit-Learn binary pipeline from disk upon service startup."""
    global model_pipeline
    model_path = "models/optimized_model.pkl"
    
    logger.info("Initializing system lifecycle hooks. Loading binary assets...")
    if not os.path.exists(model_path):
        logger.critical(f"❌ Failed to locate the compiled model file asset at path: '{model_path}'")
        raise FileNotFoundError(
            f"The binary model file was not found at '{model_path}'. "
            "Please run your training execution pipelines first."
        )
        
    try:
        model_pipeline = joblib.load(model_path)
        logger.info("📦 Machine learning model pipeline loaded successfully into virtual memory.")
    except Exception as e:
        logger.critical(f"💥 Critical failure parsing model binary file serialization data: {str(e)}")
        raise RuntimeError("Model binary parsing failure.")


# =====================================================================
# 5. CORE PRODUCTION API ROUTING ENDPOINTS
# =====================================================================
@app.get("/", tags=["Monitoring"])
def system_health_check():
    """Evaluates host server health metrics and confirms the structural validity of binary model assets."""
    is_model_valid = model_pipeline is not None
    return {
        "status": "healthy" if is_model_valid else "degraded",
        "model_loaded": is_model_valid,
        "environment": "production"
    }


@app.post("/predict", tags=["Inference"])
def evaluate_customer_churn_risk(customer: CustomerData):
    """
    Accepts a single user profile record, processes the attributes through 
    the transformation pipelines, and returns a structural evaluation prediction.
    """
    if model_pipeline is None:
        logger.error("🛑 Inference call dropped: The model memory layer is uninitialized.")
        raise HTTPException(
            status_code=503, 
            detail="The underlying model pipeline is uninitialized or in a degraded state."
        )
    
    try:
        # Convert incoming nested dictionary schema directly into a structured Pandas DataFrame 
        # matching the identical evaluation layout passed during your pipeline fitting stages.
        input_data_dict = customer.dict()
        input_df = pd.DataFrame([input_data_dict])
        
        # Calculate raw matrix evaluations
        prediction = int(model_pipeline.predict(input_df)[0])
        probability = float(model_pipeline.predict_proba(input_df)[0][1])
        
        # Dispatch structured information out to audit logging file destinations
        logger.info(
            f"🔮 Inference Executed Successfully | Contract: {customer.Contract} | "
            f"Tenure: {customer.tenure} months | Monthly Charge: ${customer.MonthlyCharges} | "
            f"Churn Prediction Outcome: {prediction} | Churn Probability: {probability:.4f}"
        )
        
        return {
            "churn_prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(probability, 4),
            "risk_status": "High Risk" if probability >= 0.5 else "Low Risk / Stable"
        }
        
    except Exception as e:
        logger.error(f"❌ Internal inference structural failure encountered during pipeline runtime: {str(e)}")
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to process request data due to internal pipeline evaluation failure: {str(e)}"
        )
