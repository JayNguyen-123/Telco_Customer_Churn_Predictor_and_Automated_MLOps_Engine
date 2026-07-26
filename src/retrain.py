import os
import sys
import joblib
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Ensure runtime paths resolve correctly across nested module locations
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import clean_telco_data, split_features_and_target
from src.drift_detector import check_numerical_drift

# 1. Configure operational execution logger handlers
os.makedirs("models", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("models/cron_execution.log"),  # Stores long-term task audits
        logging.StreamHandler()                            # Echoes runtime trace profiles
    ]
)
logger = logging.getLogger("retrain_loop")


def trigger_pipeline_retraining(reference_data_path: str, model_output_path: str):
    """
    Ingests raw training records, structures preprocessing step objects,
    trains a fresh instance of the class-balanced Random Forest model,
    and updates the serialized production binary on disk.
    """
    logger.info("🎬 Launching model retraining engine sequence...")
    
    try:
        # Load local raw source reference records
        if not os.path.exists(reference_data_path):
            raise FileNotFoundError(f"Missing baseline dataset at path: '{reference_data_path}'")
            
        raw_df = pd.read_csv(reference_data_path)
        
        # Apply your standardized cleaning and feature splitting functions
        cleaned_df = clean_telco_data(raw_df)
        X, y = split_features_and_target(cleaned_df, target_column='Churn')
        
        # Categorize column vectors for processing steps
        numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        categorical_features = [col for col in X.columns if col not in numeric_features]
        
        # Construct isolated data transformation pipeline layers
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ])
        
        # Instantiate optimized pipeline with hyperparameters identified via GridSearchCV
        optimized_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=200, 
                max_depth=10, 
                min_samples_split=5, 
                class_weight='balanced', 
                random_state=42
            ))
        ])
        
        # Execute stratified train-test splits
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Fit models on fresh data spaces
        logger.info(f"🏋️ Fitting Random Forest Classifier on {X_train.shape[0]} training samples...")
        optimized_pipeline.fit(X_train, y_train)
        
        # Atomically serialize the new model file to disk
        joblib.dump(optimized_pipeline, model_output_path)
        logger.info(f"💾 Successfully serialized updated machine learning pipeline to: '{model_output_path}'")
        
    except Exception as e:
        logger.critical(f"💥 Retraining cycle terminated prematurely due to critical failure: {str(e)}")
        raise e


if __name__ == "__main__":
    logger.info("🔄 Initiating automated background retraining checklist cycle...")
    
    # Establish operational target workspace asset tracking variables
    data_path = "data/telco_raw.csv"
    model_path = "models/optimized_model.pkl"
    
    # Check if the data path exists before proceeding
    if not os.path.exists(data_path):
        logger.error(f"🛑 Bypassing audit routine: Base file data missing at path '{data_path}'")
        sys.exit(1)
        
    # In a live production setup, you would load recent traffic records out of database tables or logs.
    # For simulation purposes, we ingest the base dataset and inject artificial drift into charges.
    base_data = pd.read_csv(data_path)
    base_data['TotalCharges'] = pd.to_numeric(base_data['TotalCharges'], errors='coerce')
    base_data.dropna(subset=['TotalCharges'], inplace=True)
    
    simulated_live_logs = base_data.copy().sample(n=500, random_state=1)
    simulated_live_logs['MonthlyCharges'] = simulated_live_logs['MonthlyCharges'] * 1.25  # Injects +25% pricing drift
    
    # Calculate dataset distribution drift metrics via Kolmogorov-Smirnov Test
    target_numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    drift_report = check_numerical_drift(base_data, simulated_live_logs, target_numerical_features)
    
    # Determine if any monitored feature breaks your statistical alpha significance thresholds
    drift_detected = any(metrics["drift_detected"] for metrics in drift_report.values())
    
    if drift_detected:
        logger.warning("🚨 System drift validation failed! Operational boundaries exceeded. Initializing re-fit sequence...")
        trigger_pipeline_retraining(reference_data_path=data_path, model_output_path=model_path)
    else:
        logger.info("✅ Operational system metrics remain completely within safe zones. Pipeline update bypassed.")
