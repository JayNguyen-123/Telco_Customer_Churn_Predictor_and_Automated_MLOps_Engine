import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Ensure runtime paths resolve correctly across nested module locations
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.api import app

# Initialize an isolated test client instance
client = TestClient(app)

@pytest.fixture
def mock_scikit_pipeline():
    """
    Generates a completely synthetic fake scikit-learn model object container.
    Mocks standard prediction array layouts expected by Scikit-Learn models.
    """
    mock_obj = MagicMock()
    # Forces model prediction to output label 0, and a safe mock probability weight matrix
    mock_obj.predict.return_value = [0]
    mock_obj.predict_proba.return_value = [[0.85, 0.15]]
    return mock_obj


@patch("joblib.load")
@patch("os.path.exists")
def test_predict_endpoint_with_mocked_pipeline(mock_exists, mock_load, mock_scikit_pipeline):
    """
    Uses python patches to intercept disk checks and asset loads, forcing
    the API setup hook to initialize smoothly using our mock pipeline.
    """
    # 1. Arrange: Enforce mock existence evaluations to bypass file error triggers
    mock_exists.return_value = True
    mock_load.return_value = mock_scikit_pipeline
    
    # Trigger the model startup sequence inside an app lifecycle context manager
    with client as initialized_client:
        valid_payload = {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 6,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 45.0,
            "TotalCharges": 270.0
        }
        
        # 2. Act: Send standard payload transaction to target endpoint
        response = initialized_client.post("/predict", json=valid_payload)
        
        # 3. Assert: Verify routing and json response structures parse perfectly
        assert response.status_code == 200
        json_resp = response.json()
        assert json_resp["churn_prediction"] == "No"
        assert json_resp["churn_probability"] == 0.15
        assert json_resp["risk_status"] == "Low Risk / Stable"
        
        # Confirm that our mock pipeline functions were called exactly once as expected
        mock_scikit_pipeline.predict.assert_called_once()
        mock_scikit_pipeline.predict_proba.assert_called_once()
