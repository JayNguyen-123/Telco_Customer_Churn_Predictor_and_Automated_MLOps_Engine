import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure runtime paths resolve correctly across nested module locations
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import clean_telco_data, split_features_and_target

def test_clean_telco_data_removes_customer_id():
    """Verifies that the high-cardinality customerID column is dropped to prevent overfitting."""
    # 1. Arrange: Create mock input data with a customerID column
    mock_data = pd.DataFrame({
        'customerID': ['1234-ABCD', '5678-EFGH'],
        'TotalCharges': ['29.85', '1889.50'],
        'Churn': ['No', 'Yes']
    })
    
    # 2. Act: Run the preprocessing function
    cleaned_df = clean_telco_data(mock_data)
    
    # 3. Assert: Verify the ID column is completely gone but data rows remain intact
    assert 'customerID' not in cleaned_df.columns
    assert len(cleaned_df) == 2


def test_clean_telco_data_handles_missing_total_charges():
    """Verifies that empty string spaces in TotalCharges are coerced to NaN and safely dropped."""
    # 1. Arrange: Create mock input data with a blank space row in TotalCharges
    mock_data = pd.DataFrame({
        'customerID': ['1111-AAAA', '2222-BBBB'],
        'TotalCharges': [' ', '50.25'],  # First row mimics an empty space entry from the raw dataset
        'Churn': ['No', 'No']
    })
    
    # 2. Act: Run the preprocessing function
    cleaned_df = clean_telco_data(mock_data)
    
    # 3. Assert: Verify the empty row was dropped and the column datatype became a numeric float
    assert len(cleaned_df) == 1
    assert cleaned_df['TotalCharges'].iloc[0] == 50.25
    assert pd.api.types.is_numeric_dtype(cleaned_df['TotalCharges'])


def test_split_features_and_target_success():
    """Verifies that features and targets are split correctly, and string labels become binary ints."""
    # 1. Arrange: Create a clean mock dataset
    mock_data = pd.DataFrame({
        'gender': ['Female', 'Male'],
        'MonthlyCharges': [29.85, 56.95],
        'Churn': ['No', 'Yes']
    })
    
    # 2. Act: Split the data into features and target arrays
    X, y = split_features_and_target(mock_data, target_column='Churn')
    
    # 3. Assert: Confirm structural shapes and mapping calculations
    assert 'Churn' not in X.columns
    assert 'gender' in X.columns
    assert len(y) == 2
    assert y.iloc[0] == 0  # 'No' maps to 0
    assert y.iloc[1] == 1  # 'Yes' maps to 1
    assert pd.api.types.is_numeric_dtype(y)


def test_split_features_and_target_missing_column_error():
    """Verifies that a KeyError is thrown if the specified target column does not exist."""
    # 1. Arrange: Create mock data missing the default target column
    mock_data = pd.DataFrame({
        'gender': ['Female', 'Male'],
        'MonthlyCharges': [29.85, 56.95]
    })
    
    # 2. Act & Assert: Ensure the function raises a KeyError
    with pytest.raises(KeyError):
        split_features_and_target(mock_data, target_column='Churn')
