import pandas as pd
import numpy as np

def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw Telco Customer Churn datasets for supervised machine learning pipelines.
    
    Processing Steps:
    1. Deep-copies the incoming DataFrame to protect against mutation side effects.
    2. Coerces the string column 'TotalCharges' into numeric floats.
    3. Drops rows containing resulting NaN values (originally blank space string values ' ').
    4. Removes the operational tracking column 'customerID' to prevent overfitting.
    
    Args:
        df (pd.DataFrame): Raw ingested Pandas DataFrame mirroring the IBM dataset layout.
        
    Returns:
        pd.DataFrame: Cleaned data framework optimized for pipeline feature extractions.
    """
    # 1. Enforce code isolation boundary by creating an independent copy
    processed_df = df.copy()
    
    # 2. Extract and resolve empty string spaces in TotalCharges column
    if 'TotalCharges' in processed_df.columns:
        # errors='coerce' forces empty strings or invalid data tokens into np.nan floats
        processed_df['TotalCharges'] = pd.to_numeric(processed_df['TotalCharges'], errors='coerce')
        
        # 3. Clean rows stripped of total revenue data bounds
        initial_row_count = len(processed_df)
        processed_df.dropna(subset=['TotalCharges'], inplace=True)
        dropped_rows = initial_row_count - len(processed_df)
        
        if dropped_rows > 0:
            # Note: In production, redirect this to your standard loggers if needed
            print(f"[DATA PREPROCESSING] Cleaned {dropped_rows} null value rows from 'TotalCharges'.")
            
    # 4. Remove uninformative high-cardinality metadata string indices
    if 'customerID' in processed_df.columns:
        processed_df.drop(columns=['customerID'], inplace=True)
        
    return processed_df


def split_features_and_target(df: pd.DataFrame, target_column: str = 'Churn') -> tuple:
    """
    Separates a cleaned DataFrame into input feature matrices and binarized target arrays.
    
    Converts string target values ('Yes'/'No') into machine-readable numeric flags (1/0).
    
    Args:
        df (pd.DataFrame): Cleaned subscriber dataset dataframe.
        target_column (str): Label string identifying the target variable column.
        
    Returns:
        tuple: (X, y) where X is a pandas DataFrame of features, and y is a pandas Series of binary targets.
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' is missing from the provided dataset.")
        
    X = df.drop(columns=[target_column])
    
    # Map target categories explicitly to binary numeric integers
    y = df[target_column].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
    
    return X, y
