import logging
import pandas as pd
from scipy.stats import ks_2samp

# Initialize logger to record monitoring diagnostics
logger = logging.getLogger("churn_api")

def check_numerical_drift(
    reference_df: pd.DataFrame, 
    production_df: pd.DataFrame, 
    numeric_cols: list, 
    threshold: float = 0.05
) -> dict:
    """
    Performs a two-sample Kolmogorov-Smirnov (KS) test on numerical columns 
    to determine if the distribution of production data has drifted away 
    from the original training baseline.

    Args:
        reference_df (pd.DataFrame): Baseline data frame (e.g., historical training data).
        production_df (pd.DataFrame): Incoming production data frame (e.g., live traffic log subsets).
        numeric_cols (list): List of continuous numeric column names to analyze.
        threshold (float): Alpha significance level. If p-value < threshold, drift is confirmed.

    Returns:
        dict: A monitoring report detailing the KS statistic, p-value, and status for each feature.
    """
    drift_report = {}
    
    for col in numeric_cols:
        # Check if the feature exists in both reference and evaluation dataframes
        if col not in reference_df.columns or col not in production_df.columns:
            logger.error(f"⚠️ Feature column '{col}' is missing from comparison dataframes. Bypassing check.")
            continue
            
        # Isolate arrays and remove any missing values
        ref_data = reference_df[col].dropna()
        prod_data = production_df[col].dropna()
        
        if len(ref_data) == 0 or len(prod_data) == 0:
            logger.warning(f"⚠️ Insufficient records in feature column '{col}' to perform drift calculations.")
            continue
            
        # Execute the two-sample Kolmogorov-Smirnov statistical test
        ks_stat, p_value = ks_2samp(ref_data, prod_data)
        
        # If the p-value falls below our alpha significance value, we reject the null hypothesis
        drift_detected = p_value < threshold
        
        drift_report[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 4),
            "drift_detected": bool(drift_detected)
        }
        
        # Generate operational alerts depending on target statuses
        if drift_detected:
            logger.warning(
                f"🚨 DATA DRIFT CONFIRMED | Column: '{col}' | "
                f"KS Stat: {ks_stat:.4f} | p-value: {p_value:.4f} (Threshold: {threshold})"
            )
        else:
            logger.info(f"✅ Feature column stability verified: '{col}' (p-value: {p_value:.4f})")
            
    return drift_report
