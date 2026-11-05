import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features for the Telecom Churn dataset.
    
    New features:
    - tenure_group
    - avg_monthly_charges (TotalCharges / tenure)
    - total_services
    - has_fiber
    - has_phone_and_internet
    - is_month_to_month
    """
    df = df.copy()
    
    # 1. Tenure groups
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, np.inf],
        labels=["0-12 months", "13-24 months", "25-48 months", "49+ months"],
        include_lowest=True
    )
    
    # 2. Average monthly charges (avoid division by zero)
    df["avg_monthly_charges"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )
    
    # 3. Total number of services
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    
    # Count how many services the customer has (simplified logic)
    df["total_services"] = 0
    
    # Phone related
    df["total_services"] += (df["PhoneService"] == "Yes").astype(int)
    df["total_services"] += (df["MultipleLines"] == "Yes").astype(int)
    
    # Internet related
    df["total_services"] += (df["InternetService"] != "No").astype(int)
    df["total_services"] += (df["OnlineSecurity"] == "Yes").astype(int)
    df["total_services"] += (df["OnlineBackup"] == "Yes").astype(int)
    df["total_services"] += (df["DeviceProtection"] == "Yes").astype(int)
    df["total_services"] += (df["TechSupport"] == "Yes").astype(int)
    df["total_services"] += (df["StreamingTV"] == "Yes").astype(int)
    df["total_services"] += (df["StreamingMovies"] == "Yes").astype(int)
    
    # 4. Binary flags
    df["has_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)
    df["has_phone_and_internet"] = (
        (df["PhoneService"] == "Yes") & (df["InternetService"] != "No")
    ).astype(int)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    
    print("Feature engineering completed.")
    print(f"New shape: {df.shape}")
    
    return df


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Return lists of numerical and categorical columns
    (excluding the target 'Churn').
    """
    # Columns that should be treated as numerical
    numerical_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "avg_monthly_charges",
        "total_services"
    ]
    
    # Everything else (except target) is categorical
    categorical_cols = [
        col for col in df.columns
        if col not in numerical_cols + ["Churn"]
    ]
    
    return numerical_cols, categorical_cols


def build_preprocessor(
    numerical_cols: List[str],
    categorical_cols: List[str]
) -> ColumnTransformer:
    """
    Create a ColumnTransformer that:
    - Scales numerical features
    - One-hot encodes categorical features
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ],
        remainder="drop"
    )
    
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Get feature names after transformation (useful for feature importance / SHAP).
    """
    return list(preprocessor.get_feature_names_out())