import pandas as pd
import numpy as np
from typing import Tuple
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_config_file(config_path: str = "Config.yaml") -> dict:
    """Load configuration dictionary from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load the raw Telecom Churn dataset from a CSV file.
    
    Parameters
    ----------
    path : str
        Path to the CSV file.
        
    Returns
    -------
    pd.DataFrame
        Raw dataframe.
    """
    df = pd.read_csv(path)
    print(f"Raw data loaded successfully. Shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning of the Telecom Churn dataset.
    
    Steps:
    - Drop customerID column
    - Convert TotalCharges to numeric
    - Drop rows with missing TotalCharges
    - Encode target variable Churn (Yes → 1, No → 0)
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.
        
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df = df.copy()
    
    # Drop customerID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    
    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # Drop rows with missing TotalCharges (only 11 rows in original dataset)
    initial_rows = df.shape[0]
    df = df.dropna(subset=["TotalCharges"])
    print(f"Dropped {initial_rows - df.shape[0]} rows with missing TotalCharges.")
    
    # Encode target variable
    if df["Churn"].dtype != int:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
        
    print(f"Data cleaned successfully. Final shape: {df.shape}")
    return df


def data_quality_report(df: pd.DataFrame) -> None:
    """
    Print a simple data quality summary.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.
    """
    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nTarget distribution (Churn):")
    print(df["Churn"].value_counts())
    print(df["Churn"].value_counts(normalize=True).round(3))
    print("=" * 60)


def get_train_test_split(
    df: pd.DataFrame,
    target: str = "Churn",
    test_size: float = 0.3,
    random_state: int = 32
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified train-test split.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.
    target : str
        Name of the target column.
    test_size : float
        Proportion of the dataset to include in the test split.
    random_state : int
        Random seed for reproducibility.
        
    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target])
    y = df[target]
        
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test