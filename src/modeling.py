import joblib
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.feature_engineering import build_preprocessor

def get_models(random_state: int = 32) -> Dict[str, Any]:
    """
    Return a dictionary of models to train.
    """
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            random_state=random_state,
            eval_metric="logloss"
        ),
        "lightgbm": LGBMClassifier(
            random_state=random_state,
            n_jobs=1,
            verbosity=-1
        )
    }
    return models

def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numerical_cols: list,
    categorical_cols: list,
    param_grid: dict[str, list[Any]],
    cv_folds: int = 5,
    random_state: int = 32,
    scoring: str = "roc_auc"    
) -> GridSearchCV:
    
    models = get_models(random_state)
        
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' is not available. Choose from {list(models.keys())}")
    
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", models[model_name])
    ])
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    print(f"\nTraining {model_name}...")
    grid_search.fit(X_train, y_train)
    
    print(f"Best CV score ({scoring}): {grid_search.best_score_:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")
    
    return grid_search


def save_model(model, path: str) -> None:
    """Save a trained model to disk."""
    joblib.dump(model, path)
    print(f"Model saved to: {path}")


def load_model(path: str):
    """Load a trained model from disk."""
    model = joblib.load(path)
    print(f"Model loaded from: {path}")
    return model
