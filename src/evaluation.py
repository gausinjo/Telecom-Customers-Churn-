import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional

from sklearn import metrics
from sklearn.inspection import PartialDependenceDisplay
import shap


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> dict:
    """
    Evaluate a trained model and return key metrics.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    results = {
        "accuracy": metrics.accuracy_score(y_test, y_pred),
        "precision": metrics.precision_score(y_test, y_pred),
        "recall": metrics.recall_score(y_test, y_pred),
        "f1": metrics.f1_score(y_test, y_pred),
        "roc_auc": metrics.roc_auc_score(y_test, y_proba),
        "pr_auc": metrics.average_precision_score(y_test, y_proba)
    }
    
    print("\n" + "="*50)
    print(f"MODEL EVALUATION (threshold = {threshold})")
    print("="*50)
    print(metrics.classification_report(y_test, y_pred, target_names=["Not Churn", "Churn"]))
    print(f"ROC-AUC : {results['roc_auc']:.4f}")
    print(f"PR-AUC  : {results['pr_auc']:.4f}")
    
    return results


def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix") -> None:
    cm = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"reports/figures/{title}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_proba, title: str = "ROC Curve") -> None:
    fpr, tpr, _ = metrics.roc_curve(y_true, y_proba)
    auc = metrics.roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"reports/figures/{title}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_precision_recall_curve(y_true, y_proba, title: str = "Precision-Recall Curve") -> None:
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_proba)
    pr_auc = metrics.average_precision_score(y_true, y_proba)
    
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"reports/figures/{title}.png", dpi=300, bbox_inches="tight")
    plt.close()


def threshold_analysis(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    thresholds: List[float] = [0.2, 0.3, 0.4, 0.5]
) -> None:
    """
    Print classification reports for different probability thresholds.
    """
    y_proba = model.predict_proba(X)[:, 1]
    
    print("\n" + "="*60)
    print("THRESHOLD ANALYSIS")
    print("="*60)
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        print(f"\n----- Threshold: {thresh} -----")
        print(metrics.classification_report(y, y_pred, target_names=["Not Churn", "Churn"]))


def plot_feature_importance(model, feature_names: List[str],  top_n: int = 15, title: str = "Feature Importance") -> None:
    """
    Plot feature importance for tree-based models.
    """
    if hasattr(model.named_steps["model"], "feature_importances_"):
        importances = model.named_steps["model"].feature_importances_
        
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x="importance", y="feature", palette="viridis")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"reports/figures/{title}.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        print("Feature importance not available for this model.")


def plot_shap_summary(model, X_sample: pd.DataFrame, feature_names: List[str]) -> None:
    """
    Create a SHAP summary plot.
    """
    # Transform the data
    X_transformed = model.named_steps["preprocessor"].transform(X_sample)
    
    # Create explainer
    try:
        explainer = shap.Explainer(model.named_steps["model"], X_transformed)
        shap_values = explainer(X_transformed)
        
        shap.summary_plot(
            shap_values,
            X_transformed,
            feature_names=feature_names,
            show=True
        )
    except Exception as e:
        print(f"Could not create SHAP plot: {e}")