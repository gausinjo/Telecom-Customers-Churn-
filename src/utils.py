import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List


def plot_churn_by_categories(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Plot stacked bar charts showing Churn rate (%) for multiple categorical columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the columns and the 'Churn' target.
    columns : List[str]
        List of categorical column names to plot.
    """

    for column in columns:
        churn_rate = (
            df.groupby(column)["Churn"]
            .value_counts(normalize=True)
            .unstack(fill_value=0) * 100
        )
        
        # Ensure both '0' and '1' exist
        for col in [0, 1]:
            if col not in churn_rate.columns:
                churn_rate[col] = 0
        
        ax = churn_rate[[0, 1]].plot(kind="bar", stacked=True, figsize=(9, 5))
        plt.ylabel("Percentage (%)")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Churn", labels=["No", "Yes"])
        plt.title(f"Churn Rate by {column}")
        
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", label_type="center", fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"../reports/figures/Churn_vs_{column}.png", dpi=300, bbox_inches="tight")
        plt.show()


def plot_numerical_vs_churn(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Create boxplots and histograms for numerical features vs Churn.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the numerical columns and the 'Churn' target.
    columns : List[str]
        List of numerical column names to plot.
    """
    for col in columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Boxplot
        sns.boxplot(x="Churn", y=col, data=df, ax=axes[0])
        axes[0].set_title(f"Boxplot of {col} by Churn")
        axes[0].set_xlabel("Churn")
        
        # Histogram + KDE
        sns.histplot(data=df, x=col, hue="Churn", kde=True, ax=axes[1], palette="Set2")
        axes[1].set_title(f"Distribution of {col} by Churn")
        
        plt.tight_layout()
        plt.savefig(f"../reports/figures/Churn_vs_{col}.png", dpi=300, bbox_inches="tight")
        plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, figsize: tuple = (12, 10)) -> None:
    """
    Plot correlation heatmap for numerical columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe.
    figsize : tuple
        Figure size.
    """
    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        square=True
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()