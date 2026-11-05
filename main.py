import os
import json
import mlflow
from sklearn import metrics
from src.data_preprocessing import (
    load_config_file,
    load_raw_data,
    clean_data,
    data_quality_report,
    get_train_test_split
)
from src.feature_engineering import (
    create_features,
    get_feature_columns
)
from src.modeling import train_model, save_model
from src.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    threshold_analysis,
    plot_feature_importance
)

def main():
    # ======================
    # 0. Load Configuration File
    # ======================
    config = load_config_file("Config.yaml")

    # ======================
    # 1. Load & Clean Data
    # ======================
    DATA_PATH = config["paths"]["raw_data"] # path to the file with input data
    
    df_raw = load_raw_data(DATA_PATH) # reading input file
    df = clean_data(df_raw) # cleaning input data
    data_quality_report(df)
    
    # ======================
    # 2. Feature Engineering
    # ======================
    df = create_features(df)
    num_cols, cat_cols = get_feature_columns(df)
    
    # ======================
    # 3. Train / Test Split
    # ======================
    SEED = config["project"]["seed"]
    TEST_SIZE = config["data_split"]["test_size"]
    X_train, X_test, y_train, y_test = get_train_test_split(df, test_size=TEST_SIZE, random_state=SEED)

    # ======================
    # 4. Train Models
    # ======================
    models_to_train = list(config["modeling"].keys())
    trained_models = {}

    trusted_types = [
    "sklearn.compose._column_transformer._RemainderColsList",

    # Scikit-learn tree-based models
    "sklearn.tree._tree.Tree",

    # XGBoost
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",

    # LightGBM
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
    ]
    
    os.makedirs("models", exist_ok=True)

    experiment_full_name = "Customer Churn"
    mlflow.set_experiment(experiment_full_name)

    mlflow.sklearn.autolog(
        log_models=False, 
        log_model_signatures=False, 
        max_tuning_runs=5
    )
    
    model_run_ids = {}
    #trained_models = {}

    with mlflow.start_run(run_name="churn_model_comparison"):
        for model_name in models_to_train:
            with mlflow.start_run(run_name=model_name, nested=True) as model_run:
                grid = train_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                numerical_cols=num_cols,
                categorical_cols=cat_cols,
                param_grid = config["modeling"][model_name],
                random_state = SEED            
            )
                trained_models[model_name] = grid.best_estimator_

                # Save model
                save_model(grid.best_estimator_, f"models/{model_name}_churn.pkl")

                mlflow.log_param("model_name", model_name)
                mlflow.log_params(grid.best_params_)

                # Mean cross-validation score of the selected configuration.
                #mlflow.log_metric("best_cv_score", grid.best_score_)

                # Save the complete pipeline to MLflow.
                '''mlflow.sklearn.log_model(
                   sk_model=grid.best_estimator_,
                    name=model_name,
                    skops_trusted_types=trusted_types,
                )
                '''
                # Store this exact child run ID
                model_run_ids[model_name] = model_run.info.run_id

    # ======================
    # 5. Evaluate Best Models
    # ======================
    all_results = {}
    for name, model in trained_models.items():
        print(f"\n\n########## Evaluating {name} ##########")

        top_n = config['features']["feature_importance"]["top_n"]

        os.makedirs("reports/figures", exist_ok=True)
        os.makedirs("reports/results", exist_ok=True)
        results = evaluate_model(model, X_test, y_test, threshold=0.5)

        all_results[name] = results
        
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        
        plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix - {name}")
        plot_roc_curve(y_test, y_proba, title=f"ROC Curve - {name}")
        plot_precision_recall_curve(y_test, y_proba, title=f"PR Curve - {name}")
        
        # Threshold analysis
        threshold_analysis(model, X_test, y_test)
        
        # Feature importance (for tree-based models)
        try:
            feature_names = model.named_steps["preprocessor"].get_feature_names_out()
            plot_feature_importance(model, feature_names, top_n, title=f"Top {top_n} Feature Importance - {name}")
        except Exception as e:
            print(f"Could not plot feature importance: {e}")

        # Reopen the same training run and add test metrics
        with mlflow.start_run(
        run_id=model_run_ids[name]
        ):

            mlflow.log_metrics({
                "test_accuracy": metrics.accuracy_score(
                    y_test,
                    y_pred
                ),
                "test_precision": metrics.precision_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),
                "test_recall": metrics.recall_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),
                "test_f1": metrics.f1_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),
                "test_roc_auc": metrics.roc_auc_score(
                    y_test,
                    y_proba
                ),
                "test_pr_auc": metrics.average_precision_score(
                    y_test,
                    y_proba
                )
            })

    with open("reports/results/metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n\nPipeline completed successfully!")


if __name__ == "__main__":
    main()