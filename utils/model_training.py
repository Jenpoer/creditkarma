import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyspark
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from tqdm import tqdm
import itertools

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, fbeta_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

import pickle
import joblib

import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

def read_gold_table(table, gold_db, spark):
    """
    Helper function to read all partitions of a gold table
    """
    folder_path = os.path.join(gold_db, table)
    files_list = [os.path.join(folder_path, os.path.basename(f)) for f in glob.glob(os.path.join(folder_path, '*'))]
    df = spark.read.option("header", "true").option("mergeSchema", "true").parquet(*files_list)
    return df

##########################
# PREPROCESSING
##########################

def split_dataset(X_df, y_df, config):
    # Consider data from model training date
    y_model_df = y_df[(y_df['snapshot_date'] >= config['train_test_start_date']) & (y_df['snapshot_date'] <= config['model_train_date'])]
    X_model_df = X_df[np.isin(X_df['customer_id'], y_model_df['customer_id'].unique())]

    # Create OOT split
    oot_splits = []
    current_start_date = config["oot_start_date"]
    while current_start_date <= config["oot_end_date"]:
        current_end_date = (current_start_date + relativedelta(months=1)) - timedelta(days=1)
        y_oot = y_model_df[(y_model_df['snapshot_date'] >= current_start_date) & (y_model_df['snapshot_date'] <= current_end_date)]
        X_oot = X_model_df[np.isin(X_model_df['customer_id'], y_oot['customer_id'].unique())]
        oot_splits.append((X_oot, y_oot))
        current_start_date = current_start_date + relativedelta(months=1)

    # Everything else goes into train-test
    y_trainvaltest = y_model_df[y_model_df['snapshot_date'] <= config['train_test_end_date']]
    X_trainvaltest = X_model_df[np.isin(X_model_df['customer_id'], y_trainvaltest['customer_id'].unique())]

    X_train, X_valtest, y_train, y_valtest = train_test_split(X_trainvaltest, y_trainvaltest, 
                                                    test_size=config['train_valtest_ratio'], 
                                                    random_state=611, 
                                                    shuffle=True, 
                                                    stratify=y_trainvaltest['label'])

    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, 
                                                    test_size=config['val_test_ratio'], 
                                                    random_state=611, 
                                                    shuffle=True, 
                                                    stratify=y_valtest['label'])

    return X_train, y_train, X_val, y_val, X_test, y_test, oot_splits

def transform_into_numpy(X, y):
    X_arr = X.drop(columns=['customer_id', 'snapshot_date']).values
    y_arr = y['label'].values

    return X_arr, y_arr

def create_data_preprocessing_pipeline():
    """
    Function to create data preprocessing sklearn pipeline object
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

def data_preprocessing(config: dict, preprocessing_pipeline: Pipeline, spark):
    # Read from gold table
    X_spark = read_gold_table('feature_store', 'datamart/gold', spark)
    y_spark = read_gold_table('label_store', 'datamart/gold', spark)

    # Convert into Pandas
    X_df = X_spark.toPandas().sort_values(by='customer_id')
    y_df = y_spark.toPandas().sort_values(by='customer_id')

    # Split dataset into train, val, test, oot
    X_train, y_train, X_val, y_val, X_test, y_test, oot_splits = split_dataset(X_df, y_df, config)

    # Convert into numpy arrays
    X_train_arr, y_train_arr = transform_into_numpy(X_train, y_train)
    X_val_arr, y_val_arr = transform_into_numpy(X_val, y_val)
    X_test_arr, y_test_arr = transform_into_numpy(X_test, y_test)
    oot_arrs = []
    for X_oot, y_oot in oot_splits:
        X_oot_arr, y_oot_arr = transform_into_numpy(X_oot, y_oot)
        oot_arrs.append((X_oot_arr, y_oot_arr))

    # Use pipeline to preprocess data
    X_train_arr = preprocessing_pipeline.fit_transform(X_train_arr)
    X_val_arr = preprocessing_pipeline.transform(X_val_arr)
    X_test_arr = preprocessing_pipeline.transform(X_test_arr)

    oot_arrs_preprocessed = []
    for X_oot_arr, y_oot_arr in oot_arrs:
        X_oot_arr_preprocessed = preprocessing_pipeline.transform(X_oot_arr)
        oot_arrs_preprocessed.append((X_oot_arr_preprocessed, y_oot_arr))

    return X_train_arr, y_train_arr, X_val_arr, y_val_arr, X_test_arr, y_test_arr, oot_arrs_preprocessed

##########################
# TRAINING
##########################
# Logistic Regression
def logistic_regression_grid_search():
    param_grid = {
        'penalty':['l1','l2','elasticnet', None],
        'C' : [0.01, 0.1, 1, 10],
        'solver': ['lbfgs','newton-cg','liblinear','sag','saga'],
        'max_iter': [200]
    }

    valid_combinations = {
        'liblinear': ['l1', 'l2'],
        'lbfgs': ['l2', None],
        'newton-cg': ['l2', None],
        'sag': ['l2', None],
        'saga': ['l1', 'l2', 'elasticnet', None],
    }

    return param_grid, valid_combinations

def train_logistic_regression(param_grid, valid_combinations, X_train_arr, y_train_arr, X_val_arr, y_val_arr):
    best_score = -np.inf
    best_model = None
    best_params = {}
    best_signature = None
    best_input_example = None

    ########################
    # Model Training
    ########################
    for solver, penalty, C, max_iter in itertools.product(param_grid['solver'], param_grid['penalty'], param_grid['C'], param_grid['max_iter']):
        if penalty not in valid_combinations.get(solver, []):
            continue  # skip invalid combo

        # Start an MLflow run
        with mlflow.start_run(run_name=f"logreg_C={C:.4f}_penalty={penalty}_solver={solver}_max-iter={max_iter}"):
            try:
                ####################
                # Train Model
                ####################
                model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter)
                model.fit(X_train_arr, y_train_arr)

                ####################
                # Evaluate Metrics
                ####################

                # AUC
                y_pred_proba_train = model.predict_proba(X_train_arr)[:, 1]
                train_auc = roc_auc_score(y_train_arr, y_pred_proba_train)
                y_pred_proba_val = model.predict_proba(X_val_arr)[:, 1]
                val_auc = roc_auc_score(y_val_arr, y_pred_proba_val)

                # F1.5
                thresholds = np.arange(0.0, 1.0, 0.01)
                beta = 1.5
                fb_scores_train = [fbeta_score(y_train_arr, y_pred_proba_train > t, beta=beta) for t in thresholds]
                fb_scores_val = [fbeta_score(y_val_arr, y_pred_proba_val > t, beta=beta) for t in thresholds]

                train_fb_score = fb_scores_train[np.argmax(fb_scores_val)]
                val_fb_score = fb_scores_val[np.argmax(fb_scores_val)]

                ####################
                # Log to MLFlow
                ####################
                mlflow.log_param("C", C)
                mlflow.log_param("penalty", penalty)
                mlflow.log_param("solver", solver)
                mlflow.log_param("max_iter", max_iter)

                mlflow.log_metric("train_auc", train_auc)
                mlflow.log_metric("val_auc", val_auc)

                mlflow.log_metric(f"train_f{beta:.1f}_score", train_fb_score)
                mlflow.log_metric(f"val_f{beta:.1f}_score", val_fb_score)

                if val_auc > best_score:
                    best_score = val_auc
                    best_model = model
                    best_params = {'C': C, 'penalty': penalty, 'solver': solver, 'max_iter': max_iter}
                    best_signature = infer_signature(X_val_arr, model.predict_proba(X_val_arr))
                    best_input_example = X_val_arr[:5]
            except Exception as e:
                print(f"Skipped C={C}, penalty={penalty}, solver={solver}: {e}")
                mlflow.end_run(status="FAILED")
                continue

    return_dict = {
        "best_score": best_score,
        "best_model": best_model,
        "best_params": best_params,
        "best_signature": best_signature,
        "best_input_example": best_input_example
    }

    return return_dict

# Random Forest Classifier
def random_forest_grid_search():
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    return param_grid

def train_random_forest(param_grid, X_train_arr, y_train_arr, X_val_arr, y_val_arr):
    best_score = -np.inf
    best_model = None
    best_params = {}
    best_signature = None
    best_input_example = None

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    ########################
    # Model Training
    ########################
    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))

        run_name = f"rf_" + "_".join(f"{k}={v}" for k, v in params.items())

        with mlflow.start_run(run_name=run_name):
            try:
                ####################
                # Train Model
                ####################
                model = RandomForestClassifier(**params, random_state=42)
                model.fit(X_train_arr, y_train_arr)

                ####################
                # Evaluate Metrics
                ####################
                y_pred_proba_train = model.predict_proba(X_train_arr)[:, 1]
                train_auc = roc_auc_score(y_train_arr, y_pred_proba_train)

                y_pred_proba_val = model.predict_proba(X_val_arr)[:, 1]
                val_auc = roc_auc_score(y_val_arr, y_pred_proba_val)

                # F1.5 Score
                thresholds = np.arange(0.0, 1.0, 0.01)
                beta = 1.5
                fb_scores_train = [fbeta_score(y_train_arr, y_pred_proba_train > t, beta=beta) for t in thresholds]
                fb_scores_val = [fbeta_score(y_val_arr, y_pred_proba_val > t, beta=beta) for t in thresholds]

                train_fb_score = fb_scores_train[np.argmax(fb_scores_val)]
                val_fb_score = fb_scores_val[np.argmax(fb_scores_val)]

                ####################
                # Log to MLFlow
                ####################
                for k, v in params.items():
                    mlflow.log_param(k, v)

                mlflow.log_metric("train_auc", train_auc)
                mlflow.log_metric("val_auc", val_auc)
                mlflow.log_metric(f"train_f{beta:.1f}_score", train_fb_score)
                mlflow.log_metric(f"val_f{beta:.1f}_score", val_fb_score)

                if val_auc > best_score:
                    best_score = val_auc
                    best_model = model
                    best_params = params
                    best_signature = infer_signature(X_val_arr, model.predict_proba(X_val_arr))
                    best_input_example = X_val_arr[:5]

            except Exception as e:
                print(f"Skipped {params}: {e}")
                mlflow.end_run(status="FAILED")
                continue

    return {
        "best_score": best_score,
        "best_model": best_model,
        "best_params": best_params,
        "best_signature": best_signature,
        "best_input_example": best_input_example
    }

# Save best model
def save_best_model(model_type,
                    config, 
                    best_model_info_dict, 
                    pipeline,
                    X_train_arr, y_train_arr,
                    X_val_arr, y_val_arr,
                    X_test_arr, y_test_arr,
                    oot_arrs):

    best_score = best_model_info_dict["best_score"]
    best_model = best_model_info_dict["best_model"]
    best_params = best_model_info_dict["best_params"]
    best_signature = best_model_info_dict["best_signature"]
    best_input_example = best_model_info_dict["best_input_example"]

    with mlflow.start_run(run_name=f"{model_type}_best_model"):
        ####################
        # Evaluate Metrics
        ####################
        # AUC
        y_pred_proba_train = best_model.predict_proba(X_train_arr)[:, 1]
        train_auc = roc_auc_score(y_train_arr, y_pred_proba_train)
        y_pred_proba_val = best_model.predict_proba(X_val_arr)[:, 1]
        val_auc = roc_auc_score(y_val_arr, y_pred_proba_val)
        y_pred_proba_test = best_model.predict_proba(X_test_arr)[:, 1]
        test_auc = roc_auc_score(y_test_arr, y_pred_proba_test)

        y_pred_proba_oots = []
        oot_aucs = []
        for i, (X_oot_arr, y_oot_arr) in enumerate(oot_arrs):
            y_pred_proba_oot = best_model.predict_proba(X_oot_arr)[:, 1]
            oot_auc = roc_auc_score(y_oot_arr, y_pred_proba_oot)
            y_pred_proba_oots.append(y_pred_proba_oot)
            oot_aucs.append(oot_auc)

        # F1.5
        thresholds = np.arange(0.0, 1.0, 0.01)
        beta = 1.5
        fb_scores_train = [fbeta_score(y_train_arr, y_pred_proba_train > t, beta=beta) for t in thresholds]
        fb_scores_val = [fbeta_score(y_val_arr, y_pred_proba_val > t, beta=beta) for t in thresholds]
        fb_scores_test = [fbeta_score(y_test_arr, y_pred_proba_test > t, beta=beta) for t in thresholds]
        fb_scores_oots = []
        for i, y_pred_proba_oot in enumerate(y_pred_proba_oots):
            y_oot_arr = oot_arrs[i][1]
            fb_scores_oot = [fbeta_score(y_oot_arr, y_pred_proba_oot > t, beta=beta) for t in thresholds]
            fb_scores_oots.append(fb_scores_oot)

        best_threshold = thresholds[np.argmax(fb_scores_val)]

        train_fb_score = fb_scores_train[np.argmax(fb_scores_val)]
        val_fb_score = fb_scores_val[np.argmax(fb_scores_val)]
        test_fb_score = fb_scores_train[np.argmax(fb_scores_test)]
        oot_fb_scores = [fb_scores_oot[np.argmax(fb_scores_oot)] for fb_scores_oot in fb_scores_oots]
        
        ####################
        # Log to MLFlow
        ####################
        
        # Params
        mlflow.log_params(config)
        mlflow.log_params(best_params)
        mlflow.log_param("best_fb_threshold", best_threshold)

        # AUC
        mlflow.log_metric("train_auc", train_auc)
        mlflow.log_metric("val_auc", val_auc)
        mlflow.log_metric("test_auc", test_auc)
        for i, auc in enumerate(oot_fb_scores):
            mlflow.log_metric(f"oot{i + 1}_auc", auc)

        # F-Beta
        mlflow.log_metric(f"train_f{beta:.1f}_score", train_fb_score)
        mlflow.log_metric(f"val_f{beta:.1f}_score", val_fb_score)
        mlflow.log_metric(f"test_f{beta:.1f}_score", test_fb_score)
        for i, fb_score in enumerate(oot_fb_scores):
            mlflow.log_metric(f"oot{i + 1}_f{beta:.1f}_score", fb_score)

        full_pipeline = make_pipeline(
            pipeline,
            best_model
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            artifact_path=f"{config['model_train_date_str']}",
            input_example=best_input_example,
            signature=best_signature,
            registered_model_name="creditkarma-scorer"
        )

        client = MlflowClient()

        client.set_model_version_tag(
            name="creditkarma-scorer",
            version=model_info.registered_model_version,
            key="train_date",
            value=config["model_train_date_str"]
        )

        client.set_model_version_tag(
            name="creditkarma-scorer",
            version=model_info.registered_model_version,
            key="model_type",
            value=model_type
        )

        print(f"✅ Logged and registered best model: {model_info.model_uri}")

##########################
# MAIN FUNCTIONS
##########################

def model_training_logreg_main(config: dict):
    print("======== Training Logistic Regression =========")
    spark = pyspark.sql.SparkSession.builder \
    .appName("model-training") \
    .master("local[*]") \
    .getOrCreate()

    pipeline = create_data_preprocessing_pipeline()

    # Preprocess data
    X_train_arr, y_train_arr, X_val_arr, y_val_arr, X_test_arr, y_test_arr, oot_arrs = data_preprocessing(config, pipeline, spark)

    print("Data preprocessing complete!")

    # Connect to MLFlow
    mlflow.set_tracking_uri(uri="http://mlflow:5001") # Set our tracking server uri for logging
    mlflow.set_experiment(config["model_train_date_str"]) # Create a new experiment for that training date

    print("Connected to MLFlow!")

    # Train logistic regression model
    param_grid, valid_combinations = logistic_regression_grid_search()
    best_model_info_dict = train_logistic_regression(param_grid, valid_combinations, X_train_arr, y_train_arr, X_val_arr, y_val_arr)
    save_best_model("logreg",
                    config, 
                    best_model_info_dict, 
                    pipeline,
                    X_train_arr, y_train_arr,
                    X_val_arr, y_val_arr,
                    X_test_arr, y_test_arr,
                    oot_arrs)
    
   

def model_training_rf_main(config: dict):
    print("======== Training RandomForestClassifier =========")
    spark = pyspark.sql.SparkSession.builder \
    .appName("model-training") \
    .master("local[*]") \
    .getOrCreate()

    pipeline = create_data_preprocessing_pipeline()

    # Preprocess data
    X_train_arr, y_train_arr, X_val_arr, y_val_arr, X_test_arr, y_test_arr, oot_arrs = data_preprocessing(config, pipeline, spark)

    print("Data preprocessing complete!")

    # Connect to MLFlow
    mlflow.set_tracking_uri(uri="http://mlflow:5001") # Set our tracking server uri for logging
    mlflow.set_experiment(config["model_train_date_str"]) # Create a new experiment for that training date

    print("Connected to MLFlow!")

    # Train random forest classifier model
    param_grid = random_forest_grid_search()
    best_model_info_dict = train_random_forest(param_grid, X_train_arr, y_train_arr, X_val_arr, y_val_arr)
    save_best_model("rf",
                    config, 
                    best_model_info_dict, 
                    pipeline,
                    X_train_arr, y_train_arr,
                    X_val_arr, y_val_arr,
                    X_test_arr, y_test_arr,
                    oot_arrs)