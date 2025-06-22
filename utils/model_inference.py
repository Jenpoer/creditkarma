import os
import glob
import numpy as np
import pyspark
from pyspark.sql.functions import col
import mlflow
from datetime import datetime
from mlflow.tracking import MlflowClient

from utils.data_processing_gold_table import build_feature_store

############################
# Utils
############################
def read_silver_table(table, silver_db, spark):
    """
    Helper function to read all partitions of a silver table
    """
    folder_path = os.path.join(silver_db, table)
    files_list = [os.path.join(folder_path, os.path.basename(f)) for f in glob.glob(os.path.join(folder_path, '*'))]
    df = spark.read.option("header", "true").parquet(*files_list)
    return df

def read_gold_table(snapshot_date_str, table, gold_db, spark):
    """
    Helper function to read gold table features from this day
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()
    folder_path = os.path.join(gold_db, table)
    files_list = [os.path.join(folder_path, os.path.basename(f)) for f in glob.glob(os.path.join(folder_path, '*'))]
    df = spark.read.option("header", "true").option("mergeSchema", "true").parquet(*files_list)
    df = df.filter(col("snapshot_date")==snapshot_date)
    return df

def read_online_feature_store(snapshot_date_str, gold_db, spark):
    """
    Helper function to read online feature store
    """
    partition_name = snapshot_date_str.replace('-','_') + '.parquet'
    folder_path = os.path.join(gold_db, 'feature_store', 'online', partition_name)
    df = spark.read.option("header", "true").option("mergeSchema", "true").parquet(folder_path)
    return df

def create_online_feature_store(snapshot_date_str, gold_db, silver_db, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()

    print("Trying to retrieve records from gold table...")
    df_gold_online = read_gold_table(snapshot_date_str, 'feature_store', gold_db, spark)

    if df_gold_online.count() == 0:
        print("Building online feature store...")
        df_attributes = read_silver_table('attributes', silver_db, spark).filter(col("snapshot_date")==snapshot_date)
        df_clickstream = read_silver_table('clickstream', silver_db, spark)
        df_financials = read_silver_table('financials', silver_db, spark).filter(col("snapshot_date")==snapshot_date)
        df_loan_type = read_silver_table('loan_type', silver_db, spark).filter(col("snapshot_date")==snapshot_date)
        
        # create online feature store
        df_gold_online = build_feature_store(df_attributes, df_financials, df_loan_type, df_clickstream)

    # Save into online feature store
    partition_name = snapshot_date_str.replace('-','_') + '.parquet'
    feature_filepath = os.path.join(gold_db, 'feature_store', 'online', partition_name)
    df_gold_online.write.mode('overwrite').parquet(feature_filepath)

############################
# Main
############################

def model_inference_main(snapshot_date_str: str):
    model_name = "creditkarma-scorer"

    # Retrieve Pyspark session
    spark = pyspark.sql.SparkSession.builder \
    .appName("model-inference") \
    .master("local[*]") \
    .getOrCreate()

    # Retrieve champion model from MLFlow
    mlflow.set_tracking_uri(uri="http://mlflow:5001")
    model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}@champion")

    client = MlflowClient()
    model_version = client.get_model_version_by_alias(model_name, "champion")
    model_train_date = model_version.tags['train_date']
    model_type = model_version.tags['model_type']

    print(f"Current deployed version: {model_train_date}")

    # Get inference data
    create_online_feature_store(snapshot_date_str, 'datamart/gold', 'datamart/silver', spark)
    df_spark = read_online_feature_store(snapshot_date_str, 'datamart/gold', spark)

    # Turn into pandas
    df_pd = df_spark.toPandas().sort_values(by='customer_id')

    # Turn into numpy array
    df_arr = df_pd.drop(columns=['customer_id', 'snapshot_date']).values

    # Do inference
    df_pred_pd = df_pd.copy()[['customer_id', 'snapshot_date']]
    pred = model.predict_proba(df_arr)
    df_pred_pd['model_version'] = f'{model_type}_{model_train_date}'
    df_pred_pd['default'] = np.argmax(pred, axis=1)
    df_pred_pd['probability_no_default'] = pred[:, 0]
    df_pred_pd['probability_default'] = pred[:, 1]

    # Save to gold table
    gold_directory = f"datamart/gold/model_predictions/{model_name}_{model_type}_{model_train_date}"

    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)

    df_pred = spark.createDataFrame(df_pred_pd)
    partition_name = snapshot_date_str.replace('-','_') + '.parquet'
    filepath = os.path.join(gold_directory, partition_name)
    df_pred.write.mode("overwrite").parquet(filepath) 

