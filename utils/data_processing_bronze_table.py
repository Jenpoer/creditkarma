import os
from datetime import datetime

import pyspark
from pyspark.sql.functions import col

def process_bronze_table(table_name, source, db, snapshot_date_str, spark):
    """
    Function to read data from source and create bronze tables
    """
    # create bronze table 
    bronze_table = os.path.join(db, table_name)
    if not os.path.exists(bronze_table):
        os.makedirs(bronze_table)

    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # load data - IRL ingest from back end source system
    df = spark.read.csv(source, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)

    # save bronze table to datamart - IRL connect to database to write
    partition_name = snapshot_date_str.replace('-','_') + '.csv'
    filepath = os.path.join(bronze_table, partition_name)
    df.toPandas().to_csv(filepath, index=False)

    return df

def process_bronze_table_main(table_name, source, bronze_db, snapshot_date_str):
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Create bronze database
    if not os.path.exists(bronze_db):
        os.makedirs(bronze_db)
    
    process_bronze_table(table_name, source, bronze_db, snapshot_date_str, spark)
    
