import os
from datetime import datetime
import pyspark

from tqdm import tqdm

from utils.data_processing_bronze_table import process_bronze_table
from utils.data_processing_silver_table import process_silver_table
from utils.data_processing_gold_table import process_gold_table

def generate_first_of_month_dates(start_date_str, end_date_str):
    """
    Generate list of dates to process
    """
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

if __name__ == "__main__":
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # Generate partitions
    start_date_str = "2023-01-01"
    end_date_str = "2024-12-01"

    dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)

    ############################
    # BRONZE
    ############################
    print("Building bronze tables...")
    # Create bronze datalake
    bronze_directory = "datamart/bronze"

    if not os.path.exists(bronze_directory):
        os.makedirs(bronze_directory)

    # Build bronze tables
    for date_str in tqdm(dates_str_lst, total=len(dates_str_lst), desc="Processing clickstream"):
        process_bronze_table('clickstream', 'data/feature_clickstream.csv', bronze_directory, date_str, spark)

    for date_str in tqdm(dates_str_lst, total=len(dates_str_lst), desc="Processing attributes"):
        process_bronze_table('attributes', 'data/features_attributes.csv', bronze_directory, date_str, spark)

    for date_str in tqdm(dates_str_lst, total=len(dates_str_lst), desc="Processing financials"):
        process_bronze_table('financials', 'data/features_financials.csv', bronze_directory, date_str, spark)

    for date_str in tqdm(dates_str_lst, total=len(dates_str_lst), desc="Processing lms"):
        process_bronze_table('lms', 'data/lms_loan_daily.csv', bronze_directory, date_str, spark)

    ############################
    # SILVER
    ############################
    print("Building silver tables...")
    # Create silver datalake
    silver_directory = "datamart/silver"

    if not os.path.exists(silver_directory):
        os.makedirs(silver_directory)

    # Build silver tables
    for date_str in tqdm(dates_str_lst, total=len(dates_str_lst), desc="Processing clickstream"):
        process_silver_table('clickstream', bronze_directory, silver_directory, date_str, spark)

    for date_str in tqdm(dates_str_lst, total=len(dates_str_lst), desc="Processing attributes"):
        process_silver_table('attributes', bronze_directory, silver_directory, date_str, spark)

    for date_str in tqdm(dates_str_lst, total=len(dates_str_lst), desc="Processing financials"):
        process_silver_table('financials', bronze_directory, silver_directory, date_str, spark)

    for date_str in tqdm(dates_str_lst, total=len(dates_str_lst), desc="Processing lms"):
        process_silver_table('lms', bronze_directory, silver_directory, date_str, spark)

    ############################
    # GOLD
    ############################
    print("Building gold tables...")
    # Create gold datalake
    gold_directory = "datamart/gold"

    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)

    # Build gold tables
    X, y = process_gold_table(silver_directory, gold_directory, dates_str_lst, spark)

    print("X: ")
    X.show(5)

    print("y: ")
    y.show(5)
    
