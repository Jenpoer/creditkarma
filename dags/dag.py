from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from utils.data_processing_bronze_table import process_bronze_table_main
from utils.data_processing_silver_table import process_silver_table_main
from utils.data_processing_gold_table import process_gold_table_main

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag',
    default_args=default_args,
    description='data pipeline run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
) as dag:

    ####################################
    # Data Pipeline
    ####################################

    dep_check_source_label_data = DummyOperator(task_id="dep_check_source_label_data")

    # BRONZE

    bronze_clickstream = PythonOperator(
        task_id='run_bronze_clickstream',
        python_callable=process_bronze_table_main,
        op_kwargs={
            'table_name': 'clickstream', 
            'source': 'data/feature_clickstream.csv', 
            'bronze_db': '/app/datamart/bronze',
            'snapshot_date_str': '{{ ds }}'
        }
    )
    bronze_attributes = PythonOperator(
        task_id='run_bronze_attributes',
        python_callable=process_bronze_table_main,
        op_kwargs={
            'table_name': 'attributes', 
            'source': 'data/features_attributes.csv', 
            'bronze_db': '/app/datamart/bronze',
            'snapshot_date_str': '{{ ds }}'
        }
    )
    bronze_financials = PythonOperator(
        task_id='run_bronze_financials',
        python_callable=process_bronze_table_main,
        op_kwargs={
            'table_name': 'financials', 
            'source': 'data/features_financials.csv', 
            'bronze_db': '/app/datamart/bronze',
            'snapshot_date_str': '{{ ds }}'
        }
    )
    bronze_lms = PythonOperator(
        task_id='run_bronze_lms',
        python_callable=process_bronze_table_main,
        op_kwargs={
            'table_name': 'lms', 
            'source': 'data/lms_loan_daily.csv', 
            'bronze_db': '/app/datamart/bronze',
            'snapshot_date_str': '{{ ds }}'
        }
    )
    

    # SILVER

    silver_clickstream = PythonOperator(
        task_id='run_silver_clickstream',
        python_callable=process_silver_table_main,
        op_kwargs={
            'table_name': 'clickstream',  
            'silver_db': '/app/datamart/silver',
            'bronze_db': '/app/datamart/bronze',
            'snapshot_date_str': '{{ ds }}'
        }
    )
    silver_attributes = PythonOperator(
        task_id='run_silver_attributes',
        python_callable=process_silver_table_main,
        op_kwargs={
            'table_name': 'attributes',  
            'silver_db': '/app/datamart/silver',
            'bronze_db': '/app/datamart/bronze',
            'snapshot_date_str': '{{ ds }}'
        }
    )
    silver_financials = PythonOperator(
        task_id='run_silver_financials',
        python_callable=process_silver_table_main,
        op_kwargs={
            'table_name': 'financials',  
            'silver_db': '/app/datamart/silver',
            'bronze_db': '/app/datamart/bronze',
            'snapshot_date_str': '{{ ds }}'
        }
    )
    silver_lms = PythonOperator(
        task_id='run_silver_lms',
        python_callable=process_silver_table_main,
        op_kwargs={
            'table_name': 'lms',  
            'silver_db': '/app/datamart/silver',
            'bronze_db': '/app/datamart/bronze',
            'snapshot_date_str': '{{ ds }}'
        }
    )

    # GOLD
    gold_table = PythonOperator(
        task_id='run_gold_table',
        python_callable=process_gold_table_main,
        op_kwargs={  
            'gold_db': '/app/datamart/gold',
            'silver_db': '/app/datamart/silver',
            'snapshot_date_str': '{{ ds }}'
        }
    )


    data_pipeline_completed = DummyOperator(task_id="data_pipeline_completed")

    # Define task dependencies
    dep_check_source_label_data >> bronze_clickstream >> silver_clickstream
    dep_check_source_label_data >> bronze_attributes >> silver_attributes
    dep_check_source_label_data >> bronze_financials >> silver_financials
    dep_check_source_label_data >> bronze_lms >> silver_lms
    silver_clickstream >> gold_table
    silver_attributes >> gold_table
    silver_financials >> gold_table
    silver_lms >> gold_table
    gold_table >> data_pipeline_completed
 
 
    