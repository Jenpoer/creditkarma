from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.python import BranchPythonOperator
from airflow.exceptions import AirflowFailException

import mlflow
from mlflow.tracking import MlflowClient

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from utils.data_processing_bronze_table import process_bronze_table_main
from utils.data_processing_silver_table import process_silver_table_main
from utils.data_processing_gold_table import process_gold_table_main
from utils.model_training import model_training_logreg_main, model_training_rf_main
from utils.model_inference import model_inference_main

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

####################################
# Data Pipeline
####################################

def should_trigger_training(**kwargs):
    exec_date = kwargs['execution_date']
    # Allow triggering only if month >= 6 (June) — or after May 2023
    if exec_date.year > 2023:
        return 'trigger_training_dag'
    else:
        return 'skip_training'

with DAG(
    'data_pipeline_dag',
    default_args=default_args,
    description='data pipeline run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
) as dag:

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

    trigger_training = TriggerDagRunOperator(
        task_id='trigger_training_dag',
        trigger_dag_id='model_training_dag',
        conf={
            'model_train_date_str': '{{ ds }}',
            'train_test_period_months': 12,
            'oot_period_months': 3,
            'train_valtest_ratio': 0.2,
            'val_test_ratio': 0.5
        },
        wait_for_completion=False
    )

    skip_training = DummyOperator(task_id='skip_training')

    check_should_trigger = BranchPythonOperator(
        task_id='check_should_trigger',
        python_callable=should_trigger_training,
        provide_context=True
    )


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

    data_pipeline_completed >> check_should_trigger
    check_should_trigger >> trigger_training
    check_should_trigger >> skip_training

####################################
# Model Training
####################################
def start_model_training(model_type, **kwargs):
    conf = kwargs['dag_run'].conf
    model_train_date_str = conf['model_train_date_str']
    train_test_period_months = conf['train_test_period_months']
    oot_period_months = conf['oot_period_months']
    train_valtest_ratio = conf['train_valtest_ratio']
    val_test_ratio = conf['val_test_ratio']

    config = {}
    config["model_train_date_str"] = model_train_date_str
    config["train_test_period_months"] = train_test_period_months
    config["oot_period_months"] =  oot_period_months
    config["model_train_date"] =  datetime.strptime(model_train_date_str, "%Y-%m-%d").date()
    config["oot_end_date"] =  config['model_train_date'] - timedelta(days = 1)
    config["oot_start_date"] =  config['model_train_date'] - relativedelta(months = oot_period_months)
    config["train_test_end_date"] =  config["oot_start_date"] - timedelta(days = 1)
    config["train_test_start_date"] =  config["oot_start_date"] - relativedelta(months = train_test_period_months)
    config["train_valtest_ratio"] = train_valtest_ratio 
    config["val_test_ratio"] = val_test_ratio
    
    if model_type == 'logreg':
        model_training_logreg_main(config)
    elif model_type == 'rf':
        model_training_rf_main(config)

with DAG(
    dag_id='model_training_dag',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,  # Only triggered by other dag
    catchup=False,
    description='Train model',
) as training_dag:
    
    train_model_started = DummyOperator(task_id="train_model_started")

    train_logistic_regression = PythonOperator(
        task_id='train_model_logreg',
        python_callable=start_model_training,
        op_kwargs={
            'model_type': 'logreg'
        }
    )

    train_random_forest = PythonOperator(
        task_id='train_model_rf',
        python_callable=start_model_training,
        op_kwargs={
            'model_type': 'rf'
        }
    )

    train_model_completed = DummyOperator(task_id="train_model_completed")

    train_model_started >> train_logistic_regression >> train_model_completed
    train_model_started >> train_random_forest >> train_model_completed

####################################
# Model Inference
####################################

def check_champion_exists():
    MODEL_NAME = "creditkarma-scorer"

    mlflow.set_tracking_uri(uri="http://mlflow:5001")

    client = MlflowClient()
    model_version = client.get_model_version_by_alias(MODEL_NAME, "champion")
    model_train_date = model_version.tags['train_date']
    model_type = model_version.tags['model_type']

    if not model_version:
        raise AirflowFailException(f"No champion model found for model '{MODEL_NAME}'")
    
    print(f"Champion model found:{model_type}_{model_train_date}")
    return f"{model_type}_{model_train_date}"

def run_model_inference(snapshot_date_str: str):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    first_day_of_month = snapshot_date.replace(day=1)
    first_day_str = first_day_of_month.strftime("%Y-%m-%d")

    model_inference_main(snapshot_date_str=first_day_str)

with DAG(
    dag_id="inference_on_new_champion",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 31),
    schedule_interval='0 0 7 * *',
    catchup=True
) as dag:

    inference_started = DummyOperator(task_id="inference_started")

    check_champion_model = PythonOperator(
        task_id='check_champion_model',
        python_callable=check_champion_exists
    )

    inference_task = PythonOperator(
        task_id='run_model_inference',
        python_callable=run_model_inference,
        op_kwargs={
            'snapshot_date_str': '{{ ds }}'
        }
    )

    inference_completed = DummyOperator(task_id="inference_ended")

    inference_started >> check_champion_model >> inference_task >> inference_completed