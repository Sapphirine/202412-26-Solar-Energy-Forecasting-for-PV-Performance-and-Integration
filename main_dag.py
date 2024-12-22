from datetime import datetime, timedelta
from textwrap import dedent
import time

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

from eecse_6893_g26_final_project.model_scripts import model, analysis

# Operators; we need this to operate!
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization

FORECAST_DATE = "2022-12-25"

############################################
# DEFINE AIRFLOW DAG (SETTINGS + SCHEDULE)
############################################

default_args = {
    'owner': 'to2359',
    'depends_on_past': False,
    'email': ['to2359@columbia.edu'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(seconds=30),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}

with DAG(
    'main_dag',
    default_args=default_args,
    description='Day Ahead Solar GHI Forecasting DAG',
    schedule_interval=timedelta(days=1), # Daily Job 
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=['example'],
) as dag:

##########################################
# DEFINE AIRFLOW OPERATORS
##########################################

    # t* examples of tasks created by instantiating operators

    
    t1 = PythonOperator(
        task_id='Cluster_3_Model',
        python_callable=model.build_cluster_model_and_forecast,
        op_kwargs={'cluster_id': 3, 'date_string': FORECAST_DATE, 'local': False}
    )

    t2 = PythonOperator(
        task_id='Cluster_3_Point_Analysis',
        python_callable=analysis.cluster_points_analysis,
        op_kwargs={'cluster_id': 3, 'date_string': FORECAST_DATE, 'local': False}
    )

    t3 = PythonOperator(
        task_id='Cluster_6_Model',
        python_callable=model.build_cluster_model_and_forecast,
        op_kwargs={'cluster_id': 6, 'date_string': FORECAST_DATE, 'local': False}
    )

    t4 = PythonOperator(
        task_id='Cluster_6_Point_Analysis',
        python_callable=analysis.cluster_points_analysis,
        op_kwargs={'cluster_id': 6, 'date_string': FORECAST_DATE, 'local': False}
    )

    t1 >> [t2]
    t3 >> [t4]

    