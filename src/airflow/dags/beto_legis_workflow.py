from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from src.legis_workflow.utils import toSpacy, SpacytoConLL
from datetime import datetime, timedelta

default_args = {
    'owner': 'Alexis Alva',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retry': False,
}

file_path = "dataset/dataset.json"
