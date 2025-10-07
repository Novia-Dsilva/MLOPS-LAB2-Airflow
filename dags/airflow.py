# Import necessary libraries and modules
from airflow import DAG
# from airflow.operators.python import PythonOperator
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.lab import load_data, data_preprocessing, build_save_model, evaluate_model, load_model_elbow, send_notification, create_visual_report

# NOTE:
# In Airflow 3.x, enabling XCom pickling should be done via environment variable:
# export AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True
# The old airflow.configuration API is deprecated.

# Define default arguments for your DAG
default_args = {
    'owner': 'Novia Dsilva',
    'start_date': datetime(2025, 1, 15),
    'retries': 0,  # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5),  # Delay before retries
}

# Create a DAG instance named 'Airflow_Lab1' with the defined default arguments
with DAG(
    'Airflow_Lab1',
    default_args=default_args,
    description='Dag example for Lab 1 of Airflow series',
    catchup=False,
) as dag:

    # Task to load data, calls the 'load_data' Python function
    load_data_task = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data,
    )

    # Task to perform data preprocessing, depends on 'load_data_task'
    data_preprocessing_task = PythonOperator(
        task_id='data_preprocessing_task',
        python_callable=data_preprocessing,
        op_args=[load_data_task.output],
    )

    # Task to build and save a model, depends on 'data_preprocessing_task'
    build_save_model_task = PythonOperator(
        task_id='build_save_model_task',
        python_callable=build_save_model,
        op_args=[data_preprocessing_task.output, "model.sav"],
    )

    # Task to Evaluate model (runs in parallel with load_model)
    evaluate_model_task = PythonOperator(
        task_id='evaluate_model_task',
        python_callable=evaluate_model,
        op_args=["model.sav", data_preprocessing_task.output],
    )

    # Task to load a model using the 'load_model_elbow' function, depends on 'build_save_model_task'
    load_model_task = PythonOperator(
        task_id='load_model_task',
        python_callable=load_model_elbow,
        op_args=["model.sav", build_save_model_task.output],
    )
    # Task to Visualtize the model
    visual_task = PythonOperator(
        task_id='create_visual',
        python_callable=create_visual_report,
        op_args=["model.sav", build_save_model_task.output],
    )
   #Task to send notification
    notification_task = PythonOperator(
        task_id='send_notification_task',
        python_callable=send_notification,
        #op_args=[load_model_task.output, build_save_model_task.output],
        op_args=[
            load_model_task.output,           # prediction
            build_save_model_task.output,     # sse
            evaluate_model_task.output        # eval_metrics ← MAKE SURE THIS IS HERE
        ],
    )


    # Set task dependencies
    # load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task >> notification_task

    
    # Changed To:
    load_data_task >> data_preprocessing_task >> build_save_model_task >> [evaluate_model_task, load_model_task, visual_task] >> notification_task  

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.test()
