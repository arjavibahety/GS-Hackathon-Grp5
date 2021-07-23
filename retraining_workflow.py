import sys

!{sys.executable} -m pip install --upgrade stepfunctions

import boto3
import logging
import numpy as np
import pandas as pd
import sagemaker
import stepfunctions

from sagemaker import RandomCutForest
from sagemaker.amazon.amazon_estimator import image_uris
from sagemaker.inputs import TrainingInput
from sagemaker.s3 import S3Uploader
from sklearn import preprocessing
from stepfunctions import steps
from stepfunctions.steps import TrainingStep, ModelStep
from stepfunctions.inputs import ExecutionInput
from stepfunctions.workflow import Workflow

session = sagemaker.Session()
stepfunctions.set_stream_logger(level=logging.INFO)

region = boto3.Session().region_name
s3 = boto3.client("s3")

workflow_execution_role = "arn:aws:iam::706876185239:role/AmazonSageMaker-StepFunctionsWorkflowExecutionRole"
train_data = "s3://hackathon-train"

bucket = "hackathon-extract" 
prefix = "rcf-trained-income"

data_filename = "df_income_train.csv"

s3.download_file("combine-train-except-last", f"{data_filename}", data_filename)
df = pd.read_csv(data_filename, delimiter=",")
df_train = df.select_dtypes(include="float")

scaler = preprocessing.MinMaxScaler()
min_max_scaler = preprocessing.MinMaxScaler()
df_train.loc[:,df_train.columns] = min_max_scaler.fit_transform(df_train.loc[:,df_train.columns])
df_train.isna().sum()/df_train.shape[0]
df_train = df_train.fillna(df_train.mean())


# Specify training job information
rcf = RandomCutForest(
    role=workflow_execution_role,
    instance_count=1,
    instance_type="ml.m4.xlarge",
    data_location=f"s3://{bucket}/{prefix}/",
    output_path=f"s3://{bucket}/{prefix}/test-output",
    num_samples_per_tree=512,
    num_trees=50,
)

rcf.fit(rcf.record_set(df_train.values.reshape(-1,1)))

rcf.deploy(initial_instance_count=1, instance_type="ml.m4.xlarge")

execution_input = ExecutionInput(
    schema={
        "TrainingJobName": "randomcutforest-2021-07-22-09-09-14-573",
        "EndpointName": "test-endpoint",
    }
)

training_step = steps.TrainingStep(
    "RetrainingModel",
    estimator=rcf,
    inputs= "s3://combine-train-except-last",
    data={
        "train": TrainingInput(train_data, content_type="text/csv")
    },
    job_name=execution_input['TrainingJobName']
)


# Visualize workflow
workflow_definition = steps.Chain([
    training_step
])

workflow = Workflow(
    name='MyInferenceRoutine_{}'.format(id),
    definition=workflow_definition,
    role=workflow_execution_role,
    execution_input=execution_input
)

workflow.create()
