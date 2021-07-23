# This file processes training data and calculates mean and standard deviation
# for all files, grouped by each company. This data is subsequently needed for
# training.

import boto3
import botocore
import pandas as pd
import sagemaker
import sys

s3_bucket = "hackathon-train"
execution_role = sagemaker.get_execution_role()
region = boto3.Session().region_name
s3 = boto3.client("s3")

train_data = {
    'balance': "df_balance_train.csv",
    'cashflow': "df_cashflow_train.csv",
    'income': "df_income_train.csv"
}

mean_data = {
    'df_balance_train.csv': "mean_balance.csv",
    'df_cashflow_train.csv': "mean_cashflow.csv",
    'df_income_train.csv': "mean_income.csv"
}

std_data = {
    'df_balance_train.csv': "std_balance.csv",
    'df_cashflow_train.csv': "std_cashflow.csv",
    'df_income_train.csv': "std_income.csv"
}

local_data = []


# Checks if bucket exists
def check_bucket_permission(bucket):

    permission = False

    try:
        boto3.Session().client("s3").head_bucket(Bucket=bucket)

    except botocore.exceptions.ParamValidationError as e:
        print(
            "Hey! You either forgot to specify your S3 bucket"
            " or you gave your bucket an invalid name!"
        )

    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "403":
            print(f"Hey! You don't have permission to access the bucket, {bucket}.")
        elif e.response["Error"]["Code"] == "404":
            print(f"Hey! Your bucket, {bucket}, doesn't exist!")
        else:
            raise
    
    else:
        permission = True
    
    return permission


# Calculates mean of each column, grouped by company, and uploads to S3 
# training bucket.
def upload_mean(f, df_train):

    mean_file = mean_data[f]
    df_mean = df_train.groupby('Ticker').mean()
    df_mean.to_csv(mean_file, index=True)
    s3.upload_file(mean_file, s3_bucket, mean_file)


# Calculates standard devation of each column, grouped by company, and 
# processed to remove zeroes to avoid INF errors during training.
def upload_std(f, df_train):

    std_file = std_data[f]
    df_std = df_train.groupby('Ticker').std()
    df_std = df_std.replace(0, 0.0000001)
    df_std.to_csv(std_file, index=True)
    s3.upload_file(std_file, s3_bucket, std_file)


if check_bucket_permission(s3_bucket):
    print(
        f"Downloaded training data will be read from s3://{downloaded_data_bucket}"
    )

# Downloads files from S3 bucket to local storage
for key in train_data:
    file_name = f"df_{key}_train.csv" 
    s3.download_file(s3_bucket, file_name, file_name)
    local_data.append(file_name)


for f in local_data:
    df = pd.read_csv(f, delimiter=',')
    tickers = df['Ticker']
    df_train = df.select_dtypes(include=["float"])
    df_train = pd.concat([df_train, tickers], axis=1)

    upload_mean(f, df_train)
    upload_std(f, df_train)
