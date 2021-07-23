import json
import urllib.parse
import boto3
import os

def lambda_handler(event, context):
    print("Event: ", event)
    print("Context: ", context)
    
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    filePath = 's3://{}/{}'.format(bucket, key)
    
    print("Bucket: ", bucket)
    print("Key: ", key)
    print("Filepath: ", filePath)
    
    folder = key.split('/')[0]
    inferenceEndpoint = ''
    meanPath = ''
    sdPath = ''
    
    if folder == os.environ['BALANCE_FOLDER']:
        inferenceEndpoint = os.environ['BALANCE_INFERENCE_ENDPOINT']
        meanPath = os.environ['BALANCE_MEAN_PATH']
        sdPath = os.environ['BALANCE_SD_PATH']
    elif folder == os.environ['INCOME_FOLDER']:
        inferenceEndpoint = os.environ['INCOME_INFERENCE_ENDPOINT']
        meanPath = os.environ['INCOME_MEAN_PATH']
        sdPath = os.environ['INCOME_SD_PATH']
    elif folder == os.environ['CASHFLOW_FOLDER']:
        inferenceEndpoint = os.environ['CASHFLOW_INFERENCE_ENDPOINT']
        meanPath = os.environ['CASHFLOW_MEAN_PATH']
        sdPath = os.environ['CASHFLOW_SD_PATH']
    else:
        return "Glue inference job could not be triggered!"
    
    print(filePath, inferenceEndpoint, meanPath, sdPath)
    glueClient = boto3.client("glue")
    response = glueClient.start_job_run(
             JobName = 'Inference',
             Arguments = {
                '--filePath': filePath,
                '--inferenceEndpoint': inferenceEndpoint,
                '--meanPath': meanPath,
                '--sdPath': sdPath,
                '--anomThreshold': os.environ['ANOM_THRESHOLD']
             })
    
    return "Glue Inference job has been triggered"
