import os
import io
import boto3
import json
import csv
import statistics
from botocore.exceptions import ClientError
import sys
from awsglue.utils import getResolvedOptions
import pandas as pd

validCols = ['Shares (Basic)', 'Shares (Diluted)', 'Revenue', 'Cost of Revenue',
       'Gross Profit', 'Operating Expenses',
       'Selling, General & Administrative', 'Operating Income (Loss)',
       'Non-Operating Income (Loss)', 'Interest Expense, Net',
       'Income Tax (Expense) Benefit, Net']
       

def sendEmail(finalScores):
    SENDER = "Anomaly Detector <arjavibahety@gmail.com>"
    RECIPIENT = "arjavibahety@gmail.com"
    AWS_REGION = "ap-southeast-1"
    SUBJECT = "Anomaly Detection Results"
    
    emailText = "Anomalies were found in the following: "
    emailHTML = "<tr><th>Row Index</th><th>Column</th><th>Value</th><th>Anomaly Score</th></tr>"
    for row in finalScores:
        emailText += "\n" + str(row['row_index']) + str(row['col']) + str(row['csvContent']) + str(row['anomalousScore'])
        emailHTML += "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(str(row['row_index']), str(row['col']), str(row['csvContent']), str(row['anomalousScore']))
    
    BODY_TEXT = (emailText)
    BODY_HTML = """
    <html>
    <head>
    <style>
    table, th, td {
      border: 1px solid black;
      border-collapse: collapse;
    }
    th, td {
      padding: 5px;
    }
    </style>
    </head>
    <body>
        <p>Anomalies were found in the following rows: </p>
        <table> """ + emailHTML + """</table>
    </body>
    </html>
    """         
    
    # The character encoding for the email.
    CHARSET = "UTF-8"
    
    # Create a new SES resource and specify a region.
    client = boto3.client('ses',region_name=AWS_REGION)
    
    # Try to send the email.
    try:
        #Provide the contents of the email.
        response = client.send_email(
            Destination={
                'ToAddresses': [
                    RECIPIENT,
                ],
            },
            Message={
                'Body': {
                    'Html': {
                        'Charset': CHARSET,
                        'Data': BODY_HTML,
                    },
                    'Text': {
                        'Charset': CHARSET,
                        'Data': BODY_TEXT,
                    },
                },
                'Subject': {
                    'Charset': CHARSET,
                    'Data': SUBJECT,
                },
            },
            Source=SENDER,
        )
    except ClientError as e:
        return e.response['Error']['Message']
    else:
        return "Email sent! Message ID: \n" + str(response['MessageId'])

def getFiles(folderName):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=folderName)
    print("Pages: ", pages)
    files = []
    for page in pages:
        for file in page['Contents']:
            if file['Key'] != folderName + '/':
                print(file['Key'], "\n")
                files.append(file['Key'])
    
    print("Files: ", files)
    return files

def getCSVData(filePath, meanPath, sdPath):
    df_test = pd.read_csv(filePath)
    mean_data = pd.read_csv(meanPath)
    sd_data = pd.read_csv(sdPath)
      
    payload = ''
    csvContent = []
    payloadCols = []
    for index, row in df_test.iterrows():
        ticker = row['Ticker']
        mean_row = mean_data[mean_data['Ticker'] == ticker]
        sd_row = sd_data[sd_data['Ticker'] == ticker]
        
        for col in validCols:
            processedValue = 0
            if col != 'Ticker':
                mean = mean_row[col].iloc[0]
                sd = sd_row[col].iloc[0]
                processedValue = (row[col] - mean) / sd
                csvContent.append(row[col])
                payloadCols.append(col)
                if payload == '':
                    payload += str(processedValue)
                else:
                    payload += '\n' + str(processedValue)
                
    return payload, csvContent, payloadCols

def getFinalScores(rawScores, csvContent, payloadCols, anomThreshold):
    scores = [row['score'] for row in rawScores['scores']]
    finalScores = []
    index = 0
    for row in rawScores['scores']:
        if (row['score'] > anomThreshold):
            finalScores.append({"row_index" : int(index / len(validCols)), "anomalousScore" : row['score'], "csvContent": csvContent[index], "col": payloadCols[index]})
        index += 1
    
    return finalScores

def main(filePath, inferenceEndpoint, meanPath, sdPath, anomThreshold):
    payload, csvContent, payloadCols = getCSVData(filePath, meanPath, sdPath)
    print("Payload: ", payload)
    print("CsvContent: ", csvContent)
    print("PayloadCols: ", payloadCols)
    runtime= boto3.client('runtime.sagemaker')
    response = runtime.invoke_endpoint(EndpointName = inferenceEndpoint,
                                      ContentType='text/csv',
                                      Body=payload,
                                      Accept = "application/json")
    print("\n")
    print("Inference response: ", response)
    rawScores = json.loads(response['Body'].read().decode())
    print("rawScores: ", rawScores)
    finalScores = getFinalScores(rawScores, csvContent, payloadCols, anomThreshold)
    print("finalScores: ", finalScores)
    emailResponse = sendEmail(finalScores)
    print(emailResponse)

if __name__== "__main__" :
    args = getResolvedOptions(sys.argv, ['filePath', 'inferenceEndpoint', 'meanPath', 'sdPath', 'anomThreshold'])
    # args = {}
    # args['filePath'] = 's3://hackathon-extract/income/df_income_DXCM_ZpKMs2BhQHL9NPf9knsiys_anomalous.csv'
    # args['inferenceEndpoint'] = 'randomcutforest-2021-07-23-07-55-32-998'
    # args['meanPath'] = 's3://hackathon-train/mean_income.csv'
    # args['sdPath'] = 's3://hackathon-train/std_income.csv'
    # args['anomThreshold'] = '2'
    
    main(args['filePath'], args['inferenceEndpoint'], args['meanPath'], args['sdPath'], float(args['anomThreshold']))
