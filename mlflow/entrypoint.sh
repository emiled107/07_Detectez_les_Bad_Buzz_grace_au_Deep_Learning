#!/bin/bash

# Configure AWS credentials for IDrive e2
export AWS_ACCESS_KEY_ID=$MLFLOW_S3_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$MLFLOW_S3_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION=us-east-1
export MLFLOW_S3_ENDPOINT_URL=$MLFLOW_S3_ENDPOINT_URL

# Start the MLflow server with the necessary environment variables for authentication
mlflow server \
    --backend-store-uri $BACKEND_STORE_URI \
    --default-artifact-root $DEFAULT_ARTIFACT_ROOT \
    --host 0.0.0.0 \
    --port 5000
