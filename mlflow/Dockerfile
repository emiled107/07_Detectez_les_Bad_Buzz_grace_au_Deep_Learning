# Base image
FROM python:3.8-slim

# Install MLflow
RUN pip install mlflow

# Set up the entrypoint
CMD ["mlflow", "server", "--host", "0.0.0.0"]
