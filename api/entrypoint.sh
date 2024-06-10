#!/bin/bash

# Activer l'environnement Conda
source /opt/conda/etc/profile.d/conda.sh
conda activate bert_sentiment_analysis

# Démarrer le serveur MLflow en arrière-plan
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://$MLFLOW_S3_ENDPOINT_URL --host 0.0.0.0 --port 5006 &

# Démarrer l'application principale en arrière-plan
conda run --no-capture-output -n bert_sentiment_analysis python main.py &

# Attendre que le serveur MLflow et l'application soient opérationnels
echo "Attente du démarrage de MLflow et de l'application principale..."
sleep 10

# Exécuter les tests
echo "Exécution des tests..."
pytest test_app.py

# Garder le script en vie pour permettre à l'application principale de continuer à fonctionner
wait
