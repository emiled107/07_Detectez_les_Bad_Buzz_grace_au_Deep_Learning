#!/bin/bash

# Activer l'environnement Conda
source /opt/conda/etc/profile.d/conda.sh
conda activate bert_sentiment_analysis

# Démarrer le serveur MLflow en arrière-plan
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://$MLFLOW_S3_ENDPOINT_URL --host 0.0.0.0 &

# Attendre que le serveur MLflow soit opérationnel
echo "Attente du démarrage de MLflow..."
sleep 10

# Exécuter les tests
echo "Exécution des tests..."
pytest test_app.py

# Exécuter l'application principale
echo "Démarrage de l'application principale..."
exec conda run --no-capture-output -n bert_sentiment_analysis python main.py
