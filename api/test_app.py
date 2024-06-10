import pytest
from flask import json, jsonify
from main import app  

# Configuration de MLflow pour enregistrer les résultats de test
import mlflow

mlflow.set_tracking_uri("http://localhost:5006")
mlflow.set_experiment("air_paradis_api_tests")

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_positive(client):
    with mlflow.start_run():
        response = client.post('/predict', json={'tweet': 'I love this!'})
        assert response.status_code == 200
        data = json.loads(response.data.decode('utf-8'))
        assert data['sentiment'] == 'positive'
        mlflow.log_metric("predict_positive_status", response.status_code)

def test_predict_negative(client):
    with mlflow.start_run():
        response = client.post('/predict', json={'tweet': 'I hate this!'})
        assert response.status_code == 200
        data = json.loads(response.data.decode('utf-8'))
        assert data['sentiment'] == 'negative'
        mlflow.log_metric("predict_negative_status", response.status_code)

def test_predict_no_tweet(client):
    with mlflow.start_run():
        response = client.post('/predict', json={})
        assert response.status_code == 400
        data = json.loads(response.data.decode('utf-8'))
        assert data['error'] == 'No tweet provided'
        mlflow.log_metric("predict_no_tweet_status", response.status_code)
