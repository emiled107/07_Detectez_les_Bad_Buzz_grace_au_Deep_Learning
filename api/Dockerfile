FROM python:3.9-slim

WORKDIR /api

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py main.py
COPY cnn_model.keras cnn_model.keras
COPY cnn_model_tokenizer.pickle cnn_model_tokenizer.pickle

CMD ["python", "main.py"]
