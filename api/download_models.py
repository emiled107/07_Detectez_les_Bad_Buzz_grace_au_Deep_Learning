import boto3
import os

# Configuration AWS
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_S3_ENDPOINT_URL = os.getenv('AWS_S3_ENDPOINT_URL')
BUCKET_NAME = os.getenv('BUCKET_NAME')

# Configuration du client S3
s3_client = boto3.client('s3', 
                         endpoint_url=AWS_S3_ENDPOINT_URL,
                         aws_access_key_id=AWS_ACCESS_KEY_ID,
                         aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

# Chemins des fichiers à télécharger
model_file_path = 'cnn_model.keras'
tokenizer_file_path = 'cnn_model_tokenizer.pickle'

# Fonctions pour télécharger les fichiers
def download_file_from_s3(bucket_name, object_name, file_name):
    with open(file_name, 'wb') as f:
        s3_client.download_fileobj(bucket_name, object_name, f)

# Télécharger les fichiers
download_file_from_s3(BUCKET_NAME, f'{model_file_path}', model_file_path)
download_file_from_s3(BUCKET_NAME, f'{tokenizer_file_path}', tokenizer_file_path)
