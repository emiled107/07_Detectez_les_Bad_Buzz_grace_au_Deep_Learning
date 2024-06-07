import os
import re
import zipfile
import requests
import pandas as pd
import numpy as np
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import ftfy
import fasttext
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Télécharger les ressources nécessaires de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Paramètres configurables
data_directory = "../data/sources"  # Répertoire pour stocker les données
zip_file_url = "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+7%C2%A0-+D%C3%A9tectez+les+Bad+Buzz+gr%C3%A2ce+au+Deep+Learning/sentiment140.zip"  # URL du fichier ZIP à télécharger
zip_file_name = "sentiment140.zip"  # Nom du fichier ZIP à télécharger

# Vérifiez si le répertoire et les données existent déjà pour éviter le téléchargement et l'extraction répétés
if not os.path.exists(os.path.join(data_directory, zip_file_name.replace('.zip', ''))):
    # Création du répertoire s'il n'existe pas
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # Téléchargement du fichier ZIP
    response = requests.get(zip_file_url)
    zip_content = BytesIO(response.content)

    # Extraction du contenu du fichier ZIP
    with zipfile.ZipFile(zip_content, 'r') as zip_ref:
        zip_ref.extractall(data_directory)
else:
    print("Les données sont déjà téléchargées et extraites.")

# Recherche du fichier CSV dans le répertoire extrait
data_path = None
for root, dirs, files in os.walk(data_directory):
    for file in files:
        if file.endswith('.csv'):
            data_path = os.path.join(root, file)
            break

if data_path:
    # Chargement et affichage des premières lignes du fichier CSV pour vérifier
    data = pd.read_csv(data_path, encoding="latin_1", names=['target', 'ids', 'date', 'flag', 'user', 'text'])
    # Supposons que la colonne contenant le texte est la 5ème colonne
    data['text'] = data['text'].apply(ftfy.fix_text)

    # Affichage des informations sur les données
    print(data.info())
    print(data.describe(include='all'))
    print(data.head())
else:
    print("Aucun fichier CSV trouvé dans le répertoire extrait.")

# Analyse des doublons
print("Nombre de doublons avant suppression :", data.duplicated(subset=['date', 'user', 'text']).sum())
doublons = data[data.duplicated()]

# Suppression des colonnes inutiles
data = data.drop(['ids', 'flag'], axis=1)

# Suppression des doublons
data_clean = data.drop_duplicates(subset=['date', 'user', 'text']).copy()

# Re-codage de la colonne 'target' de 0 et 4 à 0 et 1
data_clean['target'] = data_clean['target'].map({0: 0, 4: 1})

# Vérification des premières lignes pour s'assurer que les modifications sont correctes
print(data_clean.head())

# Fonction pour télécharger le modèle FastText s'il n'existe pas
def download_fasttext_model(model_url, model_path):
    if not os.path.exists(model_path):
        print(f"Modèle FastText non trouvé à {model_path}. Téléchargement en cours...")
        response = requests.get(model_url)
        with open(model_path, 'wb') as model_file:
            model_file.write(response.content)
        print("Téléchargement terminé.")

# URL du modèle FastText et chemin local pour le sauvegarder
model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
model_path = "../data/lid.176.bin"

# Téléchargez le modèle FastText si nécessaire
download_fasttext_model(model_url, model_path)

# Chargement du modèle pré-entraîné FastText pour la détection de la langue
model = fasttext.load_model(model_path)

def detect_language_fasttext(text):
    text = str(text).replace('\n', ' ')
    predictions = model.predict([text], k=1)
    return predictions[0][0][0].replace('__label__', '')

def detect_languages_parallel(texts, num_threads):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_text = {executor.submit(detect_language_fasttext, text): text for text in texts}
        results = []
        for future in as_completed(future_to_text):
            result = future.result()
            results.append(result)
        return results

# Application de la détection de langue en parallèle
num_threads = 10  # Ajustez en fonction de votre environnement
languages = detect_languages_parallel(data_clean['text'], num_threads)

# Stockage du résultat dans une nouvelle colonne 'language'
data_clean['language'] = languages

# Filtrer le DataFrame pour ne garder que les tweets en anglais
data_english_only = data_clean[data_clean['language'] == 'en'].copy()

# Fonction pour nettoyer le texte
def clean_text(text):
    text = text.lower()  # Convertir en minuscules
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Supprimer les URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Supprimer les mentions et hashtags
    text = re.sub(r'\d+', '', text)  # Supprimer les chiffres
    text = re.sub(r'[^\w\s]', '', text)  # Supprimer la ponctuation
    return text

# Application de la fonction de nettoyage à chaque tweet
data_english_only['text'] = data_english_only['text'].apply(clean_text)

# Vérifier si 'punkt' est déjà téléchargé
if 'punkt' in nltk.data.path:
    print("'punkt' tokenizer model is already downloaded.")
else:
    nltk.download('punkt')
    print("'punkt' tokenizer model has been downloaded.")

# Application de la tokenisation à chaque tweet
data_english_only['tokens'] = data_english_only['text'].apply(word_tokenize)

# Vérifier si 'stopwords' est déjà téléchargé
if 'stopwords' in nltk.data.path:
    print("'stopwords' tokenizer model is already downloaded.")
else:
    nltk.download('stopwords')
    print("'stopwords' tokenizer model has been downloaded.")

stop_words = set(stopwords.words('english'))

# Fonction pour supprimer les stop-words
def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

# Application de la suppression des stop-words à chaque tweet
data_english_only['tokens'] = data_english_only['tokens'].apply(remove_stopwords)

# Vérifier si 'wordnet' est déjà téléchargé
if 'wordnet' not in nltk.data.path:
    nltk.download('wordnet')
    print("'wordnet' tokenizer model has been downloaded.")
else:
    print("'wordnet' tokenizer model is already downloaded.")

lemmatizer = WordNetLemmatizer()

# Fonction pour la lemmatisation
def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

# Application de la lemmatisation à chaque tweet
data_english_only['text_lemmatized'] = data_english_only['tokens'].apply(lemmatize_tokens)

# Enregistrement du DataFrame nettoyé pour les modèles CNN et BERT
cleaned_data_path = "../data/data_english_only.csv"
data_english_only.to_csv(cleaned_data_path, index=False)

# Diviser le dataset en ensembles d'entraînement et de test
train_df, test_df = train_test_split(data_english_only, test_size=0.2, random_state=42)

# Créer les répertoires pour enregistrer les datasets si nécessaire
processed_data_path = '../data/processed_data'
os.makedirs(processed_data_path, exist_ok=True)

cnn_train_path = os.path.join(processed_data_path, 'cnn_train.csv')
cnn_test_path = os.path.join(processed_data_path, 'cnn_test.csv')
bert_train_path = os.path.join(processed_data_path, 'bert_train.csv')
bert_test_path = os.path.join(processed_data_path, 'bert_test.csv')

# Enregistrer les datasets
train_df.to_csv(cnn_train_path, index=False)
test_df.to_csv(cnn_test_path, index=False)
train_df.to_csv(bert_train_path, index=False)
test_df.to_csv(bert_test_path, index=False)

print(f"Datasets saved:\nCNN Train: {cnn_train_path}\nCNN Test: {cnn_test_path}\nBERT Train: {bert_train_path}\nBERT Test: {bert_test_path}")
