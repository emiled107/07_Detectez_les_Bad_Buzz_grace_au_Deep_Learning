import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Configuration pour réduire les logs verbeux de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Charger les données nettoyées
data_path = 'data/data_english_only.csv'
data_english_only = pd.read_csv(data_path)

# Fonction pour charger le tokenizer et préparer les données pour le modèle CNN
def prepare_data_for_cnn(data, model_name):
    tokenizer_path = f'../models/{model_name}_tokenizer.pickle'
    if not os.path.exists(tokenizer_path):
        print(f"Le fichier tokenizer à {tokenizer_path} n'existe pas.")
        return None, None

    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    label_encoder = LabelEncoder()
    data_labels = label_encoder.fit_transform(data['target'])

    X_seq = tokenizer.texts_to_sequences(data['text_lemmatized'].apply(eval).apply(' '.join))
    X_pad = tf.keras.preprocessing.sequence.pad_sequences(X_seq, maxlen=100)
    X_np = np.array(X_pad, dtype=np.int32)
    y_np = np.array(data_labels, dtype=np.int32)

    return X_np, y_np

# Évaluation du modèle CNN
def evaluate_cnn_model(model_path, data, model_name):
    X_test, y_test = prepare_data_for_cnn(data, model_name)
    if X_test is None or y_test is None:
        print("Erreur de préparation des données.")
        return

    if not os.path.exists(model_path):
        print(f"Le modèle à {model_path} n'existe pas.")
        return

    model = tf.keras.models.load_model(model_path)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])

    print(f"CNN Model Evaluation:\nLoss: {loss}\nAccuracy: {accuracy}\n{report}")

# Chemin du modèle CNN
cnn_model_path = 'models/cnn_model.keras'

# Évaluation du modèle CNN
evaluate_cnn_model(cnn_model_path, data_english_only, 'cnn_model')
