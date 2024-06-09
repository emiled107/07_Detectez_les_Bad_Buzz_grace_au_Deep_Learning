from flask import Flask, request, jsonify
import os
os.environ['TF_USE_LEGACY_KERAS'] = 'True'  # Ajouter cette ligne

import tensorflow as tf
import pickle
import json

app = Flask(__name__)

# Chemins des fichiers
model_file_path = 'cnn_model.keras'  # Utilisez le nom de dossier pour SavedModel
tokenizer_file_path = 'cnn_model_tokenizer.json'

print("TensorFlow version:", tf.__version__)

# Fonction pour charger le modèle
def load_model(path):
    if os.path.exists(path):
        try:
            model = tf.keras.models.load_model(path)
            print("Modèle chargé avec succès.")
            return model
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            return None
    else:
        print("Chemin du modèle non trouvé.")
        return None

# Fonction pour charger le tokenizer
def load_tokenizer(path):
    if os.path.exists(path):
        try:
            with open(path, 'r') as handle:
                data = json.load(handle)
                tokenizer = tf.keras.preprocessing.text.Tokenizer()
                tokenizer.word_index = data['word_index']  # Restaure l'index des mots
                print("Tokenizer chargé avec succès.")
                return tokenizer
        except Exception as e:
            print(f"Erreur lors du chargement du tokenizer: {e}")
            return None
    else:
        print("Chemin du tokenizer non trouvé.")
        return None

model = load_model(model_file_path)
tokenizer = load_tokenizer(tokenizer_file_path)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or tokenizer is None:
        return jsonify({'error': 'Modèle ou tokenizer non chargé'}), 500

    data = request.json
    tweet = data.get('tweet')
    if tweet is None:
        return jsonify({'error': 'No tweet provided'}), 400

    # Prétraiter et prédire
    try:
        sequences = tokenizer.texts_to_sequences([tweet])
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
        prediction = model.predict(padded_sequences)
        sentiment = 'negative' if prediction < 0.5 else 'positive'
        return jsonify({'sentiment': sentiment})
    except Exception as e:
        return jsonify({'error': f"Erreur lors de la prédiction: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
