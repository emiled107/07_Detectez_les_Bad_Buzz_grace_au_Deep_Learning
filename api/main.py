from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
import os

app = Flask(__name__)

# Chemins des fichiers
model_file_path = 'cnn_model.keras'
tokenizer_file_path = 'cnn_model_tokenizer.pickle'

# Charger le modèle et le tokenizer
model = tf.keras.models.load_model(model_file_path)
with open(tokenizer_file_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    tweet = data.get('tweet')
    if tweet is None:
        return jsonify({'error': 'No tweet provided'}), 400

    # Prétraiter et prédire
    sequences = tokenizer.texts_to_sequences([tweet])
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
    prediction = model.predict(padded_sequences)
    sentiment = 'negative' if prediction < 0.5 else 'positive'

    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
