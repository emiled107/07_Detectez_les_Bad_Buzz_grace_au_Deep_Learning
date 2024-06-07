import os
import pickle
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, BatchNormalization, GlobalMaxPooling1D, Dense
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configuration de la journalisation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fonction pour lister le contenu du répertoire models
def list_models_directory(directory):
    logging.info("Liste des fichiers dans le répertoire des modèles :")
    for filename in os.listdir(directory):
        logging.info(filename)

# Préparation des données
def prepare_data(data, model_name):
    label_encoder = LabelEncoder()
    data_labels = label_encoder.fit_transform(data['target'])

    tokenizer_path = f'models/{model_name}_tokenizer.pickle'
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        logging.info("Tokenizer chargé.")
    else:
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(data['text_lemmatized'])
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info("Nouveau tokenizer créé et sauvegardé.")

    data_seq = tokenizer.texts_to_sequences(data['text_lemmatized'])
    data_pad = tf.keras.preprocessing.sequence.pad_sequences(data_seq, maxlen=100)
    data_np = np.array(data_pad, dtype=np.int32)

    return data_np, data_labels, tokenizer

def build_cnn_model(input_dim, output_dim, input_length):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length, embeddings_initializer=GlorotUniform()),
        Conv1D(64, 5, activation='relu'),
        BatchNormalization(),
        GlobalMaxPooling1D(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_model(xp_name, model_name, model, X_train, y_train, X_val, y_val, model_path, batch_size=32, epochs=10):
    mlflow.set_experiment(xp_name)
    with mlflow.start_run(run_name=model_name):
        mlflow.tensorflow.autolog(log_models=True)
        mlflow.log_params({"batch_size": batch_size, "epochs": epochs})
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
        model_checkpoint = ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss', mode='min')

        logging.info("Début de l'entraînement du modèle.")
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                            callbacks=[early_stopping, model_checkpoint])
        model.save(model_path)
        logging.info("Modèle entraîné et sauvegardé.")

        best_epoch = early_stopping.stopped_epoch - 3
        best_val_loss = history.history['val_loss'][best_epoch]
        best_val_accuracy = history.history['val_accuracy'][best_epoch]
        mlflow.log_metrics({"best_val_loss": best_val_loss, "best_val_accuracy": best_val_accuracy})

tf.keras.backend.clear_session()
model_path = '../models/cnn_model.keras'
if os.path.exists(f'../models/{model_path}.keras'):
    logging.info(f"Le modèle à ../models/{model_path}.keras existe déjà. Aucun réentraînement nécessaire.")
    list_models_directory('models')  # Listez le répertoire des modèles si le modèle existe déjà
else :
    data_path = '../data/data_english_only.csv'
    data_english_only = pd.read_csv(data_path)
    data_np, data_labels, tokenizer = prepare_data(data_english_only, 'cnn_model')
    input_dim = len(tokenizer.word_index) + 1
    output_dim = 50
    input_length = 100
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_no = 1

    for train_index, val_index in kf.split(data_np, data_labels):
        X_train, X_val = data_np[train_index], data_np[val_index]
        y_train, y_val = data_labels[train_index], data_labels[val_index]
        model_cnn = build_cnn_model(input_dim, output_dim, input_length)
        model_path = f'../models/cnn_model_fold_{fold_no}.keras'
        train_and_save_model("TensorFlow/Keras avec Embedding et Convolutional Neural Network (CNN)", f'tf_keras_embedding_CNN_fold_{fold_no}', model_cnn, X_train, y_train, X_val, y_val, model_path, batch_size=16, epochs=3)
        fold_no += 1
