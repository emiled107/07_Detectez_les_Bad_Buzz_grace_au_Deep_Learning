import ast
import mlflow
import mlflow.tensorflow
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.data import Dataset
import os
import time

# Définir l'expérience MLflow
mlflow.set_experiment('BERT')

# MLflow tracking
mlflow.tensorflow.autolog()

model_path = f'../models/cnn_model.keras'
if os.path.exists(f'../models/bert'):
    logging.info(f"Le modèle à ../models/bert existe déjà. Aucun réentraînement nécessaire.")
    list_models_directory('models')  # Listez le répertoire des modèles si le modèle existe déjà
else :
    with mlflow.start_run(run_name="BERT"):
        # Chargement des données
        data_path = '../data/data_english_only.csv'
        data_english_only = pd.read_csv(data_path)

        # Conversion des chaînes de caractères représentant des listes en listes Python réelles
        data_english_only['text_lemmatized'] = data_english_only['text_lemmatized'].apply(ast.literal_eval)

        # Reconstruction des textes lemmatisés en une seule chaîne de caractères pour chaque exemple
        texts = data_english_only['text_lemmatized'].apply(' '.join)

        # Labels de sentiment
        labels = data_english_only['target'].values

        # Initialisation du tokenizer TinyBERT
        tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')

        # Tokenisation des textes
        encodings = tokenizer(texts.tolist(), truncation=True, padding='max_length', max_length=128, return_tensors='tf')

        # Convertir les tenseurs en tableaux NumPy
        input_ids = encodings['input_ids'].numpy()
        attention_masks = encodings['attention_mask'].numpy()

        # Division en train et test avec stratification
        train_input_ids, val_input_ids, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.1, stratify=labels)
        train_attention_masks, val_attention_masks = train_test_split(attention_masks, test_size=0.1, stratify=labels)

        # Convertir les tableaux NumPy en tenseurs TensorFlow
        train_input_ids = tf.convert_to_tensor(train_input_ids)
        val_input_ids = tf.convert_to_tensor(val_input_ids)
        train_attention_masks = tf.convert_to_tensor(train_attention_masks)
        val_attention_masks = tf.convert_to_tensor(val_attention_masks)
        train_labels = tf.convert_to_tensor(train_labels)
        val_labels = tf.convert_to_tensor(val_labels)

        # Création des DataLoaders
        batch_size = 64
        train_dataset = Dataset.from_tensor_slices(({'input_ids': train_input_ids, 'attention_mask': train_attention_masks}, train_labels)).shuffle(len(train_input_ids)).batch(batch_size)
        val_dataset = Dataset.from_tensor_slices(({'input_ids': val_input_ids, 'attention_mask': val_attention_masks}, val_labels)).batch(batch_size)

        # Initialisation du modèle TinyBERT
        model = TFBertForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=2, from_pt=True)

        # Utiliser l'optimiseur legacy.Adam pour meilleure performance sur M1/M2
        num_train_steps = len(train_dataset) * 5  # epochs
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=2e-5)

        # Compile the model with the appropriate loss function
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

        # Check if a checkpoint exists
        checkpoint_path = "checkpoints"
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)

        if latest_checkpoint:
            model.load_weights(latest_checkpoint)
            print(f"Resuming from checkpoint {latest_checkpoint}")

        # Entraînement du modèle
        epochs = 5
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_freq='epoch')

        for epoch in range(epochs):
            history = model.fit(train_dataset, validation_data=val_dataset, epochs=1, callbacks=[checkpoint_callback])

            # Log metrics to MLflow
            mlflow.log_metrics({"loss": history.history['loss'][0], "accuracy": history.history['accuracy'][0]}, step=epoch)
            mlflow.log_metrics({"val_loss": history.history['val_loss'][0], "val_accuracy": history.history['val_accuracy'][0]}, step=epoch)

            print(f"End of epoch {epoch+1}, pausing for 2 minutes.")
            time.sleep(120)

        # Save the model
        model_save_path = '../models/bert_tiny_finetuned'
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"Model and tokenizer saved to {model_save_path}")

    mlflow.end_run()
