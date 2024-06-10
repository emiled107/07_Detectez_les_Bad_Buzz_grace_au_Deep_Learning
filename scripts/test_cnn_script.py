import os
import pickle
import pytest
import pandas as pd
import tensorflow as tf
from main_script import prepare_data, build_cnn_model, load_model, load_tokenizer

@pytest.fixture
def sample_data():
    data = {
        'text_lemmatized': ["I love this product", "This is the worst experience ever", "Absolutely fantastic!"],
        'target': [1, 0, 1]
    }
    return pd.DataFrame(data)

def test_prepare_data(sample_data):
    model_name = 'test_model'
    data_np, data_labels, tokenizer = prepare_data(sample_data, model_name)

    assert len(data_np) == len(sample_data)
    assert len(data_labels) == len(sample_data)
    assert os.path.exists(f'models/{model_name}_tokenizer.pickle')

def test_build_cnn_model():
    input_dim = 1000
    output_dim = 50
    input_length = 100
    model = build_cnn_model(input_dim, output_dim, input_length)

    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, input_length)

def test_load_model():
    model_path = 'test_model_path'
    # Simulate a model saving to test load_model
    model = build_cnn_model(1000, 50, 100)
    model.save(model_path)
    
    loaded_model = load_model(model_path)
    assert loaded_model is not None
    assert isinstance(loaded_model, tf.keras.Model)

def test_load_tokenizer():
    tokenizer_path = 'test_tokenizer.pickle'
    # Simulate a tokenizer saving to test load_tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    loaded_tokenizer = load_tokenizer(tokenizer_path)
    assert loaded_tokenizer is not None
    assert isinstance(loaded_tokenizer, tf.keras.preprocessing.text.Tokenizer)

if __name__ == "__main__":
    pytest.main()
