import numpy as np
import tensorflow as tf
import pickle
import os
from collections import Counter
from app.fng_model import BigramPenaltyLoss

def load_model_data(model_name='my_model'):
    if model_name.startswith('custom'):
        model_path = f'app/models/custom/{model_name}.keras'
        data_path = f'app/models/custom/{model_name}_data.pkl'
    else:
        model_path = f'app/models/{model_name}.keras'
        data_path = f'app/models/{model_name}_data.pkl'

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'BigramPenaltyLoss': BigramPenaltyLoss}
    )

    with open(data_path, 'rb') as file:
        data_dict = pickle.load(file)
    
    X = data_dict['X']
    y = data_dict['y']
    char_to_idx = data_dict['char_to_idx']
    idx_to_char = data_dict['idx_to_char']
    char_set = data_dict['char_set']
    bigram_counts = data_dict.get('bigram_counts', {})
    
    return model, X, y, char_to_idx, idx_to_char, char_set, bigram_counts

def generate_quality_names_stream(model_name, count=10, prefix_text='', length=None, temperature=1.0, min_bigram_count=1, custom_names=None):
    """Generator that yields unique names one-by-one with guaranteed length and optional bigram filtering."""
    try:
        model, X, y, char_to_idx, idx_to_char, char_set, bigram_counts = load_model_data(model_name)
    except FileNotFoundError as e:
        print(f"Error loading model data: {e}")
        return

    # Get available first letters for random prefix selection
    if not prefix_text:
        if custom_names:
            # Pull first letters from the custom names
            first_letter_counts = Counter(name.strip()[0] for name in custom_names if name.strip())
            available_first_letters = list(first_letter_counts.keys())
            letter_probs = np.array([first_letter_counts[char] for char in available_first_letters], dtype=np.float32)

            # Soften letter_probs to be sensitive to temperature
            letter_logits = np.log(letter_probs + 1e-8) / temperature
            letter_probs = np.exp(letter_logits)
            letter_probs /= letter_probs.sum()  # Normalize to get a proper probability distribution
        else:
            # Load names from textfile for pretrained models
            textfile_path = os.path.join('app', 'textfiles', f"{model_name}_names.txt")
            try:
                with open(textfile_path, 'r', encoding='utf-8') as f:
                    pretrained_names = [line.strip() for line in f if line.strip()]
                available_first_letters = [name[0] for name in pretrained_names if name]

                # Compute frequency of each first letter
                first_letter_counts = Counter(available_first_letters)
                available_first_letters = list(first_letter_counts.keys())
                letter_probs = np.array([first_letter_counts[char] for char in available_first_letters], dtype=np.float32)

                # Apply temperature-sensitive softmax
                letter_logits = np.log(letter_probs + 1e-8) / temperature
                letter_probs = np.exp(letter_logits)
                letter_probs /= letter_probs.sum()
            except FileNotFoundError:
                # Fallback to char_set if textfile doesn't exist
                available_first_letters = [char for char in char_set if char.isalpha()]
                letter_probs = np.ones(len(available_first_letters), dtype=np.float32)
                letter_probs /= letter_probs.sum()

    else:
        available_first_letters = None  # Use provided prefix

    # Set default length if not provided, and ensure it's an integer
    if length is None or length == '':
        if custom_names:
            # Calculate average length from custom names
            name_lengths = [len(name.strip()) for name in custom_names if name.strip()]
            length = int(np.mean(name_lengths)) if name_lengths else 6
        else:
            # Calculate average length from pretrained model names
            textfile_path = os.path.join('app', 'textfiles', f"{model_name}_names.txt")
            try:
                with open(textfile_path, 'r', encoding='utf-8') as f:
                    pretrained_names = [line.strip() for line in f if line.strip()]
                name_lengths = [len(name) for name in pretrained_names if name]
                length = int(np.mean(name_lengths)) if name_lengths else 6
            except FileNotFoundError:
                # Fallback to default length if textfile doesn't exist
                length = 6
    else:
        # Ensure length is an integer if provided
        length = int(length)

    generated_names = set()
    attempts = 0
    max_attempts = count * 10  # More attempts in case of many near-duplicates

    while len(generated_names) < count and attempts < max_attempts:
        attempts += 1

        # Pick a random prefix for each name (if no specific prefix was provided)
        if available_first_letters:
            prefix = np.random.choice(available_first_letters, p=letter_probs).upper()
        else:
            prefix = prefix_text

        name = prefix

        while len(name) < length:
            encoded = [char_to_idx[c] for c in name if c in char_to_idx]
            if not encoded:
                encoded = [char_to_idx[np.random.choice(list(char_to_idx.keys()))]]
            encoded = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=X.shape[1], padding='pre')

            predictions = model.predict(encoded, verbose=0)[0]

            # Apply temperature-based sampling
            if temperature == 0:
                predicted_index = np.argmax(predictions)
            else:
                logits = np.log(predictions + 1e-8) / temperature
                probs = np.exp(logits)
                probs /= np.sum(probs)
                predicted_index = np.random.choice(len(probs), p=probs)

            next_char = idx_to_char[predicted_index]

            # Skip invalid or non-alphabetic characters just to be safe
            if next_char in ['\n', ' ', '', '<PAD>']:
                continue

            name += next_char

        # Ensure the generated name has the exact desired length
        if len(name) == length:
            # Optionally add the end token if it makes sense for your model's logic
            # encoded = [char_to_idx[c] for c in name if c in char_to_idx]
            # encoded = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=X.shape[1], padding='pre')
            # predictions = model.predict(encoded, verbose=0)[0]
            # end_token_idx = char_to_idx.get('<END>')
            # if end_token_idx is not None and np.argmax(predictions) == end_token_idx:
            if name not in generated_names:
                generated_names.add(name)
                yield name
        # else silently retry