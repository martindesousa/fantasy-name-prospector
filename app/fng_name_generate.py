import numpy as np
import tensorflow as tf
import pickle
import os
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
    bigram_counts = data_dict.get('bigram_counts', {})  # Get bigram counts if available
    
    return model, X, y, char_to_idx, idx_to_char, char_set, bigram_counts

def generate_quality_names_stream(model_name, count=10, prefix_text='', length=None, temperature=1.0, min_bigram_count=2, custom_names=None):
    """Generator that yields unique names one-by-one with guaranteed length and optional bigram filtering."""
    model, X, y, char_to_idx, idx_to_char, char_set, bigram_counts = load_model_data(model_name)

    # Get available first letters for random prefix selection
    if not prefix_text:
        if custom_names:
            # Pull first letters from the custom names
            available_first_letters = [name.strip()[0] for name in custom_names if name.strip()]
        else:
            # Load names from textfile for pretrained models
            textfile_path = os.path.join('textfiles', f"{model_name}_names.txt")
            try:
                with open(textfile_path, 'r', encoding='utf-8') as f:
                    pretrained_names = [line.strip() for line in f if line.strip()]
                available_first_letters = [name[0] for name in pretrained_names if name]
            except FileNotFoundError:
                # Fallback to char_set if textfile doesn't exist
                available_first_letters = [char for char in char_set if char.isalpha()]
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
            prefix = np.random.choice(available_first_letters).upper()
        else:
            prefix = prefix_text
            
        name = prefix

        for _ in range(length - len(prefix)):
            encoded = [char_to_idx[c] for c in name if c in char_to_idx]
            if not encoded:
                encoded = [char_to_idx[np.random.choice(list(char_to_idx.keys()))]]
            encoded = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=X.shape[1], padding='pre')
            predictions = model.predict(encoded, verbose=0)[0]

            if temperature == 0:
                predicted_index = np.argmax(predictions)
            else:
                predictions = np.log(predictions + 1e-8) / (temperature * 0.5)
                probs = np.exp(predictions) / np.sum(np.exp(predictions))
                sorted_indices = np.argsort(-probs)

                if bigram_counts and len(name) > 0:
                    last_char = name[-1]
                    for idx in sorted_indices[:10]:
                        if bigram_counts.get(last_char + idx_to_char[idx], 0) >= min_bigram_count:
                            predicted_index = idx
                            break
                    else:
                        predicted_index = sorted_indices[0]
                else:
                    predicted_index = np.random.choice(len(probs), p=probs)

            name += idx_to_char[predicted_index]

        if name not in generated_names:
            generated_names.add(name)
            yield name
        # else silently retry

# def generate_quality_names(model_name, count=10, seed_text='', length=7, temperature=1.0, min_bigram_count=2):
#     """Generate names of fixed length, optionally using a seed or random char, filtered by quality if needed."""
#     model, X, y, char_to_idx, idx_to_char, char_set, bigram_counts = load_model_data(model_name)

#     quality_names = []
#     attempts = 0
#     max_attempts = count * 3  # Limit to avoid infinite loops if filtering is on

#     while len(quality_names) < count and attempts < max_attempts:
#         # Use seed if provided, else pick a fresh random one each time
#         if seed_text:
#             seed = ''.join(c for c in seed_text if c in char_to_idx)
#             if not seed:
#                 seed = np.random.choice(list(char_to_idx.keys()))
#         else:
#             seed = np.random.choice(list(char_to_idx.keys()))

#         name = seed

#         for _ in range(length - len(seed)):
#             # Encode current name state
#             encoded = [char_to_idx[c] for c in name]
#             encoded = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=X.shape[1], padding='pre')

#             # Predict next character probabilities
#             predictions = model.predict(encoded, verbose=0)[0]

#             if temperature == 0:
#                 predicted_char_index = np.argmax(predictions)
#             else:
#                 predictions = np.log(predictions + 1e-8) / (temperature * 0.5)  # scale temperature
#                 probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
#                 sorted_indices = np.argsort(-probabilities)

#                 if bigram_counts and len(name) > 0:
#                     last_char = name[-1]

#                     for idx in sorted_indices[:10]:
#                         candidate_char = idx_to_char[idx]
#                         bigram = last_char + candidate_char
#                         if bigram_counts.get(bigram, 0) >= min_bigram_count:
#                             predicted_char_index = idx
#                             break
#                     else:
#                         predicted_char_index = sorted_indices[0]
#                 else:
#                     predicted_char_index = np.random.choice(len(probabilities), p=probabilities)

#             name += idx_to_char[predicted_char_index]

#         # Optionally filter names (uncomment if needed)
#         # if filter_name_by_quality(name, bigram_counts, min_count=min_bigram_count):
#         #     quality_names.append(name)
#         # else:
#         #     attempts += 1

#         quality_names.append(name)
#         attempts += 1

#     return quality_names