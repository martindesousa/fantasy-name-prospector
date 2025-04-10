import numpy as np
import tensorflow as tf
import pickle
from collections import Counter

def load_model_data(model_name='my_model'):
    model = tf.keras.models.load_model(f'models/{model_name}.keras')
    
    with open(f'models/{model_name}_data.pkl', 'rb') as file:
        data_dict = pickle.load(file)
    
    X = data_dict['X']
    y = data_dict['y']
    char_to_idx = data_dict['char_to_idx']
    idx_to_char = data_dict['idx_to_char']
    char_set = data_dict['char_set']
    bigram_counts = data_dict.get('bigram_counts', {})  # Get bigram counts if available
    
    return model, X, y, char_to_idx, idx_to_char, char_set, bigram_counts

# NOT IN USE <-------------------
# Function to generate names (based on the input seed text) with randomness
def generate_name(model_name, seed_text, length=6, temperature=1.0, min_bigram_count=2):
    model, X, y, char_to_idx, idx_to_char, char_set, bigram_counts = load_model_data(model_name)

    # Handle case where the seed text contains characters not in the training set
    valid_seed = ''.join(char for char in seed_text if char in char_to_idx)
    if valid_seed != seed_text:
        seed_text = valid_seed
        if not seed_text:  # If no valid characters, use a random character from the set
            seed_text = np.random.choice(list(char_to_idx.keys()))

    for _ in range(length - len(seed_text)):
        # Encode the seed text
        encoded = [char_to_idx[char] for char in seed_text]
        encoded = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=X.shape[1], padding='pre')
        
        # Predict probabilities for the next character
        predictions = model.predict(encoded, verbose=0)[0]

        # Apply temperature scaling to control randomness
        if temperature == 0:
            predicted_char_index = np.argmax(predictions)
        else:
            predictions = np.log(predictions + 1e-8) / temperature
            probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
            
            # Sort indices by probability (highest first)
            sorted_indices = np.argsort(-probabilities)
            
            # Try to avoid rare bigrams if we have bigram data
            if bigram_counts and len(seed_text) > 0:
                last_char = seed_text[-1]
                
                # Try the top 10 most likely characters
                for idx in sorted_indices[:10]:
                    candidate_char = idx_to_char[idx]
                    bigram = last_char + candidate_char
                    
                    # If this forms a common enough bigram, use it
                    if bigram_counts.get(bigram, 0) >= min_bigram_count:
                        predicted_char_index = idx
                        break
                else:
                    # If no good bigram found in top choices, use the most likely character
                    predicted_char_index = sorted_indices[0]
            else:
                # No bigram filtering, just sample based on probability
                predicted_char_index = np.random.choice(len(probabilities), p=probabilities)
        
        # Append the predicted character to the seed text
        seed_text += idx_to_char[predicted_char_index]
        final_name = seed_text
    
    return final_name

# Filter generated names based on quality criteria
def filter_name_by_quality(name, bigram_counts, min_count=2):
    """Check if a name meets quality criteria based on bigram frequencies"""
        
    # Check if it contains rare bigrams
    for i in range(len(name) - 1):
        bigram = name[i:i+2]
        if bigram_counts.get(bigram, 0) < min_count:
            return False
    
    return True

def generate_quality_names(model_name, count=10, seed_text='', length=7, temperature=1.0, min_bigram_count=2):
    """Generate names of fixed length, optionally using a seed or random char, filtered by quality if needed."""
    model, X, y, char_to_idx, idx_to_char, char_set, bigram_counts = load_model_data(model_name)

    quality_names = []
    attempts = 0
    max_attempts = count * 3  # Limit to avoid infinite loops if filtering is on

    while len(quality_names) < count and attempts < max_attempts:
        # Use seed if provided, else pick a fresh random one each time
        if seed_text:
            seed = ''.join(c for c in seed_text if c in char_to_idx)
            if not seed:
                seed = np.random.choice(list(char_to_idx.keys()))
        else:
            seed = np.random.choice(list(char_to_idx.keys()))

        name = seed

        for _ in range(length - len(seed)):
            # Encode current name state
            encoded = [char_to_idx[c] for c in name]
            encoded = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=X.shape[1], padding='pre')

            # Predict next character probabilities
            predictions = model.predict(encoded, verbose=0)[0]

            if temperature == 0:
                predicted_char_index = np.argmax(predictions)
            else:
                predictions = np.log(predictions + 1e-8) / (temperature * 0.5)  # scale temperature
                probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
                sorted_indices = np.argsort(-probabilities)

                if bigram_counts and len(name) > 0:
                    last_char = name[-1]

                    for idx in sorted_indices[:10]:
                        candidate_char = idx_to_char[idx]
                        bigram = last_char + candidate_char
                        if bigram_counts.get(bigram, 0) >= min_bigram_count:
                            predicted_char_index = idx
                            break
                    else:
                        predicted_char_index = sorted_indices[0]
                else:
                    predicted_char_index = np.random.choice(len(probabilities), p=probabilities)

            name += idx_to_char[predicted_char_index]

        # Optionally filter names (uncomment if needed)
        # if filter_name_by_quality(name, bigram_counts, min_count=min_bigram_count):
        #     quality_names.append(name)
        # else:
        #     attempts += 1

        quality_names.append(name)
        attempts += 1

    return quality_names

def generate_quality_names_stream(model_name, count=10, seed_text='', length=7, temperature=1.0, min_bigram_count=2):
    """Generator that yields names one-by-one with guaranteed length and optional bigram filtering."""
    model, X, y, char_to_idx, idx_to_char, char_set, bigram_counts = load_model_data(model_name)

    generated_count = 0
    attempts = 0
    max_attempts = count * 3

    while generated_count < count and attempts < max_attempts:
        seed = ''.join(c for c in seed_text if c in char_to_idx) if seed_text else ''
        if not seed:
            seed = np.random.choice(list(char_to_idx.keys()))
        
        name = seed
        for _ in range(length - len(seed)):
            encoded = [char_to_idx[c] for c in name]
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

        generated_count += 1
        yield name

