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

def prepare_generation_config(model_name, custom_names, gender, prefix_text, length, temperature):
    """Prepare all configuration data for generation including gender proportions and length."""
    config = {}
    
    # Get gender proportions and length statistics from training data
    gender_stats, length_stats = analyze_training_data(model_name, custom_names)
    
    config['gender_token_probs'] = calculate_gender_probabilities(gender_stats, gender)
    config['target_length'] = determine_target_length(length_stats, length)
    config['first_letter_info'] = prepare_first_letter_distribution(gender_stats, prefix_text, temperature)
    config['temperature'] = temperature
    
    return config

def analyze_training_data(model_name, custom_names):
    """Analyze training data to get gender proportions and length statistics."""
    gender_stats = {"<F>": [], "<M>": [], "<N>": []}
    
    if custom_names:
        names_to_analyze = custom_names
    else:
        # Load from textfile for pretrained models
        textfile_path = os.path.join('app', 'textfiles', f"{model_name}_names.txt")
        try:
            with open(textfile_path, 'r', encoding='utf-8') as f:
                names_to_analyze = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            # Return default stats if no file found
            return {
                "<F>": ["Anna", "Emma", "Sophie"] * 1000,
                "<M>": ["John", "Mike", "David"] * 1000, 
                "<N>": ["Riley", "Alex", "Jordan"] * 300
            }, {"<F>": [4, 4, 6], "<M>": [4, 4, 5], "<N>": [5, 4, 6]}
    
    # Parse names and categorize by gender
    for name in names_to_analyze:
        name = name.strip()
        if not name:
            continue
            
        if name.startswith('<F>'):
            clean_name = name[3:].strip()
            if clean_name:
                gender_stats["<F>"].append(clean_name)
        elif name.startswith('<M>'):
            clean_name = name[3:].strip()
            if clean_name:
                gender_stats["<M>"].append(clean_name)
        elif name.startswith('<N>'):
            clean_name = name[3:].strip()
            if clean_name:
                gender_stats["<N>"].append(clean_name)
        else:
            # Names without gender tags go to neutral
            gender_stats["<N>"].append(name)
    
    # Calculate length statistics for each gender
    length_stats = {}
    for gender_token, names_list in gender_stats.items():
        if names_list:
            length_stats[gender_token] = [len(name) for name in names_list]
        else:
            length_stats[gender_token] = [6]  # Default length
    
    return gender_stats, length_stats

def calculate_gender_probabilities(gender_stats, gender_preference):
    """Calculate gender token probabilities based on training data and user preference."""
    # Count occurrences of each gender
    gender_counts = {token: len(names) for token, names in gender_stats.items()}
    total = sum(gender_counts.values())
    
    if total == 0:
        base_probs = {"<F>": 0.4, "<M>": 0.4, "<N>": 0.2}
    else:
        base_probs = {token: count / total for token, count in gender_counts.items()}
    
    # Select tokens based on preference
    if gender_preference == "female":
        tokens = ["<F>", "<N>"]
    elif gender_preference == "male":
        tokens = ["<M>", "<N>"]
    else:  # neutral or any other value
        tokens = ["<F>", "<M>", "<N>"]
    
    # Calculate probabilities for selected tokens
    selected_probs = [base_probs[token] for token in tokens]
    total_selected = sum(selected_probs)
    
    if total_selected == 0:
        normalized_probs = [1.0 / len(tokens)] * len(tokens)
    else:
        normalized_probs = [p / total_selected for p in selected_probs]
    
    return {'tokens': tokens, 'probabilities': normalized_probs}

def determine_target_length(length_stats, user_length):
    """Determine target length based on training data or user preference."""
    if user_length is not None and user_length != '':
        return int(user_length)
    
    # Calculate average length across all genders
    all_lengths = []
    for lengths in length_stats.values():
        all_lengths.extend(lengths)
    
    if all_lengths:
        return int(np.mean(all_lengths))
    else:
        return 6  # Default fallback

def prepare_first_letter_distribution(gender_stats, prefix_text, temperature):
    """Prepare first letter distribution with temperature adjustment."""
    if prefix_text:
        return {'use_prefix': True, 'prefix': prefix_text}
    
    # Collect all first letters from training data
    first_letter_counts = Counter()
    for names_list in gender_stats.values():
        for name in names_list:
            if name:
                first_letter_counts[name[0].upper()] += 1
    
    if not first_letter_counts:
        # Fallback to uniform distribution
        letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        probabilities = np.ones(len(letters)) / len(letters)
        return {'use_prefix': False, 'letters': letters, 'probabilities': probabilities}
    
    # Apply temperature to letter probabilities
    letters = list(first_letter_counts.keys())
    counts = np.array([first_letter_counts[char] for char in letters], dtype=np.float32)
    
    letter_logits = np.log(counts + 1e-8) / temperature
    probabilities = np.exp(letter_logits)
    probabilities /= probabilities.sum()
    
    return {'use_prefix': False, 'letters': letters, 'probabilities': probabilities}

def generate_single_name(model, X, char_to_idx, idx_to_char, config):
    """Generate a single name using the provided configuration."""
    # Choose gender token
    gender_info = config['gender_token_probs']
    chosen_gender_token = np.random.choice(gender_info['tokens'], p=gender_info['probabilities'])
    
    # Choose first letter
    letter_info = config['first_letter_info']
    if letter_info['use_prefix']:
        first_letter = letter_info['prefix']
    else:
        first_letter = np.random.choice(letter_info['letters'], p=letter_info['probabilities'])
    
    # Start generation
    name = f"{chosen_gender_token} {first_letter.upper()}"
    target_full_length = len(name) + config['target_length'] - 1  # -1 because first_letter is already included
    
    # Generate characters until target length
    while len(name) < target_full_length:
        encoded = [char_to_idx[c] for c in name if c in char_to_idx]
        if not encoded:
            break
            
        encoded = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=X.shape[1], padding='pre')
        predictions = model.predict(encoded, verbose=0)[0]
        
        # Apply temperature sampling
        next_char = sample_next_character(predictions, idx_to_char, config['temperature'])
        
        # Skip unwanted characters
        if should_skip_character(next_char, name, chosen_gender_token):
            continue
            
        name += next_char
    
    # Clean and return the name
    return clean_generated_name(name)

def sample_next_character(predictions, idx_to_char, temperature):
    """Sample next character using temperature-based sampling."""
    if temperature == 0:
        predicted_index = np.argmax(predictions)
    else:
        logits = np.log(predictions + 1e-8) / temperature
        probs = np.exp(logits)
        probs /= np.sum(probs)
        predicted_index = np.random.choice(len(probs), p=probs)
    
    return idx_to_char[predicted_index]

def should_skip_character(char, current_name, gender_token):
    """Determine if a character should be skipped during generation."""
    # Skip invalid characters
    if char in ['\n', ' ', '', '<PAD>']:
        return True
    
    # Skip gender token characters if we're past the initial token
    if char in ['<', '>', 'F', 'M', 'N'] and len(current_name) > len(gender_token) + 1:
        return True
    
    return False

def clean_generated_name(raw_name):
    """Clean the generated name by removing gender tokens."""
    cleaned = raw_name.replace('<F>', '').replace('<M>', '').replace('<N>', '').strip()
    return cleaned if cleaned else None

def generate_quality_names_stream(model_name, count=10, gender='neutral', prefix_text='', length=None, temperature=1.0, min_bigram_count=1, custom_names=None):
    """Generator that yields unique names one-by-one with guaranteed length and optional bigram filtering."""
    try:
        model, X, y, char_to_idx, idx_to_char, char_set, bigram_counts = load_model_data(model_name)
    except FileNotFoundError as e:
        print(f"Error loading model data: {e}")
        return

    # Prepare all configurations at once
    config = prepare_generation_config(model_name, custom_names, gender, prefix_text, length, temperature)
    
    generated_names = set()
    yielded = 0
    attempts = 0
    max_attempts = count * 10

    while yielded < count and attempts < max_attempts:
        attempts += 1

        # Generate a single name
        name = generate_single_name(model, X, char_to_idx, idx_to_char, config)
        
        if name and len(name) == config['target_length'] and name not in generated_names:
            generated_names.add(name)
            yielded += 1
            yield name