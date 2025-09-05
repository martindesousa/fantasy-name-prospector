import numpy as np
import tensorflow as tf
import pickle
import os
from collections import Counter
from app.fng_model import BigramPenaltyLoss

# Global cache for trigram data
_trigram_endings = {}

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
    avg_length = data_dict.get('avg_length', 6)
    
    return model, X, y, char_to_idx, idx_to_char, char_set, bigram_counts, avg_length

def get_avg_length(model_name):
    """Get average length and whether it's a default or actual average."""
    try:
        if model_name.startswith('custom'):
            data_path = f'app/models/custom/{model_name}_data.pkl'
        else:
            data_path = f'app/models/{model_name}_data.pkl'
        
        with open(data_path, 'rb') as file:
            data_dict = pickle.load(file)
        
        # Check if avg_length exists in the data
        if 'avg_length' in data_dict:
            return data_dict['avg_length'], False  # Actual length
        else:
            return 6, True  # Default fallback
            
    except FileNotFoundError:
        return 6, True  # Default fallback

def analyze_training_data(model_name, custom_names):
    """Analyze training data to get gender proportions."""
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
            default_stats = {
                "<F>": ["Anna", "Emma", "Sophie"] * 1000,
                "<M>": ["John", "Mike", "David"] * 1000, 
                "<N>": ["Riley", "Alex", "Jordan"] * 300
            }
            return default_stats

    
    # Parse names and categorize by gender
    for name in names_to_analyze:
        name = name.strip()
        if not name:
            continue
            
        clean_name = None

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

    return gender_stats

def analyze_trigram_endings(model_name, custom_names):
    """Analyze training data to extract valid ending trigrams."""
    cache_key = f"{model_name}_trigrams"
    
    if cache_key in _trigram_endings:
        return _trigram_endings[cache_key]
    
    ending_trigrams = set()
    
    if custom_names:
        names_to_analyze = custom_names
    else:
        # Load from textfile for pretrained models
        textfile_path = os.path.join('app', 'textfiles', f"{model_name}_names.txt")
        try:
            with open(textfile_path, 'r', encoding='utf-8') as f:
                names_to_analyze = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            # Return no trigrams for fallback
            return {}
    
    # Extract ending trigrams from all names
    for name in names_to_analyze:
        name = name.strip()
        if not name:
            continue
        
        # Remove gender tokens
        clean_name = name.replace('<F>', '').replace('<M>', '').replace('<N>', '').strip()
        
        # Extract ending trigram (last 3 characters)
        if len(clean_name) >= 3:
            ending_trigram = clean_name[-3:].lower()
            ending_trigrams.add(ending_trigram)
    
    _trigram_endings[cache_key] = ending_trigrams
    return ending_trigrams

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

def calculate_hyphen_penalty(current_pos, target_length, base_penalty=5.0):
    """Calculate hyphen penalty based on position relative to end of name."""
    chars_from_end = target_length - current_pos
    
    if chars_from_end <= 1:  # Last character
        return 100.0  # Essentially impossible
    elif chars_from_end == 2:  # Second to last
        return base_penalty * 8.0  # Very high penalty
    elif chars_from_end == 3:  # Third to last
        return base_penalty * 2.0
    else:
        return base_penalty  # Normal penalty for middle of name

def calculate_trigram_penalty(current_name, candidate_char, valid_trigrams, trigram_penalty=20.0):
    """Calculate penalty for characters that would create invalid ending trigrams."""
    if len(current_name) < 2:
        return 0.0  # Need at least 2 chars to form a trigram
    
    # Get the last 2 characters and combine with candidate to form trigram
    last_two = current_name[-2:]
    potential_trigram = (last_two + candidate_char).lower()
    
    # Remove any gender tokens or spaces from the trigram check
    clean_trigram = potential_trigram.replace('<', '').replace('>', '').replace(' ', '')
    
    if len(clean_trigram) >= 3:
        ending_trigram = clean_trigram[-3:]
        if ending_trigram not in valid_trigrams:
            return trigram_penalty
    
    return 0.0

def generate_single_name(model, X, char_to_idx, idx_to_char, gender_probs, first_letter_info, target_length, temperature, valid_trigrams=None):
    """Generate a single name using the provided configuration."""
    # Choose gender token
    chosen_gender_token = np.random.choice(gender_probs['tokens'], p=gender_probs['probabilities'])
    
    # Handle prefix vs first letter selection
    if first_letter_info['use_prefix']:
        # Use the full prefix
        prefix = first_letter_info['prefix']
        formatted_prefix = prefix[0].upper() + prefix[1:].lower() if len(prefix) > 1 else prefix.upper()
        name = f"{chosen_gender_token} {formatted_prefix}"
        prefix_length = len(prefix)
    else:
        # Choose single first letter
        first_letter = np.random.choice(first_letter_info['letters'], p=first_letter_info['probabilities'])
        name = f"{chosen_gender_token} {first_letter.upper()}"
        prefix_length = 1

    # Calculate target length accounting for gender token, space, and prefix
    gender_token_length = len(chosen_gender_token)  # e.g., "<F>" = 3 chars
    space_length = 1
    
    # Target total length should be gender token + space + desired name length
    target_full_length = gender_token_length + space_length + target_length
    
    # Generate characters until full target length with smart ending logic
    while len(name) < target_full_length:
        encoded = [char_to_idx[c] for c in name if c in char_to_idx]
        if not encoded:
            break
            
        encoded = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=X.shape[1], padding='pre')
        predictions = model.predict(encoded, verbose=0)[0]
        
        chars_remaining = target_full_length - len(name)
        
        # Apply different logic based on position
        if chars_remaining == 1:
            # Last character - apply trigram validation
            prev_char = name[-1] if name else None
            next_char = sample_next_character(predictions, idx_to_char, temperature, prev_char, 
                                            is_final_char=True, current_name=name, valid_trigrams=valid_trigrams)
        else:
            # Not the last character - apply hyphen penalties and normal sampling
            prev_char = name[-1] if name else None
            next_char = sample_next_character(predictions, idx_to_char, temperature, prev_char,
                                            position_from_end=chars_remaining, target_length=target_length,
                                            current_name=name, valid_trigrams=valid_trigrams)
        
        # Skip unwanted characters
        if should_skip_character(next_char, name, chosen_gender_token):
            continue
            
        name += next_char
    
    # Clean and return the name
    return clean_generated_name(name)

def sample_next_character(predictions, idx_to_char, temperature, prev_char=None, capital_penalty=2.0, 
                         position_from_end=None, target_length=None, is_final_char=False,
                         current_name=None, valid_trigrams=None, trigram_penalty=3.0):
    """Sampling with capital letter penalties, position-aware penalties, and trigram validation."""
    
    # Standard character sampling with penalties
    if temperature == 0:
        logits = np.log(predictions + 1e-8)
    else:
        logits = np.log(predictions + 1e-8) / temperature

    # Apply capital letter penalty (avoid capitals mid-name)
    for i in range(len(logits)):
        char = idx_to_char[i]
        if char.isupper() and prev_char not in (None, '-', '<', '>', ' '):
            logits[i] -= capital_penalty
    
    # Apply hyphen penalties based on position
    if position_from_end is not None and target_length is not None:
        hyphen_chars = ['-']
        current_pos = target_length - position_from_end + 1
        hyphen_penalty = calculate_hyphen_penalty(current_pos, target_length)
        
        for i in range(len(logits)):
            char = idx_to_char[i]
            if char in hyphen_chars:
                logits[i] -= hyphen_penalty
    
    # Apply trigram penalty for final characters
    if valid_trigrams and current_name and is_final_char:
        for i in range(len(logits)):
            char = idx_to_char[i]
            penalty = calculate_trigram_penalty(current_name, char, valid_trigrams, trigram_penalty)
            if penalty > 0:
                logits[i] -= penalty

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
        model, X, y, char_to_idx, idx_to_char, char_set, bigram_counts, avg_length = load_model_data(model_name)
    except FileNotFoundError as e:
        print(f"Error loading model data: {e}")
        return

    # Prepare configurations
    gender_stats = analyze_training_data(model_name, custom_names)
    valid_trigrams = analyze_trigram_endings(model_name, custom_names)
    gender_probs = calculate_gender_probabilities(gender_stats, gender)
    first_letter_info = prepare_first_letter_distribution(gender_stats, prefix_text, temperature)

    # Use provided length or fall back to model's average length
    target_length = int(length) if length is not None and length != '' else avg_length
    
    generated_names = set()
    yielded = 0
    attempts = 0
    max_attempts = count * 10

    while yielded < count and attempts < max_attempts:
        attempts += 1

        # Generate a single name
        name = generate_single_name(model, X, char_to_idx, idx_to_char, gender_probs, first_letter_info, 
                                   target_length, temperature, valid_trigrams)
        
        if name and len(name) == target_length and name not in generated_names:
            generated_names.add(name)
            yielded += 1
            yield name