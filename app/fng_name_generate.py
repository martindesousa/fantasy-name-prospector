import numpy as np
import tensorflow as tf
import pickle
import os
from collections import Counter
from app.fng_model import BigramPenaltyLoss

# Global cache for ending distributions
_ending_distributions = {}

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
    """Analyze training data to get gender proportions and ending patterns."""
    gender_stats = {"<F>": [], "<M>": [], "<N>": []}
    gender_ending_chars = {"<F>": Counter(), "<M>": Counter(), "<N>": Counter()}
    
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
            default_endings = {
                "<F>": Counter({'a': 3000, 'e': 1500, 'n': 500, 'h': 300, 'y': 200}),
                "<M>": Counter({'n': 2000, 'e': 1500, 'r': 800, 'd': 600, 'l': 400, 's': 300}),
                "<N>": Counter({'n': 600, 'r': 300, 'l': 200, 'y': 150, 'x': 100})
            }
            return default_stats, default_endings

    
    # Parse names and categorize by gender, collect ending characters
    for name in names_to_analyze:
        name = name.strip()
        if not name:
            continue
            
        clean_name = None
        gender_token = None

        if name.startswith('<F>'):
            clean_name = name[3:].strip()
            gender_token = "<F>"
            if clean_name:
                gender_stats["<F>"].append(clean_name)
        elif name.startswith('<M>'):
            clean_name = name[3:].strip()
            gender_token = "<M>"
            if clean_name:
                gender_stats["<M>"].append(clean_name)
        elif name.startswith('<N>'):
            clean_name = name[3:].strip()
            gender_token = "<N>"
            if clean_name:
                gender_stats["<N>"].append(clean_name)
        else:
            # Names without gender tags go to neutral
            clean_name = name
            gender_stats["<N>"].append(name)
        
        # Collect ending characters for natural ending distribution
        if clean_name and len(clean_name) > 0 and gender_token:
            gender_ending_chars[gender_token][clean_name[-1].lower()] += 1

    return gender_stats, gender_ending_chars

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

def prepare_ending_distribution(gender_ending_chars, selected_genders, char_to_idx, temperature):
    """Prepare gender-specific ending character distribution with caching."""
    # Create cache key based on selected genders, character vocab, and temperature
    gender_key = "_".join(sorted(selected_genders))
    vocab_key = hash(tuple(sorted(char_to_idx.keys())))
    cache_key = f"{gender_key}_{vocab_key}_{temperature}"
    
    if cache_key in _ending_distributions:
        return _ending_distributions[cache_key]
    
    # Combine ending characters from selected genders
    combined_endings = Counter()
    for gender in selected_genders:
        if gender in gender_ending_chars:
            combined_endings.update(gender_ending_chars[gender])
    
    if not combined_endings:
        # Fallback based on selected genders
        if "<F>" in selected_genders and "<M>" not in selected_genders:
            # Female-only fallback
            combined_endings = Counter({'a': 25, 'e': 20, 'h': 10, 'n': 8, 'y': 8, 'l': 5, 'r': 4})
        elif "<M>" in selected_genders and "<F>" not in selected_genders:
            # Male-only fallback
            combined_endings = Counter({'n': 20, 'e': 15, 'r': 12, 'd': 10, 'l': 8, 's': 8, 't': 5, 'h': 2})
        else:
            # Mixed or neutral fallback
            combined_endings = Counter({'a': 20, 'e': 18, 'n': 15, 'r': 10, 'l': 8, 's': 8, 'h': 6, 'y': 6, 't': 4, 'd': 3, 'x': 2})
    
    # Filter to only characters in our vocabulary
    valid_endings = {char: count for char, count in combined_endings.items() if char in char_to_idx}
    
    if not valid_endings:
        _ending_distributions[cache_key] = None
        return None
    
    # Create probability distribution
    chars = list(valid_endings.keys())
    counts = np.array([valid_endings[char] for char in chars], dtype=np.float32)
    
    # Apply temperature
    logits = np.log(counts + 1e-8) / temperature
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    
    result = {'chars': chars, 'probabilities': probabilities, 'char_indices': [char_to_idx[c] for c in chars]}
    _ending_distributions[cache_key] = result
    return result

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

def generate_single_name(model, X, char_to_idx, idx_to_char, gender_probs, first_letter_info, target_length, temperature, ending_distribution=None):
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
            # Last character - use 70/30 blend if ending distribution exists, otherwise normal sampling
            prev_char = name[-1] if name else None
            next_char = sample_next_character(predictions, idx_to_char, temperature, prev_char, 
                                            is_final_char=True, ending_distribution=ending_distribution)
        else:
            # Not the last character - apply hyphen penalties and normal sampling
            prev_char = name[-1] if name else None
            next_char = sample_next_character(predictions, idx_to_char, temperature, prev_char,
                                            position_from_end=chars_remaining, target_length=target_length)
        
        # Skip unwanted characters
        if should_skip_character(next_char, name, chosen_gender_token):
            continue
            
        name += next_char
    
    # Clean and return the name
    return clean_generated_name(name)

def sample_next_character(predictions, idx_to_char, temperature, prev_char=None, capital_penalty=2.0, 
                         position_from_end=None, target_length=None, is_final_char=False, ending_distribution=None):
    """Sampling with capital letter penalties, position-aware penalties, and ending logic."""
    
    if is_final_char and ending_distribution:
        # For final character, strongly prefer natural endings
        ending_probs = np.zeros_like(predictions)
        for i, char_idx in enumerate(ending_distribution['char_indices']):
            if char_idx < len(ending_probs):
                ending_probs[char_idx] = ending_distribution['probabilities'][i]

        # Blend with model predictions (40% ending distribution, 60% model)
        blended_probs = 0.1 * ending_probs + 0.9 * predictions
        blended_probs /= np.sum(blended_probs)
        
        if temperature == 0:
            predicted_index = np.argmax(blended_probs)
        else:
            logits = np.log(blended_probs + 1e-8) / temperature
            probs = np.exp(logits)
            probs /= np.sum(probs)
            predicted_index = np.random.choice(len(probs), p=probs)
    else:
        # Normal character sampling with penalties
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
    gender_stats, gender_ending_chars = analyze_training_data(model_name, custom_names)
    gender_probs = calculate_gender_probabilities(gender_stats, gender)
    first_letter_info = prepare_first_letter_distribution(gender_stats, prefix_text, temperature)

    selected_genders = gender_probs['tokens']
    ending_distribution = prepare_ending_distribution(gender_ending_chars, selected_genders, char_to_idx, temperature)

    # Use provided length or fall back to model's average length
    target_length = int(length) if length is not None and length != '' else avg_length
    
    generated_names = set()
    yielded = 0
    attempts = 0
    max_attempts = count * 10

    while yielded < count and attempts < max_attempts:
        attempts += 1

        # Generate a single name
        name = generate_single_name(model, X, char_to_idx, idx_to_char, gender_probs, first_letter_info, target_length, temperature, ending_distribution)
        
        if name and len(name) == target_length and name not in generated_names:
            generated_names.add(name)
            yielded += 1
            yield name