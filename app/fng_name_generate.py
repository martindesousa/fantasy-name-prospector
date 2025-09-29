import numpy as np
import json
import os
from collections import Counter
from app.fng_model import BigramPenaltyLoss
from app.storage import load_model_from_s3
import tempfile
import tensorflow as tf

# Global cache for trigram data
_trigram_endings = {}


def _normalize_custom_names(names):
    """Ensure each provided custom name has a gender tag and an END token.

    - Strips whitespace
    - Prepends '<N> ' for names that don't start with '<F>', '<M>', or '<N>'
    - Appends explicit END char '¶' if not already present
    Returns the normalized list.
    """
    if not names:
        return []
    normalized = []
    END_CHAR = '¶'
    for n in names:
        if not isinstance(n, str):
            continue
        s = n.strip()
        if not s:
            continue
        if not (s.startswith('<F>') or s.startswith('<M>') or s.startswith('<N>')):
            s = f"<N> {s}"
        if not s.endswith(END_CHAR):
            s = s + END_CHAR
        normalized.append(s)
    return normalized

def load_model_data(model_name='my_model', user_id=None):
    if model_name.startswith('custom'):
        if user_id is None:
            raise ValueError("user_id is required for custom models")

        with tempfile.TemporaryDirectory() as tmpdir:
            keras_path = os.path.join(tmpdir, f"{model_name}.keras")
            json_path = os.path.join(tmpdir, f"{model_name}_data.json")

            # use helper to download and load metadata
            data_dict = load_model_from_s3(user_id, model_name, keras_path, json_path)

            model = tf.keras.models.load_model(
                keras_path,
                custom_objects={'BigramPenaltyLoss': BigramPenaltyLoss}
            )

    else:
        # built-in (non-custom) models still on local disk
        model_path = f'app/models/{model_name}.keras'
        data_path = f'app/models/{model_name}_data.json'

        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'BigramPenaltyLoss': BigramPenaltyLoss}
        )

        with open(data_path, 'r', encoding='utf-8') as file:
            data_dict = json.load(file)

    # Rebuild placeholders
    X_shape = data_dict.get('X_shape') or [1, 20]
    X = np.zeros(tuple(X_shape), dtype=np.int32)
    char_to_idx = data_dict.get('char_to_idx', {})
    idx_to_char_list = data_dict.get('idx_to_char', [])
    idx_to_char = {i: ch for i, ch in enumerate(idx_to_char_list)}
    char_set = data_dict.get('char_set', [])
    bigram_counts = data_dict.get('bigram_counts', {})
    avg_length = int(data_dict.get('avg_length', 6))
    y = None

    return model, X, y, char_to_idx, idx_to_char, char_set, bigram_counts, avg_length

def get_avg_length(model_name):
    """Get average length and whether it's a default or actual average."""
    try:
        if model_name.startswith('custom'):
            data_path = f'app/models/custom/{model_name}_data.json'
        else:
            data_path = f'app/models/{model_name}_data.json'
        with open(data_path, 'r', encoding='utf-8') as file:
            data_dict = json.load(file)

        if 'avg_length' in data_dict:
            return int(data_dict['avg_length']), False
        else:
            return 6, True
    except FileNotFoundError:
        return 6, True

def analyze_training_data(model_name, custom_names):
    """Analyze training data to get gender proportions."""
    gender_stats = {"<F>": [], "<M>": [], "<N>": []}
    
    if custom_names:
        names_to_analyze = _normalize_custom_names(custom_names)
    else:
        # Load from textfile for pretrained models
        textfile_path = os.path.join('app', 'textfiles', f"{model_name}_names.txt")
        try:
            with open(textfile_path, 'r', encoding='utf-8') as f:
                names_to_analyze = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            # Return empty stats if no file found
            return {"<F>": [], "<M>": [], "<N>": []}

    
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
        names_to_analyze = _normalize_custom_names(custom_names)
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

def generate_single_name(model, X, char_to_idx, idx_to_char, gender_probs, first_letter_info, target_length, temperature, valid_trigrams=None, auto_mode=False, avg_length=6, length_mode='average'):
    """Generate a single name using the provided configuration."""
    # Choose gender token
    chosen_gender_token = np.random.choice(gender_probs['tokens'], p=gender_probs['probabilities'])
    
    # Handle prefix vs first letter selection
    # If the model's vocabulary doesn't include the angle-bracket tokens or a space,
    # don't inject them into the starting string because the model cannot represent
    # them and that mismatch can change the initial sampling distribution.
    vocab_has_tokens = all(tok in char_to_idx for tok in ['<', '>', ' '])

    if first_letter_info['use_prefix']:
        # Use the full prefix
        prefix = first_letter_info['prefix']
        formatted_prefix = prefix[0].upper() + prefix[1:].lower() if len(prefix) > 1 else prefix.upper()
        if vocab_has_tokens:
            name = f"{chosen_gender_token} {formatted_prefix}"
        else:
            # Start with prefix only if tokens are missing
            name = formatted_prefix
        prefix_length = len(prefix)
    else:
        # Choose single first letter
        first_letter = np.random.choice(first_letter_info['letters'], p=first_letter_info['probabilities'])
        if vocab_has_tokens:
            name = f"{chosen_gender_token} {first_letter.upper()}"
        else:
            name = first_letter.upper()
        prefix_length = 1

    # Calculate target length accounting for gender token and space (only if present in vocab)
    gender_token_length = len(chosen_gender_token) if vocab_has_tokens else 0
    space_length = 1 if vocab_has_tokens else 0

    # If auto_mode, we'll use heuristics to decide when to stop; otherwise use explicit target
    if not auto_mode:
        # Target total length should be gender token + space + desired name length
        target_full_length = gender_token_length + space_length + target_length

        # For custom length mode, just generate until we hit the target - no end token logic needed
        if length_mode == 'custom':
            while len(name) < target_full_length:
                encoded = [char_to_idx[c] for c in name if c in char_to_idx]
                if not encoded:
                    break
                    
                encoded = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=X.shape[1], padding='pre')
                predictions = model.predict(encoded, verbose=0)[0]
                
                prev_char = name[-1] if name else None
                next_char = sample_next_character(predictions, idx_to_char, temperature, prev_char,
                                                current_name=name, valid_trigrams=valid_trigrams, 
                                                avg_length=avg_length, suppress_end_tokens=True)

                # Skip unwanted characters
                if should_skip_character(next_char, name, chosen_gender_token):
                    continue
                    
                name += next_char
        else:
            # For average length mode, use the original logic with end token awareness
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
                                                    is_final_char=True, current_name=name, valid_trigrams=valid_trigrams, 
                                                    avg_length=avg_length)
                else:
                    # Not the last character - apply hyphen penalties and normal sampling
                    prev_char = name[-1] if name else None
                    next_char = sample_next_character(predictions, idx_to_char, temperature, prev_char,
                                                    position_from_end=chars_remaining, target_length=target_length,
                                                    current_name=name, valid_trigrams=valid_trigrams, 
                                                    avg_length=avg_length)

                # Skip unwanted characters
                if should_skip_character(next_char, name, chosen_gender_token):
                    continue
                    
                name += next_char
    else:
        # Auto mode: rely only on model sampling. Stop when the explicit END char is sampled
        # or when a hard upper bound is reached to avoid runaway generation.
        max_cap = max(12, int(avg_length * 2) + 4)

        while len(name) < (gender_token_length + space_length + max_cap):
            encoded = [char_to_idx[c] for c in name if c in char_to_idx]
            if not encoded:
                break

            encoded = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=X.shape[1], padding='pre')
            predictions = model.predict(encoded, verbose=0)[0]

            prev_char = name[-1] if name else None
            next_char = sample_next_character(predictions, idx_to_char, temperature, prev_char,
                                              position_from_end=None, target_length=None,
                                              current_name=name, valid_trigrams=valid_trigrams, 
                                              avg_length=avg_length, suppress_end_tokens=False)

            if should_skip_character(next_char, name, chosen_gender_token):
                # Skip invalid/undesirable chars but do not apply heuristics beyond that
                continue

            # If model outputs the explicit END character, stop generation
            if next_char == '¶' or next_char == '<END>':
                break

            name += next_char
    
    # Clean and return the name
    return clean_generated_name(name)

def sample_next_character(predictions, idx_to_char, temperature, prev_char=None, capital_penalty=2.5, 
                         position_from_end=None, target_length=None, is_final_char=False,
                         current_name=None, valid_trigrams=None, trigram_penalty=3.0, avg_length=None, 
                         end_boost=0.10, suppress_end_tokens=False):
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

    # Suppress end tokens if we haven't reached target length yet
    if suppress_end_tokens:
        END_CHARS = ['¶', '<END>']  # Add any other end tokens your model might use
        for i in range(len(logits)):
            char = idx_to_char[i]
            if char in END_CHARS:
                logits[i] -= 100.0  # Essentially impossible to select

    # Slightly boost the logit for an explicit END token if present and we've reached avg_length
    # (but only if we're not suppressing end tokens)
    try:
        if not suppress_end_tokens:
            END_CHAR = '¶'
            if avg_length and current_name:
                # compute core name length excluding gender tokens and spaces
                core_name = current_name.replace('<F>', '').replace('<M>', '').replace('<N>', '').replace(' ', '')
                if len(core_name) >= avg_length:
                    # find end char index and boost its logit slightly
                    for i in range(len(logits)):
                        if idx_to_char[i] == END_CHAR:
                            logits[i] += end_boost
                            break
    except Exception:
        pass

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
    cleaned = raw_name.replace('<F>', '').replace('<M>', '').replace('<N>', '').replace('¶','').strip()
    return cleaned if cleaned else None

def generate_quality_names_stream(model_name, count=10, gender='neutral', prefix_text='', length=None, temperature=1.0, min_bigram_count=1, custom_names=None, length_mode='average', user_id=None):
    """Generator that yields unique names one-by-one with guaranteed length and optional bigram filtering."""
    try:
        model, X, y, char_to_idx, idx_to_char, char_set, bigram_counts, avg_length = load_model_data(model_name, user_id=user_id)
    except FileNotFoundError as e:
        print(f"Error loading model data: {e}")
        return
    
    # Prepare configurations
    gender_stats = analyze_training_data(model_name, custom_names)
    valid_trigrams = analyze_trigram_endings(model_name, custom_names)
    gender_probs = calculate_gender_probabilities(gender_stats, gender)
    first_letter_info = prepare_first_letter_distribution(gender_stats, prefix_text, temperature)

    # Determine target_length and auto_mode based on length_mode
    if length_mode == 'custom' and length is not None:
        target_length = int(length)
        auto_mode = False
    elif length_mode == 'average':
        target_length = avg_length
        auto_mode = False
    else:
        target_length = None
        auto_mode = True
    
    generated_names = set()
    yielded = 0
    attempts = 0
    max_attempts = count * 10

    while yielded < count and attempts < max_attempts:
        attempts += 1

        # Generate a single name - pass length_mode to the function
        name = generate_single_name(model, X, char_to_idx, idx_to_char, gender_probs, first_letter_info,
                                   target_length, temperature, valid_trigrams, auto_mode=auto_mode, 
                                   avg_length=avg_length, length_mode=length_mode)

        if not name:
            continue

        if auto_mode:
            # accept any non-empty generated name
            if name not in generated_names:
                generated_names.add(name)
                yielded += 1
                yield name
        else:
            # require exact length match for non-auto modes
            if target_length is not None and len(name) == target_length and name not in generated_names:
                generated_names.add(name)
                yielded += 1
                yield name