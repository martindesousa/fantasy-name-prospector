import os
import numpy as np
import tensorflow as tf
import keras
import pickle
from collections import Counter

# HELPER METHODS FOR load_data #############################################################################
def load_names(input_text=None, input_file=None):
    if input_text:
        lines = input_text.splitlines()  # If input_text is provided, split it into names
    elif input_file and os.path.exists(input_file):
        with open(input_file, 'r', encoding='utf-8') as file: # Open the file with UTF-8 encoding to handle special characters correctly
            lines = file.read().splitlines()
    else:
        raise ValueError("Must provide either input_text or input_file.")
    
    # Parse gender-tagged names
    names = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for gender tags
        if line.startswith('<F>') or line.startswith('<M>') or line.startswith('<N>'):
            # Keep the gender tag as part of the name for training
            names.append(line)
        else:
            # Automatically add <N> tag for names without gender tags
            names.append(f"<N> {line}")
    
    return names

def create_char_mappings(names):
    # Extract all characters including gender tokens
    all_text = ''.join(names)
    char_set = sorted(set(all_text))
    
    # Ensure gender tokens are properly handled as individual characters
    # The angle brackets and letters will be in char_set automatically
    
    char_to_idx = {char: idx for idx, char in enumerate(char_set)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char, char_set

def prepare_training_data(names, char_to_idx):
    sequences = [] 
    for name in names:
        # Convert each character (including those in gender tags) to indices
        seq = [char_to_idx[char] for char in name]
        if seq:  # Only add non-empty sequences
            sequences.append(seq)
        sequences.append(seq)

    X = []
    y = []
    for seq in sequences:
        for i in range(1, len(seq)):
            X.append(seq[:i])
            y.append(seq[i])

    if not X:  # Handle empty data
        raise ValueError("No valid training sequences generated")
    
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max(len(seq) for seq in sequences), padding='pre')
    y = tf.keras.utils.to_categorical(y, num_classes=len(char_to_idx))
    return X, y

# HELPER METHODS FOR create_model ##########################################################################
def get_bigram_counts(names):
    """
    Counts the frequency of each bigram in the dataset.
    """
    bigrams = []
    for name in names:
        for i in range(len(name) - 1):
            bigrams.append(name[i:i+2])
    
    # Count bigram frequencies
    bigram_counts = Counter(bigrams)
    return bigram_counts

@keras.saving.register_keras_serializable()
class BigramPenaltyLoss:
    def __init__(self, char_to_idx=None, idx_to_char=None, bigram_counts=None, min_count=5, penalty_weight=0.5):
        self.penalty_weight = penalty_weight
        self.min_count = min_count

        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char

        if char_to_idx and idx_to_char and bigram_counts:
            self._build_penalty_table(char_to_idx, idx_to_char, bigram_counts)
        else:
            self.table = None
            self.idx_chars = None

    def _build_penalty_table(self, char_to_idx, idx_to_char, bigram_counts):
        all_bigrams = [a + b for a in char_to_idx for b in char_to_idx]
        penalty_dict = {
            bg: 1.0 if bigram_counts.get(bg, 0) < self.min_count else 0.0
            for bg in all_bigrams
        }

        keys_tensor = tf.constant(list(penalty_dict.keys()))
        values_tensor = tf.constant(list(penalty_dict.values()), dtype=tf.float32)

        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor),
            default_value=0.0
        )

        self.idx_chars = tf.constant([idx_to_char[i] for i in range(len(idx_to_char))])

    def __call__(self, y_true, y_pred):
        base_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        if self.table is None or self.idx_chars is None:
            return base_loss  # No penalty if data not injected

        pred_indices = tf.argmax(y_pred, axis=-1)
        prev_indices = tf.pad(pred_indices, [[1, 0]], constant_values=0)[:-1]

        prev_chars = tf.gather(self.idx_chars, prev_indices)
        curr_chars = tf.gather(self.idx_chars, pred_indices)

        bigrams = tf.strings.join([prev_chars, curr_chars])
        penalties = self.table.lookup(bigrams)

        return base_loss + self.penalty_weight * penalties

    def get_config(self):
        return {
            "penalty_weight": self.penalty_weight,
            "min_count": self.min_count
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
def get_avg_length(names):
    # Calculate average length (without gender tokens)
    clean_lengths = [
        len(name.replace('<F>', '').replace('<M>', '').replace('<N>', '').strip())
        for name in names
    ]
    avg_length = int(round(np.mean(clean_lengths))) if clean_lengths else 6
    return avg_length
    

def load_data(input_text=None, input_file=None):
    names = load_names(input_text, input_file)  # Load data from string or file
    char_to_idx, idx_to_char, char_set = create_char_mappings(names)  # Create character mappings
    X, y = prepare_training_data(names, char_to_idx)  # Prepare the training data
    bigram_counts = get_bigram_counts(names)  # Compute bigram counts
    avg_length = get_avg_length(names) # Compute average length
    return X, y, char_to_idx, idx_to_char, char_set, bigram_counts, avg_length

# Create and compile the model
def create_model(X, char_to_idx, idx_to_char, char_set, bigram_counts):
    # Enhanced model with bidirectional LSTM for better performance
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=len(char_set), output_dim=64, input_length=X.shape[1]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001))),
        tf.keras.layers.Dropout(0.2),  # Add dropout for regularization
        tf.keras.layers.LSTM(100),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(len(char_set), activation='softmax')
    ])
    
    # Using standard categorical crossentropy 
    model.compile(
        loss=BigramPenaltyLoss(char_to_idx, idx_to_char, bigram_counts),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    return model

# Train the model with early stopping to prevent overfitting
def train_model(X, y, model, epochs=50, batch_size=64, stream_progress=None):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # callbacks = [early_stopping] # Use if want early_stopping
    callbacks = [] # Use if don't want early_stopping
    if stream_progress:
        callbacks.append(TrainingProgressCallback(total_epochs=epochs, stream_progress=stream_progress))
    
    model.fit(
        X, y, 
        epochs=epochs, 
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.1,
        callbacks=callbacks
    )

def save_model_data(model, X, y, char_to_idx, idx_to_char, char_set, bigram_counts, avg_length, model_name='my_model'):
    # Determine save path
    base_dir = os.path.join('app', 'models', 'custom' if model_name.startswith('custom') else '')
    os.makedirs(base_dir, exist_ok=True)
    
    path = os.path.join(base_dir, model_name)

    # Save the model
    model.save(path + '.keras')

    

    # Prepare and save additional data
    data_dict = {
        'X': X,
        'y': y,
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'char_set': char_set,
        'bigram_counts': bigram_counts,
        'avg_length': avg_length
    }

    with open(path + '_data.pkl', 'wb') as file:
        pickle.dump(data_dict, file)

class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, stream_progress):
        super().__init__()
        self.total_epochs = total_epochs
        self.stream_progress = stream_progress
        self.epochs_completed = []  # Store completed epochs

    def on_epoch_end(self, epoch, logs=None):
        # Current epoch (1-based)
        current_epoch = epoch + 1
        # Add to completed epochs
        self.epochs_completed.append(current_epoch)
        # Call the progress callback
        self.stream_progress(current_epoch, self.total_epochs)

    def get_completed_epochs(self):
        return self.epochs_completed