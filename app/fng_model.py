import os
import numpy as np
import tensorflow as tf
import pickle
from collections import Counter

# HELPER METHODS FOR load_data #############################################################################
def load_names(input_text=None, input_file=None):
    if input_text:
        names = input_text.splitlines()  # If input_text is provided, split it into names
    elif input_file and os.path.exists(input_file):
        with open(input_file, 'r', encoding='utf-8') as file: # Open the file with UTF-8 encoding to handle special characters correctly
            names = file.read().splitlines()
    else:
        raise ValueError("Must provide either input_text or input_file.")
    return names

def create_char_mappings(names):
    char_set = sorted(set(''.join(names)))
    char_to_idx = {char: idx for idx, char in enumerate(char_set)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char, char_set

def prepare_training_data(names, char_to_idx):
    sequences = [] 
    for name in names:
        seq = [char_to_idx[char] for char in name]
        sequences.append(seq)
    X = []
    y = []
    for seq in sequences:
        for i in range(1, len(seq)):
            X.append(seq[:i])
            y.append(seq[i])
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

def create_bigram_penalty_function(char_to_idx, idx_to_char, bigram_counts, min_count=5):
    """
    Creates a simpler bigram penalty function that can be used during inference
    """
    # Pre-compute rare bigrams set for faster lookup
    rare_bigrams = {bigram for bigram, count in bigram_counts.items() if count < min_count}
    
    def is_rare_bigram(char1, char2):
        """Check if a bigram is rare"""
        bigram = char1 + char2
        return bigram in rare_bigrams
    
    return is_rare_bigram

def load_data(input_text=None, input_file=None):
    names = load_names(input_text, input_file)  # Load data from string or file
    char_to_idx, idx_to_char, char_set = create_char_mappings(names)  # Create character mappings
    X, y = prepare_training_data(names, char_to_idx)  # Prepare the training data
    bigram_counts = get_bigram_counts(names)  # Compute bigram counts
    return X, y, char_to_idx, idx_to_char, char_set, bigram_counts

# Create and compile the model
def create_model(X, char_to_idx, idx_to_char, char_set, bigram_counts):
    # Enhanced model with bidirectional LSTM for better performance
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=len(char_set), output_dim=64, input_length=X.shape[1]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=True)),
        tf.keras.layers.Dropout(0.2),  # Add dropout for regularization
        tf.keras.layers.LSTM(100),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(len(char_set), activation='softmax')
    ])
    
    # Using standard categorical crossentropy - simpler and more effective
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    return model

# Train the model with early stopping to prevent overfitting
def train_model(X, y, model, epochs=100, batch_size=64, stream_progress=None):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        restore_best_weights=True
    )

    callbacks = [early_stopping] #Start with early stopping
    if stream_progress:
        callbacks.append(TrainingProgressCallback(total_epochs=epochs, stream_progress=stream_progress))
    
    model.fit(
        X, y, 
        epochs=epochs, 
        batch_size=batch_size,
        callbacks=callbacks
    )

def save_model_data(model, X, y, char_to_idx, idx_to_char, char_set, model_name='my_model'):
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model.save(f'app/models/{model_name}.keras')
    
    # Save data needed for generation
    data_dict = {
        'X': X,
        'y': y,
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'char_set': char_set,
        'bigram_counts': get_bigram_counts(' '.join(idx_to_char[i] for i in range(len(idx_to_char))))
    }
    
    with open(f'app/models/{model_name}_data.pkl', 'wb') as file:
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