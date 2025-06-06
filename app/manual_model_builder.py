import os
import fng_model as fng_model

    # Main function that prompts the user for the model name and file name, then trains and saves the model
def main():

    model_name = input("Enter the name for the model (e.g., 'american'): ")
    model_path = os.path.join('models', f'{model_name}.keras')
    
    text_file = input("Enter the name of the text file contain names (e.g., 'american_names.txt'): ")
    text_file_path = os.path.join('app', 'textfiles', f'{text_file}')

    # Ensure the text file exists
    if not os.path.exists(text_file_path):
        print(text_file_path)
        print(f"Error: The text file '{text_file}' was not found.")
        return

    # Handle data and return machine-learning values
    X, y, char_to_idx, idx_to_char, char_set, bigram_counts = fng_model.load_data(input_file=text_file_path)

    # Create and compile the model
    model = fng_model.create_model(X, char_to_idx, idx_to_char, char_set, bigram_counts)

    # Train the model
    print(f"Training the '{model_name}' model...")
    fng_model.train_model(X, y, model)

    # Save the model and associated data
    print(f"Saving the {model_name} model and data...")
    fng_model.save_model_data(model, X, y, char_to_idx, idx_to_char, char_set, bigram_counts, model_name=model_name)

    print(f"Model '{model_name}' has been successfully trained and saved!")

if __name__ == "__main__":
    main()