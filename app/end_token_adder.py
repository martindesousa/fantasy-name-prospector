import os

def add_end_tokens_to_file(input_file_path, output_file_path=None):
    """
    Add <!> tokens to training data file if not already present.
    
    Args:
        input_file_path: Path to the input training file
        output_file_path: Path for output file (optional, defaults to input_file_with_end.txt)
    
    Returns:
        Path to the updated file
    """
    
    # Set default output path if not provided
    if output_file_path is None:
        output_file_path = input_file_path
    
    try:
        # Load existing training names
        with open(input_file_path, 'r', encoding='utf-8') as f:
            original_names = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Training file not found: {input_file_path}")
        return None
    
    # Process names: add <!> only if not already there
    processed_names = []
    unchanged_count = 0
    added_count = 0
    
    for name in original_names:
        name = name.strip()
        if not name:
            continue
            
        if name.endswith('<!>'):
            # Already has end token
            processed_names.append(name)
            unchanged_count += 1
        else:
            # Add end token
            processed_names.append(name + '<!>')
            added_count += 1
    
    # Save updated training data
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for name in processed_names:
            f.write(name + '\n')
    
    print(f"End token processing complete:")
    print(f"  Input file: {input_file_path}")
    print(f"  Output file: {output_file_path}")
    print(f"  Total names: {len(processed_names)}")
    print(f"  Already had <!>: {unchanged_count}")
    print(f"  Added <!>: {added_count}")
    
    return output_file_path

def add_end_tokens_to_model_file(model_name, app_path='app'):
    """
    Convenience function to add end tokens to a model's training file.
    
    Args:
        model_name: Name of the model
        app_path: Path to the app directory (default: 'app')
    
    Returns:
        Path to the updated file
    """
    input_file_path = os.path.join(app_path, 'textfiles', f"{model_name}_names.txt")
    output_file_path = os.path.join(app_path, 'textfiles', f"{model_name}_names.txt")
    
    return add_end_tokens_to_file(input_file_path, output_file_path)

def batch_process_files(directory_path, pattern="*_names.txt"):
    """
    Process all training files in a directory to add end tokens.
    
    Args:
        directory_path: Path to directory containing training files
        pattern: File pattern to match (default: "*_names.txt")
    
    Returns:
        List of processed file paths
    """
    import glob
    
    pattern_path = os.path.join(directory_path, pattern)
    input_files = glob.glob(pattern_path)
    
    processed_files = []
    
    for input_file in input_files:
        print(f"\nProcessing: {input_file}")
        output_file = add_end_tokens_to_file(input_file)
        if output_file:
            processed_files.append(output_file)
    
    print(f"\nBatch processing complete. Processed {len(processed_files)} files.")
    return processed_files

# Example usage functions
if __name__ == "__main__":
    option = input(
        "Type '1' if you would like to process a single text file with an associated model, "
        "'2' if you would like to process all text files in the textfiles directory: "
    )

    if option == '1':  # Option 1: Process a single model's file
        model = input("Type your associated model (e.g., 'american', 'chinese'): ")
        add_end_tokens_to_model_file(model)

    elif option == '2':  # Option 2: Process all files in a directory
        batch_process_files("app/textfiles/")