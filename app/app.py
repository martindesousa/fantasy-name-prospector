from flask import Flask, request, render_template, Response, stream_with_context
import numpy as np
import tensorflow as tf
import app.fng_model as fng_model
import app.fng_name_generate as fng_name_generate
import json
import time
import threading
import queue
import os

app = Flask(__name__)
application = app


# Shared queue for progress updates
progress_queue = queue.Queue()

# Function to generate a name using the imported method
def generate_name(model, seed_text, length=6, temperature=1.0):
    return fng_name_generate.generate_name(model, seed_text, length, temperature)

# Progress callback function for model training
def progress_callback(current_epoch, total_epochs):
    progress_queue.put({
        'epoch': current_epoch,
        'total': total_epochs
    })

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/stream_progress', methods=['POST'])
def stream_progress():
    """Stream progress updates for model training and name generation"""
    def generate():
        selected_model = request.form['model']
        count = int(request.form['count'])
        temperature = float(request.form['temperature'])
        seed = request.form['seed']
        length = int(request.form['length'])
        epochs = 100
        
        # Clear any old messages from the queue
        while not progress_queue.empty():
            progress_queue.get()
        
        # First, yield a preparation message
        yield f"data: {json.dumps({'type': 'preparing', 'message': 'Preparing to process your request...', 'progress': 0})}\n\n"
        
        # If custom model, train it in a separate thread
        if selected_model == 'custom':
            # Get custom names from the form
            custom_names = request.form['custom_names'].splitlines()
            
            # Send message about loading custom names
            yield f"data: {json.dumps({'type': 'loading', 'message': 'Loading and processing your custom names...', 'progress': 5})}\n\n"
            time.sleep(0.5)
            
            # Load data and create model
            X, y, char_to_idx, idx_to_char, char_set, bigram_counts = fng_model.load_data(input_text="\n".join(custom_names))
            model = fng_model.create_model(X, char_to_idx, idx_to_char, char_set, bigram_counts)

            # Training is about to start message
            yield f"data: {json.dumps({'type': 'training', 'message': 'Starting model training...', 'progress': 10})}\n\n"
            time.sleep(0.5)

            # Create a thread for model training
            callback = fng_model.TrainingProgressCallback(total_epochs=epochs, stream_progress=progress_callback)
            
            def train_thread():
                try:
                    fng_model.train_model(X, y, model, epochs=epochs, batch_size=64, stream_progress=progress_callback)
                    fng_model.save_model_data(model, X, y, char_to_idx, idx_to_char, char_set, model_name='custom')
                    # Put a completion message in the queue
                    progress_queue.put({'epoch': epochs, 'total': epochs, 'complete': True})
                except Exception as e:
                    print(f"Error in training: {e}")
                    progress_queue.put({'error': str(e)})
            
            # Start training thread
            threading.Thread(target=train_thread).start()

            # Stream progress updates for epochs, similar to name generation
            epoch_updates = []
            training_complete = False
            
            # Similar to your name generation loop
            while not training_complete:
                try:
                    update = progress_queue.get(timeout=0.5)
                    
                    # Check if there was an error
                    if 'error' in update:
                        error_msg = f"Error during training: {update['error']}"
                        yield f"data: {json.dumps({'type': 'error', 'message': error_msg, 'progress': 0})}\n\n"
                        break
                        
                    # Check if training is complete
                    if update.get('complete', False):
                        training_complete = True
                        yield f"data: {json.dumps({'type': 'training', 'message': 'Training complete!', 'progress': 100})}\n\n"
                        break
                        
                    # Process regular epoch update
                    current_epoch = update['epoch']
                    total_epochs = update['total']
                    progress = int((current_epoch / total_epochs) * 100)
                    
                    # Stream this epoch update, just like you do with names
                    yield f"data: {json.dumps({'type': 'training', 'message': f'Training model (Epoch {current_epoch}/{total_epochs})', 'progress': progress})}\n\n"
            
                except queue.Empty:
                    # No update within timeout, send heartbeat
                    yield f"data: {json.dumps({'type': 'heartbeat', 'message': 'Setting up model...'})}\n\n"
        
        # Reset progress for name generation
        yield f"data: {json.dumps({'type': 'generating', 'message': 'Starting name generation...', 'progress': 0})}\n\n"

        # Generate names 
        name_stream = fng_name_generate.generate_quality_names_stream(
            model_name=selected_model,
            count=count,
            seed_text=seed,
            length=length,
            temperature=temperature
        )

        generated_names = []
        for i, name in enumerate(name_stream, start=1):
            generated_names.append(name)
            progress = int((i / count) * 100)
            yield f"data: {json.dumps({'type': 'generating', 'message': f'Generated name {i}/{count}', 'progress': progress, 'name': name})}\n\n"

        # Final message
        yield f"data: {json.dumps({'type': 'complete', 'message': 'Complete!', 'progress': 100, 'names': generated_names})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/generate', methods=['POST'])
def generate():
    # This is now just a template rendering route, actual processing happens in stream_progress
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True) #remove debug if not in debug