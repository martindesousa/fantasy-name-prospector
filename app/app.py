from flask import Flask, request, render_template, Response, stream_with_context, jsonify, make_response
import numpy as np
import tensorflow as tf
import app.fng_model as fng_model
import app.fng_name_generate as fng_name_generate
import json
import time
import threading
import queue
import os
import hashlib
import logging
from app import storage

MODEL_DIR = 'app/models'
CUSTOM_MODEL_DIR = 'app/models/custom'

app = Flask(__name__)
application = app

# Shared queue for progress updates
progress_queue = queue.Queue()

# Function to generate a name
def generate_name(model, prefix_text, length=6, temperature=1.0):
    return fng_name_generate.generate_name(model, prefix_text, length, temperature)

# Progress callback function for model training
def progress_callback(current_epoch, total_epochs):
    progress_queue.put({
        'epoch': current_epoch,
        'total': total_epochs
    })

@app.route('/')
def home():
    # Get default avg_length from a default model (or set a fallback)
    try:
        avg_length = fng_name_generate.get_avg_length('american_classic') 
    except:
        avg_length = 6  # fallback
    
    return render_template('index.html', avg_length=avg_length)

@app.route('/get_model_avg_length')
def get_model_avg_length():
    """API endpoint to get average length for a specific model"""
    model_name = request.args.get('model', 'my_model')
    logger = logging.getLogger(__name__)
    logger.info("get_model_avg_length called for model=%s", model_name)
    print(f"[app] get_model_avg_length called for model={model_name}")

    # Custom id (e.g. custom_<hash>), therefore we need
    # to pass the user's id for the S3 lookup
    if model_name.startswith('custom'):
        try:
            try:
                user_id, cookie_resp = storage.get_user_id()
            except Exception:
                user_id = request.cookies.get('user_id') or None
                cookie_resp = None

            logger.info('Looking up avg_length for custom model %s (user=%s)', model_name, user_id)
            print(f"[app] resolved user_id={user_id} for model={model_name}")
            avg_length, is_default = fng_name_generate.get_avg_length(model_name, user_id=user_id)
            print(f"[app] get_avg_length returned avg_length={avg_length} is_default={is_default} for model={model_name} user={user_id}")
            resp = jsonify({'avg_length': avg_length, 'is_default': is_default})
            if cookie_resp:
                resp.set_cookie('user_id', user_id, max_age=60*60*24*30)
            return resp
        except Exception:
            return jsonify({'avg_length': 6, 'is_default': True})

    # Non-custom models (templates) - normal path
    try:
        avg_length, is_default = fng_name_generate.get_avg_length(model_name)
        return jsonify({'avg_length': avg_length, 'is_default': is_default})
    except Exception:
        return jsonify({'avg_length': 6, 'is_default': True})

@app.route('/check_model_exists')
def check_model_exists():
    model_name = request.args.get('model')
    model_path = os.path.join(CUSTOM_MODEL_DIR, f"{model_name}.keras")
    exists = os.path.isfile(model_path)
    return jsonify({'exists': exists})


@app.route('/api/custom_models', methods=['GET'])
def api_list_custom_models():
    """Return list of custom model metadata for current user from S3."""
    try:
        user_id, cookie_resp = storage.get_user_id()
    except Exception:
        # if storage.get_user_id expects a request context
        user_id = request.cookies.get('user_id') or None
        cookie_resp = None

    models = []
    try:
        models = storage.list_user_models(user_id)
        # strip heavy trainingData fields and normalize timestamps
        stripped = []
        for meta in models:
            if not isinstance(meta, dict):
                continue
            m = dict(meta)
            if 'trainingData' in m:
                try:
                    del m['trainingData']
                except Exception:
                    m['trainingData'] = None
            if 'createdAt' not in m or not m.get('createdAt'):
                m['createdAt'] = int(time.time() * 1000)
            stripped.append(m)
        models = stripped
    except Exception as e:
        print('Error listing models from S3:', e)

    response = jsonify({'models': models})
    if cookie_resp:
        # set cookie if get_user_id created one
        response.set_cookie('user_id', user_id, max_age=60*60*24*30)
    return response


@app.route('/api/custom_models/<model_id>', methods=['GET'])
def api_get_custom_model(model_id):
    """Return full metadata for a single custom model (including trainingData)."""
    try:
        user_id, cookie_resp = storage.get_user_id()
    except Exception:
        user_id = request.cookies.get('user_id') or None
        cookie_resp = None

    try:
        meta = storage.get_model_metadata_from_s3(user_id, model_id)
    except Exception:
        return jsonify({'error': 'Model not found'}), 404

    # Ensure id and reasonable fields
    meta['id'] = meta.get('id') or model_id
    if 'name' not in meta or not meta.get('name'):
        meta['name'] = meta['id']
    if 'createdAt' not in meta or not meta.get('createdAt'):
        meta['createdAt'] = int(time.time() * 1000)

    response = jsonify({'meta': meta})
    if cookie_resp:
        response.set_cookie('user_id', user_id, max_age=60*60*24*30)
    return response


@app.route('/api/custom_models', methods=['POST'])
def api_create_custom_model():
    """Create model metadata in S3 for current user. Expects JSON with name, description, category, trainingData."""
    payload = request.get_json() or {}
    name = payload.get('name') or 'custom_model'
    description = payload.get('description', '')
    category = payload.get('category', 'other')
    training_data = payload.get('trainingData', '')

    # create deterministic id from training data
    model_hash = hashlib.md5(training_data.encode('utf-8')).hexdigest() if training_data else hashlib.md5(name.encode('utf-8')).hexdigest()
    model_id = f"custom_{model_hash}"

    user_id, cookie_resp = storage.get_user_id()

    meta = {
        'id': model_id,
        'name': name,
        'description': description,
        'category': category,
        'nameCount': len([l for l in training_data.splitlines() if l.strip()]),
        'createdAt': int(time.time() * 1000),
        'lastUsed': int(time.time() * 1000)
    }

    # include trainingData for now (optional)
    meta['trainingData'] = training_data

    try:
        storage.save_model_metadata_to_s3(user_id, model_id, meta)
    except Exception as e:
        print('Error saving metadata to S3:', e)
        return jsonify({'error': 'Save failed'}), 500

    response = jsonify({'success': True, 'model_id': model_id, 'meta': meta})
    if cookie_resp:
        response.set_cookie('user_id', user_id, max_age=60*60*24*30)
    return response


@app.route('/api/custom_models/<model_id>', methods=['DELETE'])
def api_delete_custom_model(model_id):
    user_id, cookie_resp = storage.get_user_id()
    try:
        storage.delete_model_from_s3(user_id, model_id)
    except Exception as e:
        try:
            print(f"Error deleting model {model_id}: {e}")
        except Exception:
            pass

    return jsonify({'success': True})


@app.route('/api/custom_models/<model_id>', methods=['PUT'])
def api_update_custom_model(model_id):
    """Update metadata for an existing custom model (no retrain).
    Expects JSON with optional fields: name, description, category, trainingData
    """
    payload = request.get_json() or {}
    name = payload.get('name')
    description = payload.get('description')
    category = payload.get('category')
    training_data = payload.get('trainingData')

    try:
        user_id, cookie_resp = storage.get_user_id()
    except Exception:
        user_id = request.cookies.get('user_id') or None
        cookie_resp = None

    try:
        # Read existing metadata if present
        try:
            existing_meta = storage.get_model_metadata_from_s3(user_id, model_id)
        except Exception:
            existing_meta = {}

        meta = existing_meta or {}

        # Update fields (only when provided)
        if name is not None:
            meta['name'] = name
        if description is not None:
            meta['description'] = description
        if category is not None:
            meta['category'] = category
        if training_data is not None:
            meta['trainingData'] = training_data
            meta['nameCount'] = len([l for l in training_data.splitlines() if l.strip()])

        # Update timestamps
        meta['lastUsed'] = int(time.time() * 1000)
        if 'createdAt' not in meta:
            meta['createdAt'] = int(time.time() * 1000)

        # Persist via storage helper which writes to {user_id}/{model_id}/meta.json
        storage.save_model_metadata(user_id, model_id, meta)
    except Exception as e:
        print('Error updating metadata to S3:', e)
        return jsonify({'error': 'Update failed'}), 500

    response = jsonify({'success': True, 'meta': meta})
    if cookie_resp:
        response.set_cookie('user_id', user_id, max_age=60*60*24*30)
    return response

@app.route('/stream_progress', methods=['POST'])
def stream_progress():
    """Stream progress updates for model training and name generation"""
    def generate():
        selected_model = request.form['model']
        gender = request.form['gender']
        count = int(request.form['count'])
        temperature = float(request.form['temperature'])
        prefix = request.form['prefix']
        # length_mode: 'auto' (let model decide), 'average', 'custom'
        length_mode = request.form.get('length_mode', 'average')
        length = request.form.get('length', '')
        if length and length_mode == 'custom':
            try:
                length = int(length)
            except ValueError:
                length = None
        else:
            length = None
        # generation-only route; training happens at /train
        
        # Clear any old messages from the queue
        while not progress_queue.empty():
            progress_queue.get()
        
        # First, yield a preparation message
        yield f"data: {json.dumps({'type': 'preparing', 'message': 'Preparing to process your request...', 'progress': 0})}\n\n"

        # If a using custom model, resolve the model name and ensure we pass a user_id when loading data.
        if selected_model.startswith('custom'):
            # Resolve user_id (required for loading saved custom model artifacts)
            try:
                print("[app] setting user_id from storage")
                user_id, _ = storage.get_user_id()
                print("user id resolved to", user_id)
            except Exception:
                user_id = request.cookies.get('user_id') or None

            # Determine the model_name and training names
            if selected_model == 'custom':
                # prefer an explicitly selected custom model id from the hidden field
                selected_custom = request.form.get('selected_custom_model', '').strip()
                if selected_custom:
                    model_name = selected_custom
                    # Load the training data from metadata if available (best-effort)
                    try:
                        if user_id:
                            meta = storage.get_model_metadata_from_s3(user_id, model_name)
                            raw_custom = meta.get('trainingData', '')
                            custom_names = raw_custom.splitlines()
                        else:
                            custom_names = []
                    except Exception:
                        custom_names = []
                else:
                    raw_custom = request.form.get('custom-names-input', '')
                    custom_names = raw_custom.splitlines()
                    model_hash = hashlib.md5("\n".join(custom_names).encode('utf-8')).hexdigest()
                    model_name = f"custom_{model_hash}"
            else:
                # The form posted a concrete custom model id directly (e.g. 'custom_<hash>')
                model_name = selected_model
                # Get training names by loading metadata from S3; do so if we have user_id
                if user_id:
                    try:
                        meta = storage.get_model_metadata_from_s3(user_id, model_name)
                        raw_custom = meta.get('trainingData', '')
                        custom_names = raw_custom.splitlines()
                    except Exception:
                        custom_names = []
                else:
                    # No user_id available
                    yield f"data: {json.dumps({'type': 'error', 'message': 'user_id is required for custom models', 'progress': 0})}\n\n"
                    return

            # Attempt to load custom model data from S3; pass user_id so fng_name_generate can find per-user artifacts
            try:
                print("[app] loading custom model data for model_name=", model_name, " user_id=", user_id)
                model, X, y, char_to_idx, idx_to_char, char_set, bigram_counts, avg_length = fng_name_generate.load_model_data(model_name, user_id=user_id)
            except Exception as e:
                # Stream an error and stop
                yield f"data: {json.dumps({'type': 'error', 'message': 'Custom model not found or failed to load. Please train first.', 'progress': 0})}\n\n"
                return
        else:
            model_name = selected_model
            # load built-in model
            try:
                model, X, y, char_to_idx, idx_to_char, char_set, bigram_counts, avg_length = fng_name_generate.load_model_data(model_name)
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Failed to load model: '+str(e), 'progress': 0})}\n\n"
                return
        
        # Reset progress for name generation
        yield f"data: {json.dumps({'type': 'generating', 'message': 'Starting name generation...', 'progress': 0})}\n\n"

        # Generate names (pass user_id for custom models so load operations may access S3 if needed)
        user_id, _ = storage.get_user_id()
        name_stream = fng_name_generate.generate_quality_names_stream(
            model_name=model_name,
            count=count,
            gender=gender,
            prefix_text=prefix,
            length=length,
            temperature=temperature,
            custom_names=custom_names if selected_model == 'custom' else None,
            length_mode=length_mode,
            user_id=user_id if selected_model.startswith('custom') else None
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


@app.route('/train', methods=['POST'])
def train():
    """Train a custom model from submitted names and stream progress via SSE."""
    def generate_training():
        # Read form fields (allow both JSON and form-encoded)
        if request.is_json:
            payload = request.get_json()
            raw_names = payload.get('trainingData', '')
            model_display_name = payload.get('name', 'custom_model')
            category = payload.get('category', 'other')
            description = payload.get('description', '')
        else:
            raw_names = request.form.get('custom-names-input', '')
            model_display_name = request.form.get('new-model-name', 'custom_model')
            category = request.form.get('new-model-category', 'other')
            description = request.form.get('new-model-description', '')

        names = [l for l in raw_names.splitlines() if l.strip()]
        if not names:
            yield f"data: {json.dumps({'type': 'error', 'message': 'No training names provided', 'progress': 0})}\n\n"
            return

        # compute deterministic id
        model_hash = hashlib.md5("\n".join(names).encode('utf-8')).hexdigest()
        model_id = f"custom_{model_hash}"

        # Inform client
        yield f"data: {json.dumps({'type': 'preparing', 'message': 'Preparing training...', 'progress': 0})}\n\n"

        # Prepare training data
        yield f"data: {json.dumps({'type': 'loading', 'message': 'Processing training data...', 'progress': 5})}\n\n"
        time.sleep(0.2)
        try:
            X, y, char_to_idx, idx_to_char, char_set, bigram_counts, avg_length = fng_model.load_data(input_text="\n".join(names))
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Failed to prepare training data: '+str(e), 'progress': 0})}\n\n"
            return

        model = fng_model.create_model(X, char_to_idx, idx_to_char, char_set, bigram_counts)

        epochs = 15
        yield f"data: {json.dumps({'type': 'training', 'message': 'Starting training...', 'progress': 10})}\n\n"

        # Use a per-request progress queue
        local_queue = queue.Queue()
        def local_progress_callback(cur, tot):
            local_queue.put({'epoch': cur, 'total': tot})

        # Capture user_id here while in request context so background thread can save without accessing request
        try:
            user_id_for_thread, _ = storage.get_user_id()
        except Exception:
            user_id_for_thread = None

        def train_thread():
            try:
                fng_model.train_model(X, y, model, epochs=epochs, batch_size=64, stream_progress=local_progress_callback)
                # Save to S3 via helper — provide user_id captured above
                fng_model.save_model_data(
                    model, 
                    X, y, 
                    char_to_idx, 
                    idx_to_char, 
                    char_set, 
                    bigram_counts, 
                    avg_length=avg_length, 
                    model_name=model_id, 
                    user_id=user_id_for_thread,
                    meta_dict={
                        'id': model_id,
                        'name': model_display_name,
                        'description': description,
                        'category': category,
                        'nameCount': len(names),
                        'createdAt': int(time.time() * 1000),
                        'lastUsed': int(time.time() * 1000),
                        'trainingData': raw_names
                    }
                )
                local_queue.put({'epoch': epochs, 'total': epochs, 'complete': True})
            except Exception as e:
                local_queue.put({'error': str(e)})

        threading.Thread(target=train_thread).start()

        training_complete = False
        # track last seen epoch info so heartbeat messages can include useful context
        last_epoch = 0
        last_total = epochs if isinstance(epochs, int) and epochs > 0 else 0

        while not training_complete:
            try:
                update = local_queue.get(timeout=0.5)
                if 'error' in update:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Training failed: '+update['error'], 'progress': 0})}\n\n"
                    return
                if update.get('complete'):
                    training_complete = True
                    yield f"data: {json.dumps({'type': 'training', 'message': 'Training complete', 'progress': 100})}\n\n"
                    break
                # normal epoch update
                current = update.get('epoch', last_epoch)
                total = update.get('total', last_total)
                # update last seen
                last_epoch = current
                last_total = total
                progress = int((current / total) * 100) if total else 0
                yield f"data: {json.dumps({'type': 'training', 'message': f'Epoch {current}/{total}', 'progress': progress})}\n\n"
            except queue.Empty:
                # Queue was empty; report a heartbeat but include last known epoch/total if available
                try:
                    heartbeat_msg = f"Training in progress: Epoch {last_epoch}/{last_total}" if last_total else "Training in progress"
                except Exception:
                    heartbeat_msg = "Training in progress"
                yield f"data: {json.dumps({'type': 'heartbeat', 'message': heartbeat_msg, 'progress': int((last_epoch/last_total)*100) if last_total else 0})}\n\n"

        yield f"data: {json.dumps({'type': 'complete', 'message': 'Model trained', 'model_id': model_id, 'progress': 100})}\n\n"

    return Response(stream_with_context(generate_training()), mimetype='text/event-stream')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True) #remove debug if not in debug