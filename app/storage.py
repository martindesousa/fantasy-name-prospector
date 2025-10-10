import uuid
import json
from flask import request, make_response
import boto3
import os
from botocore.exceptions import ClientError

# Initialize S3 client
s3 = boto3.client("s3")
BUCKET = os.environ.get("S3_BUCKET_NAME", "generative-name-prospector-custom-models")

def get_user_id():
    user_id = request.cookies.get("user_id")
    if not user_id:
        user_id = uuid.uuid4().hex
        resp = make_response()
        resp.set_cookie("user_id", user_id, max_age=60*60*24*30)  # 30 days
        return user_id, resp
    return user_id, None

def save_model_to_s3(user_id, model_name, keras_path, json_path, meta_dict=None):
    base_prefix = f"{user_id}/{model_name}/"
    s3.upload_file(keras_path, BUCKET, base_prefix + "model.keras")
    s3.upload_file(json_path, BUCKET, base_prefix + "data.json")
    if meta_dict is not None:
        s3.put_object(
            Bucket=BUCKET,
            Key=base_prefix + "meta.json",
            Body=json.dumps(meta_dict).encode("utf-8")
        )

def load_model_from_s3(user_id, model_name, keras_dest, json_dest, meta_dest=None):
    base_prefix = f"{user_id}/{model_name}/"
    s3.download_file(BUCKET, base_prefix + "model.keras", keras_dest)
    s3.download_file(BUCKET, base_prefix + "data.json", json_dest)
    if meta_dest:
        s3.download_file(BUCKET, base_prefix + "meta.json", meta_dest)

    with open(json_dest, "r", encoding="utf-8") as f:
        return json.load(f)

def delete_model_from_s3(user_id, model_name):
    """Delete all objects under the canonical model prefix: {user_id}/{model_name}/

    This enforces that metadata/artifacts live under the folder and does not
    attempt to touch any legacy top-level keys.
    """
    if not user_id:
        raise ValueError('user_id is required to delete model')
    prefix = f"{user_id}/{model_name}/"
    try:
        # Use the S3 resource delete (bucket is not versioned)
        s3_resource = boto3.resource('s3')
        bucket = s3_resource.Bucket(BUCKET)
        bucket.objects.filter(Prefix=prefix).delete()
    except Exception as e:
        # log and ignore
        print(f"Error deleting model {model_name}: {e}")
        pass

def save_model_metadata_to_s3(user_id, model_name, meta_dict):
    """Save metadata JSON for a model at {user_id}/{model_name}/meta.json.

    Enforces the canonical foldered layout.
    """
    if not user_id:
        raise ValueError('user_id is required to save metadata')
    base_prefix = f"{user_id}/{model_name}/"
    s3.put_object(Bucket=BUCKET, Key=base_prefix + "meta.json", Body=json.dumps(meta_dict).encode('utf-8'))

def get_model_metadata_from_s3(user_id, model_name):
    """Load metadata JSON from the canonical location {user_id}/{model_name}/meta.json.

    Raises FileNotFoundError if meta.json is not present.
    """
    if not user_id:
        raise FileNotFoundError(f"No metadata JSON found for {user_id}/{model_name}")

    key = f"{user_id}/{model_name}/meta.json"
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        body = resp['Body'].read()
        return json.loads(body.decode('utf-8'))
    except ClientError as e:
        code = e.response.get('Error', {}).get('Code')
        if code in ('NoSuchKey', '404', 'NotFound'):
            raise FileNotFoundError(f"No metadata JSON found for {user_id}/{model_name}")
        raise
    except Exception:
        raise

def list_user_models(user_id):
    """Return a list of metadata dicts (lightweight) for all models belonging to user_id.

    This helper will discover model folders under the user's prefix and attempt to
    read their canonical meta.json files. It returns a list of metadata dicts; if
    a model has no meta.json, a minimal fallback metadata is returned.
    """
    if not user_id:
        return []
    prefix = f"{user_id}/"
    try:
        # List objects under the user's prefix and derive model ids from folder names only.
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        keys = [obj['Key'] for obj in resp.get('Contents', [])] if resp else []

        model_ids = set()
        for key in keys:
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix):]
            if '/' in rest:
                folder = rest.split('/', 1)[0]
                if folder:
                    model_ids.add(folder)

        models = []
        for model_id in sorted(model_ids):
            try:
                meta = get_model_metadata_from_s3(user_id, model_id)
                # Only include models that have canonical meta.json
                meta['id'] = meta.get('id') or model_id
                if 'name' not in meta or not meta.get('name'):
                    meta['name'] = meta['id']
                models.append(meta)
            except FileNotFoundError:
                # skip folders without meta.json
                continue
        return models
    except Exception:
        return []