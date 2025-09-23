import uuid
import json
from flask import request, make_response
import boto3
import os

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

def save_model_to_s3(user_id, model_name, keras_path, json_path):
    s3.upload_file(keras_path, BUCKET, f"{user_id}/{model_name}.keras")
    s3.upload_file(json_path, BUCKET, f"{user_id}/{model_name}.json")

def load_model_from_s3(user_id, model_name, keras_dest, json_dest):
    s3.download_file(BUCKET, f"{user_id}/{model_name}.keras", keras_dest)
    s3.download_file(BUCKET, f"{user_id}/{model_name}.json", json_dest)
    # Read JSON using UTF-8 to avoid mojibake (e.g. stray 'Â' characters)
    with open(json_dest, 'r', encoding='utf-8', errors='replace') as f:
        return json.load(f)