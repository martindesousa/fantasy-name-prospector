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