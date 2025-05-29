# app/screen_agent/utils/gcs_upload.py

import mss, io, uuid, os
from PIL import Image
import os
import tempfile           # for temp directory paths
import uuid               # for generating unique filenames
import mss
import sys
from google.cloud import storage
from datetime import datetime

def capture_and_upload_to_local_file(prefix="screenshot", suffix=".jpg") -> str:
    """
    Captures the screen and writes it to a temp file, returning the filepath.
    """
    try:
        sct = mss.mss()
        mon = sct.monitors[1]
        screenshot = sct.grab(mon)
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

        path = os.path.join(tempfile.gettempdir(), f"{prefix}-{uuid.uuid4().hex}{suffix}")
        img.save(path, format="JPEG")
        print(f"DEBUG: Screenshot saved to {path}", file=sys.stderr)
        return path
    except Exception as e:
        print(f"ERROR: Failed to capture screen: {str(e)}", file=sys.stderr)
        return None

def upload_to_gcs(local_file_path: str, bucket_name: str = "my-screenshots-bucket") -> str:
    """
    Uploads a file to Google Cloud Storage and returns the public URL.
    
    Args:
        local_file_path: Path to the local file to upload
        bucket_name: Name of the GCS bucket
        
    Returns:
        str: The public URL of the uploaded file
    """
    try:
        # Initialize the GCS client
        storage_client = storage.Client()
        
        # Get the bucket
        bucket = storage_client.bucket(bucket_name)
        
        # Generate a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"screenshots/{timestamp}_{unique_id}.jpg"
        
        # Create a new blob and upload the file
        blob = bucket.blob(filename)
        blob.upload_from_filename(local_file_path)
        
        # Return the public URL (no need to set ACLs with uniform bucket-level access)
        return f"https://storage.googleapis.com/{bucket_name}/{filename}"
        
    except Exception as e:
        print(f"ERROR: Failed to upload to GCS: {str(e)}", file=sys.stderr)
        return None
