# app/screen_agent/vision_tool.py

import os
import sys
import tempfile
from google import genai
from google.genai.types import Part
from .utils.gcs_upload import capture_and_upload_to_local_file, upload_to_gcs

async def vision_via_files_api() -> str:
    """
    Capture the screen, upload to GCS, then analyze via the GenAI Files API.
    Returns both the GCS URL and the LLM's analysis text.
    """
    try:
        # Check for API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("ERROR: GOOGLE_API_KEY environment variable not set", file=sys.stderr)
            return "Error: API key not configured"

        # 1) Capture & write to a temp JPEG on disk
        tmp_path = capture_and_upload_to_local_file(prefix="screen", suffix=".jpg")
        if not tmp_path or not os.path.exists(tmp_path):
            print(f"ERROR: Failed to capture screen image to {tmp_path}", file=sys.stderr)
            return "Error: Failed to capture screen image"

        print(f"DEBUG: Successfully captured screenshot to {tmp_path}", file=sys.stderr)

        # 2) Upload to Google Cloud Storage
        gcs_url = upload_to_gcs(tmp_path)
        if not gcs_url:
            print("ERROR: Failed to upload to Google Cloud Storage", file=sys.stderr)
            return "Error: Failed to upload to Google Cloud Storage"

        print(f"DEBUG: Successfully uploaded to GCS: {gcs_url}", file=sys.stderr)

        # 3) Analyze with Gemini API
        try:
            client = genai.Client(api_key=api_key)
            with open(tmp_path, 'rb') as f:
                file_data = f.read()
                # Create a Part object with the file data in the correct format
                image_part = Part(
                    inline_data={
                        "mime_type": "image/jpeg",
                        "data": file_data
                    }
                )
                # Generate content with the image part (removed await since it's not async)
                response = client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=[
                        "Please analyze this screenshot in detail and describe what you see:",
                        image_part
                    ]
                )
                if not response or not response.text:
                    print("ERROR: Empty response from Gemini API", file=sys.stderr)
                    return "Error: Failed to get analysis from AI"
        except Exception as e:
            print(f"ERROR: Failed to analyze with Gemini API: {str(e)}", file=sys.stderr)
            return f"Error: Failed to analyze image: {str(e)}"
        
        # Clean up the temp file
        try:
            os.remove(tmp_path)
        except Exception as e:
            print(f"WARNING: Failed to clean up temp file: {str(e)}", file=sys.stderr)
            
        # Return both the GCS URL and the analysis
        return f"Screenshot saved to: {gcs_url}\n\nAnalysis: {response.text}"
    except Exception as e:
        print(f"ERROR: Unexpected error in vision_via_files_api: {str(e)}", file=sys.stderr)
        return f"Error analyzing screen: {str(e)}"
