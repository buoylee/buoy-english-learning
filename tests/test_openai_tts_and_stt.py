import os
import boto3

import openai
import pytest
from openai import OpenAI
from pathlib import Path

# This is the test case for the combined Text-to-Speech and Speech-to-Text test.
# It requires the OPENAI_API_KEY environment variable to be set.
# The test is skipped if the API key is not available.
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY is not set")
def test_tts_and_transcription_with_timestamps():
    """
    Tests a two-step process:
    1. Generate audio from text using the TTS API.
    2. Transcribe the generated audio to get word-level timestamps using the Transcription API.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_PATH") # Can be None

    client = OpenAI(api_key=api_key, base_url=base_url)

    test_text = "The quick brown fox jumps over the lazy dog."
    
    # --- Step 1: Text-to-Speech (TTS) ---
    # Generate audio from the text. The response is an audio stream.
    try:
        speech_file_path = Path(__file__).parent / "speech.mp3"
        with openai.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="alloy",
                input="The quick brown fox jumped over the lazy dog."
        ) as response:
            response.stream_to_file(speech_file_path)
    
    except Exception as e:
        pytest.fail(f"OpenAI TTS API call failed: {e}")

    # --- Step 2: Speech-to-Text (Transcription with Timestamps) ---
    # Use the generated audio bytes to get a transcription with word timestamps.
    try:
        audio_file = open("speech.mp3", "rb")
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )

        import json
        print(json.dumps(transcript.words, indent=2, ensure_ascii=False))
            
    except Exception as e:
        pytest.fail(f"OpenAI Transcription API call failed: {e}") 


# --- Cloudflare R2 Upload Test ---

def r2_credentials_present():
    """Helper function to check if all necessary R2 env vars are set."""
    required_vars = [
        "CLOUDFLARE_R2_ACCOUNT_ID",
        "CLOUDFLARE_R2_ACCESS_KEY_ID",
        "CLOUDFLARE_R2_SECRET_ACCESS_KEY",
        "CLOUDFLARE_R2_BUCKET_NAME",
    ]
    return all(os.environ.get(var) for var in required_vars)

@pytest.mark.skipif(not r2_credentials_present(), reason="Cloudflare R2 credentials are not set")
def test_upload_mp3_to_r2():
    """
    Tests uploading the generated speech.mp3 file to a Cloudflare R2 bucket.
    """
    # 1. Get credentials from environment variables
    account_id = os.environ.get("CLOUDFLARE_R2_ACCOUNT_ID")
    access_key_id = os.environ.get("CLOUDFLARE_R2_ACCESS_KEY_ID")
    secret_access_key = os.environ.get("CLOUDFLARE_R2_SECRET_ACCESS_KEY")
    bucket_name = os.environ.get("CLOUDFLARE_R2_BUCKET_NAME")

    # 2. Define file paths and R2 details
    local_file_path = Path(__file__).parent / "speech.mp3"
    r2_object_name = f"tests/{local_file_path.name}" # Store it in a 'tests' folder in the bucket
    
    assert local_file_path.exists(), f"Prerequisite file not found: {local_file_path}. Please run the TTS test first."

    # 3. Create a boto3 client configured for R2
    r2_endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
    
    try:
        s3_client = boto3.client(
            service_name='s3',
            endpoint_url=r2_endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name='auto',  # Must be 'auto' for R2
        )
        print(f"\nAttempting to upload {local_file_path} to R2 bucket '{bucket_name}' as '{r2_object_name}'...")

        # 4. Upload the file
        s3_client.upload_file(str(local_file_path), bucket_name, r2_object_name)
        print("Upload command sent successfully.")

        # 5. Verify the upload by checking the object's metadata
        print("Verifying upload...")
        response = s3_client.head_object(Bucket=bucket_name, Key=r2_object_name)
        
        assert response['ResponseMetadata']['HTTPStatusCode'] == 200, "File should exist in R2 after upload."
        assert int(response['ContentLength']) > 0, "Uploaded file in R2 should not be empty."
        
        print(f"Successfully uploaded and verified. File size: {response['ContentLength']} bytes.")

    except Exception as e:
        pytest.fail(f"Cloudflare R2 operation failed: {e}")


# --- Diagnostic Test ---

def test_diagnose_environment_variables():
    """
    A diagnostic test to print the values of environment variables that Pytest sees.
    This helps to confirm if the .env file is being loaded correctly.
    """
    print("\n--- Environment Variable Diagnostics ---")
    
    required_vars = [
        "OPENAI_API_KEY",
        "OPENAI_BASE_PATH",
        "CLOUDFLARE_R2_ACCOUNT_ID",
        "CLOUDFLARE_R2_ACCESS_KEY_ID",
        "CLOUDFLARE_R2_SECRET_ACCESS_KEY",
        "CLOUDFLARE_R2_BUCKET_NAME",
    ]
    
    all_vars_found = True
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            # For security, only print the first few characters of secrets
            if "KEY" in var or "SECRET" in var:
                print(f"✅ Found {var}: {value[:4]}...")
            else:
                print(f"✅ Found {var}: {value}")
        else:
            print(f"❌ NOT FOUND: {var}")
            all_vars_found = False
            
    print("--- End of Diagnostics ---")
    
    # This assertion will fail if any of the R2 variables are missing,
    # giving a clear signal in the test report.
    assert all_vars_found, "One or more required environment variables were not found. Check your .env file and its location."




