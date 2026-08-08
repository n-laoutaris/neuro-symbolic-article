"""Utilities for interacting with the Gemini LLM API.

This module provides functions for initializing the Gemini client, generating text and JSON content,
handling retries for API errors and performing reflexion loops for self-improvement.
"""

# Standard library imports
import os
import time
from pathlib import Path
from typing import Any, List

# Third-party imports
from dotenv import load_dotenv
from google import genai

# Local imports
from src.parsing_utils import read_txt

# Module-Level Variables
_CLIENT = None
_MODEL_NAME = None

class GeminiExhaustedException(Exception):
    """Raised when the Gemini API returns '429 Resource Exhausted' too many times consecutively, indicating the daily quota is hit."""
    pass

def initialize_gemini_client(model_name: str) -> None:
    """Initializes the Gemini API client and loads the API key from environment variables.

    Sets up the global client and model name for subsequent API calls. Requires a valid GEMINI_API_KEY in a .env file.

    Args:
        model_name: The name of the Gemini model to use (e.g., 'gemini-2.5-flash').

    Raises:
        RuntimeError: If GEMINI_API_KEY is not found in the environment.
    """
    global _CLIENT, _MODEL_NAME
    _CLIENT = genai.Client()
    _MODEL_NAME = model_name

    load_dotenv()
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY not found. Add it to a .env file in the notebook root.")
    
# Text generation in general
def call_gemini(content: List[str]) -> str:
    """Generates text content using the Gemini API.

    Sends the provided content to the initialized Gemini model and returns the generated text response.

    Args:
        content: A list of strings representing the input content for the model.

    Returns:
        The generated text response from the model.
    """
    response = _CLIENT.models.generate_content(
        model=_MODEL_NAME,
        contents=content
    )

    return response.text

# Document understanding
def call_gemini_pdf(content: str, file_name: str) -> str:
    """Generates content from a PDF document using the Gemini API.

    Uploads the specified PDF file and sends it along with text content to the model for processing.

    Args:
        content: The text prompt or instructions for the model.
        file_name: Path to the PDF file to upload and process.

    Returns:
        The generated text response from the model based on the PDF and content.
    """
    # Upload the PDF file for processing
    file_path = Path(file_name)
    content_file = _CLIENT.files.upload(file=file_path)

    response = _CLIENT.models.generate_content(
        model=_MODEL_NAME,
        contents=[content_file, content]
    )

    return response.text

# JSON structured output
def call_gemini_json(content: List[str], schema) -> str:
    """Generates JSON-structured content using the Gemini API.

    Sends the provided content to the model with a response schema to enforce JSON output format.

    Args:
        content: A list of strings representing the input content for the model.
        schema: The schema defining the expected JSON response structure.

    Returns:
        The generated JSON response as a string from the model.
    """
    response = _CLIENT.models.generate_content(
        model=_MODEL_NAME,
        contents=content,
        config={
            "response_mime_type": "application/json",
            "response_schema": schema
        }
    )

    return response.text

# Retry wrapper to combat model overload errors
def with_retries(func: callable, *args, base_delay: float = 4.0) -> Any:
    """Retries a function call with exponential backoff for overload or exhaustion errors.

    Handles 'overloaded' and 'exhausted' exceptions from the Gemini API by retrying with increasing delays.
    Raises GeminiExhaustedException if exhaustion occurs 5+ times, indicating quota limits.

    Args:
        func: The function to call and retry.
        *args: Arguments to pass to the function.
        base_delay: Initial delay in seconds for exponential backoff (default: 4.0).

    Returns:
        The result of the successful function call.

    Raises:
        GeminiExhaustedException: If the API is exhausted 5 or more times.
        Exception: Any other exception from the function call.
    """
    overloads = 0
    exhaustions = 0
    while True:
        try:
            return func(*args)
        except Exception as e:
            msg = str(e).lower()
            overloaded = "overloaded" in msg
            exhausted = "exhausted" in msg

            if overloaded:
                overloads += 1
                wait = min(base_delay * (2 ** overloads), 60.0)  # Exponential backoff, max 60s
                print(f"Gemini overloaded {overloads} times, retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue

            elif exhausted:
                overloads = 0
                exhaustions += 1
                if exhaustions >= 5:  # Stop everything after 5 exhaustions
                    raise GeminiExhaustedException("Gemini daily quota likely exceeded (5x 429 errors).")
                # Wait a fixed time and retry for fewer exhaustions
                print(f"Gemini exhausted {exhaustions} times, waiting 1 minute and rerunning the code...")
                time.sleep(60)
                continue

            # Raise any other exception immediately
            raise e

# Reflexion loop
def reflect(original_content_list: List[str], previous_result: str, json_schema = None) -> str:
    """Asks the model to review its own work and fix potential errors.

    Loads the reflexion prompt, flattens the original input, and constructs a review payload
    for the model to critique and improve the previous result.

    Args:
        original_content_list: List of strings from the original input data.
        previous_result: The draft response to review and improve.
        json_schema: Optional JSON schema to enforce structured output.

    Returns:
        The improved response from the model after reflexion.
    """
    # Load the reflexion prompt
    reflexion_prompt = read_txt('Prompts/Reflexion/reflexion.txt')

    # Flatten the original input into a single string
    flat_content = "\n\n".join([str(item) for item in original_content_list])

    # Construct the review payload
    new_content = [
        reflexion_prompt,
        f"--- [ORIGINAL INPUT DATA START] ---\n{flat_content}\n--- [ORIGINAL INPUT DATA END] ---",
        f"--- [DRAFT RESPONSE] ---\n{previous_result}"
    ]

    # Call the API with or without JSON schema
    if json_schema:
        return with_retries(call_gemini_json, new_content, json_schema)
    else:
        return with_retries(call_gemini, new_content)