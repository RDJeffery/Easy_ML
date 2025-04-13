import base64
import io
import os
import sys
from typing import Optional

import numpy as np
from PIL import Image, UnidentifiedImageError


# Counter for saving sample images
_save_counter = 0
_max_saves = 2 # Save only the first few samples

def process_image_from_base64(base64_string: str) -> Optional[np.ndarray]:
    """Decodes a base64 string, processes the image, and returns a normalized vector."""
    global _save_counter
    try:
        # Decode the base64 string
        image_data = base64.b64decode(base64_string)
        # Create an in-memory stream
        image_stream = io.BytesIO(image_data)
        # Open the image using PIL
        img = Image.open(image_stream)

        # Convert to grayscale, resize to 28x28
        img_processed = img.convert("L").resize((28, 28), Image.Resampling.LANCZOS)

        # Convert to numpy array
        img_array = np.array(img_processed)

        # --- Debugging --- 
        print(f"DEBUG: Base64 Image Original dtype={img_array.dtype}, min={img_array.min()}, max={img_array.max()}", file=sys.stderr)
        # Save a sample before normalization
        if _save_counter < _max_saves:
            try:
                sample_path = f'debug_processed_b64_sample_{_save_counter}.png'
                Image.fromarray(img_array).save(sample_path)
                print(f"DEBUG: Saved processed base64 sample (before norm) to {sample_path}", file=sys.stderr)
                _save_counter += 1
            except Exception as save_err:
                print(f"DEBUG: Error saving sample image: {save_err}", file=sys.stderr)
        # ---------------
        
        # Flatten and normalize (assuming pixel values are 0-255)
        img_vector_flat = img_array.flatten().astype(float) / 255.0

        return img_vector_flat.reshape(784, 1) # Reshape to column vector

    except (base64.binascii.Error, UnidentifiedImageError) as decode_err:
        # Handle errors during decoding or opening image
        print(f"Warning: Could not decode/open base64 image data: {decode_err}", file=sys.stderr)
        return None
    except Exception as e:
        # Catch other potential errors during processing
        print(f"Warning: Error processing image from base64: {e}", file=sys.stderr)
        return None

def process_image_from_path(file_path: str) -> Optional[np.ndarray]:
    """Loads an image from a path, processes it, and returns a normalized vector."""
    try:
        if not os.path.exists(file_path):
            # Try resolving relative to the script's execution directory as a fallback
            # This depends on how the main script is run
            alt_path = os.path.join(os.getcwd(), file_path)
            if not os.path.exists(alt_path):
                 print(f"Warning: Image file not found at specified path: {file_path} or {alt_path}", file=sys.stderr)
                 return None
            else:
                file_path = alt_path # Use the resolved path

        img = Image.open(file_path)
        # Convert to grayscale, resize to 28x28
        img_processed = img.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
        # Convert to numpy array
        img_array = np.array(img_processed)

        # --- Debugging --- 
        print(f"DEBUG: Path Image Original dtype={img_array.dtype}, min={img_array.min()}, max={img_array.max()}", file=sys.stderr)
        # ---------------

        # Flatten and normalize (assuming pixel values are 0-255)
        img_vector_flat = img_array.flatten().astype(float) / 255.0

        return img_vector_flat.reshape(784, 1) # Reshape to column vector

    except UnidentifiedImageError:
        print(f"Warning: Cannot identify image file (corrupted or wrong format?): {file_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Warning: Error processing image from path {file_path}: {e}", file=sys.stderr)
        return None 