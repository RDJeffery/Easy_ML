import numpy as np
from PIL import Image
from io import BytesIO
import base64
import sys

TARGET_SIZE = (28, 28)
TARGET_FLAT_DIM = TARGET_SIZE[0] * TARGET_SIZE[1] # 784

def process_image_from_base64(base64_string):
    """Decodes a base64 PNG string and processes it for the NN.

    Args:
        base64_string (str): Base64 encoded PNG string (can optionally start with 'data:image/png;base64,').

    Returns:
        np.ndarray: Flattened, normalized 1D NumPy array (784 elements),
                    or None if processing fails.
    """
    if not isinstance(base64_string, str):
        print("Error: Invalid input type for base64 string.", file=sys.stderr)
        return None

    try:
        # Check for and remove optional header
        header = 'data:image/png;base64,'
        if base64_string.startswith(header):
            b64_data = base64_string[len(header):]
        else:
            b64_data = base64_string # Assume raw base64 if header is missing

        # Decode base64 data
        image_data = base64.b64decode(b64_data)
        # Open image from bytes
        img = Image.open(BytesIO(image_data))
        # Process (grayscale, resize, normalize, flatten)
        return process_image_object(img)

    except Exception as e:
        print(f"Error processing base64 image data: {e}", file=sys.stderr)
        return None

def process_image_from_path(image_path):
    """Loads an image file from a path and processes it for the NN.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Flattened, normalized 1D NumPy array (784 elements),
                    or None if processing fails.
    """
    try:
        img = Image.open(image_path)
        return process_image_object(img)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error processing image from path {image_path}: {e}", file=sys.stderr)
        return None

def process_image_object(img_obj):
    """Processes a PIL Image object (grayscale, resize, normalize, flatten).

    Args:
        img_obj (PIL.Image.Image): The PIL Image object.

    Returns:
        np.ndarray: Flattened, normalized 1D NumPy array (784 elements),
                    or None if processing fails.
    """
    try:
        # Convert to grayscale, resize to target size using Lanczos resampling
        img_processed = img_obj.convert('L').resize(TARGET_SIZE, Image.LANCZOS)

        # Convert to NumPy array and normalize to [0, 1]
        img_array = np.array(img_processed).astype(float) / 255.0

        # Flatten to a 1D vector
        img_vector = img_array.flatten()

        # Sanity check shape
        if img_vector.shape != (TARGET_FLAT_DIM,):
            print(f"Error: Incorrect final image vector shape: {img_vector.shape}. Expected ({TARGET_FLAT_DIM},)", file=sys.stderr)
            return None

        return img_vector

    except Exception as e:
        print(f"Error processing PIL image object: {e}", file=sys.stderr)
        return None 