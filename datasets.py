import numpy as np
import pandas as pd
import os
import sys
import traceback
import base64
from io import BytesIO
from PIL import Image # Requires Pillow
from typing import Tuple, Optional, Union, Dict, List, Any
import pickle # Needed for CIFAR-10

# Import the new utility functions
from utils.image_processor import process_image_from_base64, process_image_from_path

# Define a type alias for the standard return tuple of loaders
# Updated type hint to include optional raw data for debugging/inspection
LoadResult = Tuple[
    Optional[np.ndarray], # X_train
    Optional[np.ndarray], # Y_train
    Optional[np.ndarray], # X_dev
    Optional[np.ndarray], # Y_dev
    int,                  # num_classes
    Optional[np.ndarray], # raw_X_train_flattened (Optional)
    Optional[np.ndarray]  # raw_Y_train (Optional)
]

# --- Helper Function for Reshaping and Splitting --- #

def _reshape_and_split_data(X_combined: np.ndarray, Y_labels: np.ndarray,
                          validation_split: Union[int, float],
                          return_shape: str = 'flattened',
                          img_dims: Optional[Tuple[int, int, int]] = (28, 28, 1)) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffles, splits data, and reshapes features based on return_shape.

    Args:
        X_combined (np.ndarray): Combined feature data. Expected shape varies based on input source.
                                 Can be (num_samples, features) or (num_samples, H, W, C).
        Y_labels (np.ndarray): Corresponding labels (num_samples,).
        validation_split (Union[int, float]): Number or fraction of samples for validation.
        return_shape (str): Desired output shape: 'flattened' (features, num_samples) or
                           'image' (num_samples, H, W, C). Defaults to 'flattened'.
        img_dims (Optional[Tuple[int, int, int]]): The target (H, W, C) dimensions if reshaping
                                                   to 'image' format. Used if X_combined is flat.
                                                   Defaults to (28, 28, 1).

    Returns:
        tuple: (X_train, Y_train, X_dev, Y_dev)
               Feature shapes depend on return_shape. Returns appropriately shaped empty arrays on error or empty input.
    """
    # Determine default feature count/shape for empty arrays
    default_feature_count = 0
    if X_combined.ndim > 1:
        # Prioritize img_dims if reshaping to image
        if return_shape == 'image' and img_dims:
            default_feature_count = -1 # Indicate image shape needed using img_dims
        # Handle flattened target shape
        elif return_shape == 'flattened':
            if X_combined.ndim == 4 and img_dims: # Input is image, target is flat
                default_feature_count = img_dims[0] * img_dims[1] * img_dims[2]
            elif X_combined.ndim == 2: # Input is flat, target is flat
                default_feature_count = X_combined.shape[1]
        # Fallbacks based only on input dimension if target shape logic didn't apply
        elif X_combined.ndim == 4 and img_dims: # Input is image, but target wasn't 'flattened' or 'image'
             default_feature_count = img_dims[0] * img_dims[1] * img_dims[2]
        elif X_combined.ndim == 2: # Input is flat, target wasn't 'flattened' or 'image'
            default_feature_count = X_combined.shape[1]

    # Initialize with default empty shapes based on target return_shape
    dtype = X_combined.dtype if X_combined.size > 0 else np.float32 # Use input dtype or default
    if return_shape == 'image' and img_dims:
        empty_shape = (0, img_dims[0], img_dims[1], img_dims[2])
        X_train = np.empty(empty_shape, dtype=dtype)
        X_dev = np.empty(empty_shape, dtype=dtype)
    else: # Default to flattened (features, samples) or handle unknown shape
        # Use determined default_feature_count; fallback to 0 if needed
        f_count = default_feature_count if default_feature_count > 0 else 0
        X_train = np.empty((f_count, 0), dtype=dtype)
        X_dev = np.empty((f_count, 0), dtype=dtype)

    Y_train = np.array([], dtype=int)
    Y_dev = np.array([], dtype=int)

    num_samples = X_combined.shape[0]
    if num_samples == 0:
        print("Warning: No samples to split.", file=sys.stderr)
        # Already initialized with empty arrays, just return
        return X_train, Y_train, X_dev, Y_dev

    # --- Shuffle --- #
    permutation = np.random.permutation(num_samples)
    X_shuffled = X_combined[permutation]
    Y_shuffled = Y_labels[permutation]

    # --- Determine Split Index --- #
    split_idx: int
    if isinstance(validation_split, float) and 0 < validation_split < 1:
        split_idx = int(num_samples * (1.0 - validation_split))
    elif isinstance(validation_split, int) and 1 <= validation_split < num_samples:
        split_idx = num_samples - validation_split
    else:
        # Fallback split value (e.g., 10% or 1000, whichever is smaller and feasible)
        split_val = min(1000, max(1, int(num_samples * 0.1)))
        if num_samples <= split_val: # Avoid validation set larger than training set
             split_val = max(1, num_samples // 2) if num_samples > 1 else 0
        print(f"Warning: Invalid validation_split value ({validation_split}). Using fallback: {split_val} samples.", file=sys.stderr)
        split_idx = num_samples - split_val

    # --- Split Data --- #
    X_train_split, X_dev_split = None, None # Initialize split vars
    if num_samples < 2 or split_idx <= 0 or split_idx >= num_samples:
        print(f"Warning: Not enough samples ({num_samples}) or invalid split ({split_idx}) for validation. Using all for training.", file=sys.stderr)
        X_train_split = X_shuffled
        Y_train = Y_shuffled
        # Keep X_dev_split as None, Y_dev is already empty []
    else:
        X_train_split = X_shuffled[:split_idx]
        Y_train = Y_shuffled[:split_idx]
        X_dev_split = X_shuffled[split_idx:]
        Y_dev = Y_shuffled[split_idx:]

    # --- Reshape Based on return_shape --- #
    reshape_error = False # Flag to track errors during reshaping

    try: # Wrap reshaping in a try block for cleaner error handling
        if return_shape == 'image':
            # print(f"Reshaping data to 'image' format...", file=sys.stderr) # Verbose
            if not img_dims:
                 print(f"Error: Cannot reshape to image. img_dims not provided.", file=sys.stderr)
                 reshape_error = True
            elif X_train_split is not None and X_train_split.ndim == 4:
                 # Input is already image format (samples, H, W, C)
                 X_train = X_train_split
                 if X_dev_split is not None: X_dev = X_dev_split
                 # print(f"  Input data already has {X_train_split.ndim} dimensions. Shape: {X_train.shape}", file=sys.stderr) # Verbose
            elif X_train_split is not None and X_train_split.ndim == 2:
                 # Input is flattened (samples, features), reshape to (samples, H, W, C)
                 num_features = X_train_split.shape[1]
                 expected_features = img_dims[0] * img_dims[1] * img_dims[2]
                 if num_features != expected_features:
                      print(f"Error: Cannot reshape to image. Expected {expected_features} features for shape {img_dims}, but got {num_features}.", file=sys.stderr)
                      reshape_error = True
                 else:
                      target_shape_train = (X_train_split.shape[0], img_dims[0], img_dims[1], img_dims[2])
                      X_train = X_train_split.reshape(target_shape_train)
                      if X_dev_split is not None and X_dev_split.size > 0:
                          target_shape_dev = (X_dev_split.shape[0], img_dims[0], img_dims[1], img_dims[2])
                          X_dev = X_dev_split.reshape(target_shape_dev)
                      # else: X_dev remains empty image shape from init
                      # print(f"  Reshaped flattened data to {target_shape_train} and {X_dev.shape}.", file=sys.stderr) # Verbose
            elif X_train_split is not None: # Check needed if split failed
                 print(f"Error: Cannot reshape to image. Input data has unexpected {X_train_split.ndim} dimensions.", file=sys.stderr)
                 reshape_error = True
            # else: X_train_split is None (split failed), error state handled below

        elif return_shape == 'flattened':
            # print("Reshaping data to 'flattened' format (features, samples)...", file=sys.stderr) # Verbose
            if X_train_split is not None and X_train_split.ndim == 4:
                 # Input is image (samples, H, W, C), flatten to (features, samples)
                 num_train_samples = X_train_split.shape[0]
                 num_features = X_train_split.shape[1] * X_train_split.shape[2] * X_train_split.shape[3]
                 X_train = X_train_split.reshape(num_train_samples, num_features).T
                 if X_dev_split is not None and X_dev_split.size > 0:
                     num_dev_samples = X_dev_split.shape[0]
                     X_dev = X_dev_split.reshape(num_dev_samples, num_features).T
                 # else: X_dev remains empty flattened shape from init
                 # print(f"  Flattened image data to {X_train.shape} and {X_dev.shape}.", file=sys.stderr) # Verbose
            elif X_train_split is not None and X_train_split.ndim == 2:
                 # Input is already flattened (samples, features), transpose to (features, samples)
                 X_train = X_train_split.T
                 if X_dev_split is not None and X_dev_split.size > 0:
                     X_dev = X_dev_split.T
                 # else: X_dev remains empty flattened shape from init
                 # print(f"  Transposed flattened data. Shape: {X_train.shape}", file=sys.stderr) # Verbose
            elif X_train_split is not None: # Check needed if split failed
                 print(f"Error: Cannot flatten. Input data has unexpected {X_train_split.ndim} dimensions.", file=sys.stderr)
                 reshape_error = True
            # else: X_train_split is None (split failed), error state handled below

        else: # Unknown return_shape
            print(f"Error: Unknown return_shape '{return_shape}'. Defaulting to flattened (transposed).", file=sys.stderr)
            if X_train_split is not None and X_train_split.ndim == 2: # Attempt transpose as fallback
                 X_train = X_train_split.T
                 if X_dev_split is not None and X_dev_split.size > 0:
                     X_dev = X_dev_split.T
            elif X_train_split is not None: # Cannot determine shape
                 reshape_error = True # Mark error to trigger reset below
            # else: X_train_split is None (split failed), error state handled below

    except ValueError as e:
        print(f"Error during reshape operation: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr) # Add traceback for debug
        reshape_error = True

    # If reshape error occurred OR if the initial split failed (X_train_split is None),
    # reset X_train/X_dev to initial empty arrays matching target shape
    if reshape_error or X_train_split is None:
         if X_train_split is not None: # Only print reset message if error happened *after* split
             print("  Resetting X arrays to empty due to reshape error.", file=sys.stderr)
         # Re-initialize based on target shape, same as initial block
         if return_shape == 'image' and img_dims:
             empty_shape = (0, img_dims[0], img_dims[1], img_dims[2])
             X_train = np.empty(empty_shape, dtype=dtype)
             X_dev = np.empty(empty_shape, dtype=dtype)
         else: # Default to flattened (features, samples)
             # Try to determine feature count more robustly for empty array
             f_count = 0
             if default_feature_count > 0:
                  f_count = default_feature_count
             elif X_combined.ndim == 2: # Fallback from original combined data
                  f_count = X_combined.shape[1]
             elif X_combined.ndim == 4 and img_dims: # Fallback from original combined data
                  f_count = img_dims[0] * img_dims[1] * img_dims[2]

             X_train = np.empty((f_count, 0), dtype=dtype)
             X_dev = np.empty((f_count, 0), dtype=dtype)


    # --- Ensure float32 type for features (often needed by models) --- #
    if X_train is not None and X_train.dtype != np.float32:
        X_train = X_train.astype(np.float32)
    if X_dev is not None and X_dev.dtype != np.float32:
        X_dev = X_dev.astype(np.float32)

    # --- Final Check and Print --- #
    # Ensure Y arrays are numpy arrays even if split failed
    if not isinstance(Y_train, np.ndarray): Y_train = np.array(Y_train, dtype=int)
    if not isinstance(Y_dev, np.ndarray): Y_dev = np.array(Y_dev, dtype=int)

    # print(f"Final Shapes: X_train={X_train.shape}, Y_train={Y_train.shape}, X_dev={X_dev.shape}, Y_dev={Y_dev.shape}", file=sys.stderr) # Verbose

    return X_train, Y_train, X_dev, Y_dev


# --- Main Data Loaders --- #

def load_csv_dataset(csv_path: str,
                     validation_split: Union[int, float] = 1000,
                     label_col_index: int = 0,
                     image_col_index: Optional[int] = None,
                     image_col_type: Optional[str] = None,
                     return_shape: str = 'flattened') -> LoadResult:
    """Loads data from a generic CSV file.

    Handles either raw pixel data across columns or image references (base64/path)
    in a specified column. Shuffles, splits, normalizes/processes features.

    Args:
        csv_path (str): Path to the dataset CSV file.
        validation_split (Union[int, float]): Number or fraction for validation set.
        label_col_index (int): Index of the label column.
        image_col_index (Optional[int]): Index of the image data column (base64/path).
                                        If None, assumes raw pixel data.
        image_col_type (Optional[str]): Type ('base64' or 'path') if using image_col_index.
        return_shape (str): Desired output shape: 'flattened' or 'image'.

    Returns:
        LoadResult: (X_train, Y_train, X_dev, Y_dev, num_classes, raw_X_train_flat, raw_Y_train)
                    Feature shapes depend on return_shape.
                    Returns (None, ..., None, 0, None, None) on failure.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}", file=sys.stderr)
        return None, None, None, None, 0, None, None

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"Error: CSV file is empty: {csv_path}", file=sys.stderr)
            return None, None, None, None, 0, None, None

        # --- NaN Value Check (Keep existing logic - just warn) ---
        if df.isnull().values.any():
            nan_rows = df[df.isnull().any(axis=1)].index.tolist()
            print(f"Warning: Found NaN values in CSV at rows (0-based index): {nan_rows[:10]}...", file=sys.stderr)

        m: int = len(df)
        n: int = len(df.columns)
        num_classes: int = 0 # Initialize num_classes

        # --- Input Validation (Keep existing logic) ---
        if not (0 <= label_col_index < n):
            print(f"Error: label_col_index ({label_col_index}) is out of bounds for {n} columns.", file=sys.stderr)
            return None, None, None, None, 0, None, None
        if image_col_index is not None:
            if not (0 <= image_col_index < n):
                print(f"Error: image_col_index ({image_col_index}) is out of bounds for {n} columns.", file=sys.stderr)
                return None, None, None, None, 0, None, None
            if image_col_index == label_col_index:
                print(f"Error: image_col_index cannot be the same as label_col_index ({label_col_index}).", file=sys.stderr)
                return None, None, None, None, 0, None, None
            if image_col_type not in ['base64', 'path']:
                print(f"Error: Invalid image_col_type ('{image_col_type}'). Must be 'base64' or 'path'.", file=sys.stderr)
                return None, None, None, None, 0, None, None
            label_col_name: str = df.columns[label_col_index]
            image_col_name: str = df.columns[image_col_index]
        else:
             if n <= 1:
                 print(f"Error: CSV file {csv_path} has {n} columns. Needs >= 2 (label + features) if not using image_col_index.", file=sys.stderr)
                 return None, None, None, None, 0, None, None

        # --- Data Loading and Processing ---
        X_processed: Optional[np.ndarray] = None # Holds processed features (num_samples, features)
        Y_labels: Optional[np.ndarray] = None    # Holds processed labels (num_samples,)
        img_dims: Optional[Tuple[int, int, int]] = None # Will be determined by processing or guessing

        if image_col_index is not None:
            # --- Mode: Load from Image Column (Base64 or Path) ---
            print(f"Processing CSV with image references (Label: col {label_col_index}, Image: col {image_col_index}, Type: {image_col_type})...", file=sys.stderr)
            all_X_data: List[np.ndarray] = []
            all_Y_labels_raw: List[Any] = []

            for index, row in df.iterrows():
                label: Any = row.iloc[label_col_index]
                image_ref: Any = row.iloc[image_col_index]
                img_vector: Optional[np.ndarray] = None

                try:
                    if pd.isna(image_ref) or pd.isna(label):
                        continue # Skip if either image ref or label is NaN

                    image_ref_str = str(image_ref)
                    if image_col_type == 'base64':
                        img_vector = process_image_from_base64(image_ref_str)
                    elif image_col_type == 'path':
                        if not os.path.isabs(image_ref_str):
                            csv_dir = os.path.dirname(csv_path)
                            potential_path = os.path.join(csv_dir, image_ref_str)
                            if os.path.exists(potential_path):
                                image_ref_str = potential_path
                        img_vector = process_image_from_path(image_ref_str)

                    if img_vector is not None:
                        all_X_data.append(img_vector)
                        all_Y_labels_raw.append(label)

                except Exception as img_err:
                    print(f"  Warning: Skipping row {index+1} (Label: '{label}') - Error processing image ref '{str(image_ref)[:50]}...': {img_err}", file=sys.stderr)

            num_samples = len(all_X_data)
            if num_samples == 0:
                 print("Error: No valid images could be processed from the specified column.", file=sys.stderr)
                 return None, None, None, None, 0, None, None
            print(f"Successfully processed {num_samples} image references.", file=sys.stderr)

            # Process Labels (after filtering bad images/labels)
            try:
                Y_labels = np.array(all_Y_labels_raw).astype(int)
                unique_labels = np.unique(Y_labels)
                num_classes = len(unique_labels)
                print(f"Found {num_classes} unique integer labels.", file=sys.stderr)
                if num_classes > 0 and not np.all(unique_labels == np.arange(num_classes)):
                    print("Remapping labels to be contiguous from 0...", file=sys.stderr)
                    label_map = {val: i for i, val in enumerate(sorted(unique_labels))}
                    Y_labels = np.array([label_map[val] for val in Y_labels], dtype=int)
            except ValueError:
                 print("Labels appear to be non-integer. Mapping to integers...", file=sys.stderr)
                 unique_labels_list = sorted(list(set(all_Y_labels_raw)))
                 num_classes = len(unique_labels_list)
                 if num_classes == 0:
                     print("Error: No valid non-NaN labels found.", file=sys.stderr)
                     return None, None, None, None, 0, None, None
                 label_map = {name: i for i, name in enumerate(unique_labels_list)}
                 Y_labels = np.array([label_map[name] for name in all_Y_labels_raw], dtype=int)
                 print(f"Found {num_classes} unique string labels and mapped them to 0-{num_classes-1}.", file=sys.stderr)

            X_processed = np.array(all_X_data) # Shape (num_samples, 784)
            img_dims = (28, 28, 1) # Assumes image processing yields 28x28 grayscale

        else:
            # --- Mode: Load from Pixel Columns --- #
            print(f"Processing CSV with pixel data (Label: col {label_col_index})...", file=sys.stderr)
            try:
                 Y_labels_series = df.iloc[:, label_col_index]
                 if Y_labels_series.isnull().any():
                     raise ValueError(f"NaN values found in label column {label_col_index}. Clean the data.")
                 Y_labels = Y_labels_series.astype(int).to_numpy()

                 feature_indices = [i for i in range(n) if i != label_col_index]
                 X_features_df = df.iloc[:, feature_indices]
                 if X_features_df.isnull().values.any():
                     raise ValueError("NaN values found in feature columns. Clean the data or implement imputation.")
                 X_processed = X_features_df.to_numpy(dtype=float)
            except ValueError as ve:
                 print(f"Error processing pixel/label columns: {ve}", file=sys.stderr)
                 return None, None, None, None, 0, None, None
            except Exception as e:
                 print(f"Error preparing data from pixel columns: {e}", file=sys.stderr)
                 return None, None, None, None, 0, None, None

            num_samples = X_processed.shape[0]
            num_features = X_processed.shape[1]
            print(f"Loaded {num_samples} samples with {num_features} features each.", file=sys.stderr)

            # --- Infer num_classes and remap labels --- #
            unique_labels = np.unique(Y_labels)
            num_classes = len(unique_labels)
            if num_classes > 0 and not np.all(unique_labels == np.arange(num_classes)):
                print("Remapping labels to be contiguous from 0...", file=sys.stderr)
                label_map = {val: i for i, val in enumerate(sorted(unique_labels))}
                Y_labels = np.array([label_map[val] for val in Y_labels], dtype=int)
            print(f"Found {num_classes} unique integer labels.", file=sys.stderr)

            # --- Determine image dims if reshaping to image --- #
            if num_features == 784: img_dims = (28, 28, 1)
            elif num_features == 3072: img_dims = (32, 32, 3)
            elif num_features == 1024: img_dims = (32, 32, 1)
            else:
                sqrt_features = int(np.sqrt(num_features))
                if sqrt_features * sqrt_features == num_features:
                    img_dims = (sqrt_features, sqrt_features, 1)
                    print(f"Guessed image dimensions {img_dims} from {num_features} features.", file=sys.stderr)
                else:
                    img_dims = None
                    if return_shape == 'image':
                         print(f"Error: Cannot reshape pixel data to 'image' shape for {num_features} features.", file=sys.stderr)
                         return None, None, None, None, 0, None, None

            # Normalize pixel values (assuming 0-255 range)
            X_processed = X_processed / 255.0

        # --- Final Checks --- #
        if X_processed is None or Y_labels is None:
            print("Error: Data loading failed before splitting.", file=sys.stderr)
            return None, None, None, None, 0, None, None
        if len(X_processed) != len(Y_labels):
            print(f"Error: Mismatch samples between features ({len(X_processed)}) and labels ({len(Y_labels)}).", file=sys.stderr)
            return None, None, None, None, 0, None, None
        if num_classes <= 1:
               print(f"Error: Only {num_classes} unique class(es) found. Need at least 2 for training.", file=sys.stderr)
               return None, None, None, None, 0, None, None

        # --- Store raw data for debugging before split/reshape --- #
        # Ensure X_processed is 2D before calling helper
        X_for_raw = X_processed
        if X_processed.ndim == 4:
             X_for_raw = X_processed.reshape(X_processed.shape[0], -1)
        raw_X_train_flat, raw_Y_train, _, _ = _reshape_and_split_data(X_for_raw, Y_labels, validation_split, 'flattened', img_dims)
        # ---------------------------------------------------------- #

        # --- Split and Reshape --- #
        X_train, Y_train, X_dev, Y_dev = _reshape_and_split_data(
            X_processed, Y_labels, validation_split, return_shape, img_dims
        )
        # ------------------------- #

        return X_train, Y_train, X_dev, Y_dev, num_classes, raw_X_train_flat, raw_Y_train

    except pd.errors.EmptyDataError:
        print(f"Error: CSV file is empty or contains no data: {csv_path}", file=sys.stderr)
        return None, None, None, None, 0, None, None
    except Exception as e:
        print(f"Error loading or processing CSV dataset {csv_path}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None, None, None, None, 0, None, None

# --- Emoji Loader --- #
def load_emoji_dataset(csv_path: str,
                       image_column: str = 'Google',
                       validation_split: float = 0.1,
                       return_shape: str = 'flattened'
                      ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int, Optional[List[str]]]:
    """Loads the emoji dataset, targeting specific image columns (Base64).

    Args:
        csv_path (str): Path to the emoji dataset CSV file.
        image_column (str): Name of the column containing base64 image strings.
        validation_split (float): Fraction for validation set.
        return_shape (str): Desired output shape: 'flattened' or 'image'.

    Returns:
        Tuple: (X_train, Y_train, X_dev, Y_dev, num_classes, class_names)
               Feature shapes depend on return_shape.
               Returns (None, ..., None, 0, None) on failure.

    """
    print(f"--- Loading Emoji Dataset ({image_column} column) --- ", file=sys.stderr)

    if not os.path.exists(csv_path):
        print(f"Error: Emoji CSV file not found at {csv_path}", file=sys.stderr)
        return None, None, None, None, 0, None

    try:
        df = pd.read_csv(csv_path)
        # Convert image column name from input to lowercase for comparison
        image_column_lower = image_column.strip().lower()

        # Normalize DataFrame columns to lowercase stripped strings for robust checking
        original_columns = list(df.columns)
        df.columns = [col.strip().lower() for col in df.columns]
        normalized_columns = list(df.columns)

        # Check using lowercase names
        if image_column_lower not in df.columns:
            print(f"Error: Specified image_column '{image_column}' (normalized to '{image_column_lower}') not found in CSV columns: {normalized_columns}.", file=sys.stderr)
            return None, None, None, None, 0, None
        # Check for 'name' (lowercase)
        if 'name' not in df.columns:
            print(f"Error: Required 'name' column not found in CSV columns: {normalized_columns}.", file=sys.stderr)
            return None, None, None, None, 0, None

        # Drop NA using lowercase column names
        df_filtered = df.dropna(subset=[image_column_lower, 'name'])
        num_samples_original = len(df)
        num_samples_filtered = len(df_filtered)
        if num_samples_filtered < num_samples_original:
            # Use original input image_column for user message
            print(f"  Warning: Dropped {num_samples_original - num_samples_filtered} rows due to missing values in '{image_column}' or 'name'.", file=sys.stderr)

        if num_samples_filtered == 0:
            print(f"Error: No valid samples found after filtering for column '{image_column}'.", file=sys.stderr)
            return None, None, None, None, 0, None

        all_X_data = []
        all_Y_names = []
        img_dims = (28, 28, 1) # Assume processing yields 28x28 grayscale

        for index, row in df_filtered.iterrows():
            # Access using lowercase column names
            base64_str = row[image_column_lower]
            name = row['name']
            img_vector = process_image_from_base64(base64_str)
            if img_vector is not None:
                all_X_data.append(img_vector)
                all_Y_names.append(name)

        num_valid_samples = len(all_X_data)
        if num_valid_samples == 0:
            print(f"Error: Failed to process any images from column '{image_column}'.", file=sys.stderr)
            return None, None, None, None, 0, None
        print(f"Successfully processed {num_valid_samples} emoji images.", file=sys.stderr)

        X_processed = np.array(all_X_data)

        unique_names = sorted(list(set(all_Y_names)))
        num_classes = len(unique_names)
        name_to_int = {name: i for i, name in enumerate(unique_names)}
        Y_labels = np.array([name_to_int[name] for name in all_Y_names], dtype=int)

        print(f"Found {num_classes} unique emoji classes.", file=sys.stderr)

        X_train, Y_train, X_dev, Y_dev = _reshape_and_split_data(
            X_processed, Y_labels, validation_split, return_shape, img_dims
        )

        return X_train, Y_train, X_dev, Y_dev, num_classes, unique_names

    except Exception as e:
        print(f"Error loading or processing Emoji dataset {csv_path}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None, None, None, None, 0, None

# --- CIFAR-10 Loader --- #
def load_cifar10_dataset(data_dir: str,
                           validation_split: int = 5000,
                           return_shape: str = 'flattened'
                          ) -> LoadResult:
    """Loads the CIFAR-10 dataset from the directory containing the Python version batches.

    Args:
        data_dir (str): The path to the 'cifar-10-batches-py' directory.
        validation_split (int): Number of samples for the validation set.
        return_shape (str): 'flattened' (features, samples) or 'image' (samples, H, W, C).

    Returns:
        LoadResult: Tuple containing train/dev data, num_classes, and raw data.
                    Returns tuple of Nones and 0 if loading fails.
    """
    print(f"--- Loading CIFAR-10 Dataset (from {data_dir}) ---", file=sys.stderr)
    X_train_flat, Y_train, X_dev_flat, Y_dev = None, None, None, None
    raw_X_train_flattened, raw_Y_train = None, None
    num_classes = 10 # CIFAR-10 has 10 classes

    try:
        # --- Load Class Names (batches.meta) --- #
        meta_path = os.path.join(data_dir, 'batches.meta') # Path relative to data_dir
        if not os.path.exists(meta_path):
            print(f"Error: CIFAR-10 batches.meta file not found at: {meta_path}", file=sys.stderr)
            return None, None, None, None, 0, None, None

        print(f"  Loading metadata from: {meta_path}", file=sys.stderr)
        with open(meta_path, 'rb') as fo:
            meta = pickle.load(fo, encoding='bytes')
        class_names = [name.decode('utf-8') for name in meta[b'label_names']]
        print(f"  Class names: {class_names}", file=sys.stderr)

        # --- Load Training Data (data_batch_1 to data_batch_5) --- #
        all_X_train = []
        all_Y_train = []
        print("  Loading training batches...", file=sys.stderr)
        for i in range(1, 6):
            batch_path = os.path.join(data_dir, f'data_batch_{i}') # Path relative to data_dir
            if not os.path.exists(batch_path):
                 print(f"Error: CIFAR-10 training batch file not found: {batch_path}", file=sys.stderr)
                 return None, None, None, None, 0, None, None
            with open(batch_path, 'rb') as fo:
                batch = pickle.load(fo, encoding='bytes')
                all_X_train.append(batch[b'data'])
                all_Y_train.extend(batch[b'labels'])
        X_train_combined = np.vstack(all_X_train)
        Y_train_combined = np.array(all_Y_train)
        print(f"    Training data combined shape: {X_train_combined.shape}", file=sys.stderr)

        # --- Load Test Data (test_batch) --- #
        test_batch_path = os.path.join(data_dir, 'test_batch') # Path relative to data_dir
        if not os.path.exists(test_batch_path):
             print(f"Error: CIFAR-10 test batch file not found: {test_batch_path}", file=sys.stderr)
             return None, None, None, None, 0, None, None

        print("  Loading test batch...", file=sys.stderr)
        with open(test_batch_path, 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
        X_test_combined = batch[b'data']
        Y_test_combined = np.array(batch[b'labels'])
        print(f"    Test data loaded shape: {X_test_combined.shape}", file=sys.stderr)

        # --- Combine, Normalize, and Reshape/Split --- #
        # Combine all data before splitting
        X_all = np.vstack((X_train_combined, X_test_combined))
        Y_all = np.concatenate((Y_train_combined, Y_test_combined))

        # Normalize pixel values to [0, 1]
        print("  Normalizing pixel values...", file=sys.stderr)
        X_all_normalized = X_all.astype(np.float32) / 255.0

        # Store raw data BEFORE reshaping/splitting for potential inspection
        # (Shape: samples x features)
        raw_X_train_flattened = X_all_normalized # Storing all data as 'raw_train' for now
        raw_Y_train = Y_all

        # Reshape and Split using the helper function
        # CIFAR image dimensions are (32, 32, 3)
        print(f"  Splitting and reshaping to '{return_shape}'...", file=sys.stderr)
        X_train, Y_train, X_dev, Y_dev = _reshape_and_split_data(
            X_combined=X_all_normalized, # Pass normalized data
            Y_labels=Y_all,
            validation_split=validation_split,
            return_shape=return_shape,
            img_dims=(32, 32, 3) # Specify CIFAR dimensions
        )

        if X_train is None or Y_train is None or X_dev is None or Y_dev is None:
            raise ValueError("Data splitting/reshaping failed.")

        print("--- CIFAR-10 Loading Complete ---", file=sys.stderr)
        return X_train, Y_train, X_dev, Y_dev, num_classes, raw_X_train_flattened, raw_Y_train

    except FileNotFoundError as e:
        print(f"Error: CIFAR-10 file not found: {e}", file=sys.stderr)
        return None, None, None, None, 0, None, None
    except KeyError as e:
        print(f"Error: Missing expected key in CIFAR-10 batch file: {e}", file=sys.stderr)
        return None, None, None, None, 0, None, None
    except Exception as e:
        print(f"An unexpected error occurred during CIFAR-10 loading: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None, None, None, None, 0, None, None

# --- QuickDraw Loader --- #
def load_multiple_npy_datasets(npy_file_map: Dict[str, int],
                               validation_split: Union[int, float] = 1000,
                               max_items_per_class: Optional[int] = 5000,
                               return_shape: str = 'flattened'
                               ) -> LoadResult:
    """Loads multiple .npy files (presumably QuickDraw), combines, splits.

    Args:
        npy_file_map (Dict[str, int]): Dictionary mapping file paths to integer labels.
        validation_split (Union[int, float]): Size/fraction for validation set.
        max_items_per_class (Optional[int]): Max items to load from each file.
        return_shape (str): Desired output shape: 'flattened' or 'image'.

    Returns:
        LoadResult: (X_train, Y_train, X_dev, Y_dev, num_classes, raw_X_train_flat, raw_Y_train)
    """
    print(f"--- Loading Multiple NPY Datasets ({len(npy_file_map)} classes) --- ", file=sys.stderr)
    all_X_data = []
    all_Y_labels = []
    num_classes = len(npy_file_map)
    img_dims = (28, 28, 1)

    sorted_files = sorted(npy_file_map.items(), key=lambda item: item[1])

    for file_path, label_index in sorted_files:
        if not os.path.exists(file_path):
            print(f"    Warning: File not found: {file_path}. Skipping class.", file=sys.stderr)
            continue
        try:
            data = np.load(file_path)
            if data.ndim != 2 or data.shape[1] != 784:
                 print(f"    Warning: Unexpected data shape {data.shape} in {file_path}. Expected (samples, 784). Skipping.", file=sys.stderr)
                 continue

            if max_items_per_class is not None and len(data) > max_items_per_class:
                indices = np.random.choice(len(data), max_items_per_class, replace=False)
                data = data[indices]

            num_items = len(data)
            if num_items == 0:
                continue

            all_X_data.append(data)
            all_Y_labels.append(np.full(num_items, label_index, dtype=int))

        except Exception as e:
            print(f"    Error loading NPY file {file_path}: {e}. Skipping class.", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    if not all_X_data:
        print("Error: No data loaded from any NPY files.", file=sys.stderr)
        return None, None, None, None, 0, None, None

    X_processed = np.concatenate(all_X_data, axis=0)
    Y_labels = np.concatenate(all_Y_labels, axis=0)
    print(f"Combined data: X={X_processed.shape}, Y={Y_labels.shape}", file=sys.stderr)

    # Data is already normalized 0-1 from QuickDraw source, but might be float64
    if X_processed.dtype != np.float32:
         X_processed = X_processed.astype(np.float32)

    raw_X_train_flat, raw_Y_train, _, _ = _reshape_and_split_data(X_processed, Y_labels, validation_split, 'flattened', img_dims)

    X_train, Y_train, X_dev, Y_dev = _reshape_and_split_data(
        X_processed, Y_labels, validation_split, return_shape, img_dims
    )

    return X_train, Y_train, X_dev, Y_dev, num_classes, raw_X_train_flat, raw_Y_train

# Helper to get class names if meta file exists
def get_cifar10_class_names(data_dir: str) -> Optional[List[str]]:
    meta_file = os.path.join(data_dir, 'batches.meta') # Construct path directly from data_dir
    try:
        with open(meta_file, 'rb') as fo:
                meta_dict = pickle.load(fo, encoding='bytes')
        if meta_dict and b'label_names' in meta_dict:
                return [name.decode('utf-8') for name in meta_dict[b'label_names']]
    except Exception:
        pass
    return None

# Example usage:
# if __name__ == "__main__":
#     # MNIST (label is column 0)
#     print("--- Loading MNIST (label col 0) ---")
#     mnist_path = 'data/train.csv'
#     load_csv_dataset(mnist_path)
#
#     # Example with label in the last column (create a dummy csv for this)
#     # print("\n--- Loading Dummy (label col -1) ---")
#     # dummy_data = np.random.rand(100, 785)
#     # dummy_data[:, -1] = np.random.randint(0, 5, 100) # Dummy labels 0-4
#     # dummy_df = pd.DataFrame(dummy_data)
#     # dummy_path = 'data/dummy_last_label.csv'
#     # dummy_df.to_csv(dummy_path, index=False, header=False)
#     # load_csv_dataset(dummy_path, label_col_index=-1) # Use -1 for last column 