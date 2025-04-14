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
               Feature shapes depend on return_shape.
    """
    X_train, Y_train, X_dev, Y_dev = None, None, None, None # Initialize return vars

    num_samples = X_combined.shape[0]
    if num_samples == 0:
        print("Warning: No samples to split.", file=sys.stderr)
        if return_shape == 'image' and img_dims:
            empty_features = np.empty((0, img_dims[0], img_dims[1], img_dims[2]), dtype=X_combined.dtype)
        else:
            feature_count = X_combined.shape[1] if X_combined.ndim > 1 else 0
            empty_features = np.empty((feature_count, 0), dtype=X_combined.dtype)
        empty_labels = np.array([], dtype=int)
        return empty_features, empty_labels, empty_features, empty_labels

    # --- Shuffle --- #
    permutation = np.random.permutation(num_samples)
    X_shuffled = X_combined[permutation]
    Y_shuffled = Y_labels[permutation]

    # --- Determine Split Index --- #
    split_idx: int
    if isinstance(validation_split, float) and 0 < validation_split < 1:
        split_idx = int(num_samples * (1.0 - validation_split))
        # print(f"Using validation split fraction: {validation_split:.2f}", file=sys.stderr)
    elif isinstance(validation_split, int) and 1 <= validation_split < num_samples:
        split_idx = num_samples - validation_split
        # print(f"Using fixed validation split size: {validation_split}", file=sys.stderr)
    else:
        split_val = 1000
        if num_samples <= split_val * 2:
            split_val = max(1, int(num_samples * 0.1))
            # print(f"Warning: Invalid or large validation_split value. Using 10% ({split_val} samples).", file=sys.stderr)
        # else:
            # print(f"Warning: Invalid validation_split value. Using default ({split_val} samples).", file=sys.stderr)
        split_idx = num_samples - split_val

    # --- Split Data --- #
    if num_samples < 2 or split_idx <= 0 or split_idx >= num_samples:
        print(f"Warning: Not enough samples ({num_samples}) for a validation split. Using all for training.", file=sys.stderr)
        X_train_split = X_shuffled
        Y_train = Y_shuffled
        if return_shape == 'image' and img_dims:
             X_dev_split = np.empty((0, img_dims[0], img_dims[1], img_dims[2]), dtype=X_combined.dtype)
        else:
             feature_count = X_combined.shape[1] if X_combined.ndim > 1 else 0
             X_dev_split = np.empty((0, feature_count), dtype=X_combined.dtype)
        Y_dev = np.array([], dtype=int)
    else:
        X_train_split = X_shuffled[:split_idx]
        Y_train = Y_shuffled[:split_idx]
        X_dev_split = X_shuffled[split_idx:]
        Y_dev = Y_shuffled[split_idx:]

    # --- Reshape Based on return_shape --- #
    reshape_error = False # Flag to track errors during reshaping

    if return_shape == 'image':
        print(f"Reshaping data to 'image' format...", file=sys.stderr)
        if X_train_split.ndim == 4:
            X_train = X_train_split
            X_dev = X_dev_split
            print(f"  Input data already has {X_train_split.ndim} dimensions. Assuming (samples, H, W, C).", file=sys.stderr)
        elif X_train_split.ndim == 2 and img_dims:
            num_features = X_train_split.shape[1]
            expected_features = img_dims[0] * img_dims[1] * img_dims[2]
            if num_features != expected_features:
                print(f"Error: Cannot reshape to image. Expected {expected_features} features for shape {img_dims}, but got {num_features}.", file=sys.stderr)
                reshape_error = True
            else:
                target_shape_train = (X_train_split.shape[0], img_dims[0], img_dims[1], img_dims[2])
                target_shape_dev = (X_dev_split.shape[0], img_dims[0], img_dims[1], img_dims[2])
                try:
                    X_train = X_train_split.reshape(target_shape_train)
                    X_dev = X_dev_split.reshape(target_shape_dev)
                    print(f"  Reshaped flattened data to {target_shape_train} and {target_shape_dev}.", file=sys.stderr)
                except ValueError as e:
                    print(f"Error reshaping training data to image format: {e}", file=sys.stderr)
                    reshape_error = True
        else:
            print(f"Error: Cannot reshape to image. Input data has {X_train_split.ndim} dimensions or img_dims not provided.", file=sys.stderr)
            reshape_error = True
        # Handle reshape error for image format
        if reshape_error:
             if img_dims:
                 empty_img = np.empty((0, img_dims[0], img_dims[1], img_dims[2]), dtype=X_combined.dtype)
                 X_train, X_dev = empty_img, empty_img
             else:
                 print("  FATAL: Cannot provide empty image array due to missing img_dims.", file=sys.stderr)
                 empty_features = np.empty((0,0), dtype=X_combined.dtype)
                 X_train, X_dev = empty_features, empty_features

    elif return_shape == 'flattened':
        print("Reshaping data to 'flattened' format (features, samples)...", file=sys.stderr)
        if X_train_split.ndim == 4:
            num_train_samples = X_train_split.shape[0]
            num_dev_samples = X_dev_split.shape[0]
            num_features = X_train_split.shape[1] * X_train_split.shape[2] * X_train_split.shape[3]
            X_train = X_train_split.reshape(num_train_samples, num_features).T
            X_dev = X_dev_split.reshape(num_dev_samples, num_features).T
            print(f"  Flattened image data to ({num_features}, {num_train_samples}) and ({num_features}, {num_dev_samples}).", file=sys.stderr)
        elif X_train_split.ndim == 2:
            X_train = X_train_split.T
            X_dev = X_dev_split.T
            print(f"  Transposed flattened data.", file=sys.stderr)
        else:
            print(f"Error: Cannot flatten. Input data has {X_train_split.ndim} dimensions.", file=sys.stderr)
            reshape_error = True
        # Handle reshape error for flattened format
        if reshape_error:
             feature_count = X_train_split.shape[1] if X_train_split.ndim == 2 else 0
             X_train = np.empty((feature_count, 0), dtype=X_combined.dtype)
             X_dev = np.empty((feature_count, 0), dtype=X_combined.dtype)
             print(f"  Falling back to empty flattened arrays.", file=sys.stderr)

    else: # Unknown return_shape
        print(f"Error: Unknown return_shape '{return_shape}'. Defaulting to flattened (transposed).", file=sys.stderr)
        if X_train_split.ndim == 2:
            X_train = X_train_split.T
            X_dev = X_dev_split.T
        else:
            feature_count = 0
            X_train = np.empty((feature_count, 0), dtype=X_combined.dtype)
            X_dev = np.empty((feature_count, 0), dtype=X_combined.dtype)

    # --- Ensure float32 type --- #
    if X_train is not None and X_train.dtype != np.float32:
        X_train = X_train.astype(np.float32)
    if X_dev is not None and X_dev.dtype != np.float32:
        X_dev = X_dev.astype(np.float32)

    print(f"Final Shapes: X_train={X_train.shape if X_train is not None else 'None'}, Y_train={Y_train.shape}, X_dev={X_dev.shape if X_dev is not None else 'None'}, Y_dev={Y_dev.shape}", file=sys.stderr)

    # --- Final Return --- #
    # Ensure all return variables are assigned, even if empty from errors
    if X_train is None: # Handle cases where reshape failed catastrophically
         if return_shape == 'image' and img_dims:
              X_train = np.empty((0, img_dims[0], img_dims[1], img_dims[2]), dtype=X_combined.dtype)
              X_dev = np.empty((0, img_dims[0], img_dims[1], img_dims[2]), dtype=X_combined.dtype)
         else:
              feature_count = X_combined.shape[1] if X_combined.ndim > 1 else 0
              X_train = np.empty((feature_count, 0), dtype=X_combined.dtype)
              X_dev = np.empty((feature_count, 0), dtype=X_combined.dtype)

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
        if image_column not in df.columns:
            print(f"Error: Specified image_column '{image_column}' not found in CSV.", file=sys.stderr)
            return None, None, None, None, 0, None
        if 'Name' not in df.columns:
            print(f"Error: Required 'Name' column not found in CSV.", file=sys.stderr)
            return None, None, None, None, 0, None

        df_filtered = df.dropna(subset=[image_column, 'Name'])
        num_samples_original = len(df)
        num_samples_filtered = len(df_filtered)
        if num_samples_filtered < num_samples_original:
            print(f"  Warning: Dropped {num_samples_original - num_samples_filtered} rows due to missing values in '{image_column}' or 'Name'.", file=sys.stderr)

        if num_samples_filtered == 0:
            print(f"Error: No valid samples found after filtering for column '{image_column}'.", file=sys.stderr)
            return None, None, None, None, 0, None

        all_X_data = []
        all_Y_names = []
        img_dims = (28, 28, 1) # Assume processing yields 28x28 grayscale

        for index, row in df_filtered.iterrows():
            base64_str = row[image_column]
            name = row['Name']
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
    """Loads the CIFAR-10 dataset from the pickled batch files.

    Args:
        data_dir (str): The directory containing the 'cifar-10-batches-py' subdirectory.
        validation_split (int): Number of samples for the validation set (taken from training).
        return_shape (str): Desired output shape: 'flattened' or 'image'.

    Returns:
        LoadResult: (X_train, Y_train, X_dev, Y_dev, num_classes, raw_X_train_flat, raw_Y_train)
    """
    print("--- Loading CIFAR-10 Dataset --- ", file=sys.stderr)
    cifar10_subdir = os.path.join(data_dir, 'cifar-10-batches-py')
    if not os.path.isdir(cifar10_subdir):
        print(f"Error: CIFAR-10 directory not found: {cifar10_subdir}", file=sys.stderr)
        return None, None, None, None, 0, None, None

    num_classes = 10
    img_dims = (32, 32, 3)

    try:
        all_train_X = []
        all_train_Y = []
        for i in range(1, 6):
            batch_file = os.path.join(cifar10_subdir, f'data_batch_{i}')
            if not os.path.exists(batch_file):
                print(f"Error: CIFAR-10 training batch file not found: {batch_file}", file=sys.stderr)
                return None, None, None, None, 0, None, None
            with open(batch_file, 'rb') as fo:
                batch_dict = pickle.load(fo, encoding='bytes')
                all_train_X.append(batch_dict[b'data'])
                all_train_Y.extend(batch_dict[b'labels'])

        X_train_combined = np.concatenate(all_train_X)
        Y_train_combined = np.array(all_train_Y, dtype=int)
        print(f"Loaded {X_train_combined.shape[0]} training samples.", file=sys.stderr)

        num_train_total = X_train_combined.shape[0]
        if not (0 < validation_split < num_train_total):
            print(f"Warning: Invalid validation_split ({validation_split}). Using default 5000.", file=sys.stderr)
            validation_split = 5000

        permutation = np.random.permutation(num_train_total)
        X_train_shuffled = X_train_combined[permutation]
        Y_train_shuffled = Y_train_combined[permutation]

        X_dev_processed = X_train_shuffled[:validation_split]
        Y_dev = Y_train_shuffled[:validation_split]
        X_train_processed = X_train_shuffled[validation_split:]
        Y_train = Y_train_shuffled[validation_split:]

        raw_X_train_flat = (X_train_processed / 255.0).astype(np.float32).T
        raw_Y_train = Y_train.copy()

        X_train = X_train_processed.astype(np.float32) / 255.0
        X_dev = X_dev_processed.astype(np.float32) / 255.0

        if return_shape == 'image':
            print("Reshaping CIFAR-10 data to 'image' format (samples, 32, 32, 3)...", file=sys.stderr)
            try:
                X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                X_dev = X_dev.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            except ValueError as e:
                 print(f"Error reshaping CIFAR-10 data to image format: {e}", file=sys.stderr)
                 return None, None, None, None, 0, None, None
        elif return_shape == 'flattened':
            print("Reshaping CIFAR-10 data to 'flattened' format (3072, samples)...", file=sys.stderr)
            X_train = X_train.T
            X_dev = X_dev.T
        else:
             print(f"Error: Unknown return_shape '{return_shape}' for CIFAR-10. Defaulting to flattened.", file=sys.stderr)
             X_train = X_train.T
             X_dev = X_dev.T

        print(f"Final Shapes: X_train={X_train.shape}, Y_train={Y_train.shape}, X_dev={X_dev.shape}, Y_dev={Y_dev.shape}", file=sys.stderr)

        return X_train, Y_train, X_dev, Y_dev, num_classes, raw_X_train_flat, raw_Y_train

    except Exception as e:
        print(f"Error loading or processing CIFAR-10 dataset from {data_dir}: {e}", file=sys.stderr)
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
    cifar10_subdir = os.path.join(data_dir, 'cifar-10-batches-py')
    meta_file = os.path.join(cifar10_subdir, 'batches.meta')
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