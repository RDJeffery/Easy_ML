import numpy as np
import pandas as pd
import os
import sys
import traceback
import base64
from io import BytesIO
from PIL import Image # Requires Pillow
from typing import Tuple, Optional, Union, Dict, List, Any

# Import the new utility functions
from utils.image_processor import process_image_from_base64, process_image_from_path

# Define a type alias for the standard return tuple of loaders
LoadResult = Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]
# (X_train, Y_train, X_dev, Y_dev, num_classes)

# --- Helper Function for Reshaping and Splitting --- #

def _reshape_and_split_data(X_combined_flat: np.ndarray, Y_labels: np.ndarray, validation_split: Union[int, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffles, splits data into train/dev sets, and reshapes features.

    Assumes X_combined_flat is shape (num_samples, 784).
    Reshapes features to (784, num_samples) for the network.

    Args:
        X_combined_flat (np.ndarray): Combined feature data, flattened (num_samples, 784).
        Y_labels (np.ndarray): Corresponding labels (num_samples,).
        validation_split (Union[int, float]): Number or fraction of samples for validation.

    Returns:
        tuple: (X_train, Y_train, X_dev, Y_dev)
               Features are reshaped to (784, m).
    """
    num_samples = X_combined_flat.shape[0]
    if num_samples == 0:
        print("Warning: No samples to split.", file=sys.stderr)
        # Return empty arrays with correct feature dimension (784, 0)
        empty_features = np.array([]).reshape(784, 0)
        empty_labels = np.array([], dtype=int)
        return empty_features, empty_labels, empty_features, empty_labels

    # Shuffle combined data
    permutation = np.random.permutation(num_samples)
    X_shuffled_flat = X_combined_flat[permutation, :]
    Y_shuffled = Y_labels[permutation]

    # Determine split index
    split_idx: int
    if isinstance(validation_split, float) and 0 < validation_split < 1:
        split_idx = int(num_samples * (1.0 - validation_split))
        print(f"Using validation split fraction: {validation_split:.2f}", file=sys.stderr)
    elif isinstance(validation_split, int) and 1 <= validation_split < num_samples:
        split_idx = num_samples - validation_split
        print(f"Using fixed validation split size: {validation_split}", file=sys.stderr)
    else:
        split_val = 1000
        if num_samples <= split_val * 2: # Ensure train set is at least val set size
            split_val = max(1, int(num_samples * 0.1)) # Use at least 1 sample for val
            print(f"Warning: Invalid or large validation_split value. Using 10% ({split_val} samples).", file=sys.stderr)
        else:
            print(f"Warning: Invalid validation_split value. Using default ({split_val} samples).", file=sys.stderr)
        split_idx = num_samples - split_val

    # Handle edge cases for split index
    if num_samples < 2 or split_idx <= 0 or split_idx >= num_samples:
        print(f"Warning: Not enough samples ({num_samples}) for a validation split. Using all for training.", file=sys.stderr)
        X_train_flat = X_shuffled_flat
        Y_train = Y_shuffled
        X_dev_flat = np.array([]).reshape(X_combined_flat.shape[1], 0) # (784, 0)
        Y_dev = np.array([], dtype=int)
    else:
        X_train_flat = X_shuffled_flat[:split_idx, :]
        Y_train = Y_shuffled[:split_idx]
        X_dev_flat = X_shuffled_flat[split_idx:, :]
        Y_dev = Y_shuffled[split_idx:]

    # Reshape features for the network (num_features, num_samples)
    X_train = X_train_flat.T
    X_dev = X_dev_flat.T

    print(f"  Train set: X={X_train.shape}, Y={Y_train.shape}", file=sys.stderr)
    print(f"  Dev set:   X={X_dev.shape}, Y={Y_dev.shape}", file=sys.stderr)

    return X_train, Y_train, X_dev, Y_dev

# --- Main Data Loaders --- #

def load_csv_dataset(csv_path: str,
                     validation_split: Union[int, float] = 1000,
                     label_col_index: int = 0,
                     image_col_index: Optional[int] = None,
                     image_col_type: Optional[str] = None) -> LoadResult:
    """Loads data from a generic CSV file.

    Handles either raw pixel data across columns or image references (base64/path)
    in a specified column. Shuffles, splits, normalizes/processes features.

    Args:
        csv_path (str): Path to the dataset CSV file.
        validation_split (Union[int, float]): Number or fraction for validation set. Defaults to 1000 samples.
        label_col_index (int): Index of the label column. Defaults to 0.
        image_col_index (Optional[int]): Index of the image data column (base64/path).
                                        If None, assumes raw pixel data. Defaults to None.
        image_col_type (Optional[str]): Type ('base64' or 'path') if using image_col_index.
                                         Defaults to None.

    Returns:
        LoadResult: A tuple (X_train, Y_train, X_dev, Y_dev, num_classes).
                    Returns (None, None, None, None, 0) on failure.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}", file=sys.stderr)
        return None, None, None, None, 0

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"Error: CSV file is empty: {csv_path}", file=sys.stderr)
            return None, None, None, None, 0

        # --- NaN Value Check ---
        if df.isnull().values.any():
            nan_rows = df[df.isnull().any(axis=1)].index.tolist()
            print(f"Warning: Found NaN values in CSV at rows (0-based index): {nan_rows[:10]}...", file=sys.stderr)
            # Option 1: Drop rows with any NaN
            # df.dropna(inplace=True)
            # print(f"  Dropped {len(nan_rows)} rows containing NaN values.", file=sys.stderr)
            # Option 2: Fill NaNs (less safe for features/labels)
            # df.fillna(0, inplace=True) # Example: fill with 0
            # print(f"  Filled NaN values (e.g., with 0). Use with caution.", file=sys.stderr)
            # **Decision: For now, just warn. Dropping might be better later.**
            if df.empty:
                print(f"Error: CSV became empty after handling NaN values (if dropping was enabled).", file=sys.stderr)
                return None, None, None, None, 0
        # ---------------------

        m: int = len(df)
        n: int = len(df.columns)
        num_classes: int = 0 # Initialize num_classes

        # --- Input Validation ---
        if not (0 <= label_col_index < n):
            print(f"Error: label_col_index ({label_col_index}) is out of bounds for {n} columns.", file=sys.stderr)
            return None, None, None, None, 0
        if image_col_index is not None:
            if not (0 <= image_col_index < n):
                print(f"Error: image_col_index ({image_col_index}) is out of bounds for {n} columns.", file=sys.stderr)
                return None, None, None, None, 0
            if image_col_index == label_col_index:
                print(f"Error: image_col_index cannot be the same as label_col_index ({label_col_index}).", file=sys.stderr)
                return None, None, None, None, 0
            if image_col_type not in ['base64', 'path']:
                print(f"Error: Invalid image_col_type ('{image_col_type}'). Must be 'base64' or 'path'.", file=sys.stderr)
                return None, None, None, None, 0
            # Ensure column names exist if using iloc later
            label_col_name: str = df.columns[label_col_index]
            image_col_name: str = df.columns[image_col_index]
        else:
             if n <= 1:
                 print(f"Error: CSV file {csv_path} has {n} columns. Needs >= 2 (label + features) if not using image_col_index.", file=sys.stderr)
                 return None, None, None, None, 0

        # --- Data Loading and Processing ---
        X_combined_flat: Optional[np.ndarray] = None
        Y_labels: Optional[np.ndarray] = None

        if image_col_index is not None:
            # --- Mode: Load from Image Column (Base64 or Path) ---
            print(f"Processing CSV with image references (Label: col {label_col_index}, Image: col {image_col_index}, Type: {image_col_type})...", file=sys.stderr)
            all_X_data: List[np.ndarray] = []
            all_Y_labels_raw: List[Any] = [] # Store original labels first

            for index, row in df.iterrows():
                label: Any = row.iloc[label_col_index]
                image_ref: Any = row.iloc[image_col_index]
                img_vector: Optional[np.ndarray] = None

                try:
                    # Check for NaN in image reference before processing
                    if pd.isna(image_ref):
                        print(f"  Warning: Skipping row {index+1} (Label: '{label}') - NaN value found in image reference column.", file=sys.stderr)
                        continue

                    image_ref_str = str(image_ref) # Ensure string for processing funcs
                    if image_col_type == 'base64':
                        img_vector = process_image_from_base64(image_ref_str)
                    elif image_col_type == 'path':
                        # Resolve relative paths based on CSV location
                        if not os.path.isabs(image_ref_str):
                            csv_dir = os.path.dirname(csv_path)
                            potential_path = os.path.join(csv_dir, image_ref_str)
                            if os.path.exists(potential_path):
                                image_ref_str = potential_path
                            else:
                                # Try relative to workspace root as fallback (handled by process_image_from_path)
                                pass
                        img_vector = process_image_from_path(image_ref_str)

                    if img_vector is None:
                         print(f"  Warning: Skipping row {index+1} (Label: '{label}') - Image processing failed for ref: '{image_ref_str[:50]}...'", file=sys.stderr)
                         continue

                    all_X_data.append(img_vector)
                    all_Y_labels_raw.append(label)

                except Exception as img_err:
                    print(f"  Warning: Skipping row {index+1} (Label: '{label}') - Error processing image ref '{str(image_ref)[:50]}...': {img_err}", file=sys.stderr)

            num_samples = len(all_X_data)
            if num_samples == 0:
                 print("Error: No valid images could be processed from the specified column.", file=sys.stderr)
                 return None, None, None, None, 0
            print(f"Successfully processed {num_samples} image references.", file=sys.stderr)

            # Convert labels to integers (handle potential NaN in labels now)
            Y_processed_labels: List[Any] = [lbl for lbl in all_Y_labels_raw if not pd.isna(lbl)]
            if len(Y_processed_labels) != num_samples:
                print(f"Warning: {num_samples - len(Y_processed_labels)} rows were skipped due to NaN labels.", file=sys.stderr)
                # This requires filtering X_data as well - complex, maybe error out instead?
                print("Error: Handling NaN labels mixed with valid image data is complex. Please clean the label column.", file=sys.stderr)
                return None, None, None, None, 0 # Error out for now

            try:
                Y_labels = np.array(Y_processed_labels).astype(int)
                unique_labels = np.unique(Y_labels)
                num_classes = len(unique_labels)
                print(f"Found {num_classes} unique integer labels.", file=sys.stderr)
                # Remap labels to be 0 to num_classes-1 if needed
                if num_classes > 0 and not np.all(unique_labels == np.arange(num_classes)):
                    print("Remapping labels to be contiguous from 0...", file=sys.stderr)
                    label_map = {val: i for i, val in enumerate(sorted(unique_labels))}
                    Y_labels = np.array([label_map[val] for val in Y_labels], dtype=int)

            except ValueError:
                 # Labels are likely strings
                 print("Labels appear to be non-integer. Mapping to integers...", file=sys.stderr)
                 unique_labels = sorted(list(set(Y_processed_labels)))
                 num_classes = len(unique_labels)
                 if num_classes == 0:
                     print("Error: No valid non-NaN labels found.", file=sys.stderr)
                     return None, None, None, None, 0
                 label_map = {name: i for i, name in enumerate(unique_labels)}
                 Y_labels = np.array([label_map[name] for name in Y_processed_labels], dtype=int)
                 print(f"Found {num_classes} unique string labels and mapped them to 0-{num_classes-1}.", file=sys.stderr)

            X_combined_flat = np.array(all_X_data) # Shape (num_samples, 784)

        else:
            # --- Mode: Load from Pixel Columns ---
            print(f"Processing CSV with raw pixel data (Label: col {label_col_index})...", file=sys.stderr)
            Y_all_series = df.iloc[:, label_col_index]
            # Check for NaN labels
            if Y_all_series.isnull().any():
                print(f"Error: NaN values found in label column ({label_col_index}). Please clean the data.", file=sys.stderr)
                return None, None, None, None, 0

            feature_indices = [i for i in range(n) if i != label_col_index]
            X_df_features = df.iloc[:, feature_indices]
            try:
                X_numeric_check = X_df_features.apply(pd.to_numeric, errors='coerce')
                if X_numeric_check.isnull().values.any():
                    print(f"Error: Non-numeric data found in feature columns (pixel mode). Check columns: {feature_indices}.", file=sys.stderr)
                    return None, None, None, None, 0
                X_all_flat = X_numeric_check.to_numpy()
            except Exception as e:
                print(f"Error during numeric check/conversion of feature columns: {e}", file=sys.stderr)
                return None, None, None, None, 0

            # Convert labels (already checked for NaN)
            try:
                Y_labels = Y_all_series.to_numpy().astype(int)
            except ValueError:
                 print(f"Error: Non-integer labels found in label column ({label_col_index}) in pixel mode.", file=sys.stderr)
                 return None, None, None, None, 0

            # Normalize features
            X_normalized_flat = X_all_flat.astype(float) / 255.
            print(f"Raw pixel data shape: {X_normalized_flat.shape}", file=sys.stderr)

            if X_normalized_flat.shape[1] != 784:
                 print(f"Warning: Feature columns ({X_normalized_flat.shape[1]}) != 784. NN might fail.", file=sys.stderr)

            X_combined_flat = X_normalized_flat
            num_classes = int(np.max(Y_labels)) + 1 if len(Y_labels) > 0 else 0
            print(f"Determined {num_classes} classes from integer labels in pixel mode.", file=sys.stderr)

        # --- Final Check before Split --- #
        if X_combined_flat is None or Y_labels is None or len(X_combined_flat) == 0 or len(Y_labels) == 0:
            print("Error: No valid data loaded after processing.", file=sys.stderr)
            return None, None, None, None, 0
        if len(X_combined_flat) != len(Y_labels):
            print(f"Error: Mismatch between number of features ({len(X_combined_flat)}) and labels ({len(Y_labels)}).", file=sys.stderr)
            return None, None, None, None, 0
        if num_classes <= 1:
             print(f"Error: Only {num_classes} unique class(es) found. Need at least 2 for training.", file=sys.stderr)
             # Allow loading but maybe training should check this?
             # return None, None, None, None, num_classes # Return anyway?
             return None, None, None, None, 0 # Consider this an error for now

        # --- Shuffle, Split, Reshape (using helper) ---
        X_train, Y_train, X_dev, Y_dev = _reshape_and_split_data(X_combined_flat, Y_labels, validation_split)

        return X_train, Y_train, X_dev, Y_dev, num_classes

    except pd.errors.EmptyDataError:
        print(f"Error: CSV file is empty or contains only headers: {csv_path}", file=sys.stderr)
        return None, None, None, None, 0
    except pd.errors.ParserError as pe:
        print(f"Error: Failed to parse CSV file '{csv_path}'. Check formatting: {pe}", file=sys.stderr)
        return None, None, None, None, 0
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}", file=sys.stderr)
        return None, None, None, None, 0
    except Exception as e:
        print(f"An unexpected error occurred during CSV loading/processing: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None, None, None, None, 0

def load_npy_dataset(npy_path: str,
                     category_index: int,
                     num_classes: int,
                     validation_split: Union[int, float] = 0.1) -> LoadResult:
    """Loads a Quick, Draw! .npy dataset.

    Args:
        npy_path (str): Path to the .npy file.
        category_index (int): The unique integer index assigned to this category (0 to num_classes-1).
        num_classes (int): The total number of classes across all QuickDraw datasets being used.
                           (Needed for one-hot encoding later).
        validation_split (Union[int, float]): Fraction or number for validation set.
                                               Defaults to 0.1 (10%).

    Returns:
        tuple: (X_train, Y_train, X_dev, Y_dev) where Y contains integer labels (0 to num_classes-1),
               or (None, None, None, None) on error.
    """
    if not os.path.exists(npy_path):
        print(f"Error: NPY file not found at {npy_path}", file=sys.stderr)
        return None, None, None, None, 0

    try:
        # --- Determine Category Index --- 
        # Extract category name (e.g., 'cat' from 'data/quickdraw/cat.npy')
        category_name = os.path.splitext(os.path.basename(npy_path))[0]
        # Need a consistent way to map category name to index across the app.
        # For now, let's assume a simple approach based on alphabetical order
        # of all .npy files in the directory. This is fragile if files change.
        # A better approach would pass the index explicitly or use a shared map.
        # Placeholder: Assume index 0 for now, needs fixing in UI call.
        # TODO: Get category index reliably from the caller (UI)
        # category_index = 0 # <<< Placeholder - This needs to be passed correctly
        # The category_index is now passed directly as an argument.
        print(f"Loading NPY dataset: {category_name} (Index: {category_index}, Total Classes: {num_classes})", file=sys.stderr)


        # --- Load Data ---
        data = np.load(npy_path)
        print(f"  Loaded {npy_path}, shape: {data.shape}", file=sys.stderr)

        # Data is expected to be (num_samples, 784)
        if data.ndim != 2 or data.shape[1] != 784:
             print(f"Error: Unexpected NPY data shape: {data.shape}. Expected (num_samples, 784).", file=sys.stderr)
             return None, None, None, None, 0

        num_samples = data.shape[0]

        # --- Create Labels ---
        # Create a label array with the determined category index
        labels = np.full(num_samples, category_index, dtype=int)

        # --- Preprocessing ---
        # Normalize pixel values to [0, 1]
        data_normalized = data / 255.0

        # --- Shuffle Data --- 
        # Shuffle images and labels together
        permutation = np.random.permutation(num_samples)
        X_shuffled = data_normalized[permutation, :]
        Y_shuffled = labels[permutation]

        # --- Split Data (e.g., 90% train, 10% dev) ---
        # Make sure we have enough samples for a split
        if num_samples < 10:
            print(f"Warning: Very few samples ({num_samples}) in {npy_path}. Using all for training.", file=sys.stderr)
            X_train_flat = X_shuffled
            Y_train = Y_shuffled
            X_dev_flat = np.array([]).reshape(0, 784) # Empty dev set
            Y_dev = np.array([], dtype=int)
        else:
            split_idx = int(num_samples * 0.9)
            X_train_flat = X_shuffled[:split_idx, :]
            Y_train = Y_shuffled[:split_idx]
            X_dev_flat = X_shuffled[split_idx:, :]
            Y_dev = Y_shuffled[split_idx:]

        # --- Reshape for NN --- 
        # Reshape X to (features, num_samples) -> (784, m)
        X_train = X_train_flat.T
        X_dev = X_dev_flat.T
        # Y should be 1D array: (m,) for train, (m_dev,) for dev

        print(f"  Split into Train: X{X_train.shape}, Y({Y_train.shape}) | Dev: X{X_dev.shape}, Y({Y_dev.shape})", file=sys.stderr)

        # The one_hot function needs the total number of classes
        # We don't strictly need it here, but the NN expects it. Pass num_classes through.
        # Return Y as integer labels; one-hot encoding happens in the training loop.
        return X_train, Y_train, X_dev, Y_dev, num_classes

    except Exception as e:
        print(f"An unexpected error occurred loading NPY {npy_path}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None, None, None, None, 0

def load_multiple_npy_datasets(npy_path_index_map: Dict[str, int],
                               validation_split: Union[int, float] = 0.1,
                               max_categories: Optional[int] = 5) -> LoadResult:
    """Loads and combines multiple Quick, Draw! .npy datasets.

    Args:
        npy_path_index_map (dict): A dictionary mapping .npy file paths
                                    to their unique integer category index.
        validation_split (Union[int, float]): Fraction or number for validation set.
        max_categories (Optional[int]): Maximum number of categories to load.
                                         If None, loads all. Defaults to 5.

    Returns:
        tuple: (X_train, Y_train, X_dev, Y_dev, num_classes) containing combined data,
               with integer labels corresponding to the provided indices,
               or (None, None, None, None) on error.
    """
    all_X_data = []
    all_Y_data = []
    max_index_loaded = -1 # Track highest index actually loaded
    categories_loaded = 0 # Track number of categories loaded

    print(f"Combining {len(npy_path_index_map)} NPY datasets...", file=sys.stderr)
    if max_categories is not None:
        print(f"  Limiting load to a maximum of {max_categories} categories.", file=sys.stderr)

    for npy_path, category_index in npy_path_index_map.items():
        # --- Stop if max_categories reached --- #
        if max_categories is not None and categories_loaded >= max_categories:
            print(f"  Reached max_categories limit ({max_categories}). Stopping NPY load.", file=sys.stderr)
            break
        # ------------------------------------

        category_name = os.path.splitext(os.path.basename(npy_path))[0]
        print(f"  Loading: {category_name} (Path: {npy_path}, Index: {category_index})", file=sys.stderr)

        if not os.path.exists(npy_path):
            print(f"    Error: NPY file not found at {npy_path}. Skipping.", file=sys.stderr)
            continue

        try:
            # Load data for this category
            data = np.load(npy_path)
            print(f"    Loaded shape: {data.shape}", file=sys.stderr)

            if data.ndim != 2 or data.shape[1] != 784:
                 print(f"    Error: Unexpected NPY data shape: {data.shape}. Expected (num_samples, 784). Skipping.", file=sys.stderr)
                 continue

            num_samples = data.shape[0]
            if num_samples == 0:
                print(f"    Warning: No samples found in {npy_path}. Skipping.", file=sys.stderr)
                continue

            # Create labels for this category
            labels = np.full(num_samples, category_index, dtype=int)

            # Append to master lists
            all_X_data.append(data) # Append raw data, normalize later
            all_Y_data.append(labels)
            max_index_loaded = max(max_index_loaded, category_index) # Update highest loaded index
            categories_loaded += 1 # Increment count of successfully loaded categories

        except Exception as e:
            print(f"    Error loading NPY dataset from {npy_path}: {e}. Skipping.", file=sys.stderr)
            # Optional: Add traceback here if needed for debugging

    # Check if any data was loaded successfully
    if not all_X_data or not all_Y_data:
        print("Error: No valid NPY data could be loaded from the provided paths.", file=sys.stderr)
        return None, None, None, None, 0

    # --- Combine Data ---
    try:
        print("Concatenating data from all categories...", file=sys.stderr)
        X_combined_flat = np.concatenate(all_X_data, axis=0)
        Y_combined = np.concatenate(all_Y_data, axis=0)
        num_classes = max_index_loaded + 1 # Calculate based on loaded data
        print(f"  Combined raw shape: X={X_combined_flat.shape}, Y={Y_combined.shape}", file=sys.stderr)

        # --- Preprocessing ---
        # Normalize pixel values to [0, 1]
        X_normalized_flat = X_combined_flat / 255.0

        # --- Shuffle Data --- 
        # Shuffle combined images and labels together
        total_samples = X_normalized_flat.shape[0]
        permutation = np.random.permutation(total_samples)
        X_shuffled_flat = X_normalized_flat[permutation, :]
        Y_shuffled = Y_combined[permutation]
        print(f"  Shuffled {total_samples} samples.", file=sys.stderr)

        # --- Split Data (e.g., 90% train, 10% dev) ---
        if total_samples < 10:
            print(f"Warning: Very few total samples ({total_samples}). Using all for training.", file=sys.stderr)
            X_train_flat = X_shuffled_flat
            Y_train = Y_shuffled
            X_dev_flat = np.array([]).reshape(0, 784) # Empty dev set
            Y_dev = np.array([], dtype=int)
        else:
            split_idx = int(total_samples * 0.9)
            X_train_flat = X_shuffled_flat[:split_idx, :]
            Y_train = Y_shuffled[:split_idx]
            X_dev_flat = X_shuffled_flat[split_idx:, :]
            Y_dev = Y_shuffled[split_idx:]

        # --- Reshape for NN --- 
        X_train = X_train_flat.T # (784, m_train)
        X_dev = X_dev_flat.T   # (784, m_dev)
        # Y remains 1D: (m_train,) and (m_dev,)

        print(f"  Split combined data into Train: X{X_train.shape}, Y({Y_train.shape}) | Dev: X{X_dev.shape}, Y({Y_dev.shape})", file=sys.stderr)
        print(f"  Total number of classes in combined dataset: {num_classes}", file=sys.stderr)

        return X_train, Y_train, X_dev, Y_dev, num_classes

    except Exception as e:
        print(f"Error processing combined NPY data: {e}", file=sys.stderr)
        traceback.print_exc()
        return None, None, None, None, 0

def load_emoji_dataset(csv_path: str,
                       image_column: str = 'Google',
                       validation_split: Union[int, float] = 0.1) -> LoadResult:
    """Loads the Emoji dataset from a CSV containing base64 encoded images.

    Args:
        csv_path (str): Path to the emojis.csv file.
        image_column (str): The column name containing the base64 encoded PNG data
                            (e.g., 'Google', 'Apple').
        validation_split (float or int): Fraction (if float < 1) or number of samples
                                       (if int >= 1) to use for the development set.
                                       Defaults to 0.1 (10%).

    Returns:
        tuple: (X_train, Y_train, X_dev, Y_dev) containing processed image data
               and integer labels, or (None, None, None, None) on error.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Emoji CSV file not found at {csv_path}", file=sys.stderr)
        return None, None, None, None, 0

    all_X_data = []
    all_Y_labels = [] # Store original emoji names/symbols temporarily
    label_map = {}    # Map emoji name/symbol to integer index
    next_label_index = 0

    print(f"Loading Emoji dataset from {csv_path} using '{image_column}' column...", file=sys.stderr)

    try:
        df = pd.read_csv(csv_path)

        # Check if required columns exist
        if 'name' not in df.columns or image_column not in df.columns:
             print(f"Error: CSV missing required columns ('name' and '{image_column}').", file=sys.stderr)
             return None, None, None, None, 0

        total_emojis = len(df)
        print(f"Processing {total_emojis} emojis...", file=sys.stderr)

        for index, row in df.iterrows():
            emoji_name = row['name']
            base64_string = row[image_column]

            # Check if base64 string is present and non-empty string
            if pd.isna(base64_string) or not isinstance(base64_string, str) or not base64_string.strip():
                print(f"  Warning: Skipping row {index+1} ('{emoji_name}') - Invalid or missing base64 data in '{image_column}' column.", file=sys.stderr)
                continue

            try:
                # --- Use utility function to decode and process image ---
                img_vector = process_image_from_base64(base64_string)

                if img_vector is None:
                    print(f"  Warning: Skipping row {index+1} ('{emoji_name}') - Image processing failed.", file=sys.stderr)
                    continue

                # --- Image vector is already flattened and normalized ---
                # img_processed = img.convert('L').resize((28, 28), Image.LANCZOS)
                # img_array = np.array(img_processed).astype(float) / 255.0
                # img_vector = img_array.flatten() # Shape (784,)
                # --- Checks are done inside the utility function ---
                # if img_vector.shape != (784,):
                #      print(f"  Warning: Skipping row {index+1} ('{emoji_name}') - Incorrect image vector shape after processing: {img_vector.shape}", file=sys.stderr)
                #      continue

                all_X_data.append(img_vector)
                all_Y_labels.append(emoji_name)

            except Exception as img_err:
                print(f"  Warning: Skipping row {index+1} ('{emoji_name}') - Error processing image data: {img_err}", file=sys.stderr)

        # --- Convert Labels to Integers ---
        num_samples = len(all_X_data)
        if num_samples == 0:
             print("Error: No valid emoji images could be processed.", file=sys.stderr)
             return None, None, None, None, 0

        print(f"Successfully processed {num_samples} emojis.", file=sys.stderr)

        # Create integer labels and map
        unique_labels = sorted(list(set(all_Y_labels)))
        num_classes = len(unique_labels)
        label_map = {name: i for i, name in enumerate(unique_labels)}
        Y_int_labels = np.array([label_map[name] for name in all_Y_labels], dtype=int)

        print(f"Found {num_classes} unique emoji classes.", file=sys.stderr)

        # --- Combine, Shuffle, Split --- # Use Helper function
        X_combined_flat = np.array(all_X_data) # Shape (num_samples, 784)
        X_train, Y_train, X_dev, Y_dev = _reshape_and_split_data(X_combined_flat, Y_int_labels, validation_split)

        print(f"Total number of classes in Emoji dataset: {num_classes}", file=sys.stderr)

        return X_train, Y_train, X_dev, Y_dev, num_classes

    except pd.errors.EmptyDataError:
        print(f"Error: Emoji CSV file is empty: {csv_path}", file=sys.stderr)
        return None, None, None, None, 0
    except Exception as e:
        print(f"Error loading Emoji dataset from {csv_path}: {e}", file=sys.stderr)
        traceback.print_exc()
        return None, None, None, None, 0

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