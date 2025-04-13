# tests/test_datasets.py

import unittest
import numpy as np
import pandas as pd
import sys
import os
import base64
from io import BytesIO
from PIL import Image # Requires Pillow
import shutil # For cleanup

# Adjust path to import from the parent directory's 'model' module
# This assumes the tests are run from the project root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import datasets

# Define the directory where test data files will be stored
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

class TestDatasetLoaders(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up test data files before running tests."""
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        print(f"Ensured test data directory exists: {TEST_DATA_DIR}")

        # --- Create Dummy Images --- #
        cls.img_paths = []
        cls.img_base64 = []
        img_size = (4, 4) # Small images for testing
        pixel_count = img_size[0] * img_size[1]
        cls.test_image_pixels = pixel_count # Store for later tests
        for i in range(2):
            try:
                img = Image.new('L', img_size, color=i * 100) # Simple grayscale
                # Use absolute paths during creation/testing for reliability
                img_abs_path = os.path.join(TEST_DATA_DIR, f'img{i+1}.png')
                img.save(img_abs_path, format='PNG')
                cls.img_paths.append(img_abs_path) # Store absolute path

                # Get Base64 string
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                cls.img_base64.append(img_str)
                print(f"Created dummy image: {img_abs_path}")
            except Exception as e:
                print(f"Error creating dummy image {i+1}: {e}")
                raise # Re-raise to fail setup if images can't be created

        # --- Create sample_pixels.csv --- #
        cls.pixel_csv_path = os.path.join(TEST_DATA_DIR, 'sample_pixels.csv')
        pixel_cols = [f'p{j}' for j in range(pixel_count)]
        pixel_data = {
            'label': [0, 1, 0, 1],
            **{col: np.random.randint(0, 256, 4) for col in pixel_cols} # Random pixel data
        }
        df_pixels = pd.DataFrame(pixel_data)
        # Add bad rows
        df_pixels.loc[4] = [0] + [100] * (pixel_count - 1) + ['not_a_number'] # Non-numeric
        df_pixels.loc[5] = [1] + [np.nan] * pixel_count # NaN
        df_pixels.to_csv(cls.pixel_csv_path, index=False)
        print(f"Created CSV: {cls.pixel_csv_path}")

        # --- Create sample_pixels_valid.csv (only good rows) ---
        cls.pixel_valid_csv_path = os.path.join(TEST_DATA_DIR, 'sample_pixels_valid.csv')
        df_pixels_valid = pd.DataFrame(pixel_data) # Original good data
        df_pixels_valid.to_csv(cls.pixel_valid_csv_path, index=False)
        print(f"Created CSV: {cls.pixel_valid_csv_path}")

        # --- Create sample_base64.csv --- #
        cls.b64_csv_path = os.path.join(TEST_DATA_DIR, 'sample_base64.csv')
        b64_data = {
            'label': [0, 1, 0, 1],
            'base64_img': cls.img_base64 * 2 # Use the valid strings
        }
        df_b64 = pd.DataFrame(b64_data)
        df_b64.loc[4] = [0, 'this is not base64'] # Invalid string
        df_b64.loc[5] = [1, np.nan] # NaN
        df_b64.to_csv(cls.b64_csv_path, index=False)
        print(f"Created CSV: {cls.b64_csv_path}")

        # --- Create sample_paths.csv --- #
        cls.path_csv_path = os.path.join(TEST_DATA_DIR, 'sample_paths.csv')
        # Use absolute paths in the CSV for simplicity in testing
        path_data = {
            'label': [0, 1, 0, 1],
            'img_path': cls.img_paths * 2 # Use the valid absolute paths
        }
        df_paths = pd.DataFrame(path_data)
        df_paths.loc[4] = [0, os.path.join(TEST_DATA_DIR, 'non_existent.png')] # Invalid path
        df_paths.loc[5] = [1, np.nan] # NaN
        df_paths.to_csv(cls.path_csv_path, index=False)
        print(f"Created CSV: {cls.path_csv_path}")

        # --- Create sample_quickdraw.npy --- #
        cls.npy_path = os.path.join(TEST_DATA_DIR, 'sample_quickdraw.npy')
        # Create a dummy npy file with shape (samples, 784)
        dummy_npy_data = np.random.randint(0, 256, size=(5, 784), dtype=np.uint8)
        np.save(cls.npy_path, dummy_npy_data)
        print(f"Created NPY: {cls.npy_path}")

        # --- Create sample_emojis.csv --- #
        cls.emoji_csv_path = os.path.join(TEST_DATA_DIR, 'sample_emojis.csv')
        emoji_data = {
            'Name': ['Emoji A', 'Emoji B', 'Emoji C', 'Emoji D'],
            'Google': cls.img_base64 * 2, # Use valid base64
            'Apple': [np.nan] * 4 # Add another column for testing selection
        }
        df_emojis = pd.DataFrame(emoji_data)
        df_emojis.loc[4] = ['Emoji E', 'not base64', 'valid_base64_placeholder'] # Invalid Google image
        df_emojis.loc[5] = ['Emoji F', np.nan, 'valid_base64_placeholder'] # NaN Google image
        df_emojis.to_csv(cls.emoji_csv_path, index=False)
        print(f"Created CSV: {cls.emoji_csv_path}")

        cls.created_files = cls.img_paths + [
            cls.pixel_csv_path, cls.b64_csv_path, cls.path_csv_path,
            cls.npy_path, cls.emoji_csv_path, cls.pixel_valid_csv_path
        ]

    @classmethod
    def tearDownClass(cls):
        """Clean up test data files after running tests."""
        print(f"Cleaning up test data in: {TEST_DATA_DIR}")
        if hasattr(cls, 'created_files'):
            for f_path in cls.created_files:
                try:
                    if os.path.exists(f_path):
                        os.remove(f_path)
                        # print(f"  Removed: {f_path}")
                except OSError as e:
                    print(f"  Error removing file {f_path}: {e}")
        # Optionally remove the directory if it's empty and we created it
        try:
            if os.path.exists(TEST_DATA_DIR) and not os.listdir(TEST_DATA_DIR):
                shutil.rmtree(TEST_DATA_DIR)
                print(f"Removed empty test data directory: {TEST_DATA_DIR}")
        except OSError as e:
            print(f"  Error removing directory {TEST_DATA_DIR}: {e}")

    def test_placeholder(self):
        """Placeholder test."""
        self.assertEqual(1 + 1, 2)

    # --- Tests for load_csv_dataset --- #
    def test_load_csv_pixel_valid(self):
        """Test loading CSV in pixel mode with valid data."""
        # Use label_col_index=0, image_col_index=None (default)
        # Use the purely valid CSV file
        X_train, Y_train, X_dev, Y_dev, num_classes = datasets.load_csv_dataset(
            self.pixel_valid_csv_path,
            validation_split=1 # Use 1 sample for validation
        )
        self.assertIsNotNone(X_train, "X_train should not be None on valid pixel CSV load")
        self.assertEqual(num_classes, 2, "Should detect 2 classes (0, 1)")
        self.assertEqual(X_train.shape[0], self.test_image_pixels, "X_train feature count mismatch")
        self.assertEqual(X_dev.shape[0], self.test_image_pixels, "X_dev feature count mismatch")
        self.assertEqual(X_train.shape[1] + X_dev.shape[1], 4, "Total samples mismatch (should be 4 valid rows)")
        self.assertEqual(Y_train.shape[0], X_train.shape[1], "Y_train length mismatch")
        self.assertEqual(Y_dev.shape[0], X_dev.shape[1], "Y_dev length mismatch")

    def test_load_csv_pixel_non_numeric(self):
        """Test CSV pixel mode error on non-numeric feature data."""
        # Temporarily modify the CSV to ensure the non-numeric row isn't skipped by NaN check
        # (This is a bit fragile, relies on specific setup)
        # A better approach might be separate test files
        df = pd.read_csv(self.pixel_csv_path)
        df_test = df.iloc[[0, 4]] # Keep first valid row and non-numeric row
        temp_path = os.path.join(TEST_DATA_DIR, 'temp_non_numeric.csv')
        df_test.to_csv(temp_path, index=False)

        X_train, Y_train, X_dev, Y_dev, num_classes = datasets.load_csv_dataset(temp_path)
        self.assertIsNone(X_train, "X_train should be None when non-numeric pixels are present")
        self.assertEqual(num_classes, 0, "num_classes should be 0 on error")

        os.remove(temp_path) # Clean up temp file

    def test_load_csv_pixel_nan_feature(self):
        """Test CSV pixel mode warning/handling of NaN feature data."""
        # Currently, the loader only warns for NaNs, doesn't error. It might return data.
        # Let's test that it *can* load, though results might be unusable.
        # Using a file with just the NaN row:
        df = pd.read_csv(self.pixel_csv_path)
        df_test = df.iloc[[5]] # Keep only the NaN row
        temp_path = os.path.join(TEST_DATA_DIR, 'temp_nan_pixel.csv')
        df_test.to_csv(temp_path, index=False)

        # Expect loading to FAIL now because the numeric check catches NaNs coerced from `to_numeric`
        X_train, Y_train, X_dev, Y_dev, num_classes = datasets.load_csv_dataset(temp_path)
        self.assertIsNone(X_train, "X_train should be None when only NaN pixels are present due to numeric check")
        self.assertEqual(num_classes, 0, "num_classes should be 0 on error")

        os.remove(temp_path) # Clean up temp file

    def test_load_csv_base64_valid(self):
        """Test loading CSV in base64 mode with valid data."""
        # Use label_col_index=0, image_col_index=1, type='base64'
        # Expects 4 valid rows from the setup file
        X_train, Y_train, X_dev, Y_dev, num_classes = datasets.load_csv_dataset(
            self.b64_csv_path,
            label_col_index=0,
            image_col_index=1,
            image_col_type='base64',
            validation_split=1 # Use 1 sample for validation
        )
        self.assertIsNotNone(X_train, "X_train should not be None on valid base64 CSV load")
        self.assertEqual(num_classes, 2, "Should detect 2 classes (0, 1)")
        # The image processor should resize to 784 pixels
        self.assertEqual(X_train.shape[0], 784, "X_train feature count mismatch (base64)")
        self.assertEqual(X_dev.shape[0], 784, "X_dev feature count mismatch (base64)")
        self.assertEqual(X_train.shape[1] + X_dev.shape[1], 4, "Total samples mismatch (should be 4 valid base64 rows)")
        self.assertEqual(Y_train.shape[0], X_train.shape[1], "Y_train length mismatch")
        self.assertEqual(Y_dev.shape[0], X_dev.shape[1], "Y_dev length mismatch")

    def test_load_csv_base64_invalid(self):
        """Test CSV base64 mode skipping rows with invalid data."""
        # The loader should skip the row with the invalid base64 string and the NaN row
        # It should still load the 4 valid rows.
        X_train, Y_train, X_dev, Y_dev, num_classes = datasets.load_csv_dataset(
            self.b64_csv_path,
            label_col_index=0,
            image_col_index=1,
            image_col_type='base64',
            validation_split=1
        )
        self.assertIsNotNone(X_train, "X_train should not be None even with invalid base64 rows")
        self.assertEqual(num_classes, 2, "Should detect 2 classes (0, 1)")
        # Should still load the 4 valid samples
        self.assertEqual(X_train.shape[1] + X_dev.shape[1], 4, "Should load 4 valid samples, skipping invalid base64/NaN")

    def test_load_csv_base64_all_invalid(self):
        """Test CSV base64 mode fails when NO valid rows exist."""
        df = pd.read_csv(self.b64_csv_path)
        df_test = df.iloc[[4, 5]] # Keep only invalid base64 and NaN rows
        temp_path = os.path.join(TEST_DATA_DIR, 'temp_all_invalid_b64.csv')
        df_test.to_csv(temp_path, index=False)

        X_train, Y_train, X_dev, Y_dev, num_classes = datasets.load_csv_dataset(
            temp_path,
            label_col_index=0, image_col_index=1, image_col_type='base64'
        )
        self.assertIsNone(X_train, "X_train should be None when no valid base64 rows exist")
        self.assertEqual(num_classes, 0, "num_classes should be 0 on error")

        os.remove(temp_path) # Clean up temp file

    def test_load_csv_path_valid(self):
        """Test loading CSV in path mode with valid data."""
        # Use label_col_index=0, image_col_index=1, type='path'
        # Expects 4 valid rows from the setup file
        X_train, Y_train, X_dev, Y_dev, num_classes = datasets.load_csv_dataset(
            self.path_csv_path,
            label_col_index=0,
            image_col_index=1,
            image_col_type='path',
            validation_split=1 # Use 1 sample for validation
        )
        self.assertIsNotNone(X_train, "X_train should not be None on valid path CSV load")
        self.assertEqual(num_classes, 2, "Should detect 2 classes (0, 1)")
        # The image processor should resize to 784 pixels
        self.assertEqual(X_train.shape[0], 784, "X_train feature count mismatch (path)")
        self.assertEqual(X_dev.shape[0], 784, "X_dev feature count mismatch (path)")
        self.assertEqual(X_train.shape[1] + X_dev.shape[1], 4, "Total samples mismatch (should be 4 valid path rows)")
        self.assertEqual(Y_train.shape[0], X_train.shape[1], "Y_train length mismatch")
        self.assertEqual(Y_dev.shape[0], X_dev.shape[1], "Y_dev length mismatch")

    def test_load_csv_path_invalid(self):
        """Test CSV path mode skipping rows with invalid data."""
        # The loader should skip the row with the invalid path and the NaN row
        # It should still load the 4 valid rows.
        X_train, Y_train, X_dev, Y_dev, num_classes = datasets.load_csv_dataset(
            self.path_csv_path,
            label_col_index=0,
            image_col_index=1,
            image_col_type='path',
            validation_split=1
        )
        self.assertIsNotNone(X_train, "X_train should not be None even with invalid path rows")
        self.assertEqual(num_classes, 2, "Should detect 2 classes (0, 1)")
        # Should still load the 4 valid samples
        self.assertEqual(X_train.shape[1] + X_dev.shape[1], 4, "Should load 4 valid samples, skipping invalid path/NaN")

    def test_load_csv_path_all_invalid(self):
        """Test CSV path mode fails when NO valid rows exist."""
        df = pd.read_csv(self.path_csv_path)
        df_test = df.iloc[[4, 5]] # Keep only invalid path and NaN rows
        temp_path = os.path.join(TEST_DATA_DIR, 'temp_all_invalid_path.csv')
        df_test.to_csv(temp_path, index=False)

        X_train, Y_train, X_dev, Y_dev, num_classes = datasets.load_csv_dataset(
            temp_path,
            label_col_index=0, image_col_index=1, image_col_type='path'
        )
        self.assertIsNone(X_train, "X_train should be None when no valid path rows exist")
        self.assertEqual(num_classes, 0, "num_classes should be 0 on error")

        os.remove(temp_path) # Clean up temp file

    def test_load_csv_bad_index(self):
        """Test load_csv_dataset error with out-of-bounds indices."""
        # Invalid label index
        res = datasets.load_csv_dataset(self.pixel_csv_path, label_col_index=99)
        self.assertIsNone(res[0], "Should fail with invalid label_col_index")
        # Invalid image index
        res = datasets.load_csv_dataset(self.path_csv_path, image_col_index=99, image_col_type='path')
        self.assertIsNone(res[0], "Should fail with invalid image_col_index")
        # Label and image index the same
        res = datasets.load_csv_dataset(self.path_csv_path, label_col_index=1, image_col_index=1, image_col_type='path')
        self.assertIsNone(res[0], "Should fail with same label and image index")

    def test_load_csv_bad_type(self):
        """Test load_csv_dataset error with invalid image_col_type."""
        res = datasets.load_csv_dataset(self.path_csv_path, image_col_index=1, image_col_type='invalid_type')
        self.assertIsNone(res[0], "Should fail with invalid image_col_type")

    def test_load_csv_file_errors(self):
        """Test load_csv_dataset with file-related errors."""
        # Non-existent file
        res = datasets.load_csv_dataset(os.path.join(TEST_DATA_DIR, 'no_such_file.csv'))
        self.assertIsNone(res[0], "Should fail with non-existent file")

        # Empty file
        empty_path = os.path.join(TEST_DATA_DIR, 'empty.csv')
        with open(empty_path, 'w') as f:
            f.write("")
        res = datasets.load_csv_dataset(empty_path)
        self.assertIsNone(res[0], "Should fail with empty file")
        os.remove(empty_path)

        # Malformed CSV (ParserError)
        malformed_path = os.path.join(TEST_DATA_DIR, 'malformed.csv')
        with open(malformed_path, 'w') as f:
            f.write("col1,col2\n1,2,3\n4,5") # Extra comma in data row
        res = datasets.load_csv_dataset(malformed_path)
        self.assertIsNone(res[0], "Should fail with malformed CSV")
        os.remove(malformed_path)

    # --- Tests for NPY loaders --- #
    def test_load_npy_valid(self):
        """Test loading a single valid NPY file."""
        category_index = 5
        num_classes = 10
        res = datasets.load_npy_dataset(self.npy_path, category_index, num_classes, validation_split=1)
        X_train, Y_train, X_dev, Y_dev, res_num_classes = res

        self.assertIsNotNone(X_train, "X_train should not be None on valid NPY load")
        self.assertEqual(res_num_classes, num_classes, "Returned num_classes mismatch")
        self.assertEqual(X_train.shape[0], 784, "X_train feature count mismatch (NPY)")
        self.assertEqual(X_dev.shape[0], 784, "X_dev feature count mismatch (NPY)")
        self.assertEqual(X_train.shape[1] + X_dev.shape[1], 5, "Total samples mismatch (NPY)") # 5 samples in dummy file
        self.assertEqual(Y_train.shape[0], X_train.shape[1], "Y_train length mismatch")
        self.assertEqual(Y_dev.shape[0], X_dev.shape[1], "Y_dev length mismatch")
        self.assertTrue(np.all(Y_train == category_index), "Y_train labels incorrect (NPY)")
        self.assertTrue(np.all(Y_dev == category_index), "Y_dev labels incorrect (NPY)")

    def test_load_npy_file_not_found(self):
        """Test load_npy_dataset with a non-existent file."""
        res = datasets.load_npy_dataset(os.path.join(TEST_DATA_DIR, 'no_such_file.npy'), 0, 1)
        self.assertIsNone(res[0], "Should fail loading non-existent NPY file")

    def test_load_npy_bad_shape(self):
        """Test load_npy_dataset with NPY file of incorrect shape."""
        bad_shape_path = os.path.join(TEST_DATA_DIR, 'bad_shape.npy')
        bad_data = np.zeros((5, 100)) # Incorrect number of features
        np.save(bad_shape_path, bad_data)

        res = datasets.load_npy_dataset(bad_shape_path, 0, 1)
        self.assertIsNone(res[0], "Should fail loading NPY with bad shape")

        os.remove(bad_shape_path)

    def test_load_multiple_npy_valid(self):
        """Test loading multiple valid NPY files."""
        # Create a second dummy NPY
        npy_path2 = os.path.join(TEST_DATA_DIR, 'sample_quickdraw2.npy')
        dummy_npy_data2 = np.random.randint(0, 256, size=(3, 784), dtype=np.uint8)
        np.save(npy_path2, dummy_npy_data2)

        path_map = {self.npy_path: 0, npy_path2: 1}
        res = datasets.load_multiple_npy_datasets(path_map, validation_split=2)
        X_train, Y_train, X_dev, Y_dev, num_classes = res

        self.assertIsNotNone(X_train, "X_train should not be None on valid multi-NPY load")
        self.assertEqual(num_classes, 2, "Should detect 2 classes (multi-NPY)")
        self.assertEqual(X_train.shape[0], 784, "X_train feature count mismatch (multi-NPY)")
        self.assertEqual(X_dev.shape[0], 784, "X_dev feature count mismatch (multi-NPY)")
        self.assertEqual(X_train.shape[1] + X_dev.shape[1], 8, "Total samples mismatch (multi-NPY, 5+3)")
        # Check if both classes are present in labels (might depend on split)
        unique_labels = np.unique(np.concatenate((Y_train, Y_dev)))
        self.assertIn(0, unique_labels, "Class 0 missing in labels (multi-NPY)")
        self.assertIn(1, unique_labels, "Class 1 missing in labels (multi-NPY)")

        os.remove(npy_path2)

    def test_load_multiple_npy_one_missing(self):
        """Test loading multiple NPY files with one missing."""
        path_map = {self.npy_path: 0, os.path.join(TEST_DATA_DIR, 'no_such_file.npy'): 1}
        res = datasets.load_multiple_npy_datasets(path_map, validation_split=1)
        X_train, Y_train, X_dev, Y_dev, num_classes = res

        # Should still load the valid one
        self.assertIsNotNone(X_train, "X_train should not be None when one NPY is missing")
        self.assertEqual(num_classes, 1, "Should detect 1 class when one NPY is missing") # Only class 0 loaded
        self.assertEqual(X_train.shape[1] + X_dev.shape[1], 5, "Total samples mismatch (one NPY missing)")

    # --- Test for Emoji loader --- #
    def test_load_emoji_valid(self):
        """Test loading the sample emoji dataset."""
        # Expects 4 valid rows from the setup file (using Google column)
        res = datasets.load_emoji_dataset(self.emoji_csv_path, image_column='Google', validation_split=1)
        X_train, Y_train, X_dev, Y_dev, num_classes = res

        self.assertIsNotNone(X_train, "X_train should not be None on valid emoji load")
        self.assertEqual(num_classes, 4, "Should detect 4 unique emoji names")
        self.assertEqual(X_train.shape[0], 784, "X_train feature count mismatch (emoji)")
        self.assertEqual(X_dev.shape[0], 784, "X_dev feature count mismatch (emoji)")
        self.assertEqual(X_train.shape[1] + X_dev.shape[1], 4, "Total samples mismatch (should be 4 valid emojis)")

    def test_load_emoji_missing_column(self):
        """Test load_emoji_dataset fails if required columns are missing."""
        # Test missing image column
        res = datasets.load_emoji_dataset(self.emoji_csv_path, image_column='MissingColumn')
        self.assertIsNone(res[0], "Should fail if specified image_column doesn't exist")

        # Test missing Name column (create temp file)
        df = pd.read_csv(self.emoji_csv_path)
        df_test = df.drop(columns=['Name'])
        temp_path = os.path.join(TEST_DATA_DIR, 'temp_no_name_emoji.csv')
        df_test.to_csv(temp_path, index=False)
        res = datasets.load_emoji_dataset(temp_path)
        self.assertIsNone(res[0], "Should fail if 'Name' column is missing")
        os.remove(temp_path)

    def test_load_emoji_no_valid_images(self):
        """Test load_emoji_dataset fails if no images are valid in the column."""
        df = pd.read_csv(self.emoji_csv_path)
        df_test = df.iloc[[4, 5]] # Keep only rows with bad/NaN Google images
        temp_path = os.path.join(TEST_DATA_DIR, 'temp_no_valid_emoji.csv')
        df_test.to_csv(temp_path, index=False)

        res = datasets.load_emoji_dataset(temp_path, image_column='Google')
        self.assertIsNone(res[0], "Should fail if no valid images in specified column")
        os.remove(temp_path)

    # --- Test for helper --- #
    def test_reshape_and_split_data(self):
        """Test the _reshape_and_split_data helper function."""
        num_samples = 100
        num_features = 784
        X_flat = np.random.rand(num_samples, num_features)
        Y = np.random.randint(0, 5, num_samples)

        # Test split by fraction
        val_frac = 0.2
        X_tr, Y_tr, X_dev, Y_dev = datasets._reshape_and_split_data(X_flat, Y, val_frac)
        expected_dev_count = int(num_samples * val_frac)

if __name__ == '__main__':
    # This allows running the tests directly
    # You might need to run as 'python -m unittest tests.test_datasets' from root
    unittest.main() 