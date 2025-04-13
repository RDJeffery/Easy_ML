# Neural Net Playground Dataset Guide

This guide explains the types of datasets compatible with the Neural Net Playground application and how to prepare your data.

The application supports several dataset formats, which can be either automatically discovered or loaded manually.

## Dataset Location for Automatic Discovery

For the application to automatically find your datasets, place them in the correct subdirectory within the `data/` folder relative to where you run the main application script (`ui_main.py`):

*   **Standard CSVs (like MNIST):** Place the CSV file directly in `data/` (e.g., `data/train.csv`).
*   **Emoji CSV:** The specifically formatted emoji dataset should be named `emojis.csv` and placed in `data/` (e.g., `data/emojis.csv`).
*   **QuickDraw NPY files:** Place individual category `.npy` files inside a `quickdraw` subdirectory (e.g., `data/quickdraw/cat.npy`, `data/quickdraw/tree.npy`).

Datasets found in these locations will appear in the "Select Dataset" dropdown on the "Data" tab.

## Supported Formats & Preparation

### 1. CSV - Raw Pixel Values

This format is suitable when your image data is directly represented as pixel values in the CSV rows.

*   **Structure:**
    *   One row per image sample.
    *   **Label Column:** A single column containing the integer label for each image (e.g., 0, 1, 2...). By default, the application assumes this is the *first* column (index 0).
    *   **Pixel Columns:** Subsequent columns containing the raw pixel values for the image. For compatibility with the default network, it's best if these represent a 28x28 grayscale image, resulting in 784 pixel columns.
    *   Pixel values should ideally be in the range 0-255.
    *   A header row is recommended but not strictly required if using column indices.
*   **Preparation:**
    *   Ensure your image data is flattened into a single row of 784 pixel values.
    *   Place the label in its designated column.
    *   Save as a standard `.csv` file.
    *   Avoid non-numeric values in pixel columns and NaN/empty values in the label column.
*   **UI Loading (Custom Upload):**
    *   Click "Upload Custom CSV".
    *   Set "Label Col Idx" to the 0-based index of your label column.
    *   **Crucially, set "Image Col Idx" to `-1`** to indicate raw pixel mode.
    *   The "Type" dropdown will be disabled.

### 2. CSV - Image References (Base64 or Path)

This format is useful when your CSV file references image data stored elsewhere, either as base64 encoded strings or as file paths.

*   **Structure:**
    *   One row per image sample.
    *   **Label Column:** A column containing the label (can be integer or string - strings will be mapped to integers automatically).
    *   **Image Data Column:** A *separate* column containing either:
        *   **Base64:** A string representation of the image encoded in base64.
        *   **Path:** A file path (relative to the CSV file, or absolute) pointing to an image file (e.g., PNG, JPG).
    *   A header row is highly recommended.
*   **Preparation:**
    *   Ensure your labels and image references are in separate columns.
    *   For paths, ensure the image files exist and are accessible.
    *   Save as a standard `.csv` file.
    *   Avoid NaN/empty values in the label and image reference columns.
*   **Image Requirements:** The referenced images can be in various formats loadable by the Python Imaging Library (PIL), such as PNG, JPG, BMP, etc. They will be automatically:
    *   Converted to grayscale.
    *   Resized to 28x28 pixels using Lanczos resampling.
    *   Normalized (pixel values divided by 255.0).
*   **UI Loading (Custom Upload):**
    *   Click "Upload Custom CSV".
    *   Set "Label Col Idx" to the 0-based index of your label column.
    *   Set "Image Col Idx" to the 0-based index of your image reference column (must be different from the label index).
    *   Select the correct format ("base64" or "path") from the "Type" dropdown.

### 3. QuickDraw NPY Files

This format uses NumPy's `.npy` file format, typically used for the QuickDraw dataset.

*   **Structure:**
    *   Each category (e.g., 'cat', 'tree') should be in its own `.npy` file.
    *   Place these files inside the `data/quickdraw/` directory.
    *   The name of the file (without the `.npy` extension) will be used as the category name in the UI dropdown.
    *   Each `.npy` file should contain a single NumPy array.
    *   The array shape should be `(number_of_samples, 784)`, where each row represents a flattened 28x28 grayscale image.
    *   Pixel values should be in the range 0-255.
*   **Preparation:**
    *   Ensure your data is in the correct NumPy array format and shape.
    *   Save each category array to a separate `.npy` file named appropriately (e.g., `apple.npy`).
    *   Place the files in `data/quickdraw/`.
*   **UI Loading (Automatic Discovery):**
    *   The application will automatically find files in `data/quickdraw/`.
    *   It will create an entry in the "Select Dataset" dropdown for "QuickDraw (Multiple NPY)".
    *   Selecting this option loads *all* found `.npy` files, assigning integer labels based on alphabetical order of filenames (0 for the first, 1 for the second, etc.).
    *   The number of classes is determined by the number of `.npy` files found.

### 4. Emoji CSV (Built-in Format)

The application specifically looks for `data/emojis.csv` for a pre-formatted emoji dataset.

*   **Structure:** Requires specific columns:
    *   `label`: The integer label for the emoji.
    *   Columns named after image providers (e.g., `Google`, `Apple`, `Twitter`) containing the **base64 encoded string** for that provider's emoji image.
*   **UI Loading (Automatic Discovery):**
    *   If `data/emojis.csv` exists, entries like "Emoji (CSV - Google)", "Emoji (CSV - Apple)" will appear in the dropdown.
    *   Selecting one loads the labels and the base64 images from the corresponding provider column.
    *   Images are processed as described in the Base64 section above.

## Using the UI (Data Tab)

*   **Select Dataset:** Choose from automatically discovered datasets.
*   **Load Selected:** Loads the dataset chosen in the dropdown.
*   **Upload Custom CSV:** Opens a file dialog to select a CSV file not in the `data/` directory.
*   **Label Col Idx:** Specify the 0-based index of the column containing labels in your custom CSV.
*   **Image Col Idx:**
    *   Set to `-1` if your custom CSV uses raw pixel columns.
    *   Set to the 0-based index of the column containing base64 strings or file paths if using image references.
*   **Type:** (Enabled only if `Image Col Idx` is >= 0). Select `base64` or `path` to match the data in your image reference column.

By preparing your data according to these formats, you can effectively use the Neural Net Playground to train and test models on various image datasets.

## Tools for Dataset Creation and Preparation

While the Playground focuses on training, you might need tools to create or format your datasets first. Here are some suggestions:

### Creating Image Datasets

*   **Simple Drawing:** For creating custom datasets (like unique symbols, digits, or simple drawings), you can use:
    *   **Desktop:** MS Paint (Windows), GIMP (Free, Cross-platform), Krita (Free, Cross-platform), Paintbrush (macOS).
    *   **Online:** Various free online drawing tools (e.g., Google Drawings, AutoDraw, Sketchpad).
    *   **Process:** Draw your images (aiming for relatively simple, centered content suitable for 28x28 grayscale conversion), save them as individual image files (PNG or JPG are common), and then create a CSV file listing the paths and labels (see Format 2). You could also write a script to convert these saved images into base64 or raw pixel CSVs.
*   **Data Augmentation (Advanced):** If you have a smaller set of images, you can programmatically create variations (rotations, flips, zooms, brightness changes, noise) to expand your dataset. Python libraries like `Albumentations` or `imgaug` are powerful tools for this.
*   **Web Scraping (Use with Caution):** You can gather images from the web using scraping tools or scripts. **However, be extremely mindful of website terms of service, copyright restrictions, and privacy concerns.** Only scrape data you have the rights to use.

### Formatting and Preparing Datasets

*   **Spreadsheet Software:**
    *   **Tools:** Microsoft Excel, Google Sheets, LibreOffice Calc, Apple Numbers.
    *   **Uses:** Excellent for manually creating or editing CSV files. You can easily add label columns, file path columns, or even paste in base64 data (though this can become unwieldy for large datasets).
*   **Scripting (Python):** This is the most flexible and powerful approach for larger datasets.
    *   **Libraries:** `pandas` (for CSV manipulation), `os` (for file system interaction), `PIL` (Pillow - for image loading/processing), `base64` (for encoding/decoding), `numpy` (for array manipulation).
    *   **Common Tasks:**
        *   Scan a folder of images, extract labels (e.g., from subfolder names), and generate a CSV with file paths and labels.
        *   Load images, convert them to base64 strings, and create the corresponding CSV column.
        *   Load images, resize/grayscale them, flatten to pixel values, and generate a raw pixel CSV.
        *   Convert image datasets into the QuickDraw `.npy` format.
*   **Online Base64 Converters:** Many websites allow you to upload an image and get its base64 string representation. Be cautious about uploading sensitive images to third-party online tools.
*   **File Renaming Tools:** Bulk renaming utilities (available for most OS or as standalone apps) can help prepare image filenames before generating a path-based CSV.

Using these tools, you can create new datasets or convert existing ones into formats compatible with the Neural Net Playground.
