# 🧠 Neural Net Playground (PyQt Edition)

<p align="center"><img src="assets/icon.png" alt="ML Playground Icon" width="100"></p>

A lightweight PyQt GUI application for training and experimenting with simple neural networks built from scratch.

**Note:** This version focuses specifically on image-based datasets processed into a 28x28 grayscale format.

## ✨ Features

*   **Interactive UI:** Train models and test them without writing code directly.
*   **Flexible Dataset Loading:**
    *   Load standard MNIST CSV (`data/train.csv`).
    *   Load Emoji data (`data/emojis.csv`) with Base64 images.
    *   Load multiple Quick, Draw! `.npy` datasets (`data/quickdraw/`) individually or combined (limited load by default).
    *   Upload custom CSV files containing:
        *   Raw pixel data.
        *   References to images via file paths.
        *   References to images via Base64 strings.
*   **Training Control:**
    *   Select model templates (Simple MLP) or specify a custom hidden layer size.
    *   Adjust epochs, learning rate, and early stopping patience.
    *   Monitor training progress via a progress bar and detailed logs.
    *   Visualize training loss and validation accuracy curves in a separate, expandable window.
*   **Model Management:** Save trained model weights to a file (`.npz`) and load them back later.
*   **Inference/Testing:**
    *   Test the loaded/trained model on external image files (`.png`, `.jpg`).
    *   Draw digits directly on a canvas and predict them.
    *   Visualize prediction results with an image preview and probability bar graph.
*   **Logging:** Detailed log area showing timestamps and messages from the application and training process.

## 🚀 Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <repository_url> # Replace with your repo URL if applicable
    cd ml_playground
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows (cmd/powershell)
    # python -m venv venv
    # .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Data Directories:**
    *   Ensure the `data` directory exists.
    *   *(Optional)* Place the MNIST `train.csv` file (e.g., from [Kaggle](https://www.kaggle.com/competitions/digit-recognizer/data)) in the `data` directory.
    *   *(Optional)* Place the `emojis.csv` file (ensure column names match `datasets.py`) in the `data` directory.
    *   *(Optional)* Create a `data/quickdraw/` subdirectory and place any Quick, Draw! `.npy` files inside it.
    *   Ensure the `assets` directory exists (or is created) and place an `icon.png` file inside it for the window icon.

5.  **Run the application:**
    ```bash
    python main.py
    ```

## 💻 Basic Usage

1.  **Data Tab:**
    *   Use the dropdown to select a pre-discovered dataset (MNIST, Emojis, QuickDraw).
    *   Click "Load Selected" to load it.
    *   Alternatively, click "Upload CSV", select your file, configure the "Label Col Idx", "Image Col Idx" (-1 for pixels), and "Type" (base64/path if using image col), then click "Open".
2.  **Train Tab:**
    *   Select a model template or choose "(Custom)" and set the hidden layer size.
    *   Adjust epochs, learning rate, and patience.
    *   Click "Start Training". Logs will show progress.
    *   Click "Expand Plot" to view training curves in a separate window (updated after training finishes).
    *   Use "Save Weights" / "Load Weights" to manage model parameters.
3.  **Test Tab:**
    *   Click "Select & Predict File" to test on an image file.
    *   Draw a digit on the canvas and click "Predict Drawing". Clear with "Clear Canvas".
    *   Results appear in the "Image Preview" and probability graph.

## 🔧 Future Improvements

*   Add support for loading and processing basic sound datasets (e.g., converting audio clips to spectrograms).

---

Made for curious model tinkerers. 🛠️

*Icon Attribution:*
<a target="_blank" href="https://icons8.com/icon/fTkqveCX0blI/artificial-intelligence">Machine Learning</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a>
