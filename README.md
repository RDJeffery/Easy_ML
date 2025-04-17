# 🧠 EasyML — A Neural Net & CNN Playground

<p align="center"><img src="assets/icon.png" alt="EasyML Icon" width="100"></p>

**EasyML** is a lightweight PyQt GUI app for training and experimenting with simple feed-forward neural networks (NN) and Convolutional Neural Networks (CNN), built largely from scratch.

I built this because I was tired of the "click-and-hope" approach to ML tutorials. I wanted to *actually* learn how neural networks work — and the best way to do that was to build one. EasyML is my learning project turned into a working app.

**Note:** The drawing canvas feature is primarily designed for 28x28 grayscale datasets like MNIST.

---

## ✨ Features

*   **Interactive UI:** Train models and test them without writing a line of code.
*   **Multi-Model Support:** Choose between a basic Neural Network (NN) or a Convolutional Neural Network (CNN).
*   **Flexible Dataset Loading:**
    *   Load standard MNIST CSV (`data/train.csv`).
    *   Load Emoji data (`data/emojis.csv`) with Base64 images.
    *   Load CIFAR-10 and CIFAR-100 datasets (`data/cifar-10-batches-py/`, `data/cifar-100-python/`).
    *   Load multiple Quick, Draw! `.npy` datasets (`data/quickdraw/`) individually or combined.
    *   Upload custom CSVs with:
        *   Raw pixel data
        *   Image file paths
        *   Base64-encoded images
*   **Training Control:**
    *   Select model type (NN/CNN).
    *   Use NN templates or define custom hidden layers.
    *   Adjust common hyperparameters (epochs, learning rate, patience) and model-specific ones (NN activation, L2, dropout; CNN defaults used for now).
    *   Monitor progress with a live log and progress bar (*Note: UI update latency known issue*).
    *   Visualize training loss and validation accuracy in a separate plot window.
*   **Model Management:** Save and load trained model weights (`.npz` for NN, `.weights.h5` for CNN).
*   **Testing & Inference:**
    *   Test on external images (`.png`, `.jpg`).
    *   Draw digits directly in the app and get predictions (best for MNIST-like models).
    *   See predicted output alongside a probability Pie Chart with tooltips.
    *   Provide simple feedback (Yes/No) on predictions.
*   **Logs:** Detailed, timestamped logs for everything happening under the hood.

---

## 🚀 Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RDJeffery/Easy_ML.git
    cd Easy_ML
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    # python -m venv venv
    # .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    *   CNN support requires TensorFlow.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Data Directories:**
    *   Ensure a `data` directory exists.
    *   *(Optional)* Add MNIST `train.csv` from [Kaggle](https://www.kaggle.com/competitions/digit-recognizer/data).
    *   *(Optional)* Add `emojis.csv` (column format must match `datasets.py`).
    *   *(Optional)* Add Quick, Draw! `.npy` files to `data/quickdraw/`.
    *   *(Optional)* Add CIFAR-10 data (Python version) to `data/cifar-10-batches-py/`.
    *   *(Optional)* Add CIFAR-100 data (Python version) to `data/cifar-100-python/` or as `cifar-100-python.tar.gz`.
    *   Create an `assets` folder and include `icon.png` for the window icon.

5.  **Run the app:**
    ```bash
    python main.py
    ```

---

## 💻 How to Use It

### Data Tab
*   Choose a preloaded dataset (MNIST, Emojis, CIFAR-10/100, QuickDraw), or upload your own CSV.
*   If uploading CSV:
    *   Set label/image column indices.
    *   Choose how images are encoded (pixels, base64, or file paths).
    *   Click Open to load.
*   For QuickDraw, select specific categories or load a random number.

### Train Tab
*   **Select Model Type:** Choose NN or CNN.
*   **Configure Hyperparameters:** UI adapts to show relevant options (NN layer sizes, activation, etc.; common params like LR, epochs for CNN).
*   Tweak epochs, learning rate, and patience.
*   Start training — logs will update live.
*   Click "Expand Plot" after training to view loss/accuracy graphs.
*   Save or load weights (format depends on selected model type).

### Test Tab
*   Use "Select & Predict File" to run inference on an image compatible with the trained model.
*   Or draw a digit (works best for MNIST models) and predict it right inside the app.
*   Preview results and view the prediction probabilities as a Pie Chart (hover for details).
*   Use the Yes/No buttons to give feedback.

---

## 🔮 Future Ideas

Some plans on the horizon:

*   **UI/UX:** Fix progress bar update lag during training.
*   **Models:** Add more CNN hyperparameter controls; explore other model types (SVM?).
*   **Data:** Add basic audio dataset support; integrate Hugging Face Datasets.
*   **Deployment:** Package app; host online (e.g., Hugging Face Spaces).

---

## 🧪 Why This Exists

This is a learning tool disguised as a playground. It's not here to beat SOTA benchmarks — it's here to make the whole training loop *click* for you.

---

## 🙌 Credits & Attribution

*Icon:*  
<a target="_blank" href="https://icons8.com/icon/fTkqveCX0blI/artificial-intelligence">Machine Learning</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a>

---

*Built with curiosity, caffeine, and mild existential dread.*