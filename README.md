# 🧠 EasyML — A Neural Net Playground for Actually Learning Neural Nets

<p align="center"><img src="assets/icon.png" alt="EasyML Icon" width="100"></p>

**EasyML** is a lightweight PyQt GUI app for training and experimenting with simple neural networks, built entirely from scratch.

I built this because I was tired of the “click-and-hope” approach to ML tutorials. I wanted to *actually* learn how neural networks work — and the best way to do that was to build one. EasyML is my learning project turned into a working app, designed for curious tinkerers like me who want more than just Jupyter notebooks and online platforms that abstract everything away.

**Note:** This version is focused on image-based datasets that are processed into a 28x28 grayscale format.

---

## ✨ Features

*   **Interactive UI:** Train models and test them without writing a line of code.
*   **Flexible Dataset Loading:**
    *   Load standard MNIST CSV (`data/train.csv`).
    *   Load Emoji data (`data/emojis.csv`) with Base64 images.
    *   Load multiple Quick, Draw! `.npy` datasets (`data/quickdraw/`) individually or combined (limited load by default).
    *   Upload custom CSVs with:
        *   Raw pixel data
        *   Image file paths
        *   Base64-encoded images
*   **Training Control:**
    *   Use a simple MLP template or define your own hidden layer size.
    *   Adjust epochs, learning rate, and early stopping patience.
    *   Monitor progress with a live log and progress bar.
    *   Visualize training loss and validation accuracy in a separate plot window.
*   **Model Management:** Save and load trained model weights with `.npz` files.
*   **Testing & Inference:**
    *   Test on external images (`.png`, `.jpg`)
    *   Draw digits directly in the app and get predictions
    *   See predicted output alongside a probability bar chart
*   **Logs:** Detailed, timestamped logs for everything happening under the hood.

---

## 🚀 Getting Started

1. **Clone the repository:**
    ```bash
    git clone https://github.com/RDJeffery/Easy_ML.git
    cd Easy_ML
    ```

2. **Create and activate a virtual environment (recommended):**
    ```bash
    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    # python -m venv venv
    # .\venv\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Prepare Data Directories:**
    *   Ensure a `data` directory exists.
    *   *(Optional)* Add MNIST `train.csv` from [Kaggle](https://www.kaggle.com/competitions/digit-recognizer/data).
    *   *(Optional)* Add `emojis.csv` (column format must match `datasets.py`).
    *   *(Optional)* Add Quick, Draw! `.npy` files to `data/quickdraw/`.
    *   Create an `assets` folder and include `icon.png` for the window icon.

5. **Run the app:**
    ```bash
    python main.py
    ```

---

## 💻 How to Use It

### Data Tab
* Choose a preloaded dataset (MNIST, Emojis, QuickDraw), or upload your own CSV.
* If uploading:
    * Set label/image column indices.
    * Choose how images are encoded (pixels, base64, or file paths).
    * Click Open to load.

### Train Tab
* Pick a model template or go custom with your own hidden layer size.
* Tweak epochs, learning rate, and patience.
* Start training — logs will update live.
* Click “Expand Plot” after training to view loss/accuracy graphs.
* Save or load weights anytime.

### Test Tab
* Use “Select & Predict File” to run inference on an image.
* Or draw a digit and predict it right inside the app.
* Preview results and view the prediction probabilities.

---

## 🔮 Future Ideas

Some plans on the horizon:

* Add basic audio dataset support (e.g., turning audio clips into spectrograms).
* Integrate Hugging Face Datasets for one-click loading.
* Host EasyML as a web app (maybe via Hugging Face Spaces).
* Build a CNN trainer module for more advanced experiments.

---

## 🧪 Why This Exists

This is a learning tool disguised as a playground. It’s not here to beat SOTA benchmarks — it’s here to make the whole training loop *click* for you. If you’ve ever felt like ML tutorials only teach you how to use *tools*, not how things actually work — this project might help fill in the gaps.

---

## 🙌 Credits & Attribution

*Icon:*  
<a target="_blank" href="https://icons8.com/icon/fTkqveCX0blI/artificial-intelligence">Machine Learning</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a>

---

*Built with curiosity, caffeine, and mild existential dread.*