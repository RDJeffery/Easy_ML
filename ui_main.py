import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QComboBox, QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout,
    QApplication, QProgressBar, QSizePolicy, QCheckBox, QTabWidget, QDialog # Import QDialog
)
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QColor, QPen, QBrush, QFontMetrics, QIcon # Import QIcon
)
from PyQt5.QtCore import Qt, QDateTime, QObject, pyqtSignal, QThread, QRectF # Import QRectF
import numpy as np
from model import neural_net, datasets # Import datasets module
from PIL import Image
import os
from PyQt5 import QtGui # Keep QtGui import for QTextCursor
import glob # Import glob for file searching
from typing import Optional # For worker thread typing

# --- Add ui directory to path ---
ui_dir = os.path.join(os.path.dirname(__file__), 'ui')
if ui_dir not in sys.path:
    sys.path.insert(0, ui_dir)
try:
    from drawing_canvas import DrawingCanvas
except ImportError as e:
    print(f"ERROR: Could not import DrawingCanvas. Make sure ui/drawing_canvas.py exists: {e}")
    # Define a dummy class to prevent NameError if import fails
    class DrawingCanvas(QWidget):
        def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs); QLabel("Error loading DrawingCanvas", self)
        def clearCanvas(self): pass
        def getDrawingArray(self): return None
        def getPreviewPixmap(self): return None

try:
    from plot_widget import PlotWidget
except ImportError as e:
    print(f"ERROR: Could not import PlotWidget. Make sure ui/plot_widget.py exists: {e}")
    # Define a dummy class
    class PlotWidget(QWidget):
        def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs); QLabel("Error loading PlotWidget", self)
        def update_plot(self, *args, **kwargs): pass

# Worker class for handling training in a separate thread
class TrainingWorker(QObject):
    # Signals to communicate with the main thread
    progress_updated = pyqtSignal(int, int) # current_iteration, total_iterations
    training_finished = pyqtSignal(tuple)   # (W1, b1, W2, b2, loss_hist, acc_hist)
    log_message = pyqtSignal(str)       # For sending log messages from worker
    error_occurred = pyqtSignal(str)      # To signal errors

    def __init__(self, X_train, Y_train, X_dev, Y_dev, alpha, iterations, initial_params, patience):
        super().__init__()
        self.X_train, self.Y_train = X_train, Y_train
        self.X_dev, self.Y_dev = X_dev, Y_dev
        self.alpha = alpha
        self.iterations = iterations
        self.initial_params = initial_params
        self.patience = patience # Store patience

    # Slot to run the training
    def run_training(self):
        try:
            self.log_message.emit("Worker thread started gradient descent...")
            W1, b1, W2, b2 = self.initial_params

            # --- Define the callback function to emit progress signal ---
            def progress_callback(current_iter, total_iter):
                self.progress_updated.emit(current_iter, total_iter)
            # --- End callback function ---

            # Call gradient descent, passing the callback and patience
            results = neural_net.gradient_descent(
                self.X_train, self.Y_train, self.X_dev, self.Y_dev,
                self.alpha, self.iterations, W1, b1, W2, b2,
                progress_callback=progress_callback,
                patience=self.patience # Pass patience here
            )
            self.training_finished.emit(results)
        except Exception as e:
            self.error_occurred.emit(f"Error in training worker: {e}")


# --- Custom Widget for Probability Bar Graph ---
class ProbabilityBarGraph(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.probabilities = []
        self.predicted_class = -1
        self.setMinimumHeight(100) # Ensure it has some height
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred) # Allow horizontal expansion

    def set_probabilities(self, probs, predicted_class=-1):
        # Expects a 1D numpy array or list
        self.probabilities = np.asarray(probs)
        self.predicted_class = predicted_class
        self.update() # Trigger a repaint

    def clear_graph(self):
        self.probabilities = []
        self.predicted_class = -1
        self.update()

    def paintEvent(self, event):
        # Check if the probabilities container is empty (works for list or numpy array)
        if len(self.probabilities) == 0:
            return # Don't draw if no data

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        num_classes = len(self.probabilities)
        if num_classes == 0:
            return

        width = self.width()
        height = self.height()
        padding = 5 # Padding around the graph
        label_width = 40 # Space for labels like "0: 0.95"
        graph_area_width = width - padding * 2 - label_width
        bar_height = (height - padding * (num_classes + 1)) / num_classes

        if bar_height <= 0 or graph_area_width <= 0: return # Not enough space

        # Font for labels
        font = painter.font()
        fm = QFontMetrics(font)
        text_height = fm.height()

        # Define colors
        bar_color = QColor("#3498db") # Blue
        highlight_color = QColor("#e74c3c") # Red
        text_color = QColor(Qt.black)
        painter.setPen(Qt.NoPen) # No outline for bars by default

        for i, prob in enumerate(self.probabilities):
            bar_y = padding + i * (bar_height + padding)
            bar_width = max(0, prob * graph_area_width) # Ensure non-negative width

            # Choose color
            if i == self.predicted_class:
                 painter.setBrush(QBrush(highlight_color))
            else:
                 painter.setBrush(QBrush(bar_color))

            # Draw bar
            painter.drawRect(QRectF(padding + label_width, bar_y, bar_width, bar_height))

            # Draw label (Class Index: Probability)
            label_text = f"{i}: {prob:.2f}"
            text_x = padding
            # Center text vertically in the bar space
            text_y = bar_y + bar_height / 2 + text_height / 4 # Approximation for vertical centering

            painter.setPen(text_color) # Set pen for text
            painter.drawText(int(text_x), int(text_y), label_text)
            painter.setPen(Qt.NoPen) # Reset pen


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Neural Net Playground")
        self.setWindowIcon(QIcon('assets/icon.png')) # Set window icon (use icon from assets dir)
        # Adjust minimum size if needed, tabs might help
        self.setMinimumSize(600, 550)

        # --- Initialize Core Attributes ---
        self._init_attributes()
        self.expanded_plot_dialog = None # Attribute to hold the expanded plot window

        # --- Setup Main Layout ---
        self.central = QWidget()
        self.main_layout = QVBoxLayout()

        # --- Create Log Area (Must be done before widgets that log) --- #
        log_area = self._create_log_area()

        # --- Create Tab Widget --- #
        self.tabs = QTabWidget()

        # --- Create Widgets for Tabs --- #

        # Data Tab
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)
        dataset_group = self._create_dataset_group()
        data_layout.addWidget(dataset_group)
        data_layout.addStretch() # Push content up

        # Train Tab
        train_tab = QWidget()
        train_layout = QVBoxLayout(train_tab)
        training_group = self._create_training_group()
        model_mgmt_group = self._create_model_mgmt_group()
        # Layout for Expand Plot Button and Info Text
        plot_info_layout = QHBoxLayout()

        expand_plot_button = QPushButton("🔎 Expand Plot")
        expand_plot_button.clicked.connect(self._show_expanded_plot)
        plot_info_layout.addWidget(expand_plot_button)

        plot_info_label = QLabel("<- Training history visualized here")
        plot_info_label.setStyleSheet("font-style: italic; color: grey;")
        plot_info_layout.addWidget(plot_info_label)
        plot_info_layout.addStretch() # Push button and label left

        train_layout.addWidget(training_group)
        train_layout.addWidget(model_mgmt_group)
        train_layout.addLayout(plot_info_layout) # Add button and info label
        # No stretch here, let plot expand

        # Infer Tab
        infer_tab = QWidget()
        infer_layout = QVBoxLayout(infer_tab)
        inference_group = self._create_inference_group()
        infer_layout.addWidget(inference_group)
        infer_layout.addStretch()

        # Add Tabs to Tab Widget
        self.tabs.addTab(data_tab, "💾 Data")
        self.tabs.addTab(train_tab, "🚀 Train")
        self.tabs.addTab(infer_tab, "🧪 Test")

        # --- Add Tab Widget and Log Area to Main Layout --- #
        self.main_layout.addWidget(self.tabs)
        self.main_layout.addWidget(log_area)
        # Adjust stretch factor if needed (e.g., give log less space)
        self.main_layout.setStretchFactor(self.tabs, 4) # Give tabs more space
        self.main_layout.setStretchFactor(log_area, 1) # Log less space

        # --- Finalize Layout ---
        self.central.setLayout(self.main_layout)
        self.setCentralWidget(self.central)

        # --- Initial UI State Setup ---
        if hasattr(self, 'template_combo'):
            self._update_hidden_layer_input(self.template_combo.currentText())

    # --- Initialization Helper ---
    def _init_attributes(self):
        """Initialize core attributes for the main window."""
        # Model parameters
        self.model_params = None
        self.current_num_classes = 0

        # Training data
        self.X_train, self.Y_train, self.X_dev, self.Y_dev = None, None, None, None
        self.train_loss_history = []
        self.val_accuracy_history = []

        # Threading
        self.training_thread = None
        self.training_worker = None

        # QuickDraw specific
        self.quickdraw_category_map = {}
        self.num_quickdraw_classes = 0
        self.quickdraw_files_map = {}

        # Model Templates Definition
        self.model_templates = {
            "Simple MLP (10 Hidden)": 10,
            "Wider MLP (50 Hidden)": 50,
            "Deep Narrow (5 Hidden)": 5,
            "(Custom)": -1
        }

    # --- UI Creation Helper Methods ---

    def _create_dataset_group(self):
        """Creates the Dataset Selection group box and its widgets."""
        group = QGroupBox("Dataset Selection")
        v_layout = QVBoxLayout()

        # Row 1: Combo box and Load button
        load_layout = QHBoxLayout()
        self.dataset_label = QLabel("Select/Upload Dataset:")
        load_layout.addWidget(self.dataset_label)
        self.dataset_combo = QComboBox()
        load_layout.addWidget(self.dataset_combo)
        self.load_dataset_button = QPushButton("💾 Load Selected")
        self.load_dataset_button.clicked.connect(self.load_selected_dataset)
        load_layout.addWidget(self.load_dataset_button)
        v_layout.addLayout(load_layout)

        # Row 2: Upload button and Column indices/type
        upload_layout = QHBoxLayout()
        self.upload_dataset_button = QPushButton("📁 Upload CSV")
        self.upload_dataset_button.clicked.connect(self.upload_csv_dataset)
        upload_layout.addWidget(self.upload_dataset_button)

        # Label Col
        self.label_col_label = QLabel("Label Col Idx:")
        upload_layout.addWidget(self.label_col_label)
        self.label_col_input = QSpinBox()
        self.label_col_input.setRange(-1, 1000)
        self.label_col_input.setToolTip("0-based index of the label column.")
        self.label_col_input.setValue(0)      # Default to 0 (MNIST format)
        upload_layout.addWidget(self.label_col_input)

        # Image Col
        self.image_col_label = QLabel("Image Col Idx:")
        upload_layout.addWidget(self.image_col_label)
        self.image_col_input = QSpinBox()
        self.image_col_input.setRange(-1, 1000)
        self.image_col_input.setValue(-1)
        self.image_col_input.setToolTip("0-based index of image column (base64/path), or -1 to use pixel columns.")
        self.image_col_input.valueChanged.connect(self._update_image_col_type_state)
        upload_layout.addWidget(self.image_col_input)

        # Image Type
        self.image_type_label = QLabel("Type:")
        upload_layout.addWidget(self.image_type_label)
        self.image_type_combo = QComboBox()
        self.image_type_combo.addItems(["(Not Applicable)", "base64", "path"])
        self.image_type_combo.setToolTip("Select 'base64' or 'path' if using an Image Column.")
        self.image_type_combo.setEnabled(False)
        upload_layout.addWidget(self.image_type_combo)

        upload_layout.addStretch()
        v_layout.addLayout(upload_layout)

        # Populate dropdown only after all relevant widgets are created
        self.populate_dataset_dropdown()

        group.setLayout(v_layout)
        return group

    def _create_inference_group(self):
        """Creates the Inference group box and its widgets."""
        group = QGroupBox("Model Testing (Inference)")
        layout = QVBoxLayout()

        # Add explanatory text
        info_label = QLabel(
            "Use this section to test the currently trained or loaded model on new data. "
            "You can either select an image file or draw a digit yourself."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-style: italic; padding-bottom: 10px;")
        layout.addWidget(info_label)

        # --- Test with File Section ---
        file_test_layout = QHBoxLayout()
        file_label = QLabel("Test with Image File:")
        file_test_layout.addWidget(file_label)
        self.predict_file_button = QPushButton("Select & Predict File")
        self.predict_file_button.setToolTip("Load an image file (e.g., PNG, JPG) for prediction")
        self.predict_file_button.clicked.connect(self._predict_image_file)
        file_test_layout.addWidget(self.predict_file_button)
        file_test_layout.addStretch()
        layout.addLayout(file_test_layout)

        # --- Test with Drawing Section ---
        drawing_test_layout = QHBoxLayout()
        self.drawing_canvas = DrawingCanvas(width=140, height=140, parent=self) # Provide dimensions and parent
        drawing_test_layout.addWidget(self.drawing_canvas)

        drawing_buttons_layout = QVBoxLayout()
        predict_drawing_button = QPushButton("Predict Drawing")
        predict_drawing_button.setToolTip("Predict the digit currently drawn on the canvas")
        predict_drawing_button.clicked.connect(self._predict_drawing)
        drawing_buttons_layout.addWidget(predict_drawing_button)

        clear_button = QPushButton("Clear Canvas")
        clear_button.clicked.connect(self.drawing_canvas.clearCanvas)
        drawing_buttons_layout.addWidget(clear_button)
        drawing_buttons_layout.addStretch()
        drawing_test_layout.addLayout(drawing_buttons_layout)
        layout.addLayout(drawing_test_layout)

        # --- Results Display Section ---
        results_layout = QHBoxLayout()
        self.image_preview_label = QLabel("Image Preview Here")
        self.image_preview_label.setAlignment(Qt.AlignCenter)
        self.image_preview_label.setMinimumSize(100, 100)
        self.image_preview_label.setStyleSheet("border: 1px solid gray; background-color: white;")
        results_layout.addWidget(self.image_preview_label)
        self.probability_graph = ProbabilityBarGraph()
        results_layout.addWidget(self.probability_graph)
        layout.addLayout(results_layout)

        group.setLayout(layout)
        return group

    def _create_training_group(self):
        """Creates the Training Controls group box and its widgets."""
        # Assign to self.training_group so title can be updated later
        self.training_group = QGroupBox("Training Controls (No dataset loaded)")
        layout = QVBoxLayout()

        # Template Selection
        template_layout = QHBoxLayout()
        self.template_label = QLabel("Model Template:")
        template_layout.addWidget(self.template_label)
        self.template_combo = QComboBox()
        self.template_combo.addItems(self.model_templates.keys())
        self.template_combo.currentTextChanged.connect(self._update_hidden_layer_input)
        template_layout.addWidget(self.template_combo)
        layout.addLayout(template_layout)

        # Form for hyperparameters
        form_layout = QFormLayout()
        self.hidden_layer_input = QSpinBox()
        self.hidden_layer_input.setRange(1, 1000)
        # Set initial value based on first template
        first_template_name = list(self.model_templates.keys())[0]
        self.hidden_layer_input.setValue(self.model_templates[first_template_name])
        self.hidden_layer_input.setToolTip("Number of neurons in the hidden layer.")
        form_layout.addRow("Hidden Layer Size:", self.hidden_layer_input)

        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 10000)
        self.epochs_input.setValue(100)
        form_layout.addRow("Epochs:", self.epochs_input)

        self.lr_input = QDoubleSpinBox()
        self.lr_input.setDecimals(4)
        self.lr_input.setRange(0.0001, 1.0)
        self.lr_input.setSingleStep(0.01)
        self.lr_input.setValue(0.1)
        form_layout.addRow("Learning Rate:", self.lr_input)

        self.patience_input = QSpinBox()
        self.patience_input.setRange(0, 1000)
        self.patience_input.setValue(0)
        self.patience_input.setToolTip("Stop training if validation accuracy doesn't improve for this many checks (0=disabled).")
        form_layout.addRow("Early Stopping Patience:", self.patience_input)
        layout.addLayout(form_layout)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Buttons
        button_layout = QHBoxLayout()
        self.train_button = QPushButton("🚀 Start Training")
        self.train_button.clicked.connect(self.start_training)
        button_layout.addWidget(self.train_button)

        layout.addLayout(button_layout)

        self.training_group.setLayout(layout)
        return self.training_group

    def _create_model_mgmt_group(self):
        """Creates the Model Management group box and its widgets."""
        group = QGroupBox("Model Management")
        layout = QHBoxLayout()

        self.save_weights_button = QPushButton("💾 Save Weights")
        self.save_weights_button.clicked.connect(self.save_weights)
        layout.addWidget(self.save_weights_button)

        self.load_weights_button = QPushButton("📂 Load Weights")
        self.load_weights_button.clicked.connect(self.load_weights)
        layout.addWidget(self.load_weights_button)

        group.setLayout(layout)
        return group

    def _create_log_area(self):
        """Creates the QTextEdit widget for logging."""
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        return self.log

    # --- End UI Creation Helper Methods ---

    # --- Slot to handle template selection changes ---
    def _update_hidden_layer_input(self, template_name):
        if template_name in self.model_templates:
            size = self.model_templates[template_name]
            is_custom = (template_name == "(Custom)")

            # Enable/disable the input field
            self.hidden_layer_input.setEnabled(is_custom)

            # Update the value if a non-custom template is selected
            if not is_custom and size > 0:
                self.hidden_layer_input.setValue(size)
            elif not is_custom and size <= 0: # Should not happen with current templates
                self._log_message(f"Warning: Template '{template_name}' has invalid size {size}. Using default.")
                self.hidden_layer_input.setValue(10) # Fallback
            # When switching to custom, leave the current value as is

            # Log change
            mode = "(Custom)" if is_custom else f"Template '{template_name}' ({size} hidden neurons)"
            self._log_message(f"Model configuration set to: {mode}")
            # Re-initialize model if data is already loaded
            if self.X_train is not None:
                 self._log_message("Dataset loaded. Re-initializing model for new hidden layer size.")
                 self._post_load_update(self.dataset_combo.currentText()) # Trigger re-init
        # ----------------------------------------------

    # --- New method to populate dataset dropdown ---
    def populate_dataset_dropdown(self):
        self.dataset_combo.clear() # Clear existing items
        self.quickdraw_category_map = {} # Reset map: display_name -> index
        self.quickdraw_files_map = {}    # Reset map: display_name -> path
        self.num_quickdraw_classes = 0   # Reset count
        current_qd_index = 0            # Start indexing QD classes from 0
        found_datasets = []             # Temp list to hold names for sorting

        self._log_message("Scanning for datasets...")

        # 1. Check for built-in MNIST
        mnist_path = "data/train.csv"
        if os.path.exists(mnist_path):
            found_datasets.append("MNIST")
            self._log_message(f"Found built-in: MNIST ({mnist_path})")
        else:
             self._log_message(f"WARN: MNIST dataset not found at {mnist_path}")

        # 2. Check for built-in Emojis
        emoji_path = "data/emojis.csv"
        if os.path.exists(emoji_path):
            found_datasets.append("Emojis")
            self._log_message(f"Found built-in: Emojis ({emoji_path})")
        else:
            self._log_message(f"WARN: Emojis dataset not found at {emoji_path}")

        # 3. Scan data/quickdraw for .npy files
        quickdraw_dir = "data/quickdraw/"
        if os.path.isdir(quickdraw_dir):
            npy_files = sorted(glob.glob(os.path.join(quickdraw_dir, "*.npy"))) # Sort for consistent indexing
            if npy_files:
                 self._log_message(f"Found {len(npy_files)} QuickDraw datasets in {quickdraw_dir}:")
                 self.num_quickdraw_classes = len(npy_files)
                 for npy_path in npy_files:
                     # Extract category name (e.g., 'cat' from 'data/quickdraw/cat.npy')
                     category_name = os.path.splitext(os.path.basename(npy_path))[0]
                     display_name = f"QuickDraw: {category_name}"
                     # Map display name to its index and store
                     self.quickdraw_category_map[display_name] = current_qd_index
                     self.quickdraw_files_map[display_name] = npy_path # Store path
                     found_datasets.append(display_name)
                     self._log_message(f"  - {display_name} (Index: {current_qd_index}, Path: {npy_path})")
                     current_qd_index += 1
                 # Add the combined option if multiple QD files were found
                 if self.num_quickdraw_classes > 1:
                     found_datasets.append("QuickDraw: All Categories")
                     self._log_message(f"Added 'QuickDraw: All Categories' option ({self.num_quickdraw_classes} classes)")
            else:
                 self._log_message(f"No .npy files found in {quickdraw_dir}")
        else:
            self._log_message(f"QuickDraw directory not found: {quickdraw_dir}")

        # TODO: Add scanning for other dataset types/locations if needed

        # Populate dropdown from sorted list
        if found_datasets:
            for name in found_datasets:
                self.dataset_combo.addItem(name)
            self.dataset_combo.setEnabled(True)
            self.load_dataset_button.setEnabled(True)
        else:
             self._log_message("ERROR: No datasets found! Cannot train.")
             self.dataset_combo.addItem("No Datasets Found")
             self.dataset_combo.setEnabled(False)
             self.load_dataset_button.setEnabled(False)

    # --- End new method ---

    # Helper method for logging with timestamp
    def _log_message(self, message):
        timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        self.log.append(f"[{timestamp}] {message}")
        QApplication.processEvents() # Keep UI responsive

    # Modify the progress update slot
    def _update_progress(self, current_iteration, total_iterations):
        # Update progress bar based on iterations
        self.progress_bar.setValue(current_iteration)
        # QApplication.processEvents() # Generally avoid in slots connected to threads

    # New slots to handle signals from the worker
    def _handle_training_finished(self, results):
        self._log_message("Training finished successfully. Updating model parameters.")
        # Unpack results
        self.model_params = (results[0], results[1], results[2], results[3])
        self.train_loss_history = results[4]
        self.val_accuracy_history = results[5]

        # --- Cleanup handled by thread's finished signal connection --- # Keep this comment
        # self._cleanup_thread() # Don't call directly here, let signal handle it

    def _handle_worker_log(self, message):
        # Simple pass-through logging for now
        self._log_message(f"[Worker]: {message}")

    def _handle_worker_error(self, error_message):
        self._log_message(f"ERROR from worker thread: {error_message}")
        # Clean up thread and re-enable UI even on error
        self._cleanup_thread()

    def _cleanup_thread(self):
        self._log_message("Cleaning up training thread...")
        if self.training_thread is not None:
            self.training_thread.quit()
            self.training_thread.wait()
        self.training_thread = None
        self.training_worker = None
        self.train_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self._log_message("=== Training Run Finished/Aborted ===")
        QApplication.processEvents()

    def _predict_image_file(self):
        """Handles image selection, preprocessing, and prediction using the current model."""
        self._log_message("--- Starting Prediction ---")
        # Clear previous preview and graph
        self.image_preview_label.clear()
        self.image_preview_label.setText("Select an image...")
        self.probability_graph.clear_graph() # Clear the graph

        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            try:
                self._log_message(f"Loading image: {file_name}")
                img = Image.open(file_name)
                # Convert to grayscale and resize for the model
                img_processed = img.convert("L").resize((28, 28))
                img_array = np.array(img_processed).reshape(784, 1) / 255.0

                # Create QPixmap for display
                qpixmap_orig = QPixmap(file_name)
                if qpixmap_orig.isNull():
                    self._log_message(f"Warning: QPixmap failed to load {file_name} directly. Trying PIL conversion.")
                    rgb_img = img.convert('RGB')
                    qimage = QImage(rgb_img.tobytes("raw", "RGB"), rgb_img.width, rgb_img.height, QImage.Format_RGB888)
                    qpixmap_orig = QPixmap.fromImage(qimage)

                scaled_pixmap = qpixmap_orig.scaled(self.image_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_preview_label.setPixmap(scaled_pixmap)

                self._log_message("Running forward propagation...")
                if not hasattr(self, 'model_params') or self.model_params[0] is None:
                     self._log_message("ERROR: Model parameters not loaded. Load weights first.")
                     self.image_preview_label.setText("Load weights first.")
                     self.probability_graph.clear_graph()
                     return # Exit the try block and method if no model

                # Perform forward propagation
                _, _, _, output, status = neural_net.forward_prop(*self.model_params, img_array)
                # Check status from forward_prop
                if not status:
                    self._log_message("ERROR: Forward propagation failed (NaN/inf detected?).")
                    self.probability_graph.clear_graph()
                    return # Exit if forward prop failed

                prediction = np.argmax(output)
                self._log_message(f"Prediction Result: {prediction}")

                # Update probability bar graph
                self.probability_graph.set_probabilities(output.flatten(), prediction)

            except Exception as e:
                self._log_message(f"ERROR during prediction process: {e}")
                self.image_preview_label.setText("Error loading/predicting.")
                self.probability_graph.clear_graph()
        else:
             self._log_message("Prediction cancelled: No file selected.")

    def start_training(self):
        if self.X_train is None or self.Y_train is None or self.X_dev is None or self.Y_dev is None:
            self._log_message("ERROR: No dataset loaded completely.")
            return
        # Prevent starting a new thread if one is already running
        if self.training_thread is not None and self.training_thread.isRunning():
             self._log_message("WARN: Training is already in progress.")
             return

        self._log_message(f"=== Starting Training Thread for {self.dataset_combo.currentText()} ===")
        self.train_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        QApplication.processEvents()

        epochs = self.epochs_input.value()
        learning_rate = self.lr_input.value()
        patience = self.patience_input.value()
        self._log_message(f"Hyperparameters: Epochs={epochs}, Learning Rate={learning_rate}, Patience={patience}")
        self.progress_bar.setMaximum(epochs)

        # --- Threading Setup --- 
        self.training_thread = QThread() # Create a new thread
        # Create worker instance with necessary data
        self.training_worker = TrainingWorker(
            self.X_train, self.Y_train, self.X_dev, self.Y_dev,
            learning_rate, epochs, self.model_params,
            patience # Pass patience value
        )
        # Move worker to the thread
        self.training_worker.moveToThread(self.training_thread)

        # --- Connect Signals and Slots ---
        # Connect worker signals to main window slots
        self.training_worker.progress_updated.connect(self._update_progress)
        self.training_worker.training_finished.connect(self._handle_training_finished)
        self.training_worker.log_message.connect(self._handle_worker_log)
        self.training_worker.error_occurred.connect(self._handle_worker_error)

        # Connect thread signals
        self.training_thread.started.connect(self.training_worker.run_training) # Run worker task when thread starts
        self.training_worker.training_finished.connect(self.training_thread.quit)
        self.training_worker.error_occurred.connect(self.training_thread.quit)
        self.training_thread.finished.connect(self.training_worker.deleteLater) # Schedule worker for deletion
        self.training_thread.finished.connect(self.training_thread.deleteLater) # Schedule thread for deletion
        # We handle UI cleanup in _cleanup_thread via _handle_training_finished/_handle_worker_error
        # --- End Signal/Slot Connections --- 

        # Start the thread
        self.training_thread.start()
        self._log_message("Training thread started.")

    # Add write method to handle stdout redirection for logging
    def write(self, text):
        # This method handles the redirected print output from gradient_descent
        # We might want to strip leading/trailing whitespace
        text = text.strip()
        if text: # Avoid logging empty lines
            self.log.moveCursor(QtGui.QTextCursor.End)
            # Prepending timestamp to redirected output might be too noisy,
            # so we just insert the text as is.
            self.log.insertPlainText(text + '\n')
            QApplication.processEvents()

    # Add flush method (required for stdout redirection)
    def flush(self):
        pass

    # --- Main Action Methods ---

    # Add methods for dataset loading buttons
    def load_selected_dataset(self):
        selected_dataset = self.dataset_combo.currentText()
        self._log_message(f"--- Loading Dataset: {selected_dataset} ---")
        # Get label column index from UI (primarily for non-predefined, but useful if defaults change)
        label_col_idx = self.label_col_input.value()

        # --- Dispatch to specific loading handlers ---
        if selected_dataset == "MNIST":
            self._handle_load_mnist(label_col_idx)
        elif selected_dataset == "QuickDraw: All Categories":
            self._handle_load_quickdraw_all()
        elif selected_dataset.startswith("QuickDraw: "):
            self._handle_load_quickdraw_single(selected_dataset)
        elif selected_dataset == "Emojis":
            self._handle_load_emojis()
        else:
            self._handle_load_unknown(selected_dataset)

        self._log_message("--- Dataset Loading Finished ---")

    # --- Dataset Loading Handlers (Called by load_selected_dataset) ---

    def _handle_load_mnist(self, label_col_idx):
        """Handles loading the predefined MNIST dataset."""
        csv_path = "data/train.csv"
        self._load_csv_internal("MNIST", csv_path, label_col_idx)

    def _handle_load_quickdraw_all(self):
        """Handles loading the 'QuickDraw: All Categories' dataset."""
        selected_dataset = "QuickDraw: All Categories"
        if self.num_quickdraw_classes > 1 and self.quickdraw_files_map:
            # Build the map of path -> index needed by the multi-loader
            path_index_map = {path: index for display_name, path in self.quickdraw_files_map.items()
                                for dn, index in self.quickdraw_category_map.items() if dn == display_name}
            # Pass validation_split (using default 0.1 from datasets.py for NPY)
            self._load_multiple_npy_internal(selected_dataset, path_index_map)
        else:
            self._log_message("ERROR: Cannot load 'All Categories'. Need >= 2 QuickDraw .npy files.")
            self.training_group.setTitle("Training Controls (Load failed)")

    def _handle_load_quickdraw_single(self, selected_dataset):
        """Handles loading a single QuickDraw category dataset."""
        category_name = selected_dataset.replace("QuickDraw: ", "")
        npy_path = os.path.join("data/quickdraw/", f"{category_name}.npy")
        if selected_dataset in self.quickdraw_category_map:
            category_index = self.quickdraw_category_map[selected_dataset]
            num_classes = self.num_quickdraw_classes # Total classes
            # Pass validation_split (using default 0.1 from datasets.py for NPY)
            self._load_npy_internal(selected_dataset, npy_path, category_index, num_classes)
        else:
            self._log_message(f"ERROR: Could not find category index for '{selected_dataset}'. Rescanning might be needed.")
            self.training_group.setTitle("Training Controls (Load failed)")

    def _handle_load_emojis(self):
        """Handles loading the predefined Emojis dataset."""
        emoji_path = "data/emojis.csv"
        # Pass validation_split (using default 0.1 from datasets.py for emoji)
        self._load_emoji_internal("Emojis", emoji_path)

    def _handle_load_unknown(self, selected_dataset):
        """Handles cases where the selected dataset is not a known predefined one."""
        self._log_message(f"WARN: Loading logic for '{selected_dataset}' not handled by predefined loaders.")
        # Attempt to treat as already loaded if it's in the combo (might be an uploaded CSV)
        if self.X_train is not None and self.dataset_combo.currentText() == selected_dataset:
            self._log_message(f"Assuming '{selected_dataset}' refers to the currently loaded (uploaded) data. No reload performed.")
            # Optionally, re-run _post_load_update to ensure consistency?
            # self._post_load_update(selected_dataset)
        else:
            self._log_message(f"ERROR: Cannot load '{selected_dataset}'. Select a known dataset or upload a new CSV.")
            # Reset potentially partially loaded data if it wasn't the current one
            self.X_train, self.Y_train, self.X_dev, self.Y_dev = None, None, None, None
            self.current_num_classes = 0
            self.model_params = None
            self.training_group.setTitle("Training Controls (Load failed)")

    # --- Helper method for loading CSV (internal logic) ---
    def _load_csv_internal(self, dataset_name, csv_path, label_col_idx,
                         validation_split=1000, # Default validation for CSV
                         image_col_index=None, image_col_type=None):
        # Log the parameters being used
        log_params = f"LabelCol={label_col_idx}"
        if image_col_index is not None:
            log_params += f", ImageCol={image_index}, ImageType={image_type}"
        else:
            log_params += f", NumClasses(Default)={validation_split}"
        self._log_message(f"Loading CSV: {dataset_name} from {csv_path} ({log_params})")

        # Reset existing data before loading new
        self.X_train, self.Y_train, self.X_dev, self.Y_dev = None, None, None, None
        # Pass image column info to the loader function
        loaded_data = datasets.load_csv_dataset(csv_path,
                                                label_col_index=label_col_idx,
                                                image_col_index=image_col_index,
                                                image_col_type=image_col_type,
                                                validation_split=validation_split)
        # Unpack results (loader now returns 5 items)
        self.X_train, self.Y_train, self.X_dev, self.Y_dev, determined_num_classes = loaded_data

        if self.X_train is not None:
            self._log_message(f"Dataset '{dataset_name}' loaded successfully.")
            # Store num_classes (use determined value if available, else default)
            self.current_num_classes = determined_num_classes if determined_num_classes > 0 else validation_split
            self._log_message(f"Using {self.current_num_classes} classes for model initialization.")
            self._post_load_update(dataset_name)
        else:
            self._log_message(f"ERROR: Failed to load {dataset_name} from {csv_path}. Check file, format, and column indices/type.")
            self.training_group.setTitle("Training Controls (Load failed)")

    # --- Helper method for loading NPY (internal logic) ---
    def _load_npy_internal(self, dataset_name, npy_path, category_index, num_classes, validation_split=0.1):
        self._log_message(f"Loading NPY: {dataset_name} from {npy_path} (Category Index: {category_index}, Total QD Classes: {num_classes})")
        # Reset existing data before loading new
        self.X_train, self.Y_train, self.X_dev, self.Y_dev = None, None, None, None
        # Pass category_index, num_classes, and validation_split to the loader function
        loaded_data = datasets.load_npy_dataset(npy_path, category_index, num_classes, validation_split=validation_split)
        self.X_train, self.Y_train, self.X_dev, self.Y_dev, loaded_num_classes = loaded_data

        if self.X_train is not None:
            self._log_message(f"Dataset '{dataset_name}' loaded successfully.")
            # Store num_classes for potential use in training setup
            self.current_num_classes = num_classes
            self._post_load_update(dataset_name)
        else:
            self._log_message(f"ERROR: Failed to load {dataset_name} from {npy_path}. Check file format and ensure it contains valid data.")
            self.training_group.setTitle("Training Controls (Load failed)")

    # --- Helper method for loading multiple NPY (internal logic) ---
    def _load_multiple_npy_internal(self, dataset_name, path_index_map, validation_split=0.1):
        self._log_message(f"Loading Multiple NPY: {dataset_name}")
        self._log_message(f"Combining {len(path_index_map)} categories...")
        # Reset existing data before loading new
        self.X_train, self.Y_train, self.X_dev, self.Y_dev = None, None, None, None
        # Call the new loader function
        loaded_data = datasets.load_multiple_npy_datasets(path_index_map, validation_split=validation_split)
        self.X_train, self.Y_train, self.X_dev, self.Y_dev, loaded_num_classes = loaded_data

        if self.X_train is not None:
            self._log_message(f"Combined dataset '{dataset_name}' loaded successfully.")
            # Store num_classes for potential use in training setup
            self.current_num_classes = loaded_num_classes # Use num_classes returned by loader
            self._post_load_update(dataset_name)
        else:
            self._log_message(f"ERROR: Failed to load combined dataset '{dataset_name}'. Check logs for details.")
            self.training_group.setTitle("Training Controls (Load failed)")

    # --- Helper method for loading Emoji CSV (internal logic) ---
    def _load_emoji_internal(self, dataset_name, csv_path, image_col='Google', validation_split=0.1):
        self._log_message(f"Loading Emoji CSV: {dataset_name} from {csv_path} using '{image_col}' images")
        # Reset existing data
        self.X_train, self.Y_train, self.X_dev, self.Y_dev = None, None, None, None
        # Call the dataset loader
        loaded_data = datasets.load_emoji_dataset(csv_path, image_column=image_col, validation_split=validation_split)
        self.X_train, self.Y_train, self.X_dev, self.Y_dev, loaded_num_classes = loaded_data

        if self.X_train is not None:
            self._log_message(f"Dataset '{dataset_name}' loaded successfully.")
            # Determine num_classes from the loaded labels
            num_classes = 0
            if self.Y_train is not None and len(self.Y_train) > 0:
                num_classes = int(np.max(self.Y_train)) + 1 # Assumes labels are 0-based indices
                if self.Y_dev is not None and len(self.Y_dev) > 0:
                    num_classes = max(num_classes, int(np.max(self.Y_dev)) + 1)

            if num_classes > 0:
                 self.current_num_classes = num_classes
                 self._log_message(f"Determined {num_classes} classes for Emojis.")
                 self._post_load_update(dataset_name)
            else:
                 self._log_message(f"ERROR: Could not determine number of classes for {dataset_name}. Load failed.")
                 self.training_group.setTitle("Training Controls (Load failed)")

        else:
            self._log_message(f"ERROR: Failed to load {dataset_name} from {csv_path}. Check file format and image data.")
            self.training_group.setTitle("Training Controls (Load failed)")

    # --- Helper method to update UI after successful load ---
    def _post_load_update(self, dataset_name):
        self.training_group.setTitle(f"Training Controls ({dataset_name})") # Update title
        self.train_loss_history = [] # Reset history when new dataset loaded
        self.val_accuracy_history = []

        # --- Re-initialize model parameters for the new dataset size ---
        if self.current_num_classes > 0:
           # Get hidden layer size from the UI input
           hidden_size = self.hidden_layer_input.value()
           self._log_message(f"Re-initializing model: Hidden Size={hidden_size}, Output Classes={self.current_num_classes}")
           self.model_params = neural_net.init_params(
               num_classes=self.current_num_classes,
               hidden_layer_size=hidden_size
           )
           self._log_message("Model re-initialized.")
        else:
             self._log_message("ERROR: Number of classes is 0 or unknown. Cannot initialize model parameters.")
             self.model_params = None
        # -------------------------------------------------------------

        # Update combo box if needed (e.g., for uploaded)
        if self.dataset_combo.findText(dataset_name) == -1:
            self.dataset_combo.addItem(dataset_name)
        self.dataset_combo.setCurrentText(dataset_name)
        # --- Enable/disable image type based on image col input ---
        self._update_image_col_type_state()

    # --- Add slot to update image type combo enabled state ---
    def _update_image_col_type_state(self):
        image_col_idx = self.image_col_input.value()
        is_pixel_mode = (image_col_idx == -1)
        self.image_type_combo.setEnabled(not is_pixel_mode)
        if is_pixel_mode:
            self.image_type_combo.setCurrentIndex(0) # Set to "(Not Applicable)"

    def upload_csv_dataset(self):
        self._log_message("--- Uploading CSV Dataset --- ")
        csv_path, _ = QFileDialog.getOpenFileName(self, "Upload Dataset CSV", "", "CSV Files (*.csv)")
        if csv_path:
            dataset_name = os.path.basename(csv_path)
            self._log_message(f"Attempting to load uploaded dataset: {dataset_name} from {csv_path}")
            label_col_idx = self.label_col_input.value()
            self._log_message(f"Using Label Column Index: {label_col_idx}")

            # --- Get Image Column settings ---
            image_col_idx = self.image_col_input.value()
            image_col_type = None
            if image_col_idx != -1:
                image_col_type = self.image_type_combo.currentText()
                if image_col_type == "(Not Applicable)": # Should not happen if logic is right, but check
                    self._log_message("ERROR: Image column index specified, but type is invalid. Select 'base64' or 'path'.")
                    self.training_group.setTitle("Training Controls (Load failed)")
                    return # Abort loading
                self._log_message(f"Using Image Column Index: {image_col_idx}, Type: {image_col_type}")
            else:
                self._log_message("Using raw pixel columns (Image Column Index is -1).")
            # ---------------------------------

            # Use the internal CSV loader, passing image column info
            # It now returns num_classes as the 5th element
            self._load_csv_internal(dataset_name, csv_path, label_col_idx,
                                    image_col_index=image_col_idx if image_col_idx != -1 else None,
                                    image_col_type=image_col_type)
            # _post_load_update is called within _load_csv_internal if successful
        else:
             self._log_message("Upload cancelled: No file selected.")
        self._log_message("--- Upload Finished ---")

    # --- Model Save/Load Methods ---
    def save_weights(self):
        # Ensure we have model parameters to save
        if not self.model_params:
            self._log_message("ERROR: No model parameters available to save.")
            return

        # Open save file dialog
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Model Weights", "", "NumPy NPZ Files (*.npz)")

        if file_path:
            # Ensure the filename ends with .npz
            if not file_path.endswith('.npz'):
                file_path += '.npz'

            try:
                W1, b1, W2, b2 = self.model_params
                np.savez(file_path, W1=W1, b1=b1, W2=W2, b2=b2)
                self._log_message(f"Weights saved successfully to: {file_path}")
            except Exception as e:
                self._log_message(f"ERROR saving weights to {file_path}: {e}")

    def load_weights(self):
        # Open load file dialog
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Model Weights", "", "NumPy NPZ Files (*.npz)")

        if file_path:
            try:
                self._log_message(f"Attempting to load weights from: {file_path}")
                data = np.load(file_path)
                # Check if required keys are present
                if all(k in data for k in ['W1', 'b1', 'W2', 'b2']):
                    W1 = data['W1']
                    b1 = data['b1']
                    W2 = data['W2']
                    b2 = data['b2']
                    self.model_params = (W1, b1, W2, b2)
                    self._log_message("Weights loaded successfully.")
                    # Reset training history as it doesn't correspond to loaded weights
                    self.train_loss_history = []
                    self.val_accuracy_history = []
                else:
                    self._log_message(f"ERROR: Loaded file {file_path} is missing required weight keys (W1, b1, W2, b2).")
            except Exception as e:
                self._log_message(f"ERROR loading weights from {file_path}: {e}")
        else:
            self._log_message("Load weights cancelled.")

    # --- New Slot for Predicting Drawing ---
    def _predict_drawing(self):
        self._log_message("--- Starting Prediction from Drawing ---")
        # Clear previous file prediction display if any
        # self.image_preview_label.clear() # Keep previous drawing preview?
        self.probability_graph.clear_graph() # Clear graph

        # Get the drawing data
        img_array = self.drawing_canvas.getDrawingArray()
        preview_pixmap = self.drawing_canvas.getPreviewPixmap()

        if img_array is None:
            self._log_message("Prediction cancelled: Drawing canvas is empty or error occurred.")
            self.image_preview_label.setText("Draw something!")
            self.probability_graph.clear_graph()
            return

        if preview_pixmap:
            # Scale pixmap to fit the label while keeping aspect ratio
            scaled_pixmap = preview_pixmap.scaled(self.image_preview_label.size(),
                                                 Qt.KeepAspectRatio,
                                                 Qt.SmoothTransformation)
            self.image_preview_label.setPixmap(scaled_pixmap)
        else:
            self.image_preview_label.setText("Preview Error")

        # Run prediction
        try:
            self._log_message("Running forward propagation on drawing...")
            if not hasattr(self, 'model_params') or self.model_params[0] is None:
                 self._log_message("ERROR: Model parameters not loaded. Load weights first.")
                 self.image_preview_label.setText("Load weights first.")
                 self.probability_graph.clear_graph()
                 return

            _, _, _, output, status = neural_net.forward_prop(*self.model_params, img_array)

            if not status:
                self._log_message("ERROR: Forward propagation failed (NaN/inf detected?).")
                self.probability_graph.clear_graph()
                return

            prediction = np.argmax(output)
            self._log_message(f"Prediction Result: {prediction}")

            # Update probability bar graph
            self.probability_graph.set_probabilities(output.flatten(), prediction)

        except Exception as e:
            self._log_message(f"ERROR during prediction: {e}")
            self.image_preview_label.setText("Error predicting.")
            self.probability_graph.clear_graph()

        self._log_message("--- Prediction Finished ---")

    # --- Add method to show plot in a separate window ---
    def _show_expanded_plot(self):
        self._log_message("Opening expanded plot window...")

        # Create a new dialog window
        # Store it in self to prevent garbage collection if shown non-modally
        self.expanded_plot_dialog = QDialog(self) # Parent is the main window
        self.expanded_plot_dialog.setWindowTitle("Expanded Training Plot")
        self.expanded_plot_dialog.setMinimumSize(600, 400) # Give it a reasonable default size

        # Layout for the dialog
        dialog_layout = QVBoxLayout(self.expanded_plot_dialog)

        # Create a NEW PlotWidget instance for the dialog
        expanded_plot_widget = PlotWidget(self.expanded_plot_dialog)
        dialog_layout.addWidget(expanded_plot_widget)

        # Update the new plot widget with existing data (if any)
        if self.train_loss_history or self.val_accuracy_history:
            try:
                # Assume default interval for now, might need adjustment
                # if gradient_descent interval changes
                interval = 10
                expanded_plot_widget.update_plot(self.train_loss_history, self.val_accuracy_history, interval)
                self._log_message("Populated expanded plot with current history.")
            except Exception as e:
                self._log_message(f"Error updating expanded plot: {e}")
        else:
            self._log_message("No training history yet for expanded plot.")

        # Show the dialog (non-modal)
        self.expanded_plot_dialog.show()
    # --- End expanded plot method ---
