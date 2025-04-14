import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QComboBox, QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout,
    QApplication, QProgressBar, QSizePolicy, QCheckBox, QTabWidget, QDialog, QFrame # Import QDialog
)
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QColor, QPen, QBrush, QFontMetrics, QIcon # Import QIcon
)
from PyQt5.QtCore import Qt, QDateTime, QObject, pyqtSignal, QThread, QRectF, QSize # Import QSize for canvas
import numpy as np
# --- Import Model Class --- #
try:
    from model.neural_net import SimpleNeuralNetwork
except ImportError as e:
    print(f"ERROR: Cannot import SimpleNeuralNetwork from model.neural_net - {e}")
    SimpleNeuralNetwork = None # Use None as fallback if import fails
# --- Import CNN Model --- #
try:
    from model.cnn_model import CNNModel # ADD THIS IMPORT
except ImportError as e:
    print(f"ERROR: Cannot import CNNModel from model.cnn_model - {e}")
    CNNModel = None # Fallback
# ------------------------ #
import datasets # Import datasets from top level
from PIL import Image
import os
from PyQt5 import QtGui # Keep QtGui import for QTextCursor
import glob # Import glob for file searching
from typing import Optional, Dict, Any, List, Tuple # Added Tuple
import pickle # Import pickle for CIFAR-10 loading

# --- Local UI Component Imports --- #
# Moved these back to top level
try:
    from ui.drawing_canvas import DrawingCanvas
except ImportError as e:
    print(f"ERROR: Could not import DrawingCanvas. Make sure ui/drawing_canvas.py exists: {e}")
    class DrawingCanvas(QWidget): # Dummy class if import fails
        def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs); QLabel("Error loading DrawingCanvas", self)
        def clearCanvas(self): pass
        def getDrawingArray(self): return None
        def getPreviewPixmap(self): return None

try:
    from ui.plot_widget import PlotWidget
except ImportError as e:
    print(f"ERROR: Could not import PlotWidget. Make sure ui/plot_widget.py exists: {e}")
    class PlotWidget(QWidget): # Dummy class
        def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs); QLabel("Error loading PlotWidget", self)
        def update_plot(self, *args, **kwargs): pass

try:
    from ui.training_worker import TrainingWorker
except ImportError as e:
    print(f"ERROR: Could not import TrainingWorker: {e}")
    class TrainingWorker(QObject): pass # Dummy

try:
    from ui.probability_bar_graph import ProbabilityBarGraph
except ImportError as e:
    print(f"ERROR: Could not import ProbabilityBarGraph: {e}")
    class ProbabilityBarGraph(QWidget): pass # Dummy
# ---------------------------------- #

# --- Main Application Execution --- #
if __name__ == '__main__':
    app = QApplication(sys.argv)
    # --- Temporary Patch for Model Import --- #
    # Try importing the class-based model again here, just before creating MainWindow
    try:
        from model.neural_net import SimpleNeuralNetwork
    except ImportError:
        print("WARN: Could not import SimpleNeuralNetwork at runtime. Using fallback.")
        SimpleNeuralNetwork = None
    # -------------------------------------- #
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__() # Initialize the QMainWindow base class
        self.setWindowTitle("🧠 Neural Net Playground") # Set the text in the window title bar
        self.setWindowIcon(QIcon('assets/icon.png')) # Set the application icon (appears in title bar/taskbar)
        # Adjust minimum size if needed, tabs might help
        self.setMinimumSize(600, 550) # Prevent the window from being resized too small

        # --- Initialize Core Attributes ---
        # Keeps track of important application state like loaded data, model parameters, etc.
        self._init_attributes()
        self.expanded_plot_dialog = None # Holds the pop-out plot window if created

        # --- Setup Main Layout ---
        # QWidget is the base class for UI objects. We use one as the central container.
        self.central = QWidget()
        # QVBoxLayout arranges widgets vertically. This is the main layout for the central widget.
        self.main_layout = QVBoxLayout()

        # --- Create Log Area (Must be done before widgets that log) --- #
        # The log area needs to exist before other UI parts are created, in case they log messages during setup.
        log_area = self._create_log_area()

        # --- Create Tab Widget --- #
        # QTabWidget provides a tabbed interface to organize different sections of the UI.
        self.tabs = QTabWidget()

        # --- Create Widgets for Tabs --- #
        # Each tab holds a QWidget, which in turn has its own layout and contains the relevant UI groups.

        # Data Tab: For selecting and loading datasets
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab) # Layout for the Data tab
        # Create the group box containing dataset selection widgets
        dataset_group = self._create_dataset_group()
        data_layout.addWidget(dataset_group) # Add the group box to the tab's layout
        data_layout.addStretch() # Pushes content up if there's extra vertical space

        # Train Tab: For configuring and running model training
        train_tab = QWidget()
        train_layout = QVBoxLayout(train_tab) # Layout for the Train tab
        # Create group boxes for training controls and model management (save/load)
        self.training_group = self._create_training_group()
        model_mgmt_group = self._create_model_mgmt_group()
        # Layout for the Expand Plot Button and Info Text
        plot_info_layout = QHBoxLayout() # Horizontal layout for button and text

        expand_plot_button = QPushButton("🔎 Expand Plot")
        # Connect the button's 'clicked' signal to the '_show_expanded_plot' method (slot)
        expand_plot_button.clicked.connect(self._show_expanded_plot)
        plot_info_layout.addWidget(expand_plot_button)

        plot_info_label = QLabel("<- Training history visualized here")
        plot_info_label.setStyleSheet("font-style: italic; color: grey;") # Style the label
        plot_info_layout.addWidget(plot_info_label)
        plot_info_layout.addStretch() # Push button and label left

        # Add the training-related group boxes and the plot info layout to the Train tab
        train_layout.addWidget(self.training_group)
        train_layout.addWidget(model_mgmt_group)
        train_layout.addLayout(plot_info_layout)
        # No stretch here, let the training section potentially expand if needed

        # Test Tab: For running predictions on the trained model
        infer_tab = QWidget() # Renamed to 'Test' visually, but variable name is 'infer_tab'
        infer_layout = QVBoxLayout(infer_tab) # Layout for the Test tab
        # Create the group box containing inference controls (file, drawing) and results display
        inference_group = self._create_inference_group()
        infer_layout.addWidget(inference_group)
        infer_layout.addStretch() # Push content up

        # Add Tabs to Tab Widget with user-friendly names and icons
        self.tabs.addTab(data_tab, "💾 Data")
        self.tabs.addTab(train_tab, "🚀 Train")
        self.tabs.addTab(infer_tab, "🧪 Test") # This is the 'Test' tab

        # --- Add Tab Widget and Log Area to Main Layout --- #
        self.main_layout.addWidget(self.tabs) # Add the entire tab widget structure
        self.main_layout.addWidget(log_area) # Add the log area below the tabs
        # Adjust stretch factor: give tabs more vertical space (4 parts) than the log area (1 part)
        self.main_layout.setStretchFactor(self.tabs, 4)
        self.main_layout.setStretchFactor(log_area, 1)

        # --- Finalize Layout ---
        self.central.setLayout(self.main_layout) # Apply the main layout to the central widget
        self.setCentralWidget(self.central) # Set the central widget for the main window

        # --- Initialize Application State ---
        # self._populate_model_dropdown() # REMOVE - Already called in _create_training_group
        # Add connections that depend on multiple UI parts being created
        self.image_col_input.valueChanged.connect(self._update_image_col_type_state)
        self._update_image_col_type_state() # Set initial state of image type combo
        self.scan_datasets() # Find available datasets on startup
        self.populate_dataset_dropdown() # Populate dropdown AFTER scanning

    def _init_attributes(self):
        # Initialize attributes to store application state
        self.datasets_info: Dict[str, Dict[str, Any]] = {} # Discovered dataset paths and types
        self.current_dataset: Optional[np.ndarray] = None
        self.current_labels: Optional[np.ndarray] = None
        self.current_dataset_name: Optional[str] = None
        self.num_classes: int = 0 # Number of classes in the loaded dataset
        self.model_params: Optional[dict] = None # Stores trained weights and biases
        self.model_layer_dims: Optional[List[int]] = None # Stores layer dimensions of current self.model_params
        self.training_worker: Optional[TrainingWorker] = None
        self.training_thread: Optional[QThread] = None
        self.train_loss_history: List[float] = []
        self.val_accuracy_history: List[float] = []
        self.class_names: Optional[List[str]] = None # List to store class names (e.g., ["cat", "dog", ...])

        # --- Model Related Attributes ---
        self.current_model: Optional[Any] = None # Holds the instantiated model object (SimpleNN or CNN)
        self.current_model_type: Optional[str] = "Simple NN" # Default or track selected type, e.g., 'Simple NN', 'CNN'
        self.model_weights_path: Optional[str] = None # Store path for Keras weights if applicable
        # --------------------------------

    # --- UI Group Creation Methods --- #

    def _create_dataset_group(self):
        """Creates the GroupBox containing dataset selection and loading widgets."""
        dataset_group = QGroupBox("Dataset Loading")
        # QFormLayout is convenient for label-widget pairs
        form_layout = QFormLayout()

        # Dropdown for selecting pre-discovered datasets
        self.dataset_dropdown = QComboBox()
        self.dataset_dropdown.setToolTip("Select a dataset automatically found in the 'data' directory")
        # When the selection changes, update the UI state
        self.dataset_dropdown.currentIndexChanged.connect(self._on_dataset_selected)
        form_layout.addRow("Select Dataset:", self.dataset_dropdown)

        # Button to load the dataset chosen in the dropdown
        self.load_dataset_button = QPushButton("Load Selected")
        self.load_dataset_button.setToolTip("Load the dataset chosen in the dropdown above")
        # Connect the button click to the loading function
        self.load_dataset_button.clicked.connect(self.load_selected_dataset)
        # Add the button directly below the dropdown (spanning both columns of the form layout)
        form_layout.addRow(self.load_dataset_button)

        # --- Widgets for Custom CSV Upload --- #
        # Add a horizontal line separator for visual clarity
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        form_layout.addRow(separator)

        # Button to open a file dialog for uploading a CSV
        upload_button = QPushButton("Upload Custom CSV")
        upload_button.setToolTip("Upload your own CSV (pixels, base64 images, or image paths)")
        upload_button.clicked.connect(self.upload_csv_dataset)
        form_layout.addRow("Load Custom:", upload_button)

        # Input for specifying the label column index in the custom CSV
        self.label_col_input = QSpinBox()
        self.label_col_input.setRange(0, 1000) # Allow columns 0 to 1000
        self.label_col_input.setValue(0) # Default to the first column (index 0)
        self.label_col_input.setToolTip("Index (0-based) of the column containing the labels in your CSV")
        form_layout.addRow("Label Col Idx:", self.label_col_input)

        # Input for specifying the image data column index (if not using raw pixels)
        self.image_col_input = QSpinBox()
        self.image_col_input.setRange(-1, 1000) # Allow -1 to indicate pixel mode
        self.image_col_input.setValue(-1) # Default to -1 (pixel mode)
        self.image_col_input.setToolTip("Index of image data column (base64/path). Set to -1 if columns contain raw pixel values.")
        form_layout.addRow("Image Col Idx:", self.image_col_input)

        # Dropdown to specify the type of data in the image column (base64 or path)
        self.image_type_combo = QComboBox()
        self.image_type_combo.addItems(["(Not Applicable)", "base64", "path"]) # Add options
        self.image_type_combo.setToolTip("Select 'base64' or 'path' if Image Col Idx is >= 0")
        self.image_type_combo.setEnabled(False) # Disabled by default (until Image Col Idx changes)
        form_layout.addRow("Type:", self.image_type_combo)
        # ------------------------------------ #

        # Scan for datasets initially (the dropdown will be populated later)
        # self.populate_dataset_dropdown() # Moved to end of __init__ after log is ready
        dataset_group.setLayout(form_layout) # Apply the form layout to the group box
        return dataset_group

    def _create_training_group(self):
        """Creates the GroupBox for training configuration widgets."""
        # QGroupBox visually groups related widgets.
        self.training_group = QGroupBox("Training Controls (No dataset loaded)") # Use self. here
        # Use a QVBoxLayout for arranging controls vertically within the group box.
        layout = QVBoxLayout()

        # --- Model Selection --- #
        self.model_type_combo = QComboBox()
        self.model_type_combo.setToolTip("Select the type of neural network model to train.")
        available_models = []
        if SimpleNeuralNetwork:
            available_models.append("Simple NN")
        if CNNModel:
            available_models.append("CNN")
        if not available_models:
            available_models.append("No Models Found")
            self.model_type_combo.setEnabled(False)
        self.model_type_combo.addItems(available_models)
        # Connect signal if we need to dynamically change hyperparameters UI later
        # self.model_type_combo.currentTextChanged.connect(self._on_model_type_changed)
        layout.addRow("Model Type:", self.model_type_combo)
        # ----------------------- #

        # --- Add QLineEdit for Hidden Layer Configuration ---
        config_layout = QHBoxLayout()
        config_label = QLabel("Hidden Layers (neurons, comma-separated):")
        config_layout.addWidget(config_label)
        self.hidden_layers_input = QLineEdit("128, 64") # Default to one hidden layer of 10 neurons
        self.hidden_layers_input.setToolTip("Enter neuron counts for hidden layers, e.g., '100, 50' for two hidden layers.")
        config_layout.addWidget(self.hidden_layers_input)
        layout.addLayout(config_layout)
        # -----------------------------------------------------

        # --- Hyperparameters --- #
        # Layout for Hyperparameters (Epochs, Learning Rate)
        hyper_layout = QHBoxLayout()
        hyper_label = QLabel("Epochs:")
        hyper_layout.addWidget(hyper_label)
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 10000) # Min 1 epoch, max 10000
        self.epochs_input.setValue(100) # Default value
        self.epochs_input.setToolTip("Number of training iterations through the entire dataset")
        hyper_layout.addWidget(self.epochs_input)

        lr_label = QLabel("Learn Rate (α):")
        hyper_layout.addWidget(lr_label)
        self.learning_rate_input = QDoubleSpinBox()
        self.learning_rate_input.setRange(0.0001, 1.0) # Min/Max learning rate
        self.learning_rate_input.setDecimals(4) # Show 4 decimal places
        self.learning_rate_input.setSingleStep(0.001) # Step size when using arrows
        self.learning_rate_input.setValue(0.01) # Default value
        self.learning_rate_input.setToolTip("Controls how much the model weights are adjusted during training (alpha)")
        hyper_layout.addWidget(self.learning_rate_input)

        patience_label = QLabel("Patience:")
        hyper_layout.addWidget(patience_label)
        self.patience_input = QSpinBox()
        self.patience_input.setRange(0, 1000) # 0 means no early stopping
        self.patience_input.setValue(10) # Default patience
        self.patience_input.setToolTip("Epochs to wait for improvement before stopping early (0=disabled)")
        hyper_layout.addWidget(self.patience_input)

        # --- Add Activation Function Selection --- #
        activation_label = QLabel("Activation:")
        hyper_layout.addWidget(activation_label)
        self.activation_combo = QComboBox()
        self.activation_combo.addItems(["ReLU", "Sigmoid", "Tanh"]) # Add options
        self.activation_combo.setToolTip("Activation function for the hidden layers (ReLU is generally recommended)")
        hyper_layout.addWidget(self.activation_combo)
        # ----------------------------------------- #

        # --- Add Optimizer Selection --- #
        optimizer_label = QLabel("Optimizer:")
        hyper_layout.addWidget(optimizer_label)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "GradientDescent"]) # Add options
        self.optimizer_combo.setToolTip("Optimization algorithm (Adam is generally recommended for faster convergence)")
        # Set Adam as default if desired: self.optimizer_combo.setCurrentText("Adam")
        hyper_layout.addWidget(self.optimizer_combo)
        # ------------------------------- #

        # --- Add L2 Regularization Lambda --- #
        l2_lambda_label = QLabel("L2 λ:")
        hyper_layout.addWidget(l2_lambda_label)
        self.l2_lambda_input = QDoubleSpinBox()
        self.l2_lambda_input.setRange(0.0, 10.0) # Allow lambda from 0 (off) up to 10
        self.l2_lambda_input.setDecimals(5)      # More precision for lambda
        self.l2_lambda_input.setSingleStep(0.001)
        self.l2_lambda_input.setValue(0.0)       # Default is off
        self.l2_lambda_input.setToolTip("L2 regularization strength (lambda). 0.0 means disabled. Typical values are small (e.g., 0.01, 0.1, 1.0).")
        hyper_layout.addWidget(self.l2_lambda_input)
        # ---------------------------------- #

        # --- Add Dropout Keep Probability --- #
        dropout_label = QLabel("Dropout Keep Prob:")
        hyper_layout.addWidget(dropout_label)
        self.dropout_keep_prob_input = QDoubleSpinBox()
        self.dropout_keep_prob_input.setRange(0.1, 1.0) # Keep prob from 10% to 100%
        self.dropout_keep_prob_input.setDecimals(2)
        self.dropout_keep_prob_input.setSingleStep(0.1)
        self.dropout_keep_prob_input.setValue(1.0)       # Default is 1.0 (off)
        self.dropout_keep_prob_input.setToolTip("Probability of keeping a neuron active during training (Dropout). 1.0 means dropout is disabled.")
        hyper_layout.addWidget(self.dropout_keep_prob_input)
        # ------------------------------------ #

        hyper_layout.addStretch() # Pushes hyperparameter widgets to the left
        layout.addLayout(hyper_layout) # Add the hyperparameter layout to the main vertical layout

        # --- Training Action Buttons --- #
        # Layout for Training Buttons (Start, Stop)
        action_layout = QHBoxLayout()
        self.start_button = QPushButton("🚀 Start Training")
        self.start_button.clicked.connect(self.start_training)
        self.start_button.setEnabled(False) # Disabled until data is loaded
        self.start_button.setToolTip("Begin the model training process (requires loaded data)")
        action_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("🛑 Stop Training")
        self.stop_button.clicked.connect(self._stop_training)
        self.stop_button.setEnabled(False) # Disabled until training starts
        self.stop_button.setToolTip("Interrupt the currently running training process")
        action_layout.addWidget(self.stop_button)
        layout.addLayout(action_layout)

        # --- Progress Bar --- #
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True) # Show percentage text
        self.progress_bar.setValue(0) # Start at 0%
        self.progress_bar.setToolTip("Shows the progress of the current training run")
        layout.addWidget(self.progress_bar)

        # --- Accuracy Display Label --- #
        accuracy_layout = QHBoxLayout()
        self.accuracy_label = QLabel("Final Validation Accuracy: --")
        self.accuracy_label.setToolTip("Accuracy achieved on the validation set after training completes")
        accuracy_layout.addWidget(self.accuracy_label)
        accuracy_layout.addStretch() # Push label to the left
        layout.addLayout(accuracy_layout) # Add it below the plot widget

        # Add connections that depend on UI elements within this group
        # self.template_combo.currentIndexChanged[str].connect(self._update_hidden_layer_input) # Reconnect if using templates
        # self._populate_model_dropdown() # Populate dropdown now that combo exists

        self.training_group.setLayout(layout) # Apply the layout to the group box
        self.training_group.setEnabled(False) # Disable group until data is loaded
        return self.training_group # Return the created group box

    def _create_model_mgmt_group(self):
        """Creates the GroupBox for saving and loading model weights."""
        mgmt_group = QGroupBox("Model Management")
        # Simple horizontal layout for the save/load buttons
        layout = QHBoxLayout()

        self.save_button = QPushButton("💾 Save Weights")
        self.save_button.setToolTip("Save the current model weights and biases to a .npz file")
        # Connect the button click to the save_weights method (removed underscore)
        self.save_button.clicked.connect(self.save_weights)
        # Initially disabled, enabled after training finishes or weights are loaded
        self.save_button.setEnabled(False)
        layout.addWidget(self.save_button)

        self.load_button = QPushButton("📂 Load Weights")
        self.load_button.setToolTip("Load previously saved model weights and biases from a .npz file")
        # Connect the button click to the load_weights method (removed underscore)
        self.load_button.clicked.connect(self.load_weights)
        layout.addWidget(self.load_button)

        layout.addStretch() # Push buttons to the left
        mgmt_group.setLayout(layout)
        return mgmt_group

    def _create_inference_group(self):
        """Creates the GroupBox containing widgets for testing the model (inference)."""
        # Renamed this group box to be more descriptive
        inference_group = QGroupBox("Model Testing (Inference)")
        # Main vertical layout for this section
        layout = QVBoxLayout()

        # Add explanatory text at the top to guide the user
        info_label = QLabel(
            "Use this section to test the currently trained or loaded model on new data. "
            "You can either select an image file or draw a digit yourself."
        )
        info_label.setWordWrap(True) # Allow text to wrap to multiple lines
        info_label.setStyleSheet("font-style: italic; padding-bottom: 10px;") # Add some style
        layout.addWidget(info_label)

        # --- Test with File Section ---
        # Horizontal layout for the file selection label and button
        file_test_layout = QHBoxLayout()
        file_label = QLabel("Test with Image File:")
        file_test_layout.addWidget(file_label)
        # Button to trigger the file dialog and prediction process
        self.predict_file_button = QPushButton("Select & Predict File")
        self.predict_file_button.setToolTip("Load an image file (e.g., PNG, JPG, JPEG) for prediction")
        # Connect click to the method handling file prediction
        self.predict_file_button.clicked.connect(self._predict_image_file)
        file_test_layout.addWidget(self.predict_file_button)
        file_test_layout.addStretch() # Push elements left
        layout.addLayout(file_test_layout)

        # --- Test with Drawing Section ---
        # Horizontal layout to place the canvas next to its controls
        drawing_test_layout = QHBoxLayout()
        # Instantiate our custom DrawingCanvas widget
        self.drawing_canvas = DrawingCanvas(width=140, height=140, parent=self)
        drawing_test_layout.addWidget(self.drawing_canvas)

        # Vertical layout for the Predict and Clear buttons next to the canvas
        drawing_buttons_layout = QVBoxLayout()
        self.predict_drawing_button = QPushButton("Predict Drawing")
        self.predict_drawing_button.setToolTip("Predict the digit currently drawn on the canvas")
        # Connect click to the method handling drawing prediction
        self.predict_drawing_button.clicked.connect(self._predict_drawing)
        drawing_buttons_layout.addWidget(self.predict_drawing_button)

        clear_button = QPushButton("Clear Canvas")
        # Connect click to the drawing canvas's own clearCanvas method
        clear_button.clicked.connect(self.drawing_canvas.clearCanvas)
        drawing_buttons_layout.addWidget(clear_button)
        drawing_buttons_layout.addStretch() # Push buttons up
        drawing_test_layout.addLayout(drawing_buttons_layout)
        layout.addLayout(drawing_test_layout)

        # --- Results Display Section ---
        # Horizontal layout for the image preview and the probability graph
        results_layout = QHBoxLayout()
        # QLabel used to display the input image (either loaded file or drawing)
        self.image_preview_label = QLabel("Image Preview Here")
        self.image_preview_label.setAlignment(Qt.AlignCenter) # Center the placeholder text/image
        self.image_preview_label.setMinimumSize(100, 100) # Ensure it has some size
        self.image_preview_label.setStyleSheet("border: 1px solid gray; background-color: white;") # Add border and background
        results_layout.addWidget(self.image_preview_label)

        # Instantiate our custom ProbabilityBarGraph widget to show prediction confidence
        self.probability_graph = ProbabilityBarGraph()
        results_layout.addWidget(self.probability_graph)
        layout.addLayout(results_layout)

        # Apply the main vertical layout to the group box
        inference_group.setLayout(layout)
        return inference_group

    def _create_log_area(self):
        """Creates the QTextEdit widget for logging."""
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        return self.log

    # --- End UI Group Creation Methods --- #

    # --- UI Update / Helper Methods --- #

    def _update_image_col_type_state(self, value=None):
        """Enables/disables the image type combo box based on image col index."""
        # Called when the 'Image Col Idx' spinbox value changes.
        # If the index is >= 0 (meaning a specific column is chosen), enable the Type dropdown.
        # Otherwise (if index is -1, meaning pixel mode), disable it.
        img_col_idx = self.image_col_input.value()
        is_image_col_mode = (img_col_idx >= 0)
        self.image_type_combo.setEnabled(is_image_col_mode)
        if not is_image_col_mode:
            self.image_type_combo.setCurrentIndex(0) # Reset to "(Not Applicable)"

    def scan_datasets(self):
        """Scans the 'data' directory for known dataset types and populates the dropdown."""
        self._log_message("Scanning for datasets...")
        self.datasets_info = {} # Reset discovered datasets

        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_path, "data")
        self._log_message(f"Looking for datasets in: {data_dir}")

        # MNIST Check
        mnist_path = os.path.join(data_dir, "train.csv")
        if os.path.exists(mnist_path):
            self.datasets_info["MNIST (CSV)"] = {"type": "csv", "path": mnist_path}
            self._log_message("Found MNIST dataset (train.csv)")

        # Emoji Check
        emoji_path = os.path.join(data_dir, "emojis.csv")
        if os.path.exists(emoji_path):
            self.datasets_info["Emoji (CSV - Google)"] = {"type": "emoji", "path": emoji_path, "provider": "Google"}
            self.datasets_info["Emoji (CSV - Apple)"] = {"type": "emoji", "path": emoji_path, "provider": "Apple"}
            self._log_message("Found Emoji dataset (emojis.csv)")

        # QuickDraw Check
        quickdraw_dir = os.path.join(data_dir, "quickdraw")
        if os.path.isdir(quickdraw_dir):
            npy_files = glob.glob(os.path.join(quickdraw_dir, "*.npy"))
            if npy_files:
                npy_map = {path: i for i, path in enumerate(sorted(npy_files))}
                self.datasets_info["QuickDraw (Multiple NPY)"] = {"type": "quickdraw", "npy_map": npy_map}
                self._log_message(f"Found {len(npy_files)} QuickDraw NPY files.")
            else:
                self._log_message(f"No .npy files found in {quickdraw_dir}")
        else:
            self._log_message(f"QuickDraw directory not found: {quickdraw_dir}")

        # --- CIFAR-10 Check --- #
        cifar10_dir = os.path.join(data_dir, 'cifar-10-batches-py')
        # Check for existence of a key file, e.g., batches.meta
        if os.path.exists(os.path.join(cifar10_dir, 'batches.meta')):
            self.datasets_info["CIFAR-10"] = {"type": "cifar10", "path": data_dir} # Store base data dir
            self._log_message("Found CIFAR-10 dataset directory.")
        else:
            self._log_message(f"CIFAR-10 directory ('{cifar10_dir}') or key file not found.")
        # ---------------------- #

    # --- New method to populate dataset dropdown ---
    def _on_dataset_selected(self):
        """Slot called when the user selects a dataset in the dropdown."""
        # Generally enables the load button if a valid dataset is selected
        selected_text = self.dataset_dropdown.currentText()
        # Check if the selected text is one of the keys in our discovered datasets info
        is_valid_selection = selected_text in self.datasets_info
        # Enable the button only if the selection is valid
        self.load_dataset_button.setEnabled(is_valid_selection)

    def populate_dataset_dropdown(self):
        self._log_message("Populating dataset dropdown...")
        self.dataset_dropdown.clear() # Clear existing items

        if not self.datasets_info:
             self._log_message("ERROR: No datasets found by scan! Cannot train.")
             self.dataset_dropdown.addItem("No Datasets Found")
             self.dataset_dropdown.setEnabled(False)
             self.load_dataset_button.setEnabled(False)
        else:
            # Populate dropdown from the keys found by scan_datasets
            dataset_names = sorted(self.datasets_info.keys())
            self._log_message(f"Found datasets: {dataset_names}")
            for name in dataset_names:
                self.dataset_dropdown.addItem(name)
            self.dataset_dropdown.setEnabled(True)
            # Let _on_dataset_selected handle enabling the load button initially
            self._on_dataset_selected()

    # --- End new method ---

    # Helper method for logging with timestamp
    def _log_message(self, message):
        timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        self.log.append(f"[{timestamp}] {message}")
        QApplication.processEvents() # Keep UI responsive

    # Modify the progress update slot to only accept percentage
    def _update_progress(self, percentage_complete):
        # Update progress bar based on percentage from worker
        self.progress_bar.setValue(percentage_complete)
        # QApplication.processEvents() # Generally avoid in slots connected to threads

    # New slots to handle signals from the worker
    def _handle_training_finished(self, results: Optional[Tuple[Any, List[float], List[float]]]):
        self.progress_bar.setValue(100)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.training_group.setEnabled(True)

        # Stop the loading animation in the expanded plot if it exists
        if self.expanded_plot_dialog and self.expanded_plot_dialog.isVisible():
            dialog_plot_widget = self.expanded_plot_dialog.findChild(PlotWidget)
            if dialog_plot_widget and hasattr(dialog_plot_widget, 'stop_loading_animation'):
                dialog_plot_widget.stop_loading_animation() # Assuming PlotWidget has this

        if results is None:
            self._log_message("Training did not complete successfully or was stopped.")
            self.accuracy_label.setText("Final Validation Accuracy: --")
        else:
            try:
                # --- Get Updated Parameters/State from Model --- #
                # Assuming results[0] is the parameters dict for now
                # This needs refinement based on the model interface
                updated_params, loss_hist, val_acc_hist = results
                if isinstance(updated_params, dict):
                    # Update the stored raw parameters (for saving)
                    self.model_params = updated_params
                    # Update the model object itself if it has a load_params method
                    if self.current_model and hasattr(self.current_model, 'load_params'):
                        self.current_model.load_params(updated_params)
                        self._log_message("Updated current model instance with trained parameters.")
                    else:
                        self._log_message("Stored trained parameters, but couldn't update model instance.")
                else:
                     self._log_message("WARN: First element returned by worker was not a dict, cannot update parameters.")
                     # Decide how to handle this - maybe results[0] *is* the updated model?
                # ----------------------------------------------- #

                self._log_message(f"Training finished. Final Loss: {loss_hist[-1]:.4f} (Lowest: {min(loss_hist):.4f})" if loss_hist else "Training finished. No loss history.")

                # Update history and plot
                self.train_loss_history = loss_hist
                self.val_accuracy_history = val_acc_hist
                if self.expanded_plot_dialog and self.expanded_plot_dialog.isVisible():
                    dialog_plot_widget = self.expanded_plot_dialog.findChild(PlotWidget)
                    if dialog_plot_widget:
                        try:
                            dialog_plot_widget.update_plot(self.train_loss_history, self.val_accuracy_history)
                            self._log_message("Updated expanded plot with final history.")
                        except Exception as e:
                            self._log_message(f"Error updating expanded plot: {e}")

                # Update accuracy label
                if val_acc_hist:
                    final_accuracy = val_acc_hist[-1] * 100
                    self.accuracy_label.setText(f"Final Validation Accuracy: {final_accuracy:.2f}%")
                    self._log_message(f"Final Validation Accuracy: {final_accuracy:.2f}%")
                else:
                    self.accuracy_label.setText("Final Validation Accuracy: N/A")
                    self._log_message("No validation accuracy history received.")

                # Enable saving and prediction if model exists
                if self.current_model or self.model_params:
                    self.save_button.setEnabled(True)
                    self.predict_drawing_button.setEnabled(True)
                    self.predict_file_button.setEnabled(True)

            except Exception as e:
                 self._log_message(f"ERROR processing training results: {e}")
                 import traceback
                 traceback.print_exc()
                 self.accuracy_label.setText("Error processing results.")

        self._cleanup_thread() # Cleanup worker/thread regardless

    def _handle_worker_log(self, message):
        # Simple pass-through logging for now
        self._log_message(f"[Worker]: {message}")

    def _handle_worker_error(self, error_message):
        self._log_message(f"ERROR from worker thread: {error_message}")
        # Clean up thread and re-enable UI even on error
        self._cleanup_thread()

    def _cleanup_thread(self):
        """Cleans up thread/worker objects and resets UI state. Called via thread.finished signal."""
        self._log_message("Cleaning up training thread resources...")
        self.training_thread = None
        self.training_worker = None
        self.start_button.setEnabled(True) 
        self.stop_button.setText("Stop Training") # Reset text
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0) # Reset progress bar value
        self._log_message("=== Training Thread Finished/Aborted ===")
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
                    # Ensure image is RGB before creating QImage for QPixmap
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Create QImage from PIL bytes
                    qimage = QImage(img.tobytes("raw", "RGB"), img.width, img.height, QImage.Format_RGB888)
                    qpixmap_orig = QPixmap.fromImage(qimage)

                scaled_pixmap = qpixmap_orig.scaled(self.image_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_preview_label.setPixmap(scaled_pixmap)

                self._log_message("Running forward propagation on drawing...")
                if self.current_model is None or not hasattr(self.current_model, 'predict'):
                    self._log_message("ERROR: Model not loaded/trained or has no predict method.")
                    self.image_preview_label.setText("Model not ready.")
                    self.probability_graph.clear_graph()
                    return

                probabilities = self.current_model.predict(img_array)
                if probabilities is None:
                    self._log_message("ERROR: Model prediction failed.")
                    self.probability_graph.clear_graph()
                    return

                prediction = np.argmax(probabilities)
                probabilities = probabilities.flatten()

                # --- Get Class Name --- #
                predicted_name = str(prediction) # Default to index string
                display_names = None # Default to no specific names for graph
                if self.class_names and prediction < len(self.class_names):
                    predicted_name = self.class_names[prediction]
                    display_names = self.class_names # Use names for graph labels
                    self._log_message(f"Prediction Result: Index={prediction}, Name='{predicted_name}'")
                else:
                    self._log_message(f"Prediction Result: Index={prediction} (No class name mapping)")
                # -------------------- #

                # Update probability bar graph with probabilities and class names (if available)
                self.probability_graph.set_probabilities(probabilities, predicted_name, display_names)

            except Exception as e:
                self._log_message(f"ERROR during prediction: {e}")
                self.image_preview_label.setText("Error loading/predicting.")
                self.probability_graph.clear_graph()
        else:
             self._log_message("Prediction cancelled: No file selected.")

    def start_training(self):
        """Initiates the model training process based on UI settings."""
        if self.X_train is None or self.Y_train is None or self.X_dev is None or self.Y_dev is None:
            self.log("Error: Training data not loaded or incomplete.")
            return
        if self.num_classes <= 0:
             self.log("Error: Number of classes not determined from loaded data.")
             return

        if self.training_worker and self.training_thread and self.training_thread.isRunning():
            self.log("Warning: Training is already in progress.")
            return

        # --- Get Selected Model Type --- #
        selected_model_type = self.model_type_combo.currentText()
        self.current_model_type = selected_model_type # Store the selected type
        self.log(f"Selected model type: {selected_model_type}")

        # --- Prepare Data Shape Based on Model --- #
        # This is CRITICAL and needs proper implementation based on dataset + model
        X_train_model, X_dev_model = None, None
        input_shape_for_model = None

        # Example logic (NEEDS REFINEMENT):
        # Assuming data loaded is (features, samples) initially from our current loaders
        num_samples_train = self.X_train.shape[1]
        num_features = self.X_train.shape[0]
        num_samples_dev = self.X_dev.shape[1]

        if selected_model_type == "Simple NN":
            if num_features != 784 and num_features != 3072: # Add checks as needed
                 self.log(f"Warning: Simple NN might not work well with {num_features} features.")
            # Simple NN expects (features, samples) - data might already be in this shape from loader
            X_train_model = self.X_train
            X_dev_model = self.X_dev
            input_shape_for_model = (num_features,) # Tuple defining the feature dimension
            # Layer dims for SimpleNN specific structure
            try:
                hidden_layers = [int(x.strip()) for x in self.hidden_layers_input.text().split(',') if x.strip()]
                self.model_layer_dims = [num_features] + hidden_layers + [self.num_classes]
            except ValueError:
                self.log("Error: Invalid format for Hidden Layers. Use comma-separated integers.")
                return
        elif selected_model_type == "CNN":
            # CNN expects (samples, height, width, channels)
            # Determine target shape based on features (very basic heuristic)
            target_shape_train = None
            target_shape_dev = None

            if num_features == 784: # MNIST-like
                input_shape_for_model = (28, 28, 1)
                target_shape_train = (num_samples_train,) + input_shape_for_model
                target_shape_dev = (num_samples_dev,) + input_shape_for_model
            elif num_features == 3072: # CIFAR-10 like
                input_shape_for_model = (32, 32, 3)
                target_shape_train = (num_samples_train,) + input_shape_for_model
                target_shape_dev = (num_samples_dev,) + input_shape_for_model
            else:
                 self.log(f"Error: Cannot determine CNN input shape for {num_features} features.")
                 return

            try:
                # Reshape data: From (features, samples) to (samples, H, W, C)
                # Need to transpose first to (samples, features) then reshape
                self.log(f"Reshaping training data for CNN from {self.X_train.shape} to {target_shape_train}")
                X_train_model = self.X_train.T.reshape(target_shape_train)

                self.log(f"Reshaping validation data for CNN from {self.X_dev.shape} to {target_shape_dev}")
                X_dev_model = self.X_dev.T.reshape(target_shape_dev)

                self.log(f"Reshaped X_train: {X_train_model.shape}, X_dev: {X_dev_model.shape}")
            except Exception as e:
                self.log(f"Error reshaping data for CNN: {e}")
                return
        else:
            self.log(f"Error: Model type '{selected_model_type}' not recognized for training.")
            return

        if X_train_model is None or X_dev_model is None or input_shape_for_model is None:
            self.log("Error: Failed to prepare data for the selected model.")
            return

        # --- Instantiate Correct Model --- #
        self.current_model = None # Clear previous model instance
        if selected_model_type == "Simple NN":
            if not SimpleNeuralNetwork:
                self.log("Error: SimpleNeuralNetwork class not available.")
                return
            # Instantiate SimpleNeuralNetwork
            try:
                # Ensure self.model_layer_dims was set correctly above
                if not self.model_layer_dims:
                    raise ValueError("Model layer dimensions not calculated.")
                self.current_model = SimpleNeuralNetwork(self.model_layer_dims) # Uses layer_dims
                self.log(f"Instantiated SimpleNeuralNetwork with layers: {self.model_layer_dims}")
            except Exception as e:
                 self.log(f"Error instantiating SimpleNeuralNetwork: {e}")
                 return

        elif selected_model_type == "CNN":
            if not CNNModel:
                 self.log("Error: CNNModel class not available.")
                 return
            # Instantiate CNNModel
            try:
                self.current_model = CNNModel(input_shape=input_shape_for_model, num_classes=self.num_classes)
                # CNN build_model is called separately, often before training or loading weights
                self.current_model.build_model() # Build the architecture now
                self.log(f"Instantiated and built CNNModel with input shape: {input_shape_for_model}")
            except Exception as e:
                 self.log(f"Error instantiating or building CNNModel: {e}")
                 return
        # -------------------------------- #


        # --- Get Hyperparameters (Adjust based on model type if needed) --- #
        # Currently using the same UI fields for both, may need adjustment
        activation = self.activation_combo.currentText() # SimpleNN specific
        optimizer_name = self.optimizer_combo.currentText() # SimpleNN specific
        learning_rate = self.learning_rate_input.value()
        epochs = self.epochs_input.value()
        batch_size = self.batch_size_input.value()
        patience = self.patience_input.value()
        l2_lambda = self.l2_lambda_input.value() # SimpleNN specific
        dropout_keep_prob = 1.0 - self.dropout_keep_prob_input.value() # Convert rate to keep_prob for SimpleNN

        # --- Setup and Start Training Thread --- #
        if self.current_model is None:
            self.log("Error: Model could not be instantiated.")
            self._set_training_ui_enabled(True) # Re-enable UI on early exit
            return

        # Prepare params dict for the worker (adapt structure as needed)
        # NOTE: The worker needs to be adapted to handle both model types!
        training_params = {
            'model': self.current_model, # Pass the instantiated model object
            'model_type': selected_model_type,
            'X_train': X_train_model, # Pass the correctly shaped data
            'Y_train': self.Y_train,
            'X_dev': X_dev_model, # Pass the correctly shaped data
            'Y_dev': self.Y_dev,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            # Pass other relevant params (may differ based on model type)
            # Pass ALL potentially relevant params, worker will decide what to use
            'activation': activation,
            'optimizer': optimizer_name,
            'l2_lambda': l2_lambda,
            'dropout_keep_prob': dropout_keep_prob,
            'patience': patience,
            'num_classes': self.num_classes, # May be needed by worker
        }

        self.log(f"Starting training thread with params: { {k: v.shape if isinstance(v, np.ndarray) else type(v) if k=='model' else v for k, v in training_params.items()} }") # Log shapes/types

        self.training_thread = QThread()
        # Ensure TrainingWorker is imported and available
        if 'TrainingWorker' not in globals():
            self.log("Error: TrainingWorker class not found.")
            self._set_training_ui_enabled(True)
            return
        try:
            # Check if TrainingWorker can be imported dynamically if needed
            # from ui.training_worker import TrainingWorker # Might be necessary
            self.training_worker = TrainingWorker(training_params)
        except Exception as e:
            self.log(f"Error initializing TrainingWorker: {e}")
            self._set_training_ui_enabled(True)
            return

        self.training_worker.moveToThread(self.training_thread)

        # --- Connect signals from worker to main thread slots (Ensure signatures match!) ---
        self.training_worker.progress.connect(self.update_progress) # Expects (epoch, total_epochs, loss, val_acc)
        self.training_worker.finished.connect(self.training_finished) # Expects (history_dict or None)
        self.training_worker.error.connect(self.training_error) # Expects (error_message_str)
        self.training_worker.log_message.connect(self.log) # Expects (message_str)
        # ---------------------------------------------------------------------------------- #

        self.training_thread.started.connect(self.training_worker.run) # Start worker's run method
        # Clean up thread object when it finishes
        self.training_thread.finished.connect(self.training_thread.deleteLater)
        # Also disconnect worker signals when thread finishes to avoid issues if worker deleted later
        self.training_thread.finished.connect(self.training_worker.progress.disconnect)
        self.training_thread.finished.connect(self.training_worker.finished.disconnect)
        self.training_thread.finished.connect(self.training_worker.error.disconnect)
        self.training_thread.finished.connect(self.training_worker.log_message.disconnect)
        # Schedule worker object for deletion after thread finishes
        self.training_thread.finished.connect(self.training_worker.deleteLater)


        # Disable UI during training
        self._set_training_ui_enabled(False)

        self.train_loss_history = [] # Reset histories
        self.val_accuracy_history = []
        self.plot_widget.reset_plot() # Clear plot
        self.training_progress_bar.setRange(0, epochs) # Set range based on total epochs
        self.training_progress_bar.setValue(0)

        self.training_thread.start() # Start the event loop for the thread
        self.log("Training thread start requested.")

    def _stop_training(self):
        """Signals the running training thread to stop gracefully."""
        if self.training_thread is not None and self.training_thread.isRunning():
            if self.training_worker is not None:
                self._log_message("--- Stop training requested by user --- ")
                # Signal the worker to stop
                self.training_worker.stop() 
                # Update UI immediately to show stopping state
                self.stop_button.setText("Stopping...")
                self.stop_button.setEnabled(False)
                self.start_button.setEnabled(False)
                # DO NOT call cleanup/wait here - let the finished signal handle it
            else:
                self._log_message("Stop requested, but worker object not found.")
                # Fallback cleanup if worker is somehow None but thread is running
                self._cleanup_thread()
        else:
            self._log_message("Stop requested, but no training thread is currently running.")

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
        selected_dataset_key = self.dataset_dropdown.currentText()
        if selected_dataset_key == "No Datasets Found":
            self._log_message("No dataset selected.")
            return

        if selected_dataset_key not in self.datasets_info:
             self._log_message(f"ERROR: Selected dataset '{selected_dataset_key}' not found in scanned info. Rescan might be needed.")
             # Optionally disable buttons again?
             # self.load_dataset_button.setEnabled(False)
             return

        self._log_message(f"--- Loading Dataset: {selected_dataset_key} --- ")
        dataset_info = self.datasets_info[selected_dataset_key]
        dataset_type = dataset_info.get("type")
        base_data_path = dataset_info.get("path") # Path to file or base dir

        # Reset data before loading
        self.X_train, self.Y_train, self.X_dev, self.Y_dev = None, None, None, None
        self.current_num_classes = 0
        self.class_names = None
        loaded_data: Optional[Tuple] = None # Use generic tuple type

        # Dispatch based on type
        if dataset_type == "csv":
             label_col_idx = 0 # Default for MNIST
             self._log_message(f"Loading CSV: {selected_dataset_key} from {base_data_path}")
             loaded_data = datasets.load_csv_dataset(base_data_path, label_col_index=label_col_idx)
        elif dataset_type == "emoji":
             provider = dataset_info.get("provider", "Google")
             self._log_message(f"Loading Emojis ({provider}) from: {base_data_path}")
             # Emoji loader returns 6 items, including class names
             loaded_data_emoji = datasets.load_emoji_dataset(base_data_path, image_column=provider)
             if loaded_data_emoji[0] is not None: # Check if load was successful
                 self.X_train, self.Y_train, self.X_dev, self.Y_dev, self.current_num_classes, self.class_names = loaded_data_emoji
             else:
                 loaded_data = None # Signal failure
        elif dataset_type == "quickdraw":
             npy_map = dataset_info.get("npy_map")
             if npy_map:
                 self._log_message(f"Loading QuickDraw (Multiple NPY) - {len(npy_map)} categories")
                 loaded_data = datasets.load_multiple_npy_datasets(npy_map)
                 # Extract class names from the map keys
                 if loaded_data[0] is not None:
                     sorted_items = sorted(npy_map.items(), key=lambda item: item[1])
                     self.class_names = [os.path.splitext(os.path.basename(path))[0].replace("_", " ") for path, index in sorted_items]
             else:
                 self._log_message("ERROR: QuickDraw type selected, but no npy_map found.")
                 loaded_data = None
        elif dataset_type == "cifar10":
             self._log_message(f"Loading CIFAR-10 from base directory: {base_data_path}")
             loaded_data = datasets.load_cifar10_dataset(base_data_path)
             # Try to load class names (specific to CIFAR-10 loader)
             if loaded_data[0] is not None:
                  # Attempt to get class names (assuming loader might provide them or we load meta separately)
                  cifar10_dir = os.path.join(base_data_path, 'cifar-10-batches-py')
                  meta_file = os.path.join(cifar10_dir, 'batches.meta')
                  try:
                      with open(meta_file, 'rb') as fo:
                           meta_dict = pickle.load(fo, encoding='bytes')
                      if meta_dict and b'label_names' in meta_dict:
                           self.class_names = [name.decode('utf-8') for name in meta_dict[b'label_names']]
                           self._log_message(f"Loaded CIFAR-10 class names: {self.class_names}")
                  except Exception as e:
                       self._log_message(f"WARN: Could not load/read CIFAR-10 class names from meta file: {e}")
        else:
            self._log_message(f"ERROR: Unknown dataset type '{dataset_type}'.")
            loaded_data = None

        # --- Process results common to most loaders (except Emoji, handled above) ---
        if dataset_type != "emoji" and loaded_data is not None and len(loaded_data) == 5:
             self.X_train, self.Y_train, self.X_dev, self.Y_dev, self.current_num_classes = loaded_data
        elif dataset_type != "emoji" and loaded_data is None:
            # Load failed, ensure vars are None
            self.X_train, self.Y_train, self.X_dev, self.Y_dev = None, None, None, None
            self.current_num_classes = 0
        # ---------------------------------------------------------------------------

        # Update UI based on load success/failure
        if self.X_train is not None:
            self._log_message(f"Dataset '{selected_dataset_key}' loaded successfully.")
            self.current_dataset_name = selected_dataset_key
            self._post_load_update(selected_dataset_key)
        else:
            self._log_message(f"ERROR: Failed to load dataset '{selected_dataset_key}'.")
            self.training_group.setTitle("Training Controls (Load failed)")
            self.start_button.setEnabled(False)
            self.current_dataset_name = None

        self._log_message("--- Dataset Loading Attempt Finished --- ")

    # --- Helper method to update UI after successful load ---
    def _post_load_update(self, dataset_name):
        # Enable the training group now that data is loaded
        self.training_group.setEnabled(True)
        self.training_group.setTitle(f"Training Controls ({dataset_name}) ({self.current_model_type or 'No Model'})")
        self.train_loss_history = []
        self.val_accuracy_history = []

        # --- Check if model needs re-initialization based on NEW DATA --- #
        # This logic mirrors start_training's checks
        reinitialize_model = False
        target_layer_dims = None
        try:
            if self.X_train is None: raise ValueError("Training data not available")
            input_size = self.X_train.shape[0]
            if self.current_num_classes <= 0: raise ValueError("Num classes unknown")
            hidden_layers_str = self.hidden_layers_input.text().strip()
            hidden_dims = [int(s.strip()) for s in hidden_layers_str.split(',') if s.strip()] if hidden_layers_str else []
            target_layer_dims = [input_size] + hidden_dims + [self.current_num_classes]

            model_type_to_use = "SimpleNN" # Hardcoded for now
            ModelClass = SimpleNeuralNetwork # Will be None for now

            if self.current_model is None or self.model_layer_dims != target_layer_dims:
                self._log_message(f"Model needs (re)initialization for new data/architecture {target_layer_dims}.")
                reinitialize_model = True
            else:
                 self._log_message(f"Existing model architecture {self.model_layer_dims} matches data. Reusing instance.")

            if reinitialize_model:
                if ModelClass is None:
                    self._log_message("WARN: Cannot initialize model class.")
                    self.current_model = None
                    self.model_params = None
                    self.model_layer_dims = None
                else:
                    self.current_model = ModelClass(layer_dims=target_layer_dims)
                    self.model_layer_dims = target_layer_dims
                    self.current_model_type = model_type_to_use
                    self.model_params = self.current_model.get_params() if hasattr(self.current_model, 'get_params') else None
                    self._log_message(f"Re-initialized model instance: {self.current_model_type} {self.model_layer_dims}")

        except Exception as e:
            self._log_message(f"ERROR checking/re-initializing model after data load: {e}")
            self.current_model = None
            self.model_params = None
            self.model_layer_dims = None
        # ----------------------------------------------------------------- #

        # Enable training button only if a model instance exists
        self.start_button.setEnabled(self.current_model is not None)
        self.save_button.setEnabled(False) # Disable save until trained/loaded

        if self.dataset_dropdown.findText(dataset_name) == -1:
            self.dataset_dropdown.addItem(dataset_name)
        self.dataset_dropdown.setCurrentText(dataset_name)
        self._update_image_col_type_state()

    # --- Add slot to update image type combo enabled state ---
    def _update_image_col_type_state(self):
        image_col_idx = self.image_col_input.value()
        is_pixel_mode = (image_col_idx == -1)
        self.image_type_combo.setEnabled(not is_pixel_mode)
        if is_pixel_mode:
            self.image_type_combo.setCurrentIndex(0) # Set to "(Not Applicable)"
        self.progress_bar.setVisible(True)
        QApplication.processEvents()

        # --- Enable Stop Button --- #
        self.stop_button.setEnabled(True)

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
        """Saves the current model's weights to a file."""
        if self.current_model is None:
            self.log("Error: No model trained or loaded to save.")
            return

        model_type = self.current_model_type # Get the type of the current model

        # Suggest a filename based on dataset and model type
        dataset_tag = self.current_dataset_name.split(' ')[0].lower().replace('-', '_') if self.current_dataset_name else "custom"
        model_tag = "cnn" if model_type == "CNN" else "simplenn"
        default_filename = f"{dataset_tag}_{model_tag}_weights"

        # Choose file extension based on model type
        if model_type == "CNN":
            # Keras common extensions: .weights.h5 (older), .keras (newer, saves architecture too sometimes)
            file_filter = "Keras Weights (*.weights.h5 *.keras)"
            default_filename += ".weights.h5"
        elif model_type == "Simple NN":
            file_filter = "NumPy Parameters (*.npz)"
            default_filename += ".npz"
        else:
            self.log(f"Error: Unknown model type '{model_type}' for saving.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Model Weights",
            default_filename, # Suggest filename
            file_filter # Filter for relevant file types
        )

        if filepath:
            try:
                if model_type == "CNN":
                    if hasattr(self.current_model, 'save_weights'):
                        self.current_model.save_weights(filepath)
                        self.log(f"CNN weights saved to {filepath}")
                    else:
                         self.log("Error: Current model object does not have 'save_weights' method.")
                elif model_type == "Simple NN":
                    # Ensure model_params is up-to-date from the model instance
                    if hasattr(self.current_model, 'get_params'):
                        self.model_params = self.current_model.get_params()
                    else:
                        self.log("Warning: Cannot get parameters from SimpleNN instance. Saving potentially stale data.")

                    if self.model_params is not None and self.model_layer_dims is not None:
                         params_to_save = self.model_params.copy() # Avoid modifying the main dict
                         # Include layer dimensions for reconstruction
                         params_to_save['_layer_dims'] = self.model_layer_dims
                         np.savez(filepath, **params_to_save)
                         self.log(f"Simple NN parameters (including layer dims) saved to {filepath}")
                    else:
                         self.log("Error: Simple NN parameters or layer dimensions missing, cannot save.")

            except Exception as e:
                self.log(f"Error saving weights to {filepath}: {e}")


    def load_weights(self):
        """Loads model weights from a file."""
        if self.X_train is None: # Need data loaded to know input/output size potentially
             self.log("Error: Load a dataset first to define model structure.")
             return
        if self.num_classes <= 0:
            self.log("Error: Number of classes not determined from loaded data.")
            return

        # Allow loading either Keras or NumPy weights
        file_filter = "Model Weights (*.weights.h5 *.keras *.npz);;All Files (*)"
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Model Weights",
            "", # Start in default directory
            file_filter
        )

        if filepath:
            self.log(f"Attempting to load weights from: {filepath}")
            loaded_successfully = False
            try:
                # Determine model type based on file extension (heuristic)
                file_extension = os.path.splitext(filepath)[1].lower()
                loaded_model_type = None
                input_shape_for_model = None

                # --- Determine Expected Input Shape Based on Currently Loaded Data --- #
                # This logic mirrors part of start_training, needed to build model before loading weights
                num_features = self.X_train.shape[0]

                if file_extension == ".npz":
                    loaded_model_type = "Simple NN"
                    input_shape_for_model = (num_features,) # Used for validation
                elif file_extension in [".h5", ".keras", ".weights.h5"]: # Common Keras extensions
                     loaded_model_type = "CNN"
                     # Infer shape like in start_training for CNN
                     if num_features == 784: input_shape_for_model = (28, 28, 1)
                     elif num_features == 3072: input_shape_for_model = (32, 32, 3)
                     else:
                         self.log(f"Error: Cannot determine target CNN input shape for currently loaded data ({num_features} features) to load weights.")
                         return
                else:
                    self.log(f"Error: Unrecognized file extension '{file_extension}' for loading weights.")
                    return

                self.log(f"Inferred model type '{loaded_model_type}' from file extension.")
                self.current_model_type = loaded_model_type # Update current type *before* instantiation

                # --- Instantiate the Correct Model --- #
                self.current_model = None # Clear previous model
                if loaded_model_type == "Simple NN":
                    if not SimpleNeuralNetwork: raise ImportError("SimpleNeuralNetwork class not available.")
                    # Try to load layer dimensions from the file itself
                    loaded_data = np.load(filepath, allow_pickle=True)
                    if '_layer_dims' in loaded_data:
                        # Need np.array(...).item() if saved as 0-dim array
                        layer_dims_loaded = loaded_data['_layer_dims']
                        if isinstance(layer_dims_loaded, np.ndarray) and layer_dims_loaded.ndim == 0:
                           self.model_layer_dims = layer_dims_loaded.item()
                        else:
                           self.model_layer_dims = list(layer_dims_loaded) # Ensure it's a list

                        self.log(f"Loaded layer dimensions from file: {self.model_layer_dims}")
                        # Sanity check input dimension against loaded data
                        if self.model_layer_dims[0] != num_features:
                             raise ValueError(f"Loaded weights input size ({self.model_layer_dims[0]}) doesn't match currently loaded dataset features ({num_features}).")
                        # Sanity check output dimension against loaded data
                        if self.model_layer_dims[-1] != self.num_classes:
                             raise ValueError(f"Loaded weights output size ({self.model_layer_dims[-1]}) doesn't match currently loaded dataset classes ({self.num_classes}).")
                    else:
                         raise ValueError("Cannot load Simple NN: Layer dimensions ('_layer_dims') not found in '.npz' file.")

                    self.current_model = SimpleNeuralNetwork(self.model_layer_dims)
                    self.current_model.load_params(loaded_data) # Pass the loaded data dict
                    self.model_params = self.current_model.get_params() # Update internal params cache
                    self.log("Simple NN model instantiated and parameters loaded.")
                    loaded_successfully = True

                elif loaded_model_type == "CNN":
                    if not CNNModel: raise ImportError("CNNModel class not available.")
                    if input_shape_for_model is None: raise ValueError("Cannot determine input shape for CNN based on loaded data.")

                    # Instantiate and build the architecture FIRST, using shape derived from loaded data
                    self.current_model = CNNModel(input_shape=input_shape_for_model, num_classes=self.num_classes)
                    self.current_model.build_model() # Build default architecture matching expected shape
                    # Now load the weights into the built architecture
                    self.current_model.load_weights(filepath)
                    # No need to store params separately, Keras model holds them
                    self.log("CNN model instantiated, built, and weights loaded.")
                    loaded_successfully = True

                # --- Post-Load UI Updates ---
                if loaded_successfully and self.current_model:
                    self.log(f"Model weights loaded successfully into a '{loaded_model_type}' model.")
                    # Update the UI dropdown to reflect the loaded model type
                    combo_index = self.model_type_combo.findText(loaded_model_type)
                    if combo_index >= 0:
                        self.model_type_combo.setCurrentIndex(combo_index)
                    else:
                        self.log(f"Warning: Loaded model type '{loaded_model_type}' not found in dropdown.")

                    self.save_button.setEnabled(True) # Can re-save
                    self.predict_drawing_button.setEnabled(True) # Can predict
                    self.predict_file_button.setEnabled(True)
                    self.start_button.setEnabled(True) # Can continue training

                    # Potentially clear plot/history as it's from a different session
                    self.plot_widget.reset_plot()
                    self.train_loss_history = []
                    self.val_accuracy_history = []
                    self.progress_bar.setValue(0)
                else:
                    # Ensure model is None if loading failed completely
                    self.current_model = None
                    self.current_model_type = None # Reset type if load failed

            except Exception as e:
                self.log(f"Error loading weights from {filepath}: {e}")
                self.current_model = None # Ensure no partial model state
                self.current_model_type = None
                self.save_button.setEnabled(False)
                self.predict_drawing_button.setEnabled(False)
                self.predict_file_button.setEnabled(False)

    # --- New Slot for Predicting Drawing ---
    def _predict_drawing(self):
        self._log_message("--- Starting Prediction from Drawing ---")
        self.probability_graph.clear_graph()
        img_array = self.drawing_canvas.getDrawingArray()
        preview_pixmap = self.drawing_canvas.getPreviewPixmap()

        if img_array is None:
            self._log_message("Prediction cancelled: Drawing canvas is empty or error occurred.")
            self.image_preview_label.setText("Draw something!")
            self.probability_graph.clear_graph()
            return

        if preview_pixmap:
            scaled_pixmap = preview_pixmap.scaled(self.image_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_preview_label.setPixmap(scaled_pixmap)
        else:
            self.image_preview_label.setText("Preview Error")

        try:
            self._log_message("Running model prediction on drawing...")
            if self.current_model is None or not hasattr(self.current_model, 'predict'):
                self._log_message("ERROR: Model not loaded/trained or has no predict method.")
                self.image_preview_label.setText("Model not ready.")
                return

            probabilities = self.current_model.predict(img_array)
            if probabilities is None:
                self._log_message("ERROR: Model prediction failed.")
                return

            prediction = np.argmax(probabilities)
            probabilities = probabilities.flatten()

            predicted_name = str(prediction)
            display_names = None
            if self.class_names and prediction < len(self.class_names):
                predicted_name = self.class_names[prediction]
                display_names = self.class_names
                self._log_message(f"Prediction Result: Index={prediction}, Name='{predicted_name}'")
            else:
                self._log_message(f"Prediction Result: Index={prediction} (No class name mapping)")
            self.probability_graph.set_probabilities(probabilities, predicted_name, display_names)

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

    def update_progress(self, epoch: int, total_epochs: int, loss: float, val_acc: float):
        """Updates the progress bar and plot during training.

        Receives epoch (0-based), total epochs, training loss, and validation accuracy.
        Handles potential NaN values if metrics are not available for an epoch.
        """
        # Progress bar shows current epoch completion (1-based for display)
        # Ensure progress bar range is set correctly (might be done in start_training)
        if self.progress_bar.maximum() != total_epochs:
             self.progress_bar.setRange(0, total_epochs)
        self.progress_bar.setValue(epoch + 1)

        # Append history if data is valid (not NaN)
        current_epoch_num = epoch + 1
        if not np.isnan(loss):
            # Ensure history length matches epoch number if possible
            while len(self.train_loss_history) < current_epoch_num -1:
                self.train_loss_history.append(np.nan) # Pad with NaN if epochs were skipped
            if len(self.train_loss_history) == current_epoch_num -1:
                 self.train_loss_history.append(loss)
            elif epoch < len(self.train_loss_history): # Overwrite if needed (e.g., re-running epoch?)
                 self.log(f"Warning: Overwriting train loss history at epoch {current_epoch_num}")
                 self.train_loss_history[epoch] = loss
            else: # Append if list is somehow shorter than expected
                 self.train_loss_history.append(loss)

        if not np.isnan(val_acc):
            val_acc_percent = val_acc * 100.0 # Convert to percentage
            while len(self.val_accuracy_history) < current_epoch_num - 1:
                 self.val_accuracy_history.append(np.nan)
            if len(self.val_accuracy_history) == current_epoch_num - 1:
                 self.val_accuracy_history.append(val_acc_percent)
            elif epoch < len(self.val_accuracy_history):
                 self.log(f"Warning: Overwriting validation accuracy history at epoch {current_epoch_num}")
                 self.val_accuracy_history[epoch] = val_acc_percent
            else:
                 self.val_accuracy_history.append(val_acc_percent)

        # Update plot - find the minimum length of the two lists
        plot_len = min(len(self.train_loss_history), len(self.val_accuracy_history))
        epochs_axis = list(range(1, plot_len + 1))

        # Pass potentially NaN-padded lists to plot widget (let it handle plotting NaNs if it can)
        self.plot_widget.update_plot(
             epochs_axis,
             self.train_loss_history[:plot_len],
             self.val_accuracy_history[:plot_len]
         )

    def training_finished(self, history: Optional[dict]):
        """Handles successful completion of the training thread.

        Receives the history dictionary (like Keras history) or None if stopped/failed.
        """
        # Called when the worker's finished signal is emitted
        # Check if the signal originated from the currently active worker
        sender_worker = self.sender()
        if sender_worker is not self.training_worker:
            self.log(f"Warning: Received finished signal from an old worker ({type(sender_worker)}). Ignoring.")
            return

        if history is not None:
            self.log("Training finished successfully (worker emitted history)." + f" Keys: {list(history.keys())}")

            # Extract final values from history dictionary for logging
            final_loss = history.get('loss', [np.nan])[-1]
            final_val_acc = history.get('val_accuracy', [np.nan])[-1]
            if not np.isnan(final_loss):
                 self.log(f"Final Training Loss: {final_loss:.4f}")
            if not np.isnan(final_val_acc):
                 # Convert to percentage if it's not already (SimpleNN provides % directly maybe?)
                 acc_val = final_val_acc * 100.0 if final_val_acc <= 1.0 else final_val_acc
                 self.log(f"Final Validation Accuracy: {acc_val:.2f}%")

            # Ensure progress bar shows 100%
            self.progress_bar.setValue(self.progress_bar.maximum())

            # Enable save/predict buttons if model is valid
            if self.current_model:
                 self.save_button.setEnabled(True)
                 self.predict_drawing_button.setEnabled(True)
                 self.predict_file_button.setEnabled(True)
            else:
                 self.log("Warning: Training finished but self.current_model is None.")
                 self.save_button.setEnabled(False)
                 self.predict_drawing_button.setEnabled(False)
                 self.predict_file_button.setEnabled(False)

        else:
            # This path taken if worker emits finished(None) e.g., on stop or error handled within worker
            self.log("Training finished (worker emitted None - likely stopped or failed gracefully).")
            self.progress_bar.setValue(0) # Reset progress bar
            self.save_button.setEnabled(False) # Cannot save if stopped/failed
            self.predict_drawing_button.setEnabled(False)
            self.predict_file_button.setEnabled(False)

        # --- Common cleanup for both success/failure --- #
        self._set_training_ui_enabled(True) # Re-enable UI

        # Clean up thread and worker references
        # Thread cleanup is handled by connections to thread.finished
        self.training_thread = None
        self.training_worker = None
        self.log("Training thread/worker references cleared after finished signal.")


    def training_error(self, message: str):
        """Handles errors reported by the training thread via the error signal."""
        # Called when the worker's error signal is emitted
        sender_worker = self.sender()
        if sender_worker is not self.training_worker:
            self.log(f"Warning: Received error signal from an old worker ({type(sender_worker)}). Ignoring.")
            return

        self.log(f"--- Training Error Signal Received ---")
        self.log(f"ERROR: {message}")

        # Reset progress bar
        self.progress_bar.setValue(0)

        # --- Common cleanup for error --- #
        self._set_training_ui_enabled(True) # Re-enable UI

        # Clean up thread and worker references
        # Thread cleanup is handled by connections to thread.finished
        self.training_thread = None
        self.training_worker = None
        self.log("Training thread/worker references cleared after error signal.")

    # ... rest of the file ...
