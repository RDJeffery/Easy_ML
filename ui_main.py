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
from model import neural_net # Import neural_net from model
import datasets # Import datasets from top level
from PIL import Image
import os
from PyQt5 import QtGui # Keep QtGui import for QTextCursor
import glob # Import glob for file searching
from typing import Optional, Dict, Any, List # For worker thread typing

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

# --- Local UI Component Imports ---
from ui.training_worker import TrainingWorker
from ui.probability_bar_graph import ProbabilityBarGraph

# --- Main Application Execution --- #
if __name__ == '__main__':
    app = QApplication(sys.argv)
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
        self.model_params: Optional[tuple] = None # Stores trained weights and biases (W1, b1, W2, b2)
        self.training_worker: Optional[TrainingWorker] = None
        self.training_thread: Optional[QThread] = None
        self.train_loss_history: List[float] = []
        self.val_accuracy_history: List[float] = []

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

        # -- Removed Model Template and Custom Hidden Layer Size --
        # # Layout for Model Template selection (Label + Dropdown)
        # model_layout = QHBoxLayout()
        # model_label = QLabel("Model Template:")
        # model_layout.addWidget(model_label)
        # self.template_combo = QComboBox()
        # self.template_combo.setToolTip("Select a model structure or choose 'Custom'")
        # self.template_combo.currentIndexChanged[str].connect(self._update_hidden_layer_input)
        # model_layout.addWidget(self.template_combo)
        # model_layout.addStretch()
        # layout.addLayout(model_layout)
        # 
        # # Layout for Custom Hidden Layer Size (Label + SpinBox)
        # hidden_layout = QHBoxLayout()
        # hidden_label = QLabel("Hidden Layer Neurons:")
        # hidden_layout.addWidget(hidden_label)
        # self.hidden_layer_input = QSpinBox()
        # self.hidden_layer_input.setRange(1, 10000)
        # self.hidden_layer_input.setValue(10)
        # self.hidden_layer_input.setToolTip("Number of neurons in the hidden layer (if template is 'Custom')")
        # self.hidden_layer_input.setEnabled(False)
        # hidden_layout.addWidget(self.hidden_layer_input)
        # hidden_layout.addStretch()
        # layout.addLayout(hidden_layout)
        # 
        # # self._populate_model_dropdown() # Removed call

        # --- Add QLineEdit for Hidden Layer Configuration ---
        config_layout = QHBoxLayout()
        config_label = QLabel("Hidden Layers (neurons, comma-separated):")
        config_layout.addWidget(config_label)
        self.hidden_layers_input = QLineEdit("10") # Default to one hidden layer of 10 neurons
        self.hidden_layers_input.setToolTip("Enter neuron counts for hidden layers, e.g., '100, 50' for two hidden layers.")
        config_layout.addWidget(self.hidden_layers_input)
        layout.addLayout(config_layout)
        # -----------------------------------------------------

        # Layout for Training Hyperparameters (Epochs, Learning Rate, Patience)
        param_layout = QFormLayout() # Form layout for label-widget pairs
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 10000)
        self.epochs_input.setValue(50) # Default epochs
        self.epochs_input.setToolTip("Number of passes through the entire training dataset")
        param_layout.addRow("Epochs:", self.epochs_input)

        self.lr_input = QDoubleSpinBox() # Use QDoubleSpinBox for floating point learning rate
        self.lr_input.setRange(0.00001, 1.0)
        self.lr_input.setSingleStep(0.001)
        self.lr_input.setDecimals(5) # Show more precision
        self.lr_input.setValue(0.01) # Default learning rate
        self.lr_input.setToolTip("Learning Rate (alpha): Step size for weight updates during training")
        param_layout.addRow("Learning Rate:", self.lr_input)

        self.patience_input = QSpinBox()
        self.patience_input.setRange(0, 100) # 0 means no early stopping
        self.patience_input.setValue(5) # Default patience
        self.patience_input.setToolTip("Early Stopping Patience: Stop training if validation accuracy doesn't improve for this many checks (0 to disable)")
        param_layout.addRow("Patience:", self.patience_input)

        layout.addLayout(param_layout)

        # Training Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Layout for Start/Stop Training Buttons
        button_layout = QHBoxLayout()
        self.train_button = QPushButton("🚀 Start Training")
        self.train_button.clicked.connect(self.start_training)
        self.train_button.setEnabled(False) # Disabled until a dataset is loaded
        button_layout.addWidget(self.train_button)

        self.stop_button = QPushButton("🛑 Stop Training")
        self.stop_button.clicked.connect(self._stop_training)
        self.stop_button.setEnabled(False) # Disabled until training starts
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)

        # Apply the arrangement of widgets (layout) to the group box
        self.training_group.setLayout(layout)
        return self.training_group

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
        self.predict_file_button.setToolTip("Load an image file (e.g., PNG, JPG) for prediction")
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
        predict_drawing_button = QPushButton("Predict Drawing")
        predict_drawing_button.setToolTip("Predict the digit currently drawn on the canvas")
        # Connect click to the method handling drawing prediction
        predict_drawing_button.clicked.connect(self._predict_drawing)
        drawing_buttons_layout.addWidget(predict_drawing_button)

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

        # --- Determine base path for data directory --- #
        if getattr(sys, 'frozen', False):
            # Running in a PyInstaller bundle
            base_path = sys._MEIPASS
            self._log_message(f"Running bundled, using MEIPASS: {base_path}")
        else:
            # Running as a normal script
            base_path = os.path.dirname(os.path.abspath(__file__))
            self._log_message(f"Running as script, using script dir: {base_path}")
            
        data_dir = os.path.join(base_path, "data")
        self._log_message(f"Looking for datasets in: {data_dir}")
        # --------------------------------------------- #

        # --- Check for specific known files/patterns using data_dir --- #
        # MNIST Check
        mnist_path = os.path.join(data_dir, "train.csv")
        if os.path.exists(mnist_path):
            self.datasets_info["MNIST (CSV)"] = {"type": "csv", "path": mnist_path}
            self._log_message("Found MNIST dataset (train.csv)")

        # Emoji Check
        emoji_path = os.path.join(data_dir, "emojis.csv")
        if os.path.exists(emoji_path):
            # Add entries for different image providers within the emoji CSV
            self.datasets_info["Emoji (CSV - Google)"] = {"type": "emoji", "path": emoji_path, "provider": "Google"}
            self.datasets_info["Emoji (CSV - Apple)"] = {"type": "emoji", "path": emoji_path, "provider": "Apple"}
            # Add more providers here if needed (e.g., Twitter)
            self._log_message("Found Emoji dataset (emojis.csv)")

        # QuickDraw Check
        quickdraw_dir = os.path.join(data_dir, "quickdraw")
        if os.path.isdir(quickdraw_dir):
            npy_files = glob.glob(os.path.join(quickdraw_dir, "*.npy"))
            if npy_files:
                # Create a mapping from file path to a unique category index (0, 1, 2...)
                npy_map = {path: i for i, path in enumerate(sorted(npy_files))}
                self.datasets_info["QuickDraw (Multiple NPY)"] = {"type": "quickdraw", "npy_map": npy_map}
                self._log_message(f"Found {len(npy_files)} QuickDraw NPY files.")
            else:
                self._log_message(f"No .npy files found in {quickdraw_dir}")
        else:
            self._log_message(f"QuickDraw directory not found: {quickdraw_dir}")

        # self.populate_dataset_dropdown() # REMOVE call from here

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
    def _handle_training_finished(self, results):
        if results is None:
            self._log_message("Training finished with errors or was stopped.")
            # Optionally reset plot, etc., if needed
            # self.training_plot_widget.clear_plot() # Example
            # Ensure UI state is reset correctly by _cleanup_thread
            self._cleanup_thread() # Make sure cleanup happens
            return

        self._log_message("Training finished successfully. Updating model parameters and plot.")
        # Unpack results - Ensure results is not None before unpacking
        # ... (rest of the code assumes results is not None)
        self.model_params = results[0] # Parameters dictionary is the first element
        self.train_loss_history = results[1] # Loss history is the second element
        self.val_accuracy_history = results[2] # Accuracy history is the third element

        # --- Update the main plot --- #
        if hasattr(self, 'training_plot_widget') and self.training_plot_widget:
            self.training_plot_widget.update_plot(self.train_loss_history, self.val_accuracy_history)
        else:
            self._log_message("Warning: Could not find training_plot_widget to update.")

        # --- Update expanded plot if it exists --- #
        if self.expanded_plot_dialog and self.expanded_plot_dialog.isVisible():
            # Find the plot widget within the dialog (assuming it's the only PlotWidget)
            expanded_plot = self.expanded_plot_dialog.findChild(PlotWidget)
            if expanded_plot:
                expanded_plot.update_plot(self.train_loss_history, self.val_accuracy_history)
            else:
                 self._log_message("Warning: Could not find PlotWidget in expanded dialog to update.")

        # --- Cleanup handled by thread's finished signal connection --- # Keep this comment
        # self._cleanup_thread() # Don't call directly here, let signal handle it
        # Enable save button now that we have trained parameters
        self.save_button.setEnabled(True)

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
        # Thread should already be finished, no need to quit/wait
        # if self.training_thread is not None:
        #     self.training_thread.quit()
        #     self.training_thread.wait() 
        self.training_thread = None
        self.training_worker = None
        # Reset UI elements
        self.train_button.setEnabled(True) 
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
                    rgb_img = img.convert('RGB')
                    qimage = QImage(rgb_img.tobytes("raw", "RGB"), rgb_img.width, rgb_img.height, QImage.Format_RGB888)
                    qpixmap_orig = QPixmap.fromImage(qimage)

                scaled_pixmap = qpixmap_orig.scaled(self.image_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_preview_label.setPixmap(scaled_pixmap)

                self._log_message("Running forward propagation...")
                if not hasattr(self, 'model_params') or not self.model_params:
                     self._log_message("ERROR: Model parameters not loaded. Load weights first.")
                     self.image_preview_label.setText("Load weights first.")
                     self.probability_graph.clear_graph()
                     return # Exit the try block and method if no model

                # Perform forward propagation using the parameters dictionary
                # _, _, _, output, status = neural_net.forward_prop(*self.model_params, img_array) # OLD call
                output_preds = neural_net.make_predictions(img_array, self.model_params)
                if output_preds.size == 0: # make_predictions returns empty on error
                    self._log_message("ERROR: Prediction failed (forward prop error?).")
                    self.probability_graph.clear_graph()
                    return 
                
                # Need output probabilities, not just predictions, for the graph
                # Rerun forward_prop to get the final activation layer (AL)
                AL, _, status = neural_net.forward_prop(img_array, self.model_params)
                if not status:
                    self._log_message("ERROR: Forward propagation failed when getting probabilities.")
                    self.probability_graph.clear_graph()
                    return

                prediction = np.argmax(AL) # Get prediction from probabilities
                self._log_message(f"Prediction Result: {prediction}")

                # Update probability bar graph with the output probabilities
                self.probability_graph.set_probabilities(AL.flatten(), prediction)

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

        self._log_message(f"=== Starting Training Thread for {self.dataset_dropdown.currentText()} ===") # Use dataset_dropdown
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
        if self.model_params is None:
            self._log_message("ERROR: Model parameters are not initialized. Cannot start training.")
            self.train_button.setEnabled(True) # Re-enable train button
            self.stop_button.setEnabled(False)
            self.progress_bar.setVisible(False)
            return
        if self.current_num_classes <= 0:
             self._log_message("ERROR: Number of classes not determined. Cannot start training.")
             self.train_button.setEnabled(True) # Re-enable train button
             self.stop_button.setEnabled(False)
             self.progress_bar.setVisible(False)
             return

        # --- UNPACKING REMOVED - model_params is now a dictionary --- 
        # W1_init, b1_init, W2_init, b2_init = self.model_params 

        self.training_worker = TrainingWorker(
            self.X_train, self.Y_train, self.X_dev, self.Y_dev,
            # W1_init, b1_init, W2_init, b2_init, # Pass initial weights/biases
            self.model_params, # Pass the entire parameters dictionary
            epochs,         # Pass epochs
            learning_rate,  # Pass learning rate as alpha
            # self.current_num_classes, # No longer needed for worker init
            patience        # Pass patience value
        )
        # Move worker to the thread
        self.training_worker.moveToThread(self.training_thread)

        # --- Connect Signals and Slots ---
        # Connect worker signals to main window slots
        self.training_worker.progress.connect(self._update_progress)
        self.training_worker.finished.connect(self._handle_training_finished)
        self.training_worker.log_message.connect(self._handle_worker_log)

        # Connect thread signals
        self.training_thread.started.connect(self.training_worker.run)
        self.training_worker.finished.connect(self.training_thread.quit)
        self.training_worker.finished.connect(self.training_worker.deleteLater) # Schedule worker for deletion
        self.training_thread.finished.connect(self._cleanup_thread) # Connect thread finished to cleanup slot
        # We handle UI cleanup in _cleanup_thread via _handle_training_finished/_handle_worker_error
        # --- End Signal/Slot Connections --- 

        # Start the thread
        self.training_thread.start()
        self._log_message("Training thread started.")

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
                self.train_button.setEnabled(False)
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
        dataset_path = dataset_info.get("path")
        label_col_idx = self.label_col_input.value() # Get from UI for CSV

        # --- Dispatch based on type stored in datasets_info ---
        if dataset_type == "csv":
             self._log_message(f"Loading CSV from: {dataset_path}")
             # Pass default validation split, assumes standard CSV layout (label col 0, pixel data)
             # Custom CSV uploads are handled by upload_csv_dataset method
             self._load_csv_internal(selected_dataset_key, dataset_path, label_col_idx=0)
        elif dataset_type == "emoji":
             provider = dataset_info.get("provider", "Google") # Default to Google if not specified
             self._log_message(f"Loading Emojis ({provider}) from: {dataset_path}")
             self._load_emoji_internal(selected_dataset_key, dataset_path, image_col=provider)
        elif dataset_type == "quickdraw":
             npy_map = dataset_info.get("npy_map")
             if npy_map:
                 self._log_message(f"Loading QuickDraw (Multiple NPY) - {len(npy_map)} categories")
                 # Pass validation_split (using default 0.1 from datasets.py for NPY)
                 self._load_multiple_npy_internal(selected_dataset_key, npy_map)
             else:
                 self._log_message("ERROR: QuickDraw type selected, but no npy_map found in dataset info.")
        else:
            self._log_message(f"ERROR: Unknown dataset type '{dataset_type}' for '{selected_dataset_key}'. Cannot load.")
            # Handle unknown types or specific cases if needed

        self._log_message("--- Dataset Loading Attempt Finished --- ")

    # --- Dataset Loading Handlers (Called by load_selected_dataset or upload) ---

    def _handle_load_mnist(self, label_col_idx):
        # This specific handler is now likely OBSOLETE because loading is based on
        # self.datasets_info type. Keeping it for reference, but it won't be called
        # by the refactored load_selected_dataset.
        """Handles loading the predefined MNIST dataset."""
        csv_path = "data/train.csv"
        self._load_csv_internal("MNIST", csv_path, label_col_idx)

    def _handle_load_quickdraw_all(self):
        # OBSOLETE - Handled by type 'quickdraw' in load_selected_dataset
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
        # OBSOLETE - Single QuickDraw files are not handled by this structure anymore.
        # Loading multiple NPYs is the supported QuickDraw method via scan_datasets.
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
        # OBSOLETE - Handled by type 'emoji' in load_selected_dataset
        """Handles loading the predefined Emojis dataset."""
        emoji_path = "data/emojis.csv"
        # Pass validation_split (using default 0.1 from datasets.py for emoji)
        self._load_emoji_internal("Emojis", emoji_path)

    def _handle_load_unknown(self, selected_dataset):
        # OBSOLETE - Logic is now contained within load_selected_dataset
        """Handles cases where the selected dataset is not a known predefined one."""
        self._log_message(f"WARN: Loading logic for '{selected_dataset}' not handled by predefined loaders.")
        # Attempt to treat as already loaded if it's in the combo (might be an uploaded CSV)
        if self.X_train is not None and self.dataset_dropdown.currentText() == selected_dataset: # Use dataset_dropdown
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
            # --- Get layer dimensions from UI --- #
            try:
                hidden_layers_str = self.hidden_layers_input.text().strip()
                if hidden_layers_str:
                    hidden_dims = [int(s.strip()) for s in hidden_layers_str.split(',') if s.strip()]
                    if not all(dim > 0 for dim in hidden_dims):
                        raise ValueError("Hidden layer dimensions must be positive integers.")
                else:
                    hidden_dims = [] # No hidden layers if input is empty
                
                # Construct full layer dimensions: [input_size] + hidden_dims + [output_size]
                input_size = 784 # Assuming 28x28 input images
                layer_dims = [input_size] + hidden_dims + [self.current_num_classes]
                self._log_message(f"Re-initializing model with layers: {layer_dims}")
                
                # Initialize parameters using the new structure
                self.model_params = neural_net.init_params(layer_dims)
                self._log_message("Model re-initialized.")
                self.save_button.setEnabled(False) # Disable save until trained
                self.train_button.setEnabled(True) # Enable training

            except ValueError as e:
                self._log_message(f"ERROR: Invalid hidden layer configuration '{self.hidden_layers_input.text()}'. Please enter comma-separated positive integers. {e}")
                self.model_params = None
                self.train_button.setEnabled(False)
                self.save_button.setEnabled(False)
            # -------------------------------------- #
        else:
            self._log_message("ERROR: Number of classes is 0 or unknown. Cannot initialize model parameters.")
            self.model_params = None
            self.train_button.setEnabled(False)
            self.save_button.setEnabled(False)
        # -------------------------------------------------------------

        # Update combo box if needed (e.g., for uploaded)
        if self.dataset_dropdown.findText(dataset_name) == -1: # Use dataset_dropdown
            self.dataset_dropdown.addItem(dataset_name)        # Use dataset_dropdown
        self.dataset_dropdown.setCurrentText(dataset_name)       # Use dataset_dropdown
        # --- Enable/disable image type based on image col input ---
        self._update_image_col_type_state()
        # --- Enable training button --- #
        self.train_button.setEnabled(True)

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
        # Ensure we have model parameters to save
        if not self.model_params:
            self._log_message("ERROR: No model parameters available to save.")
            return

        # Open save file dialog
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Model Parameters", "", "NumPy NPZ Files (*.npz)")

        if file_path:
            # Ensure the filename ends with .npz
            if not file_path.endswith('.npz'):
                file_path += '.npz'

            try:
                # Save the entire parameters dictionary
                np.savez(file_path, **self.model_params)
                self._log_message(f"Parameters saved successfully to: {file_path}")
            except Exception as e:
                self._log_message(f"ERROR saving parameters to {file_path}: {e}")

    def load_weights(self):
        # Open load file dialog
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Model Parameters", "", "NumPy NPZ Files (*.npz)")

        if file_path:
            try:
                self._log_message(f"Attempting to load parameters from: {file_path}")
                data = np.load(file_path, allow_pickle=True) # Allow pickle for potential flexibility, though not strictly needed for standard arrays
                # Load the dictionary directly
                self.model_params = dict(data) 
                self._log_message("Parameters loaded successfully.")
                # Reset training history as it doesn't correspond to loaded parameters
                self.train_loss_history = []
                self.val_accuracy_history = []
                # Update UI based on loaded parameters (e.g., infer layer structure?)
                # This is tricky - we don't necessarily know the layer string that created these weights
                # For now, just enable the train button if data is also loaded
                self._log_message("Loaded parameters might not match the current layer configuration input.")
                if self.current_dataset is not None:
                    self.train_button.setEnabled(True)
                self.save_button.setEnabled(True) # Enable save button after loading
            except Exception as e:
                self._log_message(f"ERROR loading parameters from {file_path}: {e}")
        else:
            self._log_message("Load parameters cancelled.")

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
            # Check if model_params dictionary exists and is not empty
            if not hasattr(self, 'model_params') or not self.model_params:
                 self._log_message("ERROR: Model parameters not loaded. Load weights first.")
                 self.image_preview_label.setText("Load weights first.")
                 self.probability_graph.clear_graph()
                 return

            # Perform prediction using the parameters dictionary
            # _, _, _, output, status = neural_net.forward_prop(*self.model_params, img_array) # OLD call
            # Rerun forward_prop to get the final activation layer (AL) for probabilities
            AL, _, status = neural_net.forward_prop(img_array, self.model_params)
            if not status:
                self._log_message("ERROR: Forward propagation failed for drawing prediction.")
                self.probability_graph.clear_graph()
                return

            prediction = np.argmax(AL)
            self._log_message(f"Prediction Result: {prediction}")

            # Update probability bar graph
            self.probability_graph.set_probabilities(AL.flatten(), prediction)

        except Exception as e:
            self._log_message(f"ERROR during prediction: {e}")
            # --- Add detailed exception info ---
            import traceback
            self._log_message(f"Exception Type: {type(e)}")
            self._log_message(f"Traceback:\n{traceback.format_exc()}")
            # ------------------------------------
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
