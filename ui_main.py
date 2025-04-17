import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QComboBox, QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout,
    QApplication, QProgressBar, QSizePolicy, QCheckBox, QTabWidget, QDialog, QFrame,
    QListWidget, QAbstractItemView # Import QListWidget and selection mode
)
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QColor, QPen, QBrush, QFontMetrics, QIcon # Import QIcon
)
from PyQt5.QtCore import Qt, QDateTime, QObject, pyqtSignal, QThread, QRectF, QSize, QTimer # Import QSize for canvas and QTimer
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
import random # Import random for QuickDraw selection
import tarfile # Import tarfile to potentially handle .tar.gz
import traceback # ADDED import

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
    # Updated import for PredictionVisualizer
    from ui.probability_bar_graph import PredictionVisualizer
except ImportError as e:
    print(f"ERROR: Could not import PredictionVisualizer: {e}")
    class PredictionVisualizer(QWidget): pass # Dummy

# --- Import Tab Widgets --- #
try:
    from ui.tabs.data_tab import DataTab
except ImportError as e:
    print(f"ERROR: Could not import DataTab: {e}")
    class DataTab(QWidget): pass # Dummy

try:
    from ui.tabs.train_tab import TrainTab
except ImportError as e:
    print(f"ERROR: Could not import TrainTab: {e}")
    class TrainTab(QWidget): pass # Dummy

try:
    from ui.tabs.test_tab import TestTab
except ImportError as e:
    print(f"ERROR: Could not import TestTab: {e}")
    class TestTab(QWidget): pass # Dummy

# --- Import About Dialog --- #
try:
    from ui.about_dialog import AboutDialog
except ImportError as e:
    print(f"ERROR: Could not import AboutDialog: {e}")
    AboutDialog = None # Fallback
# ------------------------- #

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
        log_widget = self._create_log_area() # Get the QTextEdit widget

        # --- Create Tab Widget --- #
        # QTabWidget provides a tabbed interface to organize different sections of the UI.
        self.tabs = QTabWidget()

        # --- Create Widgets for Tabs --- #
        # Each tab holds a QWidget, which in turn has its own layout and contains the relevant UI groups.

        # Data Tab: For selecting and loading datasets
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab) # Layout for the Data tab
        # Instantiate the DataTab widget, passing the main window reference
        self.data_tab_widget = DataTab(parent_window=self)
        data_layout.addWidget(self.data_tab_widget)
        data_layout.addStretch() # Pushes content up if there's extra vertical space

        # Train Tab: For configuring and running model training
        train_tab = QWidget()
        train_layout = QVBoxLayout(train_tab) # Layout for the Train tab
        # Instantiate the TrainTab widget
        self.train_tab_widget = TrainTab(parent_window=self)
        train_layout.addWidget(self.train_tab_widget)
        train_layout.addStretch() # Push content towards top

        # Test Tab: For running predictions on the trained model
        infer_tab = QWidget() # Renamed to 'Test' visually, but variable name is 'infer_tab'
        infer_layout = QVBoxLayout(infer_tab) # Layout for the Test tab
        # Instantiate the TestTab widget
        self.test_tab_widget = TestTab(parent_window=self)
        infer_layout.addWidget(self.test_tab_widget)
        infer_layout.addStretch() # Push content up

        # --- ADD CONNECTIONS FOR YES/NO BUTTONS --- #
        if hasattr(self.test_tab_widget, 'yes_button') and hasattr(self.test_tab_widget, 'no_button'):
            self.test_tab_widget.yes_button.clicked.connect(
                lambda: self._log_message("User Feedback: Yes (Correct Prediction)")
            )
            self.test_tab_widget.no_button.clicked.connect(
                lambda: self._log_message("User Feedback: No (Incorrect Prediction)")
            )
        else:
            self._log_message("Warning: Could not find Yes/No buttons on TestTab to connect signals.")
        # ----------------------------------------- #

        # Add Tabs to Tab Widget with user-friendly names and icons
        self.tabs.addTab(data_tab, "💾 Data")
        self.tabs.addTab(train_tab, "🚀 Train")
        self.tabs.addTab(infer_tab, "🧪 Test") # This is the 'Test' tab

        # --- Add Tab Widget and Log Area to Main Layout --- #
        self.main_layout.addWidget(self.tabs) # Add the entire tab widget structure
        self.main_layout.addWidget(log_widget) # Add the log area below the tabs
        # Adjust stretch factor: give tabs more vertical space (4 parts) than the log area (1 part)
        self.main_layout.setStretchFactor(self.tabs, 4)
        self.main_layout.setStretchFactor(log_widget, 1)

        # --- Finalize Layout ---
        self.central.setLayout(self.main_layout) # Apply the main layout to the central widget
        self.setCentralWidget(self.central) # Set the central widget for the main window

        # --- Create Menu Bar --- #
        menu_bar = self.menuBar()
        help_menu = menu_bar.addMenu("&Help")

        about_action = help_menu.addAction("&About EasyML")
        about_action.triggered.connect(self._show_about_dialog)
        # ---------------------- #

        # --- Initialize Application State --- #
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
        self.is_training: bool = False # Track if training is active

        # --- Model Related Attributes ---
        self.current_model: Optional[Any] = None # Holds the instantiated model object (SimpleNN or CNN)
        self.current_model_type: Optional[str] = "Simple NN" # Default or track selected type, e.g., 'Simple NN', 'CNN'
        self.model_weights_path: Optional[str] = None # Store path for Keras weights if applicable
        # --------------------------------


    def _create_log_area(self):
        """Creates the QTextEdit widget for logging."""
        # Assign the QTextEdit widget to self.log_output (or a distinct name)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        return self.log_output # Return the widget itself

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
            # Get the directory containing ui_main.py
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # If __file__ is the script being run directly, script_dir IS the base path (project root)
            base_path = script_dir # CORRECTED: Don't go up one level

        data_dir = os.path.join(base_path, "data")
        self._log_message(f"Corrected - Looking for datasets in: {data_dir}") # Updated log

        # MNIST Check
        mnist_path = os.path.join(data_dir, "train.csv")
        self._log_message(f"  Checking for MNIST: {mnist_path} -> Exists? {os.path.exists(mnist_path)}") # ADDED LOG
        if os.path.exists(mnist_path):
            self.datasets_info["MNIST (CSV)"] = {"type": "csv", "path": mnist_path}
            self._log_message("    Found MNIST dataset (train.csv)")

        # Emoji Check
        emoji_path = os.path.join(data_dir, "emojis.csv")
        self._log_message(f"  Checking for Emoji: {emoji_path} -> Exists? {os.path.exists(emoji_path)}") # ADDED LOG
        if os.path.exists(emoji_path):
            self.datasets_info["Emoji (CSV - Google)"] = {"type": "emoji", "path": emoji_path, "provider": "Google"}
            self.datasets_info["Emoji (CSV - Apple)"] = {"type": "emoji", "path": emoji_path, "provider": "Apple"}
            self._log_message("    Found Emoji dataset (emojis.csv)")

        # QuickDraw Check
        quickdraw_dir = os.path.join(data_dir, "quickdraw")
        self._log_message(f"  Checking for QuickDraw Dir: {quickdraw_dir} -> IsDir? {os.path.isdir(quickdraw_dir)}") # ADDED LOG
        if os.path.isdir(quickdraw_dir):
            npy_files = glob.glob(os.path.join(quickdraw_dir, "*.npy"))
            self._log_message(f"    QuickDraw Dir found. Checking for NPY files in {quickdraw_dir} -> Found: {len(npy_files)}") # ADDED LOG
            if npy_files:
                self.datasets_info["QuickDraw (.npy)"] = {"type": "quickdraw", "path": quickdraw_dir}
                self._log_message(f"      Found QuickDraw dataset ({len(npy_files)} categories) in {quickdraw_dir}")
            else:
                self._log_message(f"      Found QuickDraw directory, but no .npy files inside: {quickdraw_dir}")

        # CIFAR-10 Check (Python version)
        cifar10_dir = os.path.join(data_dir, "cifar-10-batches-py")
        self._log_message(f"  Checking for CIFAR-10 Dir: {cifar10_dir} -> IsDir? {os.path.isdir(cifar10_dir)}") # ADDED LOG
        if os.path.isdir(cifar10_dir):
            # Check for a key file like 'data_batch_1'
            cifar10_key_file = os.path.join(cifar10_dir, "data_batch_1") # ADDED VAR
            self._log_message(f"    CIFAR-10 Dir found. Checking for key file: {cifar10_key_file} -> Exists? {os.path.exists(cifar10_key_file)}") # ADDED LOG
            if os.path.exists(cifar10_key_file):
                self.datasets_info["CIFAR-10 (Python)"] = {"type": "cifar10", "path": cifar10_dir}
                self._log_message("      Found CIFAR-10 dataset (Python version)")
            else:
                 self._log_message(f"      Found CIFAR-10 directory, but missing batch files: {cifar10_dir}")

        # CIFAR-100 Check (Python version) - ADDED
        cifar100_dir_or_tar = os.path.join(data_dir, "cifar-100-python")
        cifar100_tar_path = os.path.join(data_dir, "cifar-100-python.tar.gz")
        self._log_message(f"  Checking for CIFAR-100 Dir: {cifar100_dir_or_tar} -> IsDir? {os.path.isdir(cifar100_dir_or_tar)}") # ADDED LOG

        if os.path.isdir(cifar100_dir_or_tar):
             # Check for key files like 'train' and 'test'
             cifar100_train_file = os.path.join(cifar100_dir_or_tar, "train") # ADDED VAR
             cifar100_test_file = os.path.join(cifar100_dir_or_tar, "test") # ADDED VAR
             self._log_message(f"    CIFAR-100 Dir found. Checking train: {cifar100_train_file} -> Exists? {os.path.exists(cifar100_train_file)}") # ADDED LOG
             self._log_message(f"    CIFAR-100 Dir found. Checking test: {cifar100_test_file} -> Exists? {os.path.exists(cifar100_test_file)}") # ADDED LOG
             if os.path.exists(cifar100_train_file) and os.path.exists(cifar100_test_file):
                 self.datasets_info["CIFAR-100 (Python)"] = {"type": "cifar100", "path": cifar100_dir_or_tar}
                 self._log_message("      Found CIFAR-100 dataset (Python version, extracted)")
             else:
                  self._log_message(f"      Found CIFAR-100 directory, but missing train/test files: {cifar100_dir_or_tar}")
        else:
            self._log_message(f"  CIFAR-100 Dir not found. Checking for TAR: {cifar100_tar_path} -> Exists? {os.path.exists(cifar100_tar_path)}") # ADDED LOG
            if os.path.exists(cifar100_tar_path):
                 # Found the tar.gz, assume it's valid for now (loading will handle extraction/check)
                 self.datasets_info["CIFAR-100 (Python)"] = {"type": "cifar100", "path": cifar100_tar_path}
                 self._log_message("    Found CIFAR-100 dataset (Python version, tar.gz)")

        if not self.datasets_info:
            self._log_message("No compatible datasets found in the 'data' directory.")
        else:
            self._log_message(f"Found {len(self.datasets_info)} dataset entries.")

        self.populate_dataset_dropdown() # Update dropdown after scanning

    # --- Method to handle dataset selection changes ---
    def _on_dataset_selected(self):
        """Slot called when the user selects a dataset in the dropdown."""
        selected_text = self.dataset_dropdown.currentText()
        is_valid_selection = selected_text in self.datasets_info
        self.load_dataset_button.setEnabled(is_valid_selection)

        # --- Show/Hide QuickDraw Selection Widgets --- #
        # Use the TYPE from datasets_info to check for QuickDraw
        is_quickdraw = False
        if is_valid_selection:
             dataset_type = self.datasets_info[selected_text].get("type")
             if dataset_type == "quickdraw":
                 is_quickdraw = True

        self.quickdraw_select_label.setVisible(is_quickdraw)
        self.quickdraw_list_widget.setVisible(is_quickdraw)
        self.quickdraw_random_label.setVisible(is_quickdraw)
        self.quickdraw_random_count.setVisible(is_quickdraw)

        if is_quickdraw:
            # Populate the list widget if QuickDraw is selected
            self.quickdraw_list_widget.clear()
            quickdraw_info = self.datasets_info[selected_text]
            quickdraw_dir = quickdraw_info.get("path")
            if quickdraw_dir and os.path.isdir(quickdraw_dir):
                 npy_files = glob.glob(os.path.join(quickdraw_dir, "*.npy"))
                 if npy_files:
                      # Extract category name from filename
                      category_names = sorted([os.path.splitext(os.path.basename(f))[0].replace("_", " ") for f in npy_files])
                      self.quickdraw_list_widget.addItems(category_names)
                      self._log_message(f"Populated QuickDraw list with {len(category_names)} categories.")
                 else:
                      self._log_message(f"Warning: QuickDraw selected, but no .npy files found in directory: {quickdraw_dir}")
            else:
                 self._log_message("Warning: QuickDraw selected, but path info is missing or invalid.")
        # --------------------------------------------- #

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
                # Set both the display text and the associated data to the dataset name (key)
                self.dataset_dropdown.addItem(name, name)
            self.dataset_dropdown.setEnabled(True)
            # Let _on_dataset_selected handle enabling the load button initially
            self._on_dataset_selected()

    # --- End new method ---

    # Helper method for logging with timestamp
    def _log_message(self, message):
        timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        self.log_output.append(f"[{timestamp}] {message}")
        # QApplication.processEvents() # Keep UI responsive - REMOVED to prevent potential re-entrancy issues

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
        # REMOVED direct call: self._cleanup_thread() - Cleanup now handled by _on_thread_actually_finished

    def _on_thread_actually_finished(self):
        """Slot connected to QThread.finished signal. Performs final cleanup AFTER the thread's event loop has ended."""
        self._log_message("QThread.finished signal received. Performing final cleanup...") # Kept this informative log

        # Ensure UI reflects stopped state and is re-enabled
        try:
            # self._log_message("DEBUG: Re-enabling start button...") # REMOVED DEBUG LOG
            self.start_button.setEnabled(True) # Re-enable start button *now*
            # self._log_message("DEBUG: Resetting stop button text...") # REMOVED DEBUG LOG
            self.stop_button.setText("🛑 Stop Training") # Reset text
            # self._log_message("DEBUG: Disabling stop button...") # REMOVED DEBUG LOG
            self.stop_button.setEnabled(False)
            # self._log_message("DEBUG: Resetting progress bar...") # REMOVED DEBUG LOG
            self.progress_bar.setValue(0) # Reset progress bar value
            # Check if training group exists before enabling (might be during shutdown)
            if hasattr(self, 'training_group'):
                # self._log_message("DEBUG: Re-enabling training group...") # REMOVED DEBUG LOG
                self.training_group.setEnabled(True) # Re-enable config
            else:
                pass # No need to log this specific case for the user
                # self._log_message("DEBUG: training_group not found, skipping enable.") # REMOVED DEBUG LOG
            if hasattr(self, 'tabs'):
                # self._log_message("DEBUG: Re-enabling tabs...") # REMOVED DEBUG LOG
                self.tabs.setEnabled(True) # Re-enable tab switching
            else:
                 pass # No need to log this specific case for the user
                 # self._log_message("DEBUG: tabs not found, skipping enable.") # REMOVED DEBUG LOG
        except Exception as e:
            self._log_message(f"CRITICAL ERROR during UI reset in _on_thread_actually_finished: {e}")
            import traceback
            traceback.print_exc()

        # Reset state flags and references
        # self._log_message("DEBUG: Resetting is_training flag...") # REMOVED DEBUG LOG
        self.is_training = False
        # These might already be scheduled for deletion by deleteLater,
        # but nullifying references helps prevent accidental use.
        # self._log_message("DEBUG: Nullifying worker/thread references...") # REMOVED DEBUG LOG
        self.training_worker = None
        self.training_thread = None

        self._log_message("=== Training Thread Fully Finished - UI Reset Complete ===")

    def _predict_drawing(self):
        """Gets the drawing from the canvas, preprocesses, and predicts."""
        # Access widgets via test_tab_widget
        if self.current_model is None:
            self._log_message("No model loaded or trained yet.")
            if hasattr(self, 'test_tab_widget'): # Check if tab exists
                self.test_tab_widget.feedback_label.setText("Model not ready")
                self.test_tab_widget.prediction_visualizer.clear_graph()
                self.test_tab_widget.yes_button.setEnabled(False)
                self.test_tab_widget.no_button.setEnabled(False)
            return

        # Get the drawing as a NumPy array (e.g., 28x28)
        drawing_array = self.test_tab_widget.drawing_canvas.getDrawingArray()
        if drawing_array is None:
            self._log_message("Canvas is empty or drawing is invalid.")
            # --- Clear previous results --- #
            if hasattr(self, 'test_tab_widget'):
                self.test_tab_widget.image_preview_label.clear()
                self.test_tab_widget.image_preview_label.setText("(Canvas Empty)") # Give feedback
                if self.test_tab_widget.prediction_visualizer:
                    self.test_tab_widget.prediction_visualizer.clear_graph()
                self.test_tab_widget.feedback_label.setText("Prediction: N/A")
                self.test_tab_widget.yes_button.setEnabled(False)
                self.test_tab_widget.no_button.setEnabled(False)
            # ------------------------------ #
            return

        # --- Display Drawing Preview --- #
        preview_pixmap = self.test_tab_widget.drawing_canvas.getPreviewPixmap()
        if preview_pixmap and hasattr(self, 'test_tab_widget'):
            # Scale pixmap to fit the preview label while maintaining aspect ratio
            scaled_pixmap = preview_pixmap.scaled(self.test_tab_widget.image_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.test_tab_widget.image_preview_label.setPixmap(scaled_pixmap)
        elif hasattr(self, 'test_tab_widget'):
            self.test_tab_widget.image_preview_label.setText("(Preview Error)") # Show error if pixmap fails
        # ------------------------------ #

        # --- Preprocess and reshape based on model type --- #
        model_type = self.current_model_type # e.g., "Simple NN" or "CNN"
        input_data = None

        try:
            # [Preprocessing logic as before - Simple NN and CNN]
            if model_type == "Simple NN":
                # ... (preprocessing as before)
                self._log_message("Preprocessing drawing for Simple NN...")
                if drawing_array.ndim != 2: raise ValueError(f"Unexpected drawing array dimension: {drawing_array.ndim}")
                num_features = drawing_array.shape[0] * drawing_array.shape[1]
                input_data = drawing_array.flatten().reshape(num_features, 1)
                input_data = input_data / 255.0
                self._log_message(f"Simple NN input shape: {input_data.shape}")

            elif model_type == "CNN":
                 # ... (preprocessing as before)
                self._log_message("Preprocessing drawing for CNN...")
                H, W, C = 28, 28, 1
                if self.current_model and hasattr(self.current_model, 'input_shape') and self.current_model.input_shape:
                     try:
                         if len(self.current_model.input_shape) == 3: H, W, C = self.current_model.input_shape
                         elif len(self.current_model.input_shape) == 4: H, W, C = self.current_model.input_shape[1:]
                         self._log_message(f"Using model input shape for CNN: {(H, W, C)}")
                     except Exception as shape_err: self._log_message(f"Warning: Could not parse model input shape {self.current_model.input_shape}. Error: {shape_err}. Falling back to default.")
                elif hasattr(self, 'X_train') and self.X_train is not None and self.X_train.ndim == 4:
                     if self.X_train.shape[0] > 0: H, W, C = self.X_train.shape[1:]; self._log_message(f"Using loaded data shape for CNN: {(H, W, C)}")
                else: self._log_message(f"Warning: No data or model shape available, using default {H}x{W}x{C} for CNN input.")

                if drawing_array.ndim != 2: raise ValueError(f"Unexpected drawing array dimension: {drawing_array.ndim}")
                if drawing_array.shape != (H, W):
                     self._log_message(f"Warning: Drawing array shape {drawing_array.shape} differs from expected CNN input {(H, W)}. Resizing...")
                     img_pil = Image.fromarray(drawing_array.astype(np.uint8))
                     img_resized_pil = img_pil.resize((W, H), Image.Resampling.LANCZOS)
                     drawing_array = np.array(img_resized_pil)
                     self._log_message(f"Resized drawing array to: {drawing_array.shape}")

                input_data = drawing_array.reshape(1, H, W, C)
                input_data = input_data.astype(np.float32)
                input_data = input_data / 255.0
                self._log_message(f"CNN input shape: {input_data.shape}")
            else:
                self._log_message(f"Prediction not implemented for model type: {model_type}")
                if hasattr(self, 'test_tab_widget'): # Check if tab exists
                    self.test_tab_widget.feedback_label.setText("Prediction N/A")
                    self.test_tab_widget.prediction_visualizer.clear_graph()
                    self.test_tab_widget.yes_button.setEnabled(False)
                    self.test_tab_widget.no_button.setEnabled(False)
                return

            # --- Perform Prediction --- #
            self._log_message(f"Predicting using {model_type}...")
            prediction_output = self.current_model.predict(input_data)
            # print(f"Raw prediction output: {prediction_output}") # Debug

            # --- Process & Display Results --- #
            if prediction_output is None:
                 self._log_message("Prediction failed. Model returned None.")
                 if hasattr(self, 'test_tab_widget'): # Check if tab exists
                     self.test_tab_widget.feedback_label.setText("Prediction Failed")
                     self.test_tab_widget.prediction_visualizer.clear_graph()
                     self.test_tab_widget.yes_button.setEnabled(False)
                     self.test_tab_widget.no_button.setEnabled(False)
                 return

            probabilities = prediction_output.flatten() # Make it 1D

            if probabilities.size == self.num_classes:
                # Pass raw probabilities, visualizer handles finding max etc.
                if hasattr(self, 'test_tab_widget'): # Check if tab exists
                    self.test_tab_widget.prediction_visualizer.set_probabilities(probabilities, class_names=self.class_names)

                    # Update Feedback Label and Buttons based on visualizer's predicted name
                    feedback_name = self.test_tab_widget.prediction_visualizer.get_predicted_class_name()
                    if feedback_name:
                        self.test_tab_widget.feedback_label.setText(f"Is it a {feedback_name}?")
                        self.test_tab_widget.yes_button.setEnabled(True)
                        self.test_tab_widget.no_button.setEnabled(True)
                        # Log confidence of the predicted class
                        pred_index = self.test_tab_widget.prediction_visualizer.predicted_class_index
                        # Use nan_to_num to handle potential NaN in probabilities
                        valid_probs = np.nan_to_num(probabilities)
                        confidence = valid_probs[pred_index] * 100 if 0 <= pred_index < len(valid_probs) else float('nan')
                        self._log_message(f"Drawing Prediction: {feedback_name} (Confidence: {confidence:.2f}%) Index: {pred_index}")
                    else:
                        # Handle case where visualizer couldn't determine a name (e.g., invalid probs)
                        self.test_tab_widget.feedback_label.setText("Prediction Error")
                        self.test_tab_widget.yes_button.setEnabled(False)
                        self.test_tab_widget.no_button.setEnabled(False)
            else:
                self._log_message(f"Prediction output size ({probabilities.size}) does not match number of classes ({self.num_classes}).")
                if hasattr(self, 'test_tab_widget'): # Check if tab exists
                    self.test_tab_widget.feedback_label.setText("Prediction Size Mismatch")
                    self.test_tab_widget.prediction_visualizer.clear_graph()
                    self.test_tab_widget.yes_button.setEnabled(False)
                    self.test_tab_widget.no_button.setEnabled(False)

        # --- Separate Exception Handling --- #
        except ValueError as e:
            self._log_message(f"Error during prediction preprocessing (drawing): {e}")
            if hasattr(self, 'test_tab_widget'): # Check if tab exists
                self.test_tab_widget.feedback_label.setText("Preprocessing Error")
                self.test_tab_widget.prediction_visualizer.clear_graph()
                self.test_tab_widget.yes_button.setEnabled(False)
                self.test_tab_widget.no_button.setEnabled(False)
        except Exception as e:
            self._log_message(f"An unexpected error occurred during drawing prediction: {e}")
            if hasattr(self, 'test_tab_widget'): # Check if tab exists
                self.test_tab_widget.feedback_label.setText("Prediction Failed")
                self.test_tab_widget.prediction_visualizer.clear_graph()
                self.test_tab_widget.yes_button.setEnabled(False)
                self.test_tab_widget.no_button.setEnabled(False)
            import traceback
            traceback.print_exc()
        # --------------------------------- #

    def _predict_image_file(self):
        """Handles image selection, preprocessing, and prediction using the current model."""
        self._log_message("--- Starting Prediction ---")
        # Clear previous preview and graph
        if hasattr(self, 'test_tab_widget'): # Check if tab exists
            self.test_tab_widget.image_preview_label.clear()
            self.test_tab_widget.image_preview_label.setText("Select an image...")
            self.test_tab_widget.prediction_visualizer.clear_graph()
            self.test_tab_widget.feedback_label.setText("Prediction: N/A")
            self.test_tab_widget.yes_button.setEnabled(False)
            self.test_tab_widget.no_button.setEnabled(False)
        else:
            # Log if test_tab_widget doesn't exist for some reason
            self._log_message("Warning: test_tab_widget not found during _predict_image_file initialization.")
            return # Cannot proceed without the test tab

        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            try:
                self._log_message(f"Loading image: {file_name}")
                img = Image.open(file_name)
                # [Preprocessing logic as before - Determine H, W, C, resize, normalize]
                model_type = self.current_model_type
                H, W, C = 28, 28, 1
                if model_type == "CNN":
                     if self.current_model and hasattr(self.current_model, 'input_shape') and self.current_model.input_shape:
                         try:
                             if len(self.current_model.input_shape) == 3: H, W, C = self.current_model.input_shape
                             elif len(self.current_model.input_shape) == 4: H, W, C = self.current_model.input_shape[1:]
                             self._log_message(f"Using model input shape for file prediction: {(H, W, C)}")
                         except Exception: self._log_message(f"Could not parse model input shape {self.current_model.input_shape}, using default {H}x{W}x{C}.")
                     else: self._log_message(f"No model shape available, using default {H}x{W}x{C} for file prediction.")
                target_mode = "L" if C == 1 else "RGB"
                self._log_message(f"Preprocessing image file to {target_mode} and size {(W, H)}...")
                img_processed = img.convert(target_mode).resize((W, H), Image.Resampling.LANCZOS)
                img_array_raw = np.array(img_processed)
                input_data = None
                if model_type == "Simple NN":
                     num_features = H * W * C
                     input_data = img_array_raw.flatten().reshape(num_features, 1)
                     input_data = input_data / 255.0
                     self._log_message(f"Prepared Simple NN input data, shape: {input_data.shape}")
                elif model_type == "CNN":
                     input_data = img_array_raw.reshape(1, H, W, C)
                     input_data = input_data.astype(np.float32)
                     input_data = input_data / 255.0
                     self._log_message(f"Prepared CNN input data, shape: {input_data.shape}")
                else:
                    self._log_message(f"ERROR: Unknown model type '{model_type}' for prediction.")
                    if hasattr(self, 'test_tab_widget'): # Check if tab exists
                         self.test_tab_widget.feedback_label.setText("Prediction N/A (Bad Model Type)")
                         self.test_tab_widget.prediction_visualizer.clear_graph()
                         self.test_tab_widget.yes_button.setEnabled(False)
                         self.test_tab_widget.no_button.setEnabled(False)
                    return

                if input_data is None:
                    self._log_message("ERROR: Failed to prepare input data from file.")
                    if hasattr(self, 'test_tab_widget'): # Check if tab exists
                         self.test_tab_widget.feedback_label.setText("Input Data Prep Failed")
                         self.test_tab_widget.prediction_visualizer.clear_graph()
                         self.test_tab_widget.yes_button.setEnabled(False)
                         self.test_tab_widget.no_button.setEnabled(False)
                    return

                # Create QPixmap for display
                qpixmap_orig = QPixmap(file_name)
                if qpixmap_orig.isNull():
                    self._log_message(f"Warning: QPixmap failed to load {file_name} directly. Trying PIL conversion.")
                    if img.mode != 'RGB': img = img.convert('RGB')
                    qimage = QImage(img.tobytes("raw", "RGB"), img.width, img.height, QImage.Format_RGB888)
                    qpixmap_orig = QPixmap.fromImage(qimage)
                # Access image_preview_label via test_tab_widget
                scaled_pixmap = qpixmap_orig.scaled(self.test_tab_widget.image_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.test_tab_widget.image_preview_label.setPixmap(scaled_pixmap)

                # --- Perform Prediction --- #
                self._log_message("Running prediction on file...")
                if self.current_model is None or not hasattr(self.current_model, 'predict'):
                    self._log_message("ERROR: Model not loaded/trained or has no predict method.")
                    if hasattr(self, 'test_tab_widget'): # Check if tab exists
                        self.test_tab_widget.feedback_label.setText("Model not ready.")
                        self.test_tab_widget.prediction_visualizer.clear_graph()
                        self.test_tab_widget.yes_button.setEnabled(False)
                        self.test_tab_widget.no_button.setEnabled(False)
                    return

                prediction_output = self.current_model.predict(input_data)
                if prediction_output is None:
                    self._log_message("ERROR: Model prediction failed (returned None).")
                    if hasattr(self, 'test_tab_widget'): # Check if tab exists
                        self.test_tab_widget.feedback_label.setText("Prediction Failed")
                        self.test_tab_widget.prediction_visualizer.clear_graph()
                        self.test_tab_widget.yes_button.setEnabled(False)
                        self.test_tab_widget.no_button.setEnabled(False)
                    return

                probabilities = prediction_output.flatten()

                if probabilities.size != self.num_classes:
                     self._log_message(f"Prediction output size ({probabilities.size}) does not match number of classes ({self.num_classes}).")
                     if hasattr(self, 'test_tab_widget'): # Check if tab exists
                         self.test_tab_widget.feedback_label.setText("Prediction Size Mismatch")
                         self.test_tab_widget.prediction_visualizer.clear_graph()
                         self.test_tab_widget.yes_button.setEnabled(False)
                         self.test_tab_widget.no_button.setEnabled(False)
                     return

                # Update visualizer with probabilities and class names (if available)
                display_names = self.class_names # Pass available names
                if hasattr(self, 'test_tab_widget'): # Check if tab exists
                    self.test_tab_widget.prediction_visualizer.set_probabilities(probabilities, class_names=display_names)

                    # --- Update Feedback Label and Buttons --- #
                    feedback_name = self.test_tab_widget.prediction_visualizer.get_predicted_class_name()
                    if feedback_name:
                        self.test_tab_widget.feedback_label.setText(f"Is it a {feedback_name}?")
                        self.test_tab_widget.yes_button.setEnabled(True)
                        self.test_tab_widget.no_button.setEnabled(True)
                        # Log confidence
                        pred_index = self.test_tab_widget.prediction_visualizer.predicted_class_index
                        # Use nan_to_num to handle potential NaN in probabilities
                        valid_probs = np.nan_to_num(probabilities)
                        confidence = valid_probs[pred_index] * 100 if 0 <= pred_index < len(valid_probs) else float('nan')
                        self._log_message(f"File Prediction: {feedback_name} (Confidence: {confidence:.2f}%) Index: {pred_index}")
                    else:
                        self.test_tab_widget.feedback_label.setText("Prediction Error")
                        self.test_tab_widget.yes_button.setEnabled(False)
                        self.test_tab_widget.no_button.setEnabled(False)
                    # ---------------------------------------- #

            # --- Separate Exception Handling --- #
            except Exception as e:
                self._log_message(f"ERROR during file prediction: {e}")
                if hasattr(self, 'test_tab_widget'): # Check if tab exists
                    self.test_tab_widget.feedback_label.setText("Error loading/predicting.")
                    self.test_tab_widget.prediction_visualizer.clear_graph()
                    self.test_tab_widget.yes_button.setEnabled(False)
                    self.test_tab_widget.no_button.setEnabled(False)
                import traceback
                traceback.print_exc()
            # --------------------------------- #
        else:
             self._log_message("Prediction cancelled: No file selected.")


    # --- Post Load Update --- #
    def _post_load_update(self, dataset_name):
        # Enable the training group now that data is loaded
        self.training_group.setEnabled(True)
        self.training_group.setTitle(f"Training Controls ({dataset_name})")
        self.start_button.setEnabled(True)

        # ... (rest of the correct _post_load_update method) ...

        self.load_button.setEnabled(True)
        # --- Enable/Disable Prediction Buttons Based on Model and Data --- #
        drawing_prediction_enabled = False # Default to disabled
        file_prediction_enabled = False # Default to disabled

        if self.current_model is not None:
            # Model exists, enable file prediction
            file_prediction_enabled = True

            # Check if model/data is suitable for drawing canvas (e.g., 28x28x1)
            is_drawing_compatible = False
            if self.current_model_type == 'Simple NN':
                # Check if input features match 28*28
                if input_features == 784:
                    is_drawing_compatible = True
            elif self.current_model_type == 'CNN':
                # Check if input shape matches (H, W, C) == (28, 28, 1)
                try:
                    h, w, c = self.X_train.shape[1:]
                    if h == 28 and w == 28 and c == 1:
                         is_drawing_compatible = True
                except Exception:
                    # If shape check fails, assume incompatible
                    pass # Keep is_drawing_compatible False

            if is_drawing_compatible:
                drawing_prediction_enabled = True

        # Update TestTab buttons
        if hasattr(self, 'test_tab_widget'):
             self.test_tab_widget.predict_drawing_button.setEnabled(drawing_prediction_enabled)
             self.test_tab_widget.predict_file_button.setEnabled(file_prediction_enabled)
             # Always disable feedback buttons after load
             self.test_tab_widget.yes_button.setEnabled(False)
             self.test_tab_widget.no_button.setEnabled(False)
        # --- End Prediction Button Logic --- #

        # This else corresponds to `if self.X_train is not None:`
        else:
            self._log_message("Data loading failed. Training/Testing controls remain disabled.")
             # Keep buttons on MainWindow/TrainTab disabled
            self.training_group.setEnabled(False)
            self.start_button.setEnabled(False)
            self.load_button.setEnabled(False)
            self.save_button.setEnabled(False)
             # --- Disable buttons on TestTab ---
            if hasattr(self, 'test_tab_widget'):
                 self.test_tab_widget.predict_drawing_button.setEnabled(False)
                 self.test_tab_widget.predict_file_button.setEnabled(False)
                 self.test_tab_widget.yes_button.setEnabled(False)
                 self.test_tab_widget.no_button.setEnabled(False)
             # --- End of TestTab button disabling ---

        # Clear Prediction Visualizer and reset feedback UI (This should happen regardless of data load success/failure)
        if hasattr(self, 'test_tab_widget') and self.test_tab_widget.prediction_visualizer is not None:
            self.test_tab_widget.prediction_visualizer.clear_graph()
            self.test_tab_widget.feedback_label.setText("Prediction: N/A")
            self.test_tab_widget.yes_button.setEnabled(False)
            self.test_tab_widget.no_button.setEnabled(False)

    def start_training(self):
        """Initiates the model training process based on UI settings."""
        if self.X_train is None or self.Y_train is None or self.X_dev is None or self.Y_dev is None:
            self._log_message("Error: Training data not loaded or incomplete.")
            return
        if self.num_classes <= 0:
             self._log_message("Error: Number of classes not determined from loaded data.")
             return

        # --- Check if training is already running using the flag --- #
        if self.is_training:
            self._log_message("Warning: Training is already in progress.")
            return
        # ---------------------------------------------------------- #

        # --- Get Selected Model Type --- #
        selected_model_type = self.model_type_combo.currentText()
        self.current_model_type = selected_model_type # Store the selected type
        self._log_message(f"Selected model type: {selected_model_type}")

        # --- Prepare Data Shape Based on Model --- #
        X_train_model, X_dev_model = None, None
        input_shape_for_model = None
        self.model_layer_dims = None # Reset layer dims here

        # Check the dimensions of the loaded data
        data_ndim = self.X_train.ndim
        self._log_message(f"Loaded data dimensions: {data_ndim}, Shape: {self.X_train.shape}")

        try:
            if selected_model_type == "Simple NN":
                if data_ndim == 4: # Data is (samples, H, W, C), needs flattening for NN
                    self._log_message("Data is 4D, flattening for Simple NN...")
                    num_samples_train, H, W, C = self.X_train.shape
                    num_samples_dev = self.X_dev.shape[0]
                    num_features = H * W * C
                    # Reshape to (samples, features) then transpose to (features, samples)
                    X_train_model = self.X_train.reshape(num_samples_train, num_features).T
                    X_dev_model = self.X_dev.reshape(num_samples_dev, num_features).T
                    self._log_message(f"Flattened shapes: X_train={X_train_model.shape}, X_dev={X_dev_model.shape}")
                elif data_ndim == 2: # Data is already (features, samples)
                    self._log_message("Data is 2D, using directly for Simple NN.")
                    num_features = self.X_train.shape[0]
                    X_train_model = self.X_train
                    X_dev_model = self.X_dev
                else:
                    raise ValueError(f"Unexpected data dimensions for Simple NN: {data_ndim}")

                input_shape_for_model = (num_features,) # Shape for NN input layer

                # Calculate layer dims for SimpleNN
                hidden_layers_str = self.hidden_layers_input.text()
                hidden_layers = [int(x.strip()) for x in hidden_layers_str.split(',') if x.strip()]
                self.model_layer_dims = [num_features] + hidden_layers + [self.num_classes]
                self._log_message(f"Calculated Simple NN layer dims: {self.model_layer_dims}")

            elif selected_model_type == "CNN":
                if data_ndim == 4: # Data is already (samples, H, W, C)
                    self._log_message("Data is 4D, using directly for CNN.")
                    num_samples_train, H, W, C = self.X_train.shape
                    input_shape_for_model = (H, W, C) # Shape for CNN input layer
                    X_train_model = self.X_train
                    X_dev_model = self.X_dev
                elif data_ndim == 2: # Data is (features, samples), needs reshaping for CNN
                    self._log_message("Data is 2D, reshaping for CNN...")
                    num_features, num_samples_train = self.X_train.shape
                    num_samples_dev = self.X_dev.shape[1]

                    # Infer H, W, C from num_features
                    if num_features == 784: # MNIST-like
                        input_shape_for_model = (28, 28, 1)
                    elif num_features == 3072: # CIFAR-10 like
                        input_shape_for_model = (32, 32, 3)
                    else:
                        raise ValueError(f"Cannot determine CNN image shape from {num_features} features.")

                    target_shape_train = (num_samples_train,) + input_shape_for_model
                    target_shape_dev = (num_samples_dev,) + input_shape_for_model

                    # Reshape: Transpose (features, samples) -> (samples, features) then reshape
                    self._log_message(f"Reshaping train data {self.X_train.shape} -> {target_shape_train}")
                    X_train_model = self.X_train.T.reshape(target_shape_train)
                    self._log_message(f"Reshaping dev data {self.X_dev.shape} -> {target_shape_dev}")
                    X_dev_model = self.X_dev.T.reshape(target_shape_dev)
                else:
                    raise ValueError(f"Unexpected data dimensions for CNN: {data_ndim}")

            else:
                raise ValueError(f"Model type '{selected_model_type}' not recognized for data preparation.")

        except ValueError as e:
            self._log_message(f"Error preparing data: {e}")
            return # Stop training if data prep fails
        except Exception as e:
            self._log_message(f"Unexpected error during data preparation: {e}")
            import traceback
            traceback.print_exc()
            return

        # --- Check if data preparation succeeded --- #
        if X_train_model is None or X_dev_model is None or input_shape_for_model is None:
            self._log_message("Error: Failed to prepare data for the selected model (result was None).")
            return

        # --- Instantiate Correct Model --- #
        self.current_model = None # Clear previous model instance
        if selected_model_type == "Simple NN":
            if not SimpleNeuralNetwork:
                self._log_message("Error: SimpleNeuralNetwork class not available.")
                return
            # Instantiate SimpleNeuralNetwork
            try:
                # Ensure self.model_layer_dims was set correctly above
                if not self.model_layer_dims:
                    raise ValueError("Model layer dimensions not calculated.")
                self.current_model = SimpleNeuralNetwork(self.model_layer_dims) # Uses layer_dims
                self._log_message(f"Instantiated SimpleNeuralNetwork with layers: {self.model_layer_dims}")
                # Load the actual parameters
                if hasattr(self.current_model, 'load_params') and self.model_params is not None:
                    self.current_model.load_params(self.model_params)
                    self._log_message("Loaded parameters into SimpleNN instance.")
                    # Enable buttons after successful load
                    self.save_button.setEnabled(True)
                    self.predict_drawing_button.setEnabled(True)
                    self.predict_file_button.setEnabled(True)
                else:
                    self._log_message("Warning: Could not load parameters into SimpleNN instance (missing method or no params).")
            except Exception as e:
                 self._log_message(f"Error instantiating or loading params for SimpleNeuralNetwork: {e}")
                 # Reset model if instantiation/load fails

        elif selected_model_type == "CNN":
            if not CNNModel:
                 self._log_message("Error: CNNModel class not available.")
                 return
            # Instantiate CNNModel
            try:
                self.current_model = CNNModel(input_shape=input_shape_for_model, num_classes=self.num_classes)
                # CNN build_model is called separately, often before training or loading weights
                self.current_model.build_model() # Build the architecture now
                self._log_message(f"Instantiated and built CNNModel with input shape: {input_shape_for_model}")
            except Exception as e:
                 self._log_message(f"Error instantiating or building CNNModel: {e}")
                 return
        # -------------------------------- #

        # --- Get Hyperparameters from UI (based on model type) --- #
        training_params = {
            'model': self.current_model, # Pass the instantiated model object
            'model_type': selected_model_type,
            'X_train': X_train_model,
            'Y_train': self.Y_train,
            'X_dev': X_dev_model,
            'Y_dev': self.Y_dev,
            'num_classes': self.num_classes,
        }

        if selected_model_type == "Simple NN":
            training_params.update({
                'activation': self.activation_combo.currentText(),
                'optimizer': self.optimizer_combo.currentText(),
                # Use values directly from UI elements
                'learning_rate': self.learning_rate_input.value(),
                'epochs': self.epochs_input.value(),
                'batch_size': self.batch_size_input.value() if hasattr(self, 'batch_size_input') else 64, # Keep using CNN's batch size input for now
                'patience': self.patience_input.value(),
                'l2_lambda': self.l2_lambda_input.value(),
                'dropout_rate': 1.0 - self.dropout_keep_prob_input.value(),
            })
            # Note: Simple NN currently uses some params intended for CNN UI group (LR, Epochs, Patience, BatchSize)
            #       Consider adding separate inputs for NN if defaults/ranges need to differ significantly.
            self._log_message("Gathered Simple NN hyperparameters from UI.")

        elif selected_model_type == "CNN":
            training_params.update({
                'learning_rate': self.learning_rate_input.value(),
                'epochs': self.epochs_input.value(),
                'batch_size': self.batch_size_input.value(),
                'patience': self.patience_input.value(),
                # CNN-specific params (e.g., activation, optimizer) are often part of the model architecture itself
                # or handled by the Keras fit method defaults if not explicitly passed.
                # We pass None for params specific to Simple NN
                'activation': None,
                'optimizer': None,
                'l2_lambda': None,
                'dropout_rate': None, # Dropout is usually added as a Layer in Keras
            })
            self._log_message("Using CNN hyperparameters from UI.")
        else:
            self._log_message(f"ERROR: Unknown model type '{selected_model_type}' for hyperparameter gathering.")
            return

        # --- Setup and Start Training Thread --- #
        if self.current_model is None:
            self._log_message("Error: Model could not be instantiated.")
            self._set_training_ui_enabled(True) # Re-enable UI on early exit
            return

        # Prepare the params string for logging *outside* the f-string
        params_log_str = str({k: v.shape if isinstance(v, np.ndarray) else type(v) if k=='model' else v for k, v in training_params.items()})
        self._log_message(f"Starting training thread with params: {params_log_str}")

        self.training_thread = QThread()
        # self._log_message(f"DEBUG: Created QThread object: {self.training_thread}") # REMOVED DEBUG LOG

        # Ensure TrainingWorker is imported and available
        if 'TrainingWorker' not in globals():
            self._log_message("Error: TrainingWorker class not found.")
            self._set_training_ui_enabled(True)
            return
        try:
            # Check if TrainingWorker can be imported dynamically if needed
            # from ui.training_worker import TrainingWorker # Might be necessary
            self.training_worker = TrainingWorker(training_params)
        except Exception as e:
            self._log_message(f"Error initializing TrainingWorker: {{e}}")
            self._set_training_ui_enabled(True)
            return

        self.training_worker.moveToThread(self.training_thread)

        # --- Connect signals from worker to main thread slots (Ensure signatures match!) --- # Requires TrainingWorker signals
        self.training_worker.progress.connect(self.update_progress) # Expects (epoch, total_epochs, loss, val_acc)
        self.training_worker.finished.connect(self.training_finished) # Expects (history_dict or None)
        self.training_worker.error.connect(self.training_error) # Expects (error_message_str)
        self.training_worker.log_message.connect(self._handle_worker_log) # Connect to helper slot
        # ---------------------------------------------------------------------------------- #

        # --- Crucial Connection: Worker finished -> Thread quits -> Thread finished --- #
        self.training_worker.finished.connect(self.training_thread.quit) # Tell thread's event loop to stop when worker is done
        # ----------------------------------------------------------------------------- #

        self.training_thread.started.connect(self.training_worker.run) # Start worker's run method
        # Clean up thread object when it finishes (RE-ADD deleteLater)
        self.training_thread.finished.connect(self.training_thread.deleteLater)
        # Schedule worker object for deletion after thread finishes (RE-ADD deleteLater)
        self.training_thread.finished.connect(self.training_worker.deleteLater)
        # Connect thread finished signal to the NEW final cleanup slot
        self.training_thread.finished.connect(self._on_thread_actually_finished)
        # self._log_message(f"DEBUG: Connected QThread.finished to _on_thread_actually_finished. Connections: {self.training_thread.receivers(self.training_thread.finished)}") # REMOVED DEBUG LOG

        # Disable UI during training
        self._set_training_ui_enabled(False)

        self.train_loss_history = [] # Reset histories
        self.val_accuracy_history = []
         # Reset progress bar
        self.progress_bar.setRange(0, training_params['epochs'])
        self.progress_bar.setValue(0)

        # --- Set Training Flag and Start Thread --- #
        self.is_training = True # Set flag to indicate training started
        # self._log_message(f"DEBUG: Starting QThread object: {self.training_thread}") # REMOVED DEBUG LOG
        self.training_thread.start()
        self._log_message("Training thread started.")

    def _disconnect_worker_signals(self):
        """Safely disconnects signals from the training worker."""
        self._log_message("Attempting to disconnect worker signals...")
        if self.training_worker:
            try: self.training_worker.progress.disconnect(self.update_progress)
            except TypeError: pass # Signal already disconnected or never connected
            try: self.training_worker.finished.disconnect(self.training_finished)
            except TypeError: pass
            try: self.training_worker.error.disconnect(self.training_error)
            except TypeError: pass
            try: self.training_worker.log_message.disconnect(self._handle_worker_log)
            except TypeError: pass
        else:
             self._log_message("Worker object not found, cannot disconnect signals.")

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
            self.log_output.moveCursor(QtGui.QTextCursor.End)
            # Prepending timestamp to redirected output might be too noisy,
            # so we just insert the text as is.
            self.log_output.insertPlainText(text + '\n')
            # QApplication.processEvents() # Keep UI responsive - REMOVED to prevent potential re-entrancy issues

    # Add flush method (required for stdout redirection)
    def flush(self):
        pass

    # --- Main Action Methods ---

    # Add methods for dataset loading buttons
    def load_selected_dataset(self):
        """Loads the dataset selected in the dropdown."""
        # --- ADD DEBUG LOG HERE ---
        self._log_message(f"DEBUG: load_selected_dataset CALLED. Dropdown key: {self.dataset_dropdown.currentData()}")
        # --------------------------
        dataset_key = self.dataset_dropdown.currentData()
        if not dataset_key or dataset_key not in self.datasets_info:
            self._log_message("No valid dataset selected.")
            return
        
        # --- Assign dataset_name here --- #
        dataset_name = self.dataset_dropdown.currentText() # Use the text from the dropdown

        dataset_info = self.datasets_info[dataset_key]
        dataset_type = dataset_info["type"]
        dataset_path = dataset_info["path"]

        # Initial log message moved inside the try block after path/map is determined
        self.current_dataset_name = dataset_key # Store the key name

        # --- Determine required data shape based on selected model --- #
        selected_model_type = self.model_type_combo.currentText()
        return_shape = 'flattened' # Default for Simple NN
        if selected_model_type == "CNN":
            return_shape = 'image'
        self._log_message(f"  Model type '{selected_model_type}' requires data shape: '{return_shape}'")
        # ------------------------------------------------------------ #

        X_train, Y_train, X_dev, Y_dev, num_classes, raw_X_train, raw_Y_train = None, None, None, None, 0, None, None
        self.class_names = None # Reset class names

        try:
            npy_map_to_load = None # Initialize variable for QuickDraw map
            if dataset_type == 'quickdraw':
                # --- QuickDraw Selection Logic --- #
                quickdraw_dir = dataset_info.get("path")
                self._log_message(f"Attempting to load QuickDraw from directory: {quickdraw_dir}") # Add entry log
                if not quickdraw_dir or not os.path.isdir(quickdraw_dir):
                    raise FileNotFoundError(f"QuickDraw directory not found or invalid: {quickdraw_dir}")

                # --- V3: Force Glob Check --- #
                self._log_message(f"  Scanning QuickDraw directory for .npy files using glob: {quickdraw_dir}")
                npy_files_in_dir = glob.glob(os.path.join(quickdraw_dir, "*.npy"))
                if not npy_files_in_dir:
                    # Log the directory contents if no npy files found
                    try:
                         dir_contents = os.listdir(quickdraw_dir)
                         self._log_message(f"    DEBUG: Contents of {quickdraw_dir}: {dir_contents}")
                    except Exception as list_err:
                         self._log_message(f"    DEBUG: Could not list contents of {quickdraw_dir}: {list_err}")
                    # *** MODIFIED ERROR MESSAGE ***
                    raise FileNotFoundError(f"ERROR V3: No .npy files found via glob in {quickdraw_dir}")
                self._log_message(f"  Found {len(npy_files_in_dir)} .npy files via glob.")
                # ---------------------------- #

                selected_files_for_map = []
                random_count = self.quickdraw_random_count.value()

                if random_count > 0:
                    # User specified a random count
                    if random_count > len(npy_files_in_dir):
                        self._log_message(f"Warning: Requested {random_count} random categories, but only {len(npy_files_in_dir)} available. Loading all.")
                        selected_files_for_map = npy_files_in_dir
                    else:
                        selected_files_for_map = random.sample(npy_files_in_dir, random_count)
                        self._log_message(f"Randomly selected {random_count} QuickDraw categories.")
                else:
                    # User selected from the list
                    selected_items = self.quickdraw_list_widget.selectedItems()
                    if not selected_items:
                        raise ValueError("QuickDraw selected, but no categories chosen from the list (and random count is 0).")
                    # Map selected text back to file paths (assuming order is preserved)
                    selected_names = {item.text() for item in selected_items}
                    all_category_names = [os.path.splitext(os.path.basename(f))[0].replace("_", " ") for f in npy_files_in_dir]
                    for i, name in enumerate(all_category_names):
                        if name in selected_names:
                            selected_files_for_map.append(npy_files_in_dir[i])
                    self._log_message(f"Selected {len(selected_files_for_map)} QuickDraw categories from list: {[os.path.basename(f) for f in selected_files_for_map]}")

                if not selected_files_for_map:
                     raise ValueError("Failed to determine QuickDraw categories to load.")

                # Build the npy_map ONLY with the selected files, assigning indices 0 to N-1
                npy_map_to_load = {path: i for i, path in enumerate(sorted(selected_files_for_map))} # Sort selected for consistent index assignment
                # ---------------------------------- #
                self._log_message(f"Loading {len(npy_map_to_load)} selected QuickDraw categories...")
                # --- DEBUG LOG: Show the map being passed --- #
                self._log_message(f"DEBUG: npy_map_to_load contents: {npy_map_to_load}")
                # ------------------------------------------ #

                X_train, Y_train, X_dev, Y_dev, num_classes, _, _ = datasets.load_multiple_npy_datasets(
                    npy_file_map=npy_map_to_load, # Pass the filtered dict here
                    return_shape=return_shape # Pass required shape
                )
                # Extract class names from the map keys if needed
                # Class names now come from the filtered list/map
                if npy_map_to_load:
                     # Sort the selected files based on the assigned index in npy_map_to_load
                     sorted_items = sorted(npy_map_to_load.items(), key=lambda item: item[1])
                     self.class_names = [os.path.splitext(os.path.basename(p))[0].replace("_", " ") for p, index in sorted_items]
                     self._log_message(f"QuickDraw Class names loaded: {self.class_names[:5]}... ({len(self.class_names)} total)")

            elif dataset_type == 'emoji':
                path = dataset_info.get('path')
                if not path:
                    raise ValueError("Could not retrieve path for emoji dataset.")
                self._log_message(f"Loading emoji data from: {path}")
                # Determine provider from dataset_key (adjust if key format changes)
                provider = "Google" # Default
                if "Apple" in dataset_key:
                    provider = "Apple"
                # Load emojis with specific function to get class names
                X_train, Y_train, X_dev, Y_dev, num_classes, class_names = datasets.load_emoji_dataset(
                    csv_path=path,
                    image_column=provider, # Pass correct provider
                    return_shape=return_shape # Pass required shape
                )
                self.class_names = class_names # Store emoji names
                if class_names:
                     self._log_message(f"Loaded emoji names: {class_names[:5]}... ({len(class_names)} total)")

            elif dataset_type == 'cifar10':
                path = dataset_info.get('path')
                if not path:
                    raise ValueError("Could not retrieve path for CIFAR-10 dataset.")
                self._log_message(f"Loading CIFAR-10 data from base directory: {path}")
                X_train, Y_train, X_dev, Y_dev, num_classes, _, _ = datasets.load_cifar10_dataset(
                    data_dir=path,
                    return_shape=return_shape # Pass required shape
                )
                # Get class names for CIFAR-10
                self.class_names = datasets.get_cifar10_class_names(path)
                if self.class_names:
                    self._log_message(f"CIFAR-10 Class names: {self.class_names}")
                else:
                     self._log_message("Could not load CIFAR-10 class names.")

            elif dataset_type == 'csv':
                path = dataset_info.get('path')
                if not path:
                    raise ValueError("Could not retrieve path for CSV dataset.")
                self._log_message(f"Loading CSV data from: {path}")
                # Use generic CSV loader
                label_col = self.label_col_input.value()
                img_col = self.image_col_input.value()
                img_type = self.image_type_combo.currentText()

                image_col_idx = img_col if img_col >= 0 else None
                image_col_type = img_type.lower() if img_type != "(Not Applicable)" else None

                X_train, Y_train, X_dev, Y_dev, num_classes, raw_X_train, raw_Y_train = datasets.load_csv_dataset(
                    csv_path=path,
                    label_col_index=label_col,
                    image_col_index=image_col_idx,
                    image_col_type=image_col_type,
                    return_shape=return_shape # Pass required shape
                )

            # --- RE-INSERTED CIFAR-100 BLOCK --- #
            elif dataset_type == "cifar100":
                self._log_message(f"---> Entered CIFAR-100 loading logic block for {dataset_name}.")
                model_req_shape = 'image' if self.current_model_type == 'CNN' else 'flattened'
                self._log_message(f"Requesting shape '{model_req_shape}' from CIFAR-100 loader.")
                self._log_message(f"---> About to call self._load_cifar100_data for path: {dataset_path}")
                loaded_data = self._load_cifar100_data(dataset_path, return_shape=model_req_shape)
                self._log_message(f"<--- Returned from self._load_cifar100_data. Result type: {type(loaded_data)}")

                if loaded_data:
                    self._log_message(f"    DEBUG: Unpacking result from _load_cifar100_data...")
                    X_train_part, _, X_test_part, _, self.class_names, fine_labels = loaded_data
                    # Combine train and test for simplicity
                    X_train = np.vstack((X_train_part, X_test_part)) # Use combined X for training data
                    Y_train = np.concatenate((fine_labels["train"], fine_labels["test"])) # Use combined Y for training labels
                    # Note: We are not creating separate X_dev, Y_dev here. The split happens later in the model training if needed.
                    X_dev, Y_dev = None, None # Explicitly set dev sets to None for now
                    num_classes = 100
                    if not self.class_names:
                        self._log_message("Warning: Fine label names for CIFAR-100 not loaded, using numbers.")
                        self.class_names = [str(i) for i in range(num_classes)]
                    self._log_message(f"Successfully processed CIFAR-100 data. Shape: {X_train.shape}")
                else:
                     self._log_message("Raising error because _load_cifar100_data returned None/False.")
                     raise ValueError("CIFAR-100 loading failed (_load_cifar100_data returned None or False).")
            # --- END RE-INSERTED BLOCK --- #

            else:
                 # Added a catch-all else to handle unexpected types
                 self._log_message(f"Error: Encountered unknown dataset type '{dataset_type}' during loading.")
                 raise ValueError(f"Unknown dataset type: {dataset_type}")

            # Store loaded data (ensure variables are assigned)
            if X_train is not None and Y_train is not None:
                self.X_train, self.Y_train = X_train, Y_train
                # Handle potentially None dev sets
                self.X_dev = X_dev if X_dev is not None else np.array([])
                self.Y_dev = Y_dev if Y_dev is not None else np.array([])
                self.num_classes = num_classes
                # Store raw data if available
                self.raw_X_train_flattened = raw_X_train if raw_X_train is not None else None
            else:
                 # This condition suggests loading failed within one of the elif blocks
                 # Error should have been logged there, but we ensure failure state here
                 raise ValueError("Data loading failed or returned empty data (X_train/Y_train are None).")

            # --- Log final class_names value before post_load_update --- #
            self._log_message(f"  DEBUG [load_selected_dataset]: FINAL class_names before _post_load_update: {self.class_names[:10] if self.class_names else 'None'}")
            # ----------------------------------------------------------- #

            # Update UI state after successful load
            # TEMP: Comment out to diagnose post-prediction clearing - RESTORED
            self._post_load_update(dataset_key)
            # self._log_message("TEMP DEBUG: Skipped _post_load_update call in load_selected_dataset") # Remove debug log

        except FileNotFoundError as e:
            self._log_message(f"Error: Dataset file not found: {e}")
        except ValueError as e:
             self._log_message(f"Error processing dataset: {e}")
        except Exception as e:
            self._log_message(f"An unexpected error occurred loading data: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging

    def _on_model_type_changed(self, model_type: str):
        """Slot called when the model type dropdown changes."""
        self._log_message(f"Model type changed to: {model_type}")
        # Only update if the type actually changed to avoid redundant loads/resets
        if self.current_model_type != model_type:
            self.current_model_type = model_type # Update the internal state

            # Update UI visibility for hyperparameters
            self._update_hyperparameter_visibility()

            # Reset any loaded model/weights as they are likely incompatible
            self._reset_model_state()
            self._log_message("Model state has been reset due to type change.")

            # Reload the current dataset (if one is selected) to get the correct shape
            current_dataset_key = self.dataset_dropdown.currentData()
            if current_dataset_key and current_dataset_key in self.datasets_info:
                self._log_message(f"Reloading dataset '{current_dataset_key}' for new model type '{model_type}'...")
                # Ensure load button is temporarily disabled during auto-reload? (Optional)
                # self.load_dataset_button.setEnabled(False)
                self.load_selected_dataset() # This will use the new self.current_model_type
                # Re-enable button after load finishes (handled by _post_load_update)
                # self.load_dataset_button.setEnabled(True)
            else:
                self._log_message("No valid dataset selected, skipping reload on model type change.")
        else:
            # Optional: Log that the same type was re-selected
            # self._log_message(f"Model type '{model_type}' re-selected.")
            pass

    def _update_hyperparameter_visibility(self):
        """Show/hide hyperparameter sections based on the selected model type."""
        model_type = self.model_type_combo.currentText()

        # Hide all model-specific groups initially
        # Use getattr to safely access groups that might not exist (if creation failed)
        layer_group = getattr(self, 'layer_sizes_group', None)
        cnn_group = getattr(self, 'cnn_params_group', None)
        common_group = getattr(self, 'common_params_group', None)

        if layer_group:
            layer_group.setVisible(False)
        if cnn_group:
            cnn_group.setVisible(False)
        # Ensure the common group is always visible (if it exists)
        if common_group:
            common_group.setVisible(True)

        # Show the relevant model-specific group
        if model_type == "Simple NN":
            if layer_group:
                layer_group.setVisible(True)
                self._log_message("Showing Simple NN hyperparameters.")
            else:
                self._log_message("Warning: Simple NN layer group not found.")
        elif model_type == "CNN":
            if cnn_group:
                cnn_group.setVisible(True)
                self._log_message("Showing CNN hyperparameters (placeholder).")
            else:
                self._log_message("Warning: CNN params group not found.")
            # Optionally hide/disable specific common params if not applicable to CNN?
        else:
            self._log_message(f"Unknown model type selected: {model_type}")

    def _reset_model_state(self):
        """Clears the current model instance and weights."""
        self.current_model = None
        self.model_params = None
        self.model_layer_dims = None

    def upload_csv_dataset(self):
        """Opens a dialog to select a custom CSV dataset."""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Custom CSV Dataset", "", "CSV Files (*.csv);;All Files (*)", options=options
        )

        if file_path:
            self._log_message(f"Custom CSV selected: {file_path}")
            # Add this path to datasets_info temporarily or handle directly
            dataset_key = f"Custom: {os.path.basename(file_path)}"
            self.datasets_info[dataset_key] = {"type": "csv", "path": file_path}
            # Add to dropdown and select it
            self.dataset_dropdown.addItem(dataset_key)
            self.dataset_dropdown.setCurrentText(dataset_key)
            # Load it immediately after adding
            self.load_selected_dataset()
        else:
            self._log_message("Custom CSV selection cancelled.")

    def save_weights(self):
        """Saves the current model's weights/parameters to a file."""
        if not self.current_model and not self.model_params:
            self._log_message("No model trained or parameters loaded to save.")
            return

        # Suggest a filename based on dataset and model type
        suggested_filename = "model_weights"
        if self.current_dataset_name:
            # Ensure dataset name is a string before processing
            dataset_name_str = str(self.current_dataset_name) if self.current_dataset_name is not None else "unknown_dataset"
            # Sanitize dataset name for filename
            safe_dataset_name = "".join(c for c in dataset_name_str if c.isalnum() or c in ('_', '-')).rstrip()
            if not safe_dataset_name: # Handle empty string case after sanitization
                safe_dataset_name = "unknown_dataset"
            suggested_filename = f"{safe_dataset_name}_{self.current_model_type.replace(' ', '')}"
        # If self.current_dataset_name is None, suggested_filename remains "model_weights"
 
        # Determine file extension based on model type
        if self.current_model_type == "Simple NN":
            filter = "NumPy NPZ Files (*.npz)"
            suggested_filename += ".npz"
        elif self.current_model_type == "CNN":
            # Keras save_weights requires the .weights.h5 suffix
            filter = "Keras Weights Files (*.weights.h5)"
            suggested_filename += ".weights.h5"
        else:
            filter = "All Files (*)"
            self._log_message(f"Unknown model type '{self.current_model_type}' for saving, using generic filter.")

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Model Weights", suggested_filename, filter, options=options)

        if file_path:
            self._log_message(f"Saving weights to: {file_path}")
            try:
                if self.current_model_type == "Simple NN":
                    if self.model_params is not None:
                        # Add model structure info to the saved file
                        save_data = {
                            'params': self.model_params,
                            'layer_dims': self.model_layer_dims,
                            'num_classes': self.num_classes,
                            'class_names': self.class_names,
                            'dataset_name': self.current_dataset_name
                        }
                        np.savez(file_path, **save_data)
                        self._log_message("Simple NN parameters saved successfully to NPZ.")
                    else:
                        self._log_message("No parameters available for Simple NN to save.")

                elif self.current_model_type == "CNN":
                    if self.current_model and hasattr(self.current_model, 'save_weights'):
                        # Keras model handles saving its own structure/weights
                        self.current_model.save_weights(file_path)
                        # Optionally save metadata separately if needed (e.g., class names)
                        # meta_path = os.path.splitext(file_path)[0] + ".meta"
                        # with open(meta_path, 'wb') as f:
                        #     pickle.dump({'class_names': self.class_names, ...}, f)
                        self._log_message("CNN weights saved successfully to H5.")
                    else:
                         self._log_message("CNN Model object not found or has no save_weights method.")

                else:
                    self._log_message(f"Saving not implemented for model type: {self.current_model_type}")

            except Exception as e:
                self._log_message(f"Error saving weights: {e}")
                import traceback
                traceback.print_exc()
        else:
            self._log_message("Save operation cancelled.")

    def load_weights(self):
        """Loads model weights/parameters from a file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Model Weights", "", "NumPy NPZ Files (*.npz);;Keras H5 Files (*.h5);;All Files (*)", options=options)

        if file_path:
            self._log_message(f"Loading weights from: {file_path}")
            try:
                if self.current_model_type == "Simple NN":
                    # Load weights for Simple NN
                    data = np.load(file_path, allow_pickle=True) # Must allow pickle for object arrays

                    # --- Extract data --- #
                    # Use .item() to extract dictionary stored as 0D object array
                    if 'params' in data and isinstance(data['params'], np.ndarray) and data['params'].shape == ():
                        loaded_params = data['params'].item()
                    else:
                        # Fallback or raise error if params format is unexpected
                        loaded_params = data.get('params') # Try direct access
                        if not isinstance(loaded_params, dict):
                           raise ValueError("Could not load 'params' dictionary from NPZ file.")

                    # Load other data (assume direct access works for these unless errors occur)
                    loaded_layer_dims = data.get('layer_dims')
                    loaded_num_classes = data.get('num_classes')
                    loaded_class_names = data.get('class_names')
                    loaded_dataset_name = data.get('dataset_name')
                    # -------------------- #

                    # --- Convert types if necessary --- #
                    if isinstance(loaded_layer_dims, np.ndarray):
                        loaded_layer_dims = loaded_layer_dims.tolist()
                    if isinstance(loaded_num_classes, np.ndarray):
                        loaded_num_classes = int(loaded_num_classes)
                    if isinstance(loaded_class_names, np.ndarray):
                         loaded_class_names = loaded_class_names.tolist()
                    # ---------------------------------- #

                    # --- Validate and apply loaded data --- #
                    if not isinstance(loaded_params, dict) or loaded_layer_dims is None or loaded_num_classes is None:
                        raise ValueError("Loaded NPZ file is missing required keys ('params', 'layer_dims', 'num_classes') or params is not a dict.")

                    self.model_layer_dims = loaded_layer_dims # Use loaded dims
                    self.num_classes = loaded_num_classes
                    # Only update class_names if the loaded value is not None
                    if loaded_class_names is not None:
                        self.class_names = loaded_class_names
                    else:
                        self._log_message("Warning: Loaded weights file did not contain class names. Using existing class names if available.")
                    self.current_dataset_name = loaded_dataset_name # Okay if None
                    self.model_params = loaded_params # Store loaded params

                    # Re-instantiate the model with loaded architecture
                    if self.current_model_type == "Simple NN":
                        if not SimpleNeuralNetwork:
                            self._log_message("Error: SimpleNeuralNetwork class not available.")
                            return
                        # Instantiate SimpleNeuralNetwork
                        try:
                            # Ensure self.model_layer_dims was set correctly above
                            if not self.model_layer_dims:
                                raise ValueError("Model layer dimensions not calculated.")
                            self.current_model = SimpleNeuralNetwork(self.model_layer_dims) # Uses layer_dims
                            self._log_message(f"Instantiated SimpleNeuralNetwork with layers: {self.model_layer_dims}")
                            # Load the actual parameters
                            if hasattr(self.current_model, 'load_params') and self.model_params is not None:
                                self.current_model.load_params(self.model_params)
                                self._log_message("Loaded parameters into SimpleNN instance.")
                                # Enable buttons after successful load
                                self.save_button.setEnabled(True)
                                # --- Corrected & Added Compatibility Check ---
                                drawing_prediction_enabled = False
                                if self.current_model_type == 'Simple NN' and self.model_layer_dims and self.model_layer_dims[0] == 784:
                                    drawing_prediction_enabled = True
                                # Add check for CNN input shape if possible (assuming self.current_model has input_shape)
                                elif self.current_model_type == 'CNN' and hasattr(self.current_model, 'input_shape') and self.current_model.input_shape == (28, 28, 1):
                                    drawing_prediction_enabled = True

                                if hasattr(self, 'test_tab_widget'):
                                     self.test_tab_widget.predict_drawing_button.setEnabled(drawing_prediction_enabled)
                                     self.test_tab_widget.predict_file_button.setEnabled(True) # File prediction always enabled if weights load
                                # ---------------------------------------------
                            else:
                                self._log_message("Warning: Could not load parameters into SimpleNN instance (missing method or no params).")
                        except Exception as e:
                             self._log_message(f"Error instantiating or loading params for SimpleNeuralNetwork: {e}")
                             # Reset model if instantiation/load fails

                    elif self.current_model_type == "CNN":
                        if not CNNModel:
                             self._log_message("Error: CNNModel class not available.")
                             return
                        # Instantiate CNNModel
                        try:
                            self.current_model = CNNModel(input_shape=input_shape_for_model, num_classes=self.num_classes)
                            # CNN build_model is called separately, often before training or loading weights
                            self.current_model.build_model() # Build the architecture now
                            self._log_message(f"Instantiated and built CNNModel with input shape: {input_shape_for_model}")
                        except Exception as e:
                             self._log_message(f"Error instantiating or building CNNModel: {e}")
                             return
                        # -------------------------------- #

                elif self.current_model_type == "CNN":
                    # Ensure CNN model is instantiated before loading weights
                    if self.current_model is None:
                        self._log_message("CNN model not instantiated. Instantiating based on loaded data...")
                        # Check if dataset is loaded
                        if not hasattr(self, 'X_train') or self.X_train is None or self.num_classes <= 0:
                            self._log_message("ERROR: Cannot load CNN weights. Load a dataset first to define input shape and classes.")
                            return

                        # Determine input shape from loaded data (replicating logic from start_training)
                        input_shape_for_model = None
                        data_ndim = self.X_train.ndim
                        if data_ndim == 4: # Data is (samples, H, W, C)
                            input_shape_for_model = self.X_train.shape[1:]
                        elif data_ndim == 2: # Data is (features, samples)
                            num_features = self.X_train.shape[0]
                            if num_features == 784: input_shape_for_model = (28, 28, 1)
                            elif num_features == 3072: input_shape_for_model = (32, 32, 3)
                            else:
                                self._log_message(f"ERROR: Cannot determine CNN shape from {num_features} features in loaded data.")
                                return
                        else:
                             self._log_message(f"ERROR: Unexpected loaded data dimensions: {data_ndim}")
                             return

                        # Instantiate and build the model
                        if not CNNModel:
                            self._log_message("ERROR: CNNModel class not available.")
                            return
                        try:
                            self.current_model = CNNModel(input_shape=input_shape_for_model, num_classes=self.num_classes)
                            self.current_model.build_model()
                            self._log_message(f"Instantiated and built CNNModel with input shape: {input_shape_for_model}")
                        except Exception as e:
                            self._log_message(f"Error instantiating/building CNNModel for weight loading: {e}")
                            self.current_model = None # Reset on failure
                            return

                    # Now, attempt to load weights into the (potentially newly created) model
                    if self.current_model and hasattr(self.current_model, 'load_weights'):

                         self.current_model.load_weights(file_path)
                         self._log_message("CNN weights loaded successfully.")
                         # After loading, enable save/predict buttons if not already
                         self.save_button.setEnabled(True)
                         # --- Corrected & Added Compatibility Check ---
                         drawing_prediction_enabled = False
                         # Check CNN input shape if possible (assuming self.current_model has input_shape)
                         if hasattr(self.current_model, 'input_shape') and self.current_model.input_shape == (28, 28, 1):
                            drawing_prediction_enabled = True

                         if hasattr(self, 'test_tab_widget'):
                              self.test_tab_widget.predict_drawing_button.setEnabled(drawing_prediction_enabled)
                              self.test_tab_widget.predict_file_button.setEnabled(True) # File prediction always enabled if weights load
                         # ---------------------------------------------
                    else:
                         
                         self._log_message("CNN Model object not found or has no load_weights method.")

                else:
                    self._log_message(f"Loading not implemented for model type: {self.current_model_type}")

            except Exception as e:
                self._log_message(f"Error loading weights: {e}")
                traceback.print_exc()
        else:
            self._log_message("Load operation cancelled.")

    def _show_expanded_plot(self):
        """Shows the training plot in a separate, resizable dialog window."""
        if self.expanded_plot_dialog is None:
            # Create the dialog only if it doesn't exist
            self.expanded_plot_dialog = QDialog(self) # Parent is the main window
            self.expanded_plot_dialog.setWindowTitle("Training History Plot")
            self.expanded_plot_dialog.setMinimumSize(500, 400) # Set a reasonable minimum size

            layout = QVBoxLayout()
            # Create a *new* PlotWidget instance specifically for this dialog
            dialog_plot_widget = PlotWidget(self.expanded_plot_dialog)
            layout.addWidget(dialog_plot_widget)
            self.expanded_plot_dialog.setLayout(layout)
            self._log_message("Created expanded plot dialog.")
        else:
             # If dialog exists, just bring it to the front
             self._log_message("Bringing existing expanded plot dialog to front.")

        # Find the PlotWidget within the dialog (whether new or existing)
        dialog_plot_widget = self.expanded_plot_dialog.findChild(PlotWidget)

        if dialog_plot_widget:
            # Update the plot with the current history data
            try:
                dialog_plot_widget.update_plot(self.train_loss_history, self.val_accuracy_history)
                self._log_message("Updating expanded plot with current history.")
                # If training is currently running, start the loading animation
                if self.training_thread and self.training_thread.isRunning() and hasattr(dialog_plot_widget, 'start_loading_animation'):
                    dialog_plot_widget.start_loading_animation()

            except Exception as e:
                 self._log_message(f"Error updating expanded plot: {e}")
        else:
            self._log_message("Error: Could not find PlotWidget in expanded dialog.")

        # Show the dialog
        self.expanded_plot_dialog.show()
        self.expanded_plot_dialog.raise_() # Bring to front
        self.expanded_plot_dialog.activateWindow() # Give it focus

    def update_progress(self, epoch: int, total_epochs: int, loss: float, val_acc: float):
        """Slot to receive progress updates from the training worker."""
        percentage = int((epoch / total_epochs) * 100)
        self.progress_bar.setValue(percentage)
        self._log_message(f"  DEBUG [update_progress]: Setting progress bar value to {percentage}% (Epoch {epoch}/{total_epochs})")
        self.progress_bar.setFormat(f"Epoch {epoch}/{total_epochs} - Loss: {loss:.4f} - Val Acc: {val_acc:.3f} ({percentage}%)")

        # Append data to history lists
        self.train_loss_history.append(loss)
        self.val_accuracy_history.append(val_acc)


    def training_finished(self, results: Optional[Any]): # Changed type hint to Any
        """Slot called when the training worker finishes successfully.
           Handles results for both SimpleNN (tuple) and CNN (dict).
        """
        # --- CORRECTED LOG CALL --- #
        self._log_message("Training worker finished successfully. Processing results...")
        # -------------------------- #
        self.is_training = False # Training is no longer active
        self._set_training_ui_enabled(True) # Re-enable train/load buttons, disable stop

        if results is None:
            self._log_message("Training finished, but no results were returned (possibly stopped early or error).")
            self.accuracy_label.setText("Training Stopped/Failed")
            self.progress_bar.setValue(0) # Reset progress
            # Keep save/predict disabled if no valid results
            self.save_button.setEnabled(False)
            self.predict_drawing_button.setEnabled(False)
            self.predict_file_button.setEnabled(False)
            return # Exit early

        # --- Process Results Based on Type --- #
        try:
            final_val_acc = float('nan') # Default to NaN
            processed_ok = False

            if self.current_model_type == "Simple NN" and isinstance(results, tuple) and len(results) == 3:
                self._log_message("Processing Simple NN results tuple...")
                # Expecting tuple: (final_params, loss_hist, acc_hist)
                model_params, self.train_loss_history, self.val_accuracy_history = results
                self.model_params = model_params # Store final trained parameters
                # Update model instance (optional, if needed elsewhere)
                if self.current_model and hasattr(self.current_model, 'load_params'):
                    self.current_model.load_params(self.model_params)
                if self.val_accuracy_history: final_val_acc = self.val_accuracy_history[-1]
                processed_ok = True

            elif self.current_model_type == "CNN" and isinstance(results, dict):
                self._log_message("Processing CNN results dictionary (Keras history)...")
                # Expecting Keras history dictionary
                keras_history = results
                self.train_loss_history = keras_history.get('loss', [])
                # Handle Keras key difference: 'val_accuracy' or 'val_acc'
                acc_key = 'val_accuracy' if 'val_accuracy' in keras_history else 'val_acc'
                self.val_accuracy_history = keras_history.get(acc_key, [])
                # CNN model instance (self.current_model) is updated internally by fit
                if self.val_accuracy_history: final_val_acc = self.val_accuracy_history[-1]
                processed_ok = True

            else:
                 self._log_message(f"Warning: Received unexpected results type ({type(results)}) for model type '{self.current_model_type}'. Cannot process.")

            # --- Update UI if processed successfully --- #
            if processed_ok:
                 # Update accuracy label
                 if not np.isnan(final_val_acc):
                     self.accuracy_label.setText(f"Final Validation Accuracy: {final_val_acc:.4f}")
                     self._log_message(f"Final Validation Accuracy: {final_val_acc:.4f}")
                 else:
                     self.accuracy_label.setText("Training Complete (No validation accuracy)")
                 self.progress_bar.setValue(100) # Mark as 100% complete

                 # Update the expanded plot if it's open
                 if self.expanded_plot_dialog:
                     # Find the PlotWidget within the dialog (whether new or existing)
                     dialog_plot_widget = self.expanded_plot_dialog.findChild(PlotWidget)
                     if dialog_plot_widget:
                         try:
                             # Pass interval=1 assuming history is per epoch now
                             dialog_plot_widget.update_plot(self.train_loss_history, self.val_accuracy_history, interval=1)
                             self._log_message("Updating expanded plot with final history.")
                         except Exception as e:
                             self._log_message(f"Error updating expanded plot: {e}")
                     else:
                         self._log_message("Error: Could not find PlotWidget in expanded dialog for final update.")

                 # Enable saving and prediction now that training is done
                 self.save_button.setEnabled(True)
                 # Enable prediction buttons on TestTab
                 if hasattr(self, 'test_tab_widget'):
                     # --- Check compatibility before enabling drawing button ---
                     drawing_prediction_enabled = False
                     if self.current_model_type == 'Simple NN' and self.model_layer_dims and self.model_layer_dims[0] == 784:
                          drawing_prediction_enabled = True
                     elif self.current_model_type == 'CNN' and hasattr(self.current_model, 'input_shape') and self.current_model.input_shape == (28, 28, 1):
                          drawing_prediction_enabled = True
                     # -------------------------------------------------------
                     self.test_tab_widget.predict_drawing_button.setEnabled(drawing_prediction_enabled)
                     self.test_tab_widget.predict_file_button.setEnabled(True) # File prediction always enabled after successful train
                     # Keep feedback buttons disabled until next prediction
                     self.test_tab_widget.yes_button.setEnabled(False)
                     self.test_tab_widget.no_button.setEnabled(False)
            else:
                 # Handle case where results type didn't match model type
                 self.accuracy_label.setText("Training finished (Result Mismatch)")
                 self.progress_bar.setValue(0) # Reset progress
                 self.save_button.setEnabled(False)
                 # Disable prediction buttons on TestTab
                 if hasattr(self, 'test_tab_widget'):
                     self.test_tab_widget.predict_drawing_button.setEnabled(False)
                     self.test_tab_widget.predict_file_button.setEnabled(False)
                     self.test_tab_widget.yes_button.setEnabled(False)
                     self.test_tab_widget.no_button.setEnabled(False)

        except Exception as e:
            self._log_message(f"Error processing training results: {e}")
            traceback.print_exc()
            self.accuracy_label.setText("Error processing results")
            self.progress_bar.setValue(0)
            self.save_button.setEnabled(False)
            # Disable prediction buttons on TestTab on error
            if hasattr(self, 'test_tab_widget'):
                self.test_tab_widget.predict_drawing_button.setEnabled(False)
                self.test_tab_widget.predict_file_button.setEnabled(False)
                self.test_tab_widget.yes_button.setEnabled(False)
                self.test_tab_widget.no_button.setEnabled(False)

    def training_error(self, message: str):
        """Slot called when the training worker emits an error."""
        self._log_message(f"[TRAINING ERROR] {message}") # Remove ', error'
        self.accuracy_label.setText("Training Failed!")
        # Stop loading animation in expanded plot if it exists
        if self.expanded_plot_dialog:
             dialog_plot_widget = self.expanded_plot_dialog.findChild(PlotWidget)
             if dialog_plot_widget and hasattr(dialog_plot_widget, 'stop_loading_animation'):
                  dialog_plot_widget.stop_loading_animation()

        # Call cleanup directly to reset UI state etc.
        # REMOVED: self._cleanup_thread() - Cleanup now handled by _on_thread_actually_finished

    def _set_training_ui_enabled(self, enabled: bool):
        """Enables/disables training-related UI elements."""
        self.start_button.setEnabled(enabled)
        self.load_button.setEnabled(enabled) # Disable loading new data during training
        self.dataset_dropdown.setEnabled(enabled) # Disable changing dataset
        self.model_type_combo.setEnabled(enabled) # Disable changing model type

        # Also disable hyperparameter inputs within their groups
        # Simple NN Group
        if hasattr(self, 'layer_sizes_group'):
            for widget in self.layer_sizes_group.findChildren(QWidget):
                 # Check if widget has setEnabled before calling
                 if hasattr(widget, 'setEnabled'):
                     widget.setEnabled(enabled)
        # Common Params Group
        if hasattr(self, 'common_params_group'):
             for widget in self.common_params_group.findChildren(QWidget):
                 if hasattr(widget, 'setEnabled'):
                     widget.setEnabled(enabled)
        # CNN Group (if applicable)
        if hasattr(self, 'cnn_params_group'):
             for widget in self.cnn_params_group.findChildren(QWidget):
                  if hasattr(widget, 'setEnabled'):
                      widget.setEnabled(enabled)

        # Stop button logic: Enable ONLY when training starts, disable otherwise
        self.stop_button.setEnabled(not enabled)

        # --- IMPORTANT: Keep these enabled ---
        self.training_group.setEnabled(enabled) # DO NOT disable the whole group
        self.progress_bar.setEnabled(enabled)
        self.accuracy_label.setEnabled(enabled)
        # ------------------------------------

        # Optionally disable tab switching
        self.tabs.setEnabled(enabled)

    def _load_cifar100_data(self, path, return_shape='image'):
        # *** LOG FUNCTION ENTRY - V2 ***
        self._log_message(f"---***>>> ENTERED _load_cifar100_data <<<***--- Path: {path}, Shape: {return_shape}")
        """Loads CIFAR-100 data from pickled files (train, test, meta). Handles tar.gz extraction.

        Args:
            path (str): Path to the 'cifar-100-python' directory OR the .tar.gz file.
            return_shape (str): 'flattened' or 'image'.

        Returns:
            Tuple containing (X_train, y_train_coarse, X_test, y_test_coarse, fine_label_names, fine_labels_dict)
            or None on failure.
        """
        self._log_message(f"--- Loading CIFAR-100 Dataset (from {path}) ---")
        self._log_message(f"*** Entered _load_cifar100_data function for path: {path} ***") # ADDED ENTRY LOG
        self._log_message(f"--- Loading CIFAR-100 Dataset (requesting shape: {return_shape}) --- ") # Log requested shape

        data_dir = path
        # Handle extraction if path is a tar.gz file
        if path.endswith(".tar.gz") and os.path.exists(path):
            extracted_dir_name = "cifar-100-python" # Standard name
            parent_dir = os.path.dirname(path)
            data_dir = os.path.join(parent_dir, extracted_dir_name)
            self._log_message(f"Target extracted directory: {data_dir}")

            # Define expected key file paths within the extracted directory
            extracted_train_path = os.path.join(data_dir, "train")
            extracted_test_path = os.path.join(data_dir, "test")

            # Check if already extracted and valid
            if not (os.path.isdir(data_dir) and os.path.exists(extracted_train_path) and os.path.exists(extracted_test_path)):
                 self._log_message(f"Attempting to extract {path} to {parent_dir}...")
                 try:
                     with tarfile.open(path, "r:gz") as tar:
                         # Check members before extracting (optional security)
                         # for member in tar.getmembers():
                         #     print(f"  Extracting: {member.name}")
                         tar.extractall(path=parent_dir)
                     self._log_message("Extraction complete.")
                     # Verify extraction success by checking directory and key files again
                     if not (os.path.isdir(data_dir) and os.path.exists(extracted_train_path) and os.path.exists(extracted_test_path)):
                         self._log_message(f"Error: Extraction seemed complete, but key files/dir not found at expected location: {data_dir}")
                         return None
                 except Exception as e:
                     self._log_message(f"Error: Failed to extract CIFAR-100 tar.gz file: {e}")
                     traceback.print_exc()
                     return None
            else:
                 self._log_message(f"CIFAR-100 directory '{data_dir}' already exists with train/test files, skipping extraction.")
        elif not os.path.isdir(data_dir):
             self._log_message(f"Error: CIFAR-100 path is not a directory and not a .tar.gz file: {path}")
             return None

        # --- Proceed with loading from data_dir (which is now guaranteed to be the directory path) --- #
        meta_path = os.path.join(data_dir, "meta")
        train_path = os.path.join(data_dir, "train")
        test_path = os.path.join(data_dir, "test")

        # --- Add Existence Checks --- #
        if not os.path.exists(meta_path):
             self._log_message(f"Error: CIFAR-100 meta file not found at: {meta_path}")
             return None
        if not os.path.exists(train_path):
            self._log_message(f"Error: CIFAR-100 train file not found at: {train_path}")
            return None
        if not os.path.exists(test_path):
            self._log_message(f"Error: CIFAR-100 test file not found at: {test_path}")
            return None
        # --------------------------- #

        try:
            # Load class names (meta file)
            self._log_message(f"  Loading metadata from: {meta_path}")
            fine_label_names = []
            coarse_label_names = []
            meta = None # Init
            with open(meta_path, 'rb') as fo:
                self._log_message(f"    DEBUG: Attempting pickle.load on {meta_path}...")
                meta = pickle.load(fo, encoding='bytes')
                self._log_message(f"    DEBUG: pickle.load completed.")
            # Check for keys before accessing
            if meta and b'fine_label_names' in meta: fine_label_names = [name.decode('utf-8') for name in meta[b'fine_label_names']]
            else: self._log_message(f"    Warning: Key b'fine_label_names' not found in meta file.")
            if meta and b'coarse_label_names' in meta: coarse_label_names = [name.decode('utf-8') for name in meta[b'coarse_label_names']]
            else: self._log_message(f"    Warning: Key b'coarse_label_names' not found in meta file.")
            self._log_message(f"    Loaded {len(fine_label_names)} fine labels and {len(coarse_label_names)} coarse labels.")

            # Load training data (single 'train' file)
            self._log_message(f"  Loading training data from: {train_path}")
            train_batch = None # Init
            with open(train_path, 'rb') as fo:
                self._log_message(f"    DEBUG: Attempting pickle.load on {train_path}...")
                train_batch = pickle.load(fo, encoding='bytes')
                self._log_message(f"    DEBUG: pickle.load completed.")
            # Check keys before access
            if not train_batch or b'data' not in train_batch or b'fine_labels' not in train_batch or b'coarse_labels' not in train_batch:
                 raise KeyError("Missing required keys (data, fine_labels, coarse_labels) in train batch.")
            X_train = train_batch[b'data']
            y_train_fine = np.array(train_batch[b'fine_labels'])
            y_train_coarse = np.array(train_batch[b'coarse_labels'])
            self._log_message(f"    Training data shape: {X_train.shape}")

            # Load test data (single 'test' file)
            self._log_message(f"  Loading test data from: {test_path}")
            test_batch = None # Init
            with open(test_path, 'rb') as fo:
                 self._log_message(f"    DEBUG: Attempting pickle.load on {test_path}...")
                 test_batch = pickle.load(fo, encoding='bytes')
                 self._log_message(f"    DEBUG: pickle.load completed.")
            # Check keys before access
            if not test_batch or b'data' not in test_batch or b'fine_labels' not in test_batch or b'coarse_labels' not in test_batch:
                 raise KeyError("Missing required keys (data, fine_labels, coarse_labels) in test batch.")
            X_test = test_batch[b'data']
            y_test_fine = np.array(test_batch[b'fine_labels'])
            y_test_coarse = np.array(test_batch[b'coarse_labels'])
            self._log_message(f"    Test data shape: {X_test.shape}")

            # --- Reshape data based on return_shape --- #
            # CIFAR format: N x 3072 -> N x 3 x 32 x 32 -> N x 32 x 32 x 3
            def reshape_cifar(data):
                data = data.reshape(-1, 3, 32, 32)
                data = data.transpose(0, 2, 3, 1)
                return data

            if return_shape == 'image':
                 self._log_message("  Reshaping CIFAR-100 data to 'image' format (samples, 32, 32, 3)...")
                 X_train = reshape_cifar(X_train)
                 X_test = reshape_cifar(X_test)
            elif return_shape == 'flattened':
                 self._log_message("  Keeping CIFAR-100 data in 'flattened' format (samples, 3072)...")
                 # Data is already (N, 3072), no reshape needed before splitting
                 # Normalization and split handles transposition if needed by NN
                 pass
            else:
                 self._log_message(f"Warning: Unknown return_shape '{return_shape}' for CIFAR-100. Keeping as (samples, 3072).")

            self._log_message(f"--- CIFAR-100 Loaded Successfully. Final shapes (before combine/split): Train={X_train.shape}, Test={X_test.shape} ---")

            fine_labels_dict = {"train": y_train_fine, "test": y_test_fine}
            # Return components separately, main loader combines/splits
            return X_train, y_train_coarse, X_test, y_test_coarse, fine_label_names, fine_labels_dict

        except FileNotFoundError as e:
             self._log_message(f"Error loading CIFAR-100 component file: {e}")
             return None
        except (KeyError, pickle.UnpicklingError) as e: # Catch pickle errors too
             self._log_message(f"Error accessing data or unpickling CIFAR-100 file: {e}")
             traceback.print_exc() # Log traceback for pickle/key errors
             return None
        except Exception as e:
             self._log_message(f"An unexpected error occurred during CIFAR-100 loading: {e}")
             traceback.print_exc()
             return None

    def _show_about_dialog(self):
        """Shows the About dialog."""
        if AboutDialog:
            dialog = AboutDialog(self) # Parent is the main window
            dialog.exec_() # Show modal dialog
        else:
            self._log_message("Error: AboutDialog could not be loaded.")
            # Optionally show a basic QMessageBox as fallback
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", "Could not load the About dialog.")
