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
                # Store the list of files found, sorted for consistency
                sorted_npy_files = sorted(npy_files)
                self.datasets_info["QuickDraw (Select Categories)"] = {
                    "type": "quickdraw",
                    "files": sorted_npy_files # Store the list of full paths
                }
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

        # --- Show/Hide QuickDraw Selection Widgets --- #
        is_quickdraw = selected_text == "QuickDraw (Select Categories)"
        self.quickdraw_select_label.setVisible(is_quickdraw)
        self.quickdraw_list_widget.setVisible(is_quickdraw)
        self.quickdraw_random_label.setVisible(is_quickdraw)
        self.quickdraw_random_count.setVisible(is_quickdraw)

        if is_quickdraw and is_valid_selection:
            # Populate the list widget if QuickDraw is selected
            self.quickdraw_list_widget.clear()
            quickdraw_info = self.datasets_info[selected_text]
            npy_files = quickdraw_info.get("files", [])
            if npy_files:
                # Extract category name from filename (e.g., "the_mona_lisa.npy" -> "the mona lisa")
                category_names = [os.path.splitext(os.path.basename(f))[0].replace("_", " ") for f in npy_files]
                self.quickdraw_list_widget.addItems(category_names)
                self._log_message(f"Populated QuickDraw list with {len(category_names)} categories.")
                # Optionally set a default selection (e.g., first 5 items)
                # for i in range(min(5, len(category_names))):
                #     self.quickdraw_list_widget.item(i).setSelected(True)
            else:
                self._log_message("Warning: QuickDraw selected, but no NPY files found in datasets_info.")
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
        if self.current_model is None:
            self._log_message("No model loaded or trained yet.")
            return

        # Get the drawing as a NumPy array (e.g., 28x28)
        drawing_array = self.drawing_canvas.getDrawingArray()
        if drawing_array is None:
            self._log_message("Canvas is empty or drawing is invalid.")
            # --- Clear previous results --- #
            self.image_preview_label.clear()
            self.image_preview_label.setText("(Canvas Empty)") # Give feedback
            if hasattr(self, 'probability_graph') and self.probability_graph:
                self.probability_graph.clear_graph()
            self.prediction_result_label.setText("Prediction: N/A")
            # ------------------------------ #
            return

        # --- Display Drawing Preview --- #
        preview_pixmap = self.drawing_canvas.getPreviewPixmap()
        if preview_pixmap:
            # Scale pixmap to fit the preview label while maintaining aspect ratio
            scaled_pixmap = preview_pixmap.scaled(self.image_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_preview_label.setPixmap(scaled_pixmap)
        else:
            self.image_preview_label.setText("(Preview Error)") # Show error if pixmap fails
        # ------------------------------ #

        # --- Preprocess and reshape based on model type --- #
        model_type = self.current_model_type # e.g., "Simple NN" or "CNN"
        input_data = None

        try:
            if model_type == "Simple NN":
                self._log_message("Preprocessing drawing for Simple NN...")
                # Expected shape: (features, 1)
                # Assuming the drawing array is 2D (H, W)
                if drawing_array.ndim != 2:
                    raise ValueError(f"Unexpected drawing array dimension: {drawing_array.ndim}")
                num_features = drawing_array.shape[0] * drawing_array.shape[1]
                # Normalize (0-1 range), flatten, and reshape
                input_data = drawing_array.flatten().reshape(num_features, 1)
                # Normalize *after* flattening
                input_data = input_data / 255.0
                self._log_message(f"Simple NN input shape: {input_data.shape}")

            elif model_type == "CNN":
                self._log_message("Preprocessing drawing for CNN...")
                # Expected shape: (1, H, W, C)
                # Determine H, W, C. Try from loaded data, fallback to defaults.
                H, W, C = 28, 28, 1 # Default (e.g., MNIST)

                # Try to get expected shape from the model itself if possible
                if self.current_model and hasattr(self.current_model, 'input_shape') and self.current_model.input_shape:
                    try:
                        # Keras input_shape is often (H, W, C) or (None, H, W, C)
                        if len(self.current_model.input_shape) == 3:
                            H, W, C = self.current_model.input_shape
                        elif len(self.current_model.input_shape) == 4:
                            H, W, C = self.current_model.input_shape[1:]
                        self._log_message(f"Using model input shape for CNN: {(H, W, C)}")
                    except Exception as shape_err:
                        self._log_message(f"Warning: Could not parse model input shape {self.current_model.input_shape}. Error: {shape_err}. Falling back to default.")
                # Fallback: Infer from loaded data shape (less reliable)
                elif hasattr(self, 'X_train') and self.X_train is not None and self.X_train.ndim == 4:
                    if self.X_train.shape[0] > 0:
                        H, W, C = self.X_train.shape[1:]
                        self._log_message(f"Using loaded data shape for CNN: {(H, W, C)}")
                else:
                    # No data loaded, use default
                     self._log_message(f"Warning: No data or model shape available, using default {H}x{W}x{C} for CNN input.")

                # Ensure drawing array is 2D (H, W)
                if drawing_array.ndim != 2:
                    raise ValueError(f"Unexpected drawing array dimension: {drawing_array.ndim}")

                # Resize drawing if it doesn't match the target H, W
                if drawing_array.shape != (H, W):
                    self._log_message(f"Warning: Drawing array shape {drawing_array.shape} differs from expected CNN input {(H, W)}. Resizing...")
                    # Use PIL for resizing to maintain consistency with file loading
                    img_pil = Image.fromarray(drawing_array.astype(np.uint8)) # Convert to PIL Image
                    img_resized_pil = img_pil.resize((W, H), Image.Resampling.LANCZOS) # Resize (PIL uses W, H order)
                    drawing_array = np.array(img_resized_pil) # Convert back to numpy array
                    self._log_message(f"Resized drawing array to: {drawing_array.shape}")

                # Normalize (0-1 range) and reshape to (1, H, W, C)
                input_data = drawing_array.reshape(1, H, W, C)
                # Ensure dtype is float32 for TensorFlow
                input_data = input_data.astype(np.float32)
                # Normalize *after* reshaping
                input_data = input_data / 255.0
                self._log_message(f"CNN input shape: {input_data.shape}")

            else:
                self._log_message(f"Prediction not implemented for model type: {model_type}")
                return

            # --- Perform Prediction --- #
            self._log_message(f"Predicting using {model_type}...")
            prediction = self.current_model.predict(input_data)
            # print(f"Raw prediction output: {prediction}") # Debug

            # --- Process & Display Results --- #
            if prediction is None:
                 self._log_message("Prediction failed. Model returned None.")
                 return

            # Output shape might vary (e.g., (num_classes, 1) for NN, (1, num_classes) for TF Keras)
            # We need probabilities per class
            probabilities = prediction.flatten() # Make it 1D

            if probabilities.size == self.num_classes:
                predicted_class_index = np.argmax(probabilities)
                confidence = probabilities[predicted_class_index] * 100 # As percentage

                # Get class label (use index if names not available)
                if self.class_names and 0 <= predicted_class_index < len(self.class_names):
                    predicted_label = self.class_names[predicted_class_index]
                else:
                    predicted_label = f"Class {predicted_class_index}"

                self.prediction_result_label.setText(f"Prediction: {predicted_label} ({confidence:.1f}%)")
                self._log_message(f"Drawing Prediction: {predicted_label} (Confidence: {confidence:.2f}%) Index: {predicted_class_index}")

                # Update probability bar graph
                if hasattr(self, 'probability_graph'):
                     # Call the correct method: set_probabilities
                     # Pass predicted_label as the second argument, class_names as third
                     self.probability_graph.set_probabilities(probabilities, predicted_label, self.class_names)
            else:
                self._log_message(f"Prediction output size ({probabilities.size}) does not match number of classes ({self.num_classes}).")
                self.prediction_result_label.setText("Prediction Error")

        except ValueError as e:
             self._log_message(f"Error preprocessing drawing for prediction: {e}")
             self.prediction_result_label.setText("Preprocessing Error")
        except Exception as e:
            self._log_message(f"An error occurred during drawing prediction: {e}")
            self.prediction_result_label.setText("Prediction Failed")
            import traceback
            traceback.print_exc()

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
                # Get model expected shape
                model_type = self.current_model_type
                H, W, C = 28, 28, 1  # Default
                if model_type == "CNN":
                    if self.current_model and hasattr(self.current_model, 'input_shape') and self.current_model.input_shape:
                        try:
                            if len(self.current_model.input_shape) == 3: H, W, C = self.current_model.input_shape
                            elif len(self.current_model.input_shape) == 4: H, W, C = self.current_model.input_shape[1:]
                            self._log_message(f"Using model input shape for file prediction: {(H, W, C)}")
                        except Exception:
                            self._log_message(f"Could not parse model input shape {self.current_model.input_shape}, using default {H}x{W}x{C}.")
                    else:
                        self._log_message(f"No model shape available, using default {H}x{W}x{C} for file prediction.")
                # Preprocess image: Convert color, resize
                target_mode = "L" if C == 1 else "RGB"  # Grayscale or Color
                self._log_message(f"Preprocessing image file to {target_mode} and size {(W, H)}...")
                img_processed = img.convert(target_mode).resize((W, H), Image.Resampling.LANCZOS)

                # Convert to numpy array
                img_array_raw = np.array(img_processed)

                # Prepare data for the specific model
                input_data = None
                if model_type == "Simple NN":
                    num_features = H * W * C
                    input_data = img_array_raw.flatten().reshape(num_features, 1)
                    input_data = input_data / 255.0  # Normalize
                    self._log_message(f"Prepared Simple NN input data, shape: {input_data.shape}")
                elif model_type == "CNN":
                    input_data = img_array_raw.reshape(1, H, W, C)
                    input_data = input_data.astype(np.float32)  # Ensure float32 for TF
                    input_data = input_data / 255.0  # Normalize
                    self._log_message(f"Prepared CNN input data, shape: {input_data.shape}")
                else:
                    self._log_message(f"ERROR: Unknown model type '{model_type}' for prediction.")
                    return

                if input_data is None:
                    self._log_message("ERROR: Failed to prepare input data from file.")
                    return

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

                probabilities = self.current_model.predict(input_data)
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
            QApplication.processEvents()

    # Add flush method (required for stdout redirection)
    def flush(self):
        pass

    # --- Main Action Methods ---

    # Add methods for dataset loading buttons
    def load_selected_dataset(self):
        """Loads the dataset selected in the dropdown."""
        dataset_key = self.dataset_dropdown.currentData()
        if not dataset_key or dataset_key not in self.datasets_info:
            self._log_message("No valid dataset selected.")
            return

        dataset_info = self.datasets_info[dataset_key]
        dtype = dataset_info.get('type') # Use .get for safety

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
            if dtype == 'quickdraw':
                # --- QuickDraw Selection Logic --- #
                all_npy_files = dataset_info.get('files', [])
                if not all_npy_files:
                    raise ValueError("No QuickDraw NPY files found in dataset info.")

                selected_files_for_map = []
                random_count = self.quickdraw_random_count.value()

                if random_count > 0:
                    # User specified a random count
                    if random_count > len(all_npy_files):
                        self._log_message(f"Warning: Requested {random_count} random categories, but only {len(all_npy_files)} available. Loading all.")
                        selected_files_for_map = all_npy_files
                    else:
                        selected_files_for_map = random.sample(all_npy_files, random_count)
                        self._log_message(f"Randomly selected {random_count} QuickDraw categories.")
                else:
                    # User selected from the list
                    selected_items = self.quickdraw_list_widget.selectedItems()
                    if not selected_items:
                        raise ValueError("QuickDraw selected, but no categories chosen from the list (and random count is 0).")
                    # Map selected text back to file paths (assuming order is preserved)
                    selected_names = {item.text() for item in selected_items}
                    all_category_names = [os.path.splitext(os.path.basename(f))[0].replace("_", " ") for f in all_npy_files]
                    for i, name in enumerate(all_category_names):
                        if name in selected_names:
                            selected_files_for_map.append(all_npy_files[i])
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

            elif dtype == 'emoji':
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

            elif dtype == 'cifar10':
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

            elif dtype == 'csv':
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
            # Removed the final else block as unknown types handled earlier

            # Store loaded data
            self.X_train, self.Y_train = X_train, Y_train
            self.X_dev, self.Y_dev = X_dev, Y_dev # Assign validation data too
            self.num_classes = num_classes

            # For debugging/inspection: Store raw flattened training data if available
            self.raw_X_train_flattened = raw_X_train

            if X_train is None or Y_train is None:
                 raise ValueError("Data loading returned None. Check logs.")

            # Update UI state after successful load
            self._post_load_update(dataset_key)

        except FileNotFoundError as e:
            self._log_message(f"Error: Dataset file not found: {e}")
        except ValueError as e:
             self._log_message(f"Error processing dataset: {e}")
        except Exception as e:
            self._log_message(f"An unexpected error occurred loading data: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging

    def _post_load_update(self, dataset_name):
        # Enable the training group now that data is loaded
        self.training_group.setEnabled(True)
        self.training_group.setTitle(f"Training Controls ({dataset_name})")
        self.start_button.setEnabled(True)
        # Keep stop button disabled until training actually starts
        # self.stop_button.setEnabled(False)
        # self.save_button.setEnabled(False) # Keep save disabled until training finishes

        # Update model parameter display based on loaded data
        input_features = 0
        output_classes = self.num_classes
        if self.X_train is not None:
            if self.current_model_type == 'Simple NN':
                # NN expects (features, samples)
                input_features = self.X_train.shape[0]
            elif self.current_model_type == 'CNN':
                # CNN expects (samples, H, W, C) -> We don't directly set features from here
                input_features = -1 # Indicate shape is H, W, C
                pass
            else:
                 input_features = self.X_train.shape[0] # Fallback? Or handle error

            self._log_message(f"Dataset loaded: '{dataset_name}'")
            self._log_message(f"  Training samples: {self.X_train.shape[1] if self.X_train.ndim == 2 else self.X_train.shape[0]}")
            self._log_message(f"  Validation samples: {self.X_dev.shape[1] if self.X_dev.ndim == 2 else self.X_dev.shape[0]}")
            if input_features > 0:
                 self._log_message(f"  Input features: {input_features}")
            elif input_features == -1:
                 h,w,c = self.X_train.shape[1:]
                 self._log_message(f"  Input shape (H, W, C): ({h}, {w}, {c})")
            self._log_message(f"  Output classes: {output_classes}")

            # Enable inference buttons IF a model is also loaded/trained
            if self.current_model is not None:
                self.predict_drawing_button.setEnabled(True)
                self.predict_file_button.setEnabled(True)
            # Enable load weights button now that we know num_classes etc.
            self.load_button.setEnabled(True)
        else:
            self._log_message("Data loading failed. Training controls remain disabled.")
            # Keep buttons disabled
            self.training_group.setEnabled(False)
            self.start_button.setEnabled(False)
            self.load_button.setEnabled(False)
            self.save_button.setEnabled(False)
            self.predict_drawing_button.setEnabled(False)
            self.predict_file_button.setEnabled(False)

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
                    self.class_names = loaded_class_names # Okay if None
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
                                self.predict_drawing_button.setEnabled(True)
                                self.predict_file_button.setEnabled(True)
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
                         self.predict_drawing_button.setEnabled(True)
                         self.predict_file_button.setEnabled(True)
                    else:
                         
                         self._log_message("CNN Model object not found or has no load_weights method.")

                else:
                    self._log_message(f"Loading not implemented for model type: {self.current_model_type}")

            except Exception as e:
                self._log_message(f"Error loading weights: {e}")
                import traceback
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
        self.progress_bar.setFormat(f"Epoch {epoch}/{total_epochs} - Loss: {loss:.4f} - Val Acc: {val_acc:.3f} ({percentage}%)")

        # Append data to history lists
        self.train_loss_history.append(loss)
        self.val_accuracy_history.append(val_acc)

        # Update embedded plot (if it exists - though it was removed)
        # if hasattr(self, 'plot_widget') and self.plot_widget:
        #      self.plot_widget.update_plot(self.train_loss_history, self.val_accuracy_history)

        # Update expanded plot if it exists and is visible
        if self.expanded_plot_dialog and self.expanded_plot_dialog.isVisible():
            dialog_plot_widget = self.expanded_plot_dialog.findChild(PlotWidget)
            if dialog_plot_widget:
                try:
                    dialog_plot_widget.update_plot(self.train_loss_history, self.val_accuracy_history)
                except Exception as e:
                    self._log_message(f"Error updating expanded plot during progress: {e}")

        QApplication.processEvents() # Keep UI responsive during updates

    def training_finished(self, results: Optional[Any]): # Changed type hint to Any
        """Slot called when the training worker finishes successfully."""
        self._log_message("Training worker finished successfully. (Inside training_finished slot)")

        # --- Restore Original Code --- #
        train_loss_history = []
        val_accuracy_history = []
        processed_ok = False

        if results is not None:
            if self.current_model_type == "Simple NN":
                try:
                    # Expecting tuple: (final_params, loss_hist, acc_hist)
                    model_params, train_loss_history, val_accuracy_history = results
                    self.model_params = model_params # Store final trained parameters (for saving Simple NN)
                    # Re-instantiate the model object with the trained parameters
                    if self.model_layer_dims: # Ensure layer dimensions are available
                        self.current_model = SimpleNeuralNetwork(self.model_layer_dims)
                        self.current_model.load_params(self.model_params)
                        self._log_message("Updated self.current_model with trained Simple NN parameters.")
                    else:
                        self._log_message("Error: Could not update model instance, layer_dims missing.")
                        self.current_model = None # Ensure model is None if update failed
                    processed_ok = True
                except (TypeError, ValueError) as e:
                    self._log_message(f"Error unpacking Simple NN results tuple: {e}. Results: {results}")

            elif self.current_model_type == "CNN":
                if isinstance(results, dict):
                    # Expecting Keras history dictionary
                    train_loss_history = results.get('loss', [])
                    # Handle potential key diff: 'val_accuracy' or 'val_acc'
                    val_accuracy_history = results.get('val_accuracy', results.get('val_acc', []))
                    # Params are already updated within self.current_model (the CNNModel instance)
                    processed_ok = True
                else:
                    self._log_message(f"Error: Expected a dictionary for CNN results, got {type(results)}")
                    processed_ok = False
            else:
                self._log_message(f"Warning: Unknown model type '{self.current_model_type}' in training_finished.")
        # -------------------------------------------- #

        if processed_ok:
            self.train_loss_history = train_loss_history
            self.val_accuracy_history = val_accuracy_history

            final_val_acc = val_accuracy_history[-1] if val_accuracy_history else float('nan')
            self.accuracy_label.setText(f"Final Validation Accuracy: {final_val_acc:.4f}")
            self._log_message(f"Training complete. Final Validation Accuracy: {final_val_acc:.4f}")

            # Update plots one last time
            if self.expanded_plot_dialog:
                dialog_plot_widget = self.expanded_plot_dialog.findChild(PlotWidget)
                if dialog_plot_widget:
                    dialog_plot_widget.update_plot(self.train_loss_history, self.val_accuracy_history)
                    if hasattr(dialog_plot_widget, 'stop_loading_animation'):
                        dialog_plot_widget.stop_loading_animation()

            # Enable relevant buttons post-training
            # Check if either model object or params (for SimpleNN) exist
            if self.current_model or self.model_params:
                 self.save_button.setEnabled(True)
                 self.predict_drawing_button.setEnabled(True)
                 self.predict_file_button.setEnabled(True)
            else:
                 self._log_message("Post-training: Model/Params not available, keeping Save/Predict disabled.")

        else: # Handle cases where results were None or processing failed
            if results is None:
                self._log_message("Training finished, but no results were returned (possibly stopped early).")
            else:
                 self._log_message("Training finished, but results could not be processed.")
            self.accuracy_label.setText("Training finished (Check Logs).")
            # Ensure save/predict buttons are disabled if results failed
            self.save_button.setEnabled(False)
            self.predict_drawing_button.setEnabled(False)
            self.predict_file_button.setEnabled(False)
        # --- End of Restored Code ---

        # Call cleanup directly to reset UI state etc.
        # REMOVED: self._cleanup_thread() - Cleanup now handled by _on_thread_actually_finished

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
        self.stop_button.setEnabled(enabled)
        self.training_group.setEnabled(enabled)
        self.tabs.setEnabled(enabled)
