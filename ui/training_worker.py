# ui/training_worker.py

from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
from typing import Dict, Any
# Import SimpleNeuralNetwork directly
try:
    from model.neural_net import SimpleNeuralNetwork
except ImportError:
    # Fallback if the structure requires adding the parent directory
    import sys
    import os
    # Add project root to path to find model module
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from model.neural_net import SimpleNeuralNetwork
    except ImportError as e:
        print(f"CRITICAL ERROR: Could not import SimpleNeuralNetwork in TrainingWorker: {e}")
        # Define dummy class if import fails
        class SimpleNeuralNetwork:
            def __init__(self, *args, **kwargs): pass
            def train(self, *args, **kwargs): return None

# Worker class for handling training in a separate thread
class TrainingWorker(QObject):
    # Signals to communicate with the main thread
    progress = pyqtSignal(int)              # Emit only percentage completion (int)
    finished = pyqtSignal(object)           # Use object for the result tuple (parameters_dict, loss_hist, acc_hist) or None
    log_message = pyqtSignal(str)           # For sending log messages from worker

    # Removed error_occurred signal, will emit finished(None) on error instead for simplicity

    def __init__(self, model: SimpleNeuralNetwork, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray, training_params: Dict[str, Any]):
        super().__init__()
        self.model = model # Store the model object instance
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_dev = X_dev
        self.Y_dev = Y_dev
        self.training_params = training_params # Store dict of params like alpha, epochs, etc.
        self._is_running = True # Flag to control the loop

    def stop(self):
        """Signals the worker to stop training gracefully."""
        self.log_message.emit("Stop request received by worker.")
        self._is_running = False
        # Signal the model's training loop if it supports it
        if hasattr(self.model, 'stop_training_flag'):
            self.model.stop_training_flag = True

    def run(self):
        """Runs the gradient descent training."""
        self.log_message.emit("Training worker started.")
        self._is_running = True
        # Reset stop flag on the model if possible
        if hasattr(self.model, 'stop_training_flag'):
            self.model.stop_training_flag = False

        try:
            # Define the callback function to emit progress and check for stop request
            def progress_callback(iteration, total_iterations, train_loss, val_acc):
                if not self._is_running:
                    self.log_message.emit("Stop detected during training iteration.")
                    return False # Signal gradient_descent to stop

                # Calculate percentage completion based on iterations
                # (Note: train_loss and val_acc are available here if needed for more complex progress reporting)
                percent_complete = int((iteration / total_iterations) * 100) if total_iterations > 0 else 0
                self.progress.emit(percent_complete)
                return True # Continue training

            # Pass progress callback into training_params
            self.training_params['progress_callback'] = progress_callback

            # Call the model's train method
            results_tuple = self.model.train(
                self.X_train, self.Y_train, self.X_dev, self.Y_dev,
                **self.training_params
            )

            if not self._is_running:
                 self.log_message.emit("Training stopped early by request.")
                 self.finished.emit(None) # Signal completion without results
            elif results_tuple is None:
                 self.log_message.emit("Training failed or model returned None.")
                 self.finished.emit(None) # Signal completion without results
            else:
                self.log_message.emit("Training completed successfully.")
                # Assuming results_tuple[0] contains the updated parameters/state
                # This might need adjustment based on the actual interface implementation
                self.finished.emit(results_tuple) # Emit the tuple containing the dict

        except Exception as e:
            self.log_message.emit(f"CRITICAL Error during training: {e}")
            import traceback
            traceback.print_exc() # Print traceback for debugging
            self.finished.emit(None) # Signal completion with None on error
        finally:
            self._is_running = False # Ensure flag is reset
            # Reset stop flag on the model if possible
            if hasattr(self.model, 'stop_training_flag'):
                self.model.stop_training_flag = False