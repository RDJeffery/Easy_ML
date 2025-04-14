# ui/training_worker.py

from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
from typing import Dict
# Assuming neural_net is in the model directory relative to the main script
try:
    # This relative import might work if ui_main is run directly and model is a sibling
    from model import neural_net
except ImportError:
    # Fallback if the structure requires adding the parent directory
    import sys
    import os
    # Add project root to path to find model module
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from model import neural_net
    except ImportError as e:
        print(f"CRITICAL ERROR: Could not import neural_net in TrainingWorker: {e}")
        # Define dummy functions if import fails to prevent NameError later
        class DummyNeuralNet:
            def gradient_descent(*args, **kwargs): return None
        neural_net = DummyNeuralNet()


# Worker class for handling training in a separate thread
class TrainingWorker(QObject):
    # Signals to communicate with the main thread
    progress = pyqtSignal(int)              # Emit only percentage completion (int)
    finished = pyqtSignal(object)           # Use object for the result tuple (parameters_dict, loss_hist, acc_hist) or None
    log_message = pyqtSignal(str)           # For sending log messages from worker

    # Removed error_occurred signal, will emit finished(None) on error instead for simplicity

    def __init__(self, X_train, Y_train, X_dev, Y_dev, initial_parameters: Dict[str, np.ndarray], epochs: int, alpha: float, patience: int, activation_function: str, optimizer_name: str, l2_lambda: float, dropout_keep_prob: float):
        super().__init__()
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_dev = X_dev
        self.Y_dev = Y_dev
        self.parameters = initial_parameters # Store the parameters dictionary
        self.epochs = epochs
        self.alpha = alpha
        self.patience = patience
        self.activation_function = activation_function # Store the selected activation
        self.optimizer_name = optimizer_name # Store the selected optimizer
        self.l2_lambda = l2_lambda # Store L2 lambda
        self.dropout_keep_prob = dropout_keep_prob # Store dropout keep prob
        self._is_running = True # Flag to control the loop

    def stop(self):
        """Signals the worker to stop training gracefully."""
        self.log_message.emit("Stop request received by worker.")
        self._is_running = False
        # Also signal the neural_net training loop if possible (requires modification there)
        neural_net.stop_training_flag = True # Assuming neural_net has a global flag


    def run(self):
        """Runs the gradient descent training."""
        self.log_message.emit("Training worker started.")
        self._is_running = True
        # Assuming neural_net has a stop flag capability (can be removed if not)
        if hasattr(neural_net, 'stop_training_flag'):
            neural_net.stop_training_flag = False # Reset flag before starting

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

            # Call gradient descent from the neural_net module
            results_tuple = neural_net.gradient_descent(
                self.X_train, self.Y_train, self.X_dev, self.Y_dev,
                self.alpha, self.epochs,
                self.parameters, # Pass the parameters dictionary
                hidden_activation=self.activation_function,
                optimizer_name=self.optimizer_name,
                l2_lambda=self.l2_lambda,
                dropout_keep_prob=self.dropout_keep_prob, # Pass dropout keep prob
                progress_callback=progress_callback,
                patience=self.patience
            )

            if not self._is_running:
                 self.log_message.emit("Training stopped early by request.")
                 self.finished.emit(None) # Signal completion without results
            elif results_tuple is None:
                 self.log_message.emit("Training failed or returned None.")
                 self.finished.emit(None) # Signal completion without results
            else:
                self.log_message.emit("Training completed successfully.")
                # results_tuple should be (final_parameters_dict, loss_history, val_acc_history)
                self.finished.emit(results_tuple) # Emit the tuple containing the dict

        except Exception as e:
            self.log_message.emit(f"CRITICAL Error during training: {e}")
            import traceback
            traceback.print_exc() # Print traceback for debugging
            self.finished.emit(None) # Signal completion with None on error
        finally:
            self._is_running = False # Ensure flag is reset
            neural_net.stop_training_flag = False # Reset neural_net flag