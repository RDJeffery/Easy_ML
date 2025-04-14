# ui/training_worker.py

from PyQt5.QtCore import QObject, pyqtSignal, QTimer
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import time # For potential sleep/yield
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

# --- TensorFlow/Keras Import --- #
# Add this to handle Keras History object if needed
try:
    import tensorflow as tf
    # Check if History is available
    if hasattr(tf.keras.callbacks, 'History'):
        KerasHistory = tf.keras.callbacks.History
    else:
        KerasHistory = None # Fallback
except ImportError:
    tf = None
    KerasHistory = None
    print("TensorFlow not found in TrainingWorker. CNN training will not work.")
# ----------------------------- #


# Worker class for handling training in a separate thread
class TrainingWorker(QObject):
    # Signals to communicate with the main thread
    # progress: epoch, total_epochs, loss (optional), val_acc (optional)
    progress = pyqtSignal(int, int, float, float)
    # finished: history dictionary or None
    finished = pyqtSignal(object)
    # log_message: string message
    log_message = pyqtSignal(str)
    error = pyqtSignal(str) # Separate signal for errors

    def __init__(self, training_params: Dict[str, Any]):
        super().__init__()
        self.params = training_params
        self._is_running = True # Flag to control the loop/stop request
        # Extract key parameters for easier access
        self.model = self.params.get('model')
        self.model_type = self.params.get('model_type')
        self.X_train = self.params.get('X_train')
        self.Y_train = self.params.get('Y_train')
        self.X_dev = self.params.get('X_dev')
        self.Y_dev = self.params.get('Y_dev')
        self.epochs = self.params.get('epochs', 10)

    def stop(self):
        """Signals the worker to stop training gracefully."""
        self.log_message.emit("Stop request received by worker.")
        self._is_running = False
        # For Keras models, set the stop_training flag (checked by callbacks/fit)
        if self.model_type == "CNN" and self.model and hasattr(self.model, 'model') and self.model.model:
             self.log_message.emit("Setting Keras model.stop_training = True")
             self.model.model.stop_training = True
        # For SimpleNN, rely on the flag being checked in its loop (if implemented)
        elif hasattr(self.model, 'stop_training_flag'):
            self.model.stop_training_flag = True


    def run(self):
        """Runs the training loop based on the model type."""
        self.log_message.emit(f"Training worker started for model type: {self.model_type}")
        self._is_running = True
        # Reset stop flags if they exist
        if hasattr(self.model, 'stop_training_flag'):
            self.model.stop_training_flag = False
        if self.model_type == "CNN" and self.model and hasattr(self.model, 'model') and self.model.model:
             self.model.model.stop_training = False

        if not self.model or not self.model_type:
            self.error.emit("Worker Error: Model or model_type missing in parameters.")
            self.finished.emit(None)
            return

        history_data = None # To store final history dictionary

        try:
            # --- Select Training Path Based on Model Type --- #

            if self.model_type == "Simple NN":
                self.log_message.emit("Starting Simple NN training (using gradient_descent)...")
                # Define the callback for Simple NN
                def simple_nn_callback(epoch, total_epochs, train_loss, val_acc):
                    if not self._is_running:
                        self.log_message.emit("Stop detected during Simple NN training.")
                        return False # Signal gradient_descent to stop
                    # Emit progress (use NaN or None if values are missing)
                    loss_val = float(train_loss) if train_loss is not None else np.nan
                    acc_val = float(val_acc) if val_acc is not None else np.nan
                    self.progress.emit(epoch, total_epochs, loss_val, acc_val)
                    QTimer.singleShot(0, lambda: None) # Yield to event loop
                    return True # Continue training

                # Prepare params specific to SimpleNN gradient_descent
                gd_params = {
                    'alpha': self.params.get('learning_rate', 0.01),
                    'epochs': self.epochs,
                    'batch_size': self.params.get('batch_size', 64),
                    'activation': self.params.get('activation', 'relu'),
                    'optimizer_name': self.params.get('optimizer', 'adam'),
                    'l2_lambda': self.params.get('l2_lambda', 0.0),
                    'dropout_keep_prob': self.params.get('dropout_keep_prob', 1.0),
                    'patience': self.params.get('patience', 0),
                    'progress_callback': simple_nn_callback
                }
                self.log_message.emit(f"Simple NN Params: { {k:v for k,v in gd_params.items() if k != 'progress_callback'} }")

                # Ensure the model has the gradient_descent method
                if not hasattr(self.model, 'gradient_descent'):
                     raise AttributeError("Simple NN model object does not have 'gradient_descent' method.")

                # Call gradient descent
                results_tuple = self.model.gradient_descent(
                    self.X_train, self.Y_train, self.X_dev, self.Y_dev,
                    **gd_params
                )

                if results_tuple is not None:
                    # Expecting (final_params, train_loss_hist, val_acc_hist)
                    _, train_loss_hist, val_acc_hist = results_tuple
                    # Construct a history dict compatible with Keras format if possible
                    history_data = {
                        'loss': train_loss_hist,
                        'val_accuracy': val_acc_hist # Assuming accuracy is returned
                        # Add other metrics if available
                    }
                # Final parameters are implicitly stored within self.model object now

            elif self.model_type == "CNN":
                self.log_message.emit("Starting CNN training (using Keras model.fit)...")

                # Keras fit blocks, so we get history at the end.
                # We cannot easily emit progress per epoch without callbacks.
                # TODO: Implement Keras callbacks for better progress reporting and stopping.

                # Prepare params for Keras model.train method
                keras_params = {
                    'epochs': self.epochs,
                    'batch_size': self.params.get('batch_size', 32),
                    'learning_rate': self.params.get('learning_rate', 0.001)
                    # Keras handles optimizer/loss internally during compile (called in model.train)
                }
                self.log_message.emit(f"Keras Params: {keras_params}")

                # Ensure the model has the train method
                if not hasattr(self.model, 'train'):
                    raise AttributeError("CNN model object does not have 'train' method.")

                # Call Keras training
                keras_history = self.model.train(
                    self.X_train, self.Y_train, self.X_dev, self.Y_dev,
                    **keras_params
                )

                if keras_history and hasattr(keras_history, 'history'):
                    history_data = keras_history.history
                    # Emit progress based on the final history data
                    if 'loss' in history_data and 'val_accuracy' in history_data:
                        num_epochs_trained = len(history_data['loss'])
                        for i in range(num_epochs_trained):
                            if not self._is_running: break # Check stop flag between emits
                            loss_val = history_data['loss'][i]
                            acc_val = history_data['val_accuracy'][i]
                            self.progress.emit(i, self.epochs, loss_val, acc_val)
                            QTimer.singleShot(0, lambda: None) # Yield
                            time.sleep(0.01) # Small sleep to allow UI updates
                else:
                    self.log_message.emit("Keras training finished but returned no history.")

            else:
                raise ValueError(f"Unknown model type encountered: {self.model_type}")

            # --- Training Loop Finished --- #
            if not self._is_running:
                 self.log_message.emit("Training stopped early by request.")
                 self.finished.emit(None) # Signal completion without results
            elif history_data is None:
                 self.log_message.emit("Training finished, but no history data was generated.")
                 self.finished.emit(None)
            else:
                self.log_message.emit("Training completed successfully.")
                self.finished.emit(history_data) # Emit the history dictionary

        except Exception as e:
            error_msg = f"CRITICAL Error during training: {e}"
            self.log_message.emit(error_msg)
            self.error.emit(error_msg) # Emit specific error signal
            import traceback
            traceback.print_exc() # Print traceback for debugging
            self.finished.emit(None) # Signal completion with None on error
        finally:
            self._is_running = False # Ensure flag is reset
            self.log_message.emit("Training worker run method finished.")
            # Reset stop flags again just in case
            if hasattr(self.model, 'stop_training_flag'):
                self.model.stop_training_flag = False
            if self.model_type == "CNN" and self.model and hasattr(self.model, 'model') and self.model.model:
                 self.model.model.stop_training = False