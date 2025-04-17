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

        results_to_emit = None # To store results to be emitted

        try:
            # --- Select Training Path Based on Model Type --- #

            if self.model_type == "Simple NN":
                self.log_message.emit("Starting Simple NN training (using gradient_descent)...")
                # Get total epochs for the callback
                total_epochs = self.params.get('epochs', 10)
                def simple_nn_callback(epoch, _total_epochs_ignored, train_loss, val_acc):
                    if not self._is_running:
                        self.log_message.emit("Stop detected during Simple NN training.")
                        return False # Signal gradient_descent to stop
                    # Emit progress (use NaN or None if values are missing)
                    loss_val = float(train_loss) if train_loss is not None else np.nan
                    acc_val = float(val_acc) if val_acc is not None else np.nan
                    # Pass the extracted total_epochs to the progress signal
                    self.progress.emit(epoch, total_epochs, loss_val, acc_val)
                    # --- Add small sleep/yield --- #
                    # QTimer.singleShot(0, lambda: None) # Yield - Keep or remove? Try sleep first.
                    time.sleep(0.01) 
                    # --------------------------- #
                    return True # Continue training

                # Prepare params specific to SimpleNN gradient_descent
                gd_params = {
                    'alpha': self.params.get('learning_rate', 0.01),
                    'epochs': total_epochs, # Use extracted epochs
                    'hidden_activation': self.params.get('activation', 'relu'),
                    'optimizer_name': self.params.get('optimizer', 'adam'),
                    'l2_lambda': self.params.get('l2_lambda', 0.0),
                    # NN uses keep_prob, CNN uses dropout_rate (1-keep_prob)
                    'dropout_keep_prob': 1.0 - self.params.get('dropout_rate', 0.0),
                    'patience': self.params.get('patience', 0),
                    'progress_callback': simple_nn_callback
                }
                # Create params string for logging outside the f-string
                params_log_str = str({k:v for k,v in gd_params.items() if k != 'progress_callback'})
                self.log_message.emit(f"Simple NN Params: {params_log_str}")

                # Call the correct training method ('train')
                results_tuple = self.model.train(
                    self.X_train, self.Y_train, self.X_dev, self.Y_dev,
                    **gd_params
                )

                if results_tuple is not None:
                    # Expecting (final_params, train_loss_hist, val_acc_hist)
                    # Store the full tuple to be emitted
                    results_to_emit = results_tuple
                else:
                    results_to_emit = None # Training might have been stopped

            elif self.model_type == "CNN":
                self.log_message.emit("Starting CNN training (using Keras model.fit)...")

                # Get necessary parameters for Keras training
                keras_epochs = self.params.get('epochs', 10)
                keras_batch_size = self.params.get('batch_size', 32)
                keras_learning_rate = self.params.get('learning_rate', 0.001)
                keras_patience = self.params.get('patience', 0)
                # Get the new parameters
                keras_use_batch_norm = self.params.get('use_batch_norm', False)
                keras_use_data_augmentation = self.params.get('use_data_augmentation', False)
                keras_use_lr_scheduler = self.params.get('use_lr_scheduler', False)
                
                # Check if the model needs to be built with the new parameters
                if not self.model.model:  # If model is not built yet
                    self.log_message.emit(f"Building CNN model with Batch Norm: {keras_use_batch_norm}, Data Aug: {keras_use_data_augmentation}")
                    # These parameters only take effect during build_model, so we need to set them before building
                    self.model.use_batch_norm = keras_use_batch_norm
                    self.model.use_data_augmentation = keras_use_data_augmentation
                    self.model.build_model()
                else:
                    # Model is already built
                    current_bn = getattr(self.model, 'use_batch_norm', False)
                    current_da = getattr(self.model, 'use_data_augmentation', False)
                    if current_bn != keras_use_batch_norm or current_da != keras_use_data_augmentation:
                        self.log_message.emit(f"Warning: Model architecture parameters changed but model already built. Rebuilding model.")
                        self.log_message.emit(f"  Batch Norm: {current_bn} -> {keras_use_batch_norm}, Data Aug: {current_da} -> {keras_use_data_augmentation}")
                        # Update parameters and rebuild model
                        self.model.use_batch_norm = keras_use_batch_norm
                        self.model.use_data_augmentation = keras_use_data_augmentation
                        self.model.build_model()

                # Callback to emit progress (simplified version)
                class ProgressCallback(tf.keras.callbacks.Callback):
                    def __init__(self, worker):
                        super().__init__()
                        self.worker = worker

                    def on_epoch_end(self, epoch, logs=None):
                        if not self.worker._is_running:
                            self.model.stop_training = True
                            self.worker.log_message.emit("Stop detected during Keras epoch.")
                            return

                        logs = logs or {}
                        loss = logs.get('loss', np.nan)
                        # Handle potential key differences for validation accuracy
                        acc_key = 'val_accuracy' if 'val_accuracy' in logs else 'val_acc'
                        val_acc = logs.get(acc_key, np.nan)
                        # Epoch is 0-based from Keras, emit 1-based
                        self.worker.progress.emit(epoch + 1, self.worker.params.get('epochs', 10), loss, val_acc)
                        QTimer.singleShot(0, lambda: None) # Yield

                # Prepare params for Keras model.train method
                # Pass the actual callback functions from the worker
                keras_params = {
                    'epochs': keras_epochs,
                    'batch_size': keras_batch_size,
                    'learning_rate': keras_learning_rate,
                    'patience': keras_patience,
                    'use_lr_scheduler': keras_use_lr_scheduler,  # Pass the new LR scheduler parameter
                    'log_callback': self.log_message.emit,       # Pass worker signal emit
                    'progress_callback': self.progress.emit    # Pass worker signal emit
                }
                params_log_str = str(keras_params) # Log the prepared params
                self.log_message.emit(f"Keras Params: {params_log_str}")

                # Ensure the model has the train method
                if not hasattr(self.model, 'train'):
                    raise AttributeError("CNN model object does not have 'train' method.")

                # Call Keras training using the **kwargs expansion
                keras_history_obj = self.model.train(
                    self.X_train, # Or X_train_model etc.
                    self.Y_train, # Or Y_train_model etc.
                    # REMOVED self.X_dev and self.Y_dev
                    **keras_params  # Pass parameters as keyword arguments
                )

                # --- CORRECTED RESULT HANDLING --- #
                if keras_history_obj is not None:
                    # For Keras, the result *is* the history dictionary
                    self.log_message.emit(f"Keras training successful. History keys: {list(keras_history_obj.keys())}")
                    results_to_emit = keras_history_obj # Assign the dict
                else:
                    self.log_message.emit("Keras training finished but returned None (or failed).")
                    results_to_emit = None
                # --------------------------------- #

            else:
                raise ValueError(f"Unknown model type encountered: {self.model_type}")

            # --- Training Loop Finished --- #
            if not self._is_running:
                 self.log_message.emit("Training stopped early by request.")
                 self.finished.emit(None) # Signal completion without results
            elif results_to_emit is None:
                 # Log message handled within the if/else block above
                 self.finished.emit(None)
            else:
                self.log_message.emit("Training completed successfully.")
                self.finished.emit(results_to_emit) # Emit the results (tuple for SimpleNN, dict for CNN)

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