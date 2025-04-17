import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import numpy as np
import traceback
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import sys
# --- ADD TIME IMPORT --- #
import time
# ----------------------- #

# --- DEFINE Custom Callback --- #
class TrainingProgressCallback(Callback):
    def __init__(self, epochs, log_callback=None, progress_callback=None):
        super().__init__()
        self.epochs = epochs
        # Use default print if callbacks are None
        self.log_callback = log_callback if log_callback else lambda msg: print(msg, file=sys.stderr)
        self.progress_callback = progress_callback # Can be None
        self.batch_count = 0
        self.samples_per_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.batch_count = 0
        # Estimate samples per epoch (might not be exact with validation split)
        # --- Safely calculate samples_per_epoch --- #
        steps = self.params.get('steps')
        batch_size = self.params.get('batch_size')
        samples = self.params.get('samples')
        if steps is not None and batch_size is not None:
            self.samples_per_epoch = steps * batch_size
        elif samples is not None:
             self.samples_per_epoch = samples
        else:
             self.samples_per_epoch = 0 # Fallback if info unavailable
        # ------------------------------------------ #
        self.log_callback(f"Epoch {epoch+1}/{self.epochs} starting...")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get('loss')
        val_acc = logs.get('val_accuracy') # Keras uses 'val_accuracy'
        val_loss = logs.get('val_loss')
        self.log_callback(f"Epoch {epoch+1}/{self.epochs} finished. Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        # Call main progress callback at end of epoch
        if self.progress_callback:
            try:
                # progress_callback expects (epoch, total_epochs, loss, val_acc)
                # Note: Keras epoch is 0-indexed, GUI might expect 1-indexed
                self.progress_callback(epoch + 1, self.epochs, train_loss, val_acc)
                # --- Add small sleep to yield --- # 
                time.sleep(0.01)
                # -------------------------------- #
            except Exception as e:
                self.log_callback(f"Error in progress_callback: {e}")

    # Optional: Implement on_batch_end for more granular progress
    # def on_batch_end(self, batch, logs=None):
    #     self.batch_count += 1
    #     if self.progress_callback and self.samples_per_epoch > 0:
    #         # Simple percentage based on batches (less accurate)
    #         percentage = int((self.batch_count * self.params.get('batch_size', 32) / self.samples_per_epoch) * 100)
    #         # Or calculate based on epoch progress?
    #         # Keras logs might contain batch loss/acc
    #         # self.progress_callback(...) # Needs different signature or handling
    #         pass
# ---------------------------- #

class CNNModel:
    """
    A Convolutional Neural Network model using tf.keras.
    """
    def __init__(self, input_shape, num_classes, use_batch_norm=False, use_data_augmentation=False):
        """
        Initializes the CNNModel.

        Args:
            input_shape (tuple): The shape of the input images (e.g., (28, 28, 1) or (32, 32, 3)).
            num_classes (int): The number of output classes.
            use_batch_norm (bool): Whether to include Batch Normalization layers. Defaults to False.
            use_data_augmentation (bool): Whether to include data augmentation layers. Defaults to False.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None # Keras model will be stored here
        self.use_batch_norm = use_batch_norm
        self.use_data_augmentation = use_data_augmentation
        print(f"CNNModel initialized with input shape: {self.input_shape}, num_classes: {self.num_classes}, Batch Norm: {use_batch_norm}, Data Aug: {use_data_augmentation}")

    def build_model(self):
        """
        Builds the Keras Sequential model architecture.
        Optionally includes Data Augmentation and Batch Normalization layers.
        """
        print(f"Building CNN model (Batch Norm: {self.use_batch_norm}, Data Aug: {self.use_data_augmentation})...")
        
        model_layers = [
            keras.Input(shape=self.input_shape, name="input_layer")
        ]
        
        # --- Optional Data Augmentation Block ---
        if self.use_data_augmentation:
            print("  Adding Data Augmentation layers...")
            # Create a sequential block for augmentation
            # These layers are only active during training
            data_augmentation = Sequential(
                [
                    # Use layers directly, not via preprocessing
                    layers.RandomFlip("horizontal", input_shape=self.input_shape),
                    layers.RandomRotation(0.1),
                    layers.RandomZoom(0.1),
                    layers.RandomContrast(0.1),
                    # Add more augmentation as needed (e.g., layers.RandomTranslation)
                ],
                name="data_augmentation",
            )
            model_layers.append(data_augmentation)
        # ---------------------------------------

        # --- Convolutional Block 1 ---
        model_layers.append(layers.Conv2D(32, kernel_size=(3, 3), name="conv1"))
        if self.use_batch_norm:
            model_layers.append(layers.BatchNormalization(name="bn1"))
        model_layers.append(layers.Activation("relu", name="relu1"))
        model_layers.append(layers.MaxPooling2D(pool_size=(2, 2), name="pool1"))
        
        # --- Convolutional Block 2 ---
        model_layers.append(layers.Conv2D(64, kernel_size=(3, 3), name="conv2"))
        if self.use_batch_norm:
            model_layers.append(layers.BatchNormalization(name="bn2"))
        model_layers.append(layers.Activation("relu", name="relu2"))
        model_layers.append(layers.MaxPooling2D(pool_size=(2, 2), name="pool2"))
        
        # Flatten and Dense Layers
        model_layers.append(layers.Flatten(name="flatten"))
        model_layers.append(layers.Dropout(0.5, name="dropout"))
        model_layers.append(layers.Dense(128, name="dense1"))
        if self.use_batch_norm:
            model_layers.append(layers.BatchNormalization(name="bn_dense"))
        model_layers.append(layers.Activation("relu", name="relu_dense"))
        model_layers.append(layers.Dense(self.num_classes, activation="softmax", name="output_layer"))
        
        # Create the final model
        self.model = keras.Sequential(model_layers, name="cnn_model")
        
        print("CNN model built:")
        self.model.summary() # Print model summary to console

    def train(self, X_train, Y_train, epochs, batch_size, learning_rate, patience, log_callback=None, progress_callback=None, use_lr_scheduler=False):
        """Trains the CNN model using the provided data.
           Accepts log_callback and progress_callback.
           Optionally uses learning rate scheduling.
        """
        # --- Use passed callbacks or default print --- #
        current_log_callback = log_callback if log_callback else lambda msg: print(msg, file=sys.stderr)
        current_progress_callback = progress_callback # Keep as is, can be None
        # --------------------------------------------- #

        if self.model is None:
            current_log_callback("Error: CNN model has not been built yet.")
            return None

        # Store callbacks for internal use (used by custom callback)
        self.log_callback = current_log_callback
        self.progress_callback = current_progress_callback

        current_log_callback(f"Starting CNN training for {epochs} epochs with batch size {batch_size} and LR {learning_rate}...")
        # --- LOG SHAPES BEFORE FIT --- #
        current_log_callback(f"  DEBUG [CNNModel.train]: X_train shape = {X_train.shape if X_train is not None else 'None'}")
        current_log_callback(f"  DEBUG [CNNModel.train]: Y_train shape = {Y_train.shape if Y_train is not None else 'None'}")
        # ----------------------------- #

        # --- Ensure Model is Compiled --- #
        # Check if the model needs compilation (either first time or if LR changes)
        needs_compile = False
        if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
            needs_compile = True
            current_log_callback("  Model not compiled yet.")
        else:
            # Model is compiled, check if LR needs updating
            current_lr = K.get_value(self.model.optimizer.learning_rate)
            if current_lr != learning_rate:
                current_log_callback(f"  Learning rate changed ({current_lr} -> {learning_rate}). Updating optimizer.")
                K.set_value(self.model.optimizer.learning_rate, learning_rate)
                # Note: We are NOT recompiling just for LR change, as K.set_value handles it.
            # else: LR is the same, no need to compile or update LR.

        if needs_compile:
            current_log_callback(f"  Compiling model with Adam optimizer, LR={learning_rate}")
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # -------------------------------- #

        callbacks = []
        # Add learning rate scheduler if requested
        if use_lr_scheduler:
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=max(5, patience // 2),
                min_lr=1e-6,
                verbose=1
            )
            callbacks.append(lr_scheduler)
            current_log_callback(f"  Enabled Learning Rate Scheduler (ReduceLROnPlateau)")
            
        if patience > 0:
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
            callbacks.append(early_stopping)
            current_log_callback(f"  Enabled Early Stopping with patience={patience}")

        # Add custom callback for logging/progress
        # Ensure it uses the callbacks passed to this train method
        custom_callback = TrainingProgressCallback(epochs=epochs,
                                                 log_callback=current_log_callback,    # Pass correct log func
                                                 progress_callback=current_progress_callback) # Pass correct progress func
        callbacks.append(custom_callback)

        try:
            # Ensure Y_train is integer type for sparse crossentropy
            if Y_train is not None and not np.issubdtype(Y_train.dtype, np.integer):
                 current_log_callback(f"  Warning: Converting Y_train dtype from {Y_train.dtype} to int for sparse crossentropy.")
                 Y_train = Y_train.astype(int)

            # --- FIT CALL --- #
            current_log_callback(f"  Calling model.fit with batch_size={batch_size}, epochs={epochs}")
            history = self.model.fit(X_train, Y_train,
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     validation_split=0.1, # Using internal split
                                     callbacks=callbacks, # Pass the list of callbacks
                                     verbose=0)
            # ------------- #
            current_log_callback("CNN training finished.")
            return history.history # Return the history dictionary
        except ValueError as ve:
            current_log_callback(f"ERROR during CNN model.fit: {ve}")
            current_log_callback("Check input data shapes and types.")
            traceback.print_exc()
            return None
        except Exception as e:
             current_log_callback(f"An unexpected error occurred during CNN training: {e}")
             traceback.print_exc()
             return None

    def predict(self, X):
        """
        Makes predictions on new data.

        Args:
            X: Data to predict on. Should have shape compatible with model input.

        Returns:
            np.ndarray: Predicted probabilities for each class.
        """
        if self.model is None:
            print("Error: Model not built or loaded.")
            return None

        # Keras predict expects a batch, even if it's just one sample
        if len(X.shape) < len(self.input_shape) + 1: # Check if batch dimension is missing
             # Add batch dimension if predicting a single sample
             X = tf.expand_dims(X, axis=0)


        print(f"CNN predicting on input with shape: {X.shape}")
        predictions = self.model.predict(X)
        return predictions

    def evaluate(self, X_test, Y_test):
        """
        Evaluates the model on test data.

        Args:
            X_test: Test data features.
            Y_test: Test data labels.

        Returns:
            list: Loss and metrics (e.g., [loss, accuracy]).
        """
        if self.model is None:
            print("Error: Model not built or loaded.")
            return None

        print("Evaluating CNN model...")
        results = self.model.evaluate(X_test, Y_test, verbose=0)
        print(f"CNN Evaluation - Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}")
        return results


    def save_weights(self, filepath):
        """
        Saves the model's weights.

        Args:
            filepath (str): Path to save the weights file (e.g., 'model_weights.weights.h5').
                           Keras typically uses '.weights.h5' or '.keras' (newer format).
        """
        if self.model:
            print(f"Saving CNN model weights to {filepath}")
            self.model.save_weights(filepath)
        else:
            print("Error: No model to save.")

    def load_weights(self, filepath):
        """
        Loads the model's weights. The model architecture must be defined first.

        Args:
            filepath (str): Path to the weights file.
        """
        if self.model is None:
            # Need to build the model first to load weights into it
            print("Warning: Building model architecture before loading weights.")
            self.build_model() # Build with default/initial parameters

        if self.model:
            try:
                print(f"Loading CNN model weights from {filepath}")
                self.model.load_weights(filepath)
                print("CNN weights loaded successfully.")
            except Exception as e:
                print(f"Error loading CNN weights from {filepath}: {e}")
                # Handle cases where the architecture might not match the weights
        else:
             print("Error: Model could not be built, cannot load weights.")

    # We might not need get_params/load_params if relying purely on Keras save/load
    # def get_params(self):
    #     # Keras models manage their parameters internally
    #     pass

    # def load_params(self, params):
    #     # Keras models manage their parameters internally
    #     pass 