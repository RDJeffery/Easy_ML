import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CNNModel:
    """
    A Convolutional Neural Network model using tf.keras.
    """
    def __init__(self, input_shape, num_classes):
        """
        Initializes the CNNModel.

        Args:
            input_shape (tuple): The shape of the input images (e.g., (28, 28, 1) or (32, 32, 3)).
            num_classes (int): The number of output classes.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None # Keras model will be stored here
        print(f"CNNModel initialized with input shape: {self.input_shape} and num_classes: {self.num_classes}")

    def build_model(self):
        """
        Builds the Keras Sequential model architecture.
        This uses a simple Conv->Pool->Conv->Pool->Flatten->Dense structure.
        """
        print("Building CNN model...")
        self.model = keras.Sequential(
            [
                keras.Input(shape=self.input_shape, name="input_layer"),
                # Convolutional Block 1
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="conv1"),
                layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),
                # Convolutional Block 2
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu", name="conv2"),
                layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),
                # Flatten and Dense Layers
                layers.Flatten(name="flatten"),
                layers.Dropout(0.5, name="dropout"), # Add dropout for regularization
                layers.Dense(128, activation="relu", name="dense1"),
                layers.Dense(self.num_classes, activation="softmax", name="output_layer"),
            ],
            name="cnn_model" # Give the overall model a name
        )
        print("CNN model built:")
        self.model.summary() # Print model summary to console

    def train(self, X_train, Y_train, X_val, Y_val, epochs=10, batch_size=32, learning_rate=0.001, callbacks=None):
        """
        Compiles and trains the Keras model.

        Args:
            X_train: Training data features.
            Y_train: Training data labels.
            X_val: Validation data features.
            Y_val: Validation data labels.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for the optimizer.
            callbacks (list, optional): List of Keras callbacks to use during training.
        """
        if self.model is None:
            print("Error: Model not built yet. Call build_model() first.")
            return None

        print(f"Starting CNN training for {epochs} epochs with batch size {batch_size} and LR {learning_rate}...")

        # Choose an optimizer (Adam is common)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        # Compile the model - Specify loss, optimizer, and metrics
        self.model.compile(loss="sparse_categorical_crossentropy", # Use this if Y labels are integers
                           optimizer=optimizer,
                           metrics=["accuracy"])

        # TODO: Add callbacks? (e.g., EarlyStopping, ModelCheckpoint)

        # Use the callbacks passed in argument, default to empty list if None
        callbacks_to_use = callbacks if callbacks is not None else []

        # Train the model
        history = self.model.fit(X_train, Y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_data=(X_val, Y_val),
                                 callbacks=callbacks_to_use)

        print("CNN training finished.")
        # You might want to return the history object for plotting later
        return history

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