import numpy as np
import sys
from typing import List, Tuple, Dict, Any, Optional, Callable

# --- Activation Functions & Derivatives (Standalone) ---
def ReLU(Z: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit activation."""
    return np.maximum(0, Z)

def deriv_ReLU(Z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU w.r.t. Z."""
    return Z > 0

def Sigmoid(Z: np.ndarray) -> np.ndarray:
    """Sigmoid activation."""
    Z_clipped = np.clip(Z, -500, 500) # Prevent overflow
    return 1 / (1 + np.exp(-Z_clipped))

def deriv_Sigmoid(A: np.ndarray) -> np.ndarray: # Takes activation A
    """Derivative of Sigmoid w.r.t. Z, expressed using A."""
    return A * (1 - A)

def Tanh(Z: np.ndarray) -> np.ndarray:
    """Hyperbolic Tangent activation."""
    return np.tanh(Z)

def deriv_Tanh(A: np.ndarray) -> np.ndarray: # Takes activation A
    """Derivative of Tanh w.r.t. Z, expressed using A."""
    return 1 - np.power(A, 2)

def softmax(Z: np.ndarray) -> np.ndarray:
    """Softmax activation for the output layer."""
    shift_Z = Z - np.max(Z, axis=0, keepdims=True) # Numerical stability
    A = np.exp(shift_Z) / np.sum(np.exp(shift_Z), axis=0, keepdims=True)
    epsilon = 1e-10 # Prevent log(0)
    A = np.clip(A, epsilon, 1. - epsilon)
    return A
# ------------------------------------------------------

# === Simple Neural Network Class ===

class SimpleNeuralNetwork:
    """A simple feedforward neural network implementation."""

    def __init__(self, layer_dims: List[int], log_callback: Optional[Callable[[str], None]] = None):
        """Initializes the neural network.

        Args:
            layer_dims (List[int]): List containing the size of each layer,
                                     e.g., [input_size, hidden1_size, ..., output_size].
            log_callback (Optional[Callable[[str], None]]): Function to use for logging messages.
                                                            Defaults to print if None.
        """
        if not layer_dims or len(layer_dims) < 2:
             raise ValueError("layer_dims must contain at least input and output size.")
        self.layer_dims = layer_dims
        # --- Store log_callback --- #
        self.log_callback = log_callback if log_callback is not None else lambda msg: print(msg, file=sys.stderr)
        # -------------------------- #
        self.parameters = self._init_params(layer_dims)
        self.stop_training_flag = False # Flag for stopping training externally

    # --- Internal Helper Methods ---

    def _init_params(self, layer_dims: List[int]) -> Dict[str, np.ndarray]:
        """Initializes parameters dictionary."""
        np.random.seed(1) # for consistency
        parameters: Dict[str, np.ndarray] = {}
        L = len(layer_dims)
        for l in range(1, L):
            # He initialization (good default for ReLU)
            parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2. / layer_dims[l-1])
            parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
            assert(parameters[f'W{l}'].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters[f'b{l}'].shape == (layer_dims[l], 1))
        print(f"  DEBUG [SimpleNN._init_params]: Initialized parameters for layers: {layer_dims}", file=sys.stderr)
        return parameters

    def _linear_forward(self, A_prev: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Linear part of forward propagation."""
        Z = W.dot(A_prev) + b
        linear_cache = (A_prev, W, b) # Cache values needed for backprop
        assert(Z.shape == (W.shape[0], A_prev.shape[1]))
        return Z, linear_cache

    def _activation_forward(self, A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, activation: str) -> Tuple[np.ndarray, Tuple[Any, Any]]:
        """LINEAR -> ACTIVATION step."""
        Z, linear_cache = self._linear_forward(A_prev, W, b)
        activation_cache = Z # Cache Z needed for backprop derivative calculation

        if activation.lower() == "relu": A = ReLU(Z)
        elif activation.lower() == "sigmoid": A = Sigmoid(Z); activation_cache = A # Cache A for sigmoid deriv
        elif activation.lower() == "tanh": A = Tanh(Z); activation_cache = A # Cache A for tanh deriv
        elif activation.lower() == "softmax": A = softmax(Z) # Softmax uses Z cache, deriv calculated differently
        else: raise ValueError(f"Unknown activation function: {activation}")

        cache = (linear_cache, activation_cache) # linear_cache=(A_prev,W,b), activation_cache=Z or A
        return A, cache

    def _forward_prop(self, X: np.ndarray, keep_prob: float = 1.0, hidden_activation: str = "relu") -> Tuple[np.ndarray, List[Tuple[Any, Any, Optional[np.ndarray]]], bool]:
        """Full forward propagation with optional dropout."""
        caches: List[Tuple[Any, Any, Optional[np.ndarray]]] = [] # Cache now includes dropout mask D
        A = X
        L = len(self.parameters) // 2 # Number of layers with parameters (W,b pairs)

        try:
            # Hidden layers (LINEAR -> ACTIVATION -> Optional Dropout)
            for l in range(1, L):
                A_prev = A
                Wl = self.parameters[f'W{l}']
                bl = self.parameters[f'b{l}']
                A, cache_l = self._activation_forward(A_prev, Wl, bl, activation=hidden_activation)
                linear_cache, activation_cache = cache_l

                # Apply Dropout
                dropout_mask = None
                if keep_prob < 1.0: # Apply dropout only if keep_prob < 1.0 (i.e., during training)
                    dropout_mask = (np.random.rand(*A.shape) < keep_prob).astype(int) # Create mask D
                    A = A * dropout_mask  # Apply mask
                    A = A / keep_prob     # Scale using inverted dropout
                caches.append((linear_cache, activation_cache, dropout_mask)) # Store cache for layer l

                if np.isnan(A).any() or np.isinf(A).any():
                    print(f"ERROR: NaN/Inf in activations layer {l}", file=sys.stderr); return A, caches, False

            # Output layer (LINEAR -> SOFTMAX)
            WL = self.parameters[f'W{L}']
            bL = self.parameters[f'b{L}']
            AL, cache_L = self._activation_forward(A, WL, bL, activation="softmax")
            linear_cache_L, activation_cache_L = cache_L
            caches.append((linear_cache_L, activation_cache_L, None)) # No dropout mask for last layer

            if np.isnan(AL).any() or np.isinf(AL).any():
                print("ERROR: NaN/Inf in output activations", file=sys.stderr); return AL, caches, False

            assert(AL.shape[0] == self.layer_dims[-1] and AL.shape[1] == X.shape[1])
            return AL, caches, True # Success

        except Exception as e:
            print(f"ERROR during _forward_prop: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            # Return dummy values on error
            output_size = self.layer_dims[-1]
            dummy_AL = np.zeros((output_size, X.shape[1]))
            if not caches: caches = [(None, None, None)] * L # Ensure caches has right length
            return dummy_AL, caches, False

    def _one_hot(self, Y: np.ndarray, num_classes: int) -> np.ndarray:
        """Converts label vector Y to one-hot matrix."""
        m = Y.size
        Y_flat = Y.flatten().astype(int)
        one_hot_Y = np.zeros((num_classes, m))
        valid_indices_mask = (Y_flat >= 0) & (Y_flat < num_classes)
        valid_labels = Y_flat[valid_indices_mask]
        valid_positions = np.arange(m)[valid_indices_mask]
        if valid_labels.size > 0:
            one_hot_Y[valid_labels, valid_positions] = 1
        if valid_labels.size < m:
             print(f"WARN [one_hot]: {m - valid_labels.size}/{m} labels out of range [0, {num_classes-1}]", file=sys.stderr)
        return one_hot_Y

    def _linear_backward(self, dZ: np.ndarray, linear_cache: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Linear part of backward propagation."""
        A_prev, W, b = linear_cache
        m = A_prev.shape[1]
        dW = 1./m * np.dot(dZ, A_prev.T)
        db = 1./m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        assert (dA_prev.shape == A_prev.shape); assert (dW.shape == W.shape); assert (db.shape == b.shape)
        return dA_prev, dW, db

    def _activation_backward(self, dA: np.ndarray, cache: Tuple[Any, Any], activation: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ACTIVATION -> LINEAR backward step."""
        linear_cache, activation_cache = cache # activation_cache = Z or A
        # Calculate dZ based on activation function
        if activation.lower() == "relu": dZ = dA * deriv_ReLU(activation_cache) # activation_cache = Z
        elif activation.lower() == "sigmoid": dZ = dA * deriv_Sigmoid(activation_cache) # activation_cache = A
        elif activation.lower() == "tanh": dZ = dA * deriv_Tanh(activation_cache) # activation_cache = A
        elif activation.lower() == "softmax": dZ = dA # dA is already AL - Y_one_hot for cross-entropy
        else: raise ValueError(f"Unknown activation: {activation}")
        # Pass dZ to linear backward step
        return self._linear_backward(dZ, linear_cache)

    def _backward_prop(self, AL: np.ndarray, Y: np.ndarray, caches: List[Tuple[Any, Any, Optional[np.ndarray]]], hidden_activation: str = "relu", l2_lambda: float = 0.0, keep_prob: float = 1.0) -> Dict[str, np.ndarray]:
        """Full backward propagation with L2 and Dropout."""
        grads: Dict[str, np.ndarray] = {}
        L = len(caches) # number of layers (input is layer 0)
        m = AL.shape[1]
        Y = Y.reshape(1, m) # Ensure Y is shape (1, m)

        # --- Output layer (Softmax + Cross Entropy) ---
        num_classes = AL.shape[0]
        Y_one_hot = self._one_hot(Y.flatten(), num_classes)
        dZL = AL - Y_one_hot # Grad of loss w.r.t. Z for softmax output layer

        # Grads for layer L
        current_cache = caches[L-1]
        linear_cache_L, activation_cache_L, _ = current_cache # No dropout on last layer
        # Get grads dAL_prev, dWL, dbL from linear part of softmax layer
        grads[f"dA{L-1}"], dW_L, db_L = self._activation_backward(dZL, (linear_cache_L, activation_cache_L), activation="softmax")
        # Add L2 regularization gradient to dWL
        grads[f"dW{L}"] = dW_L + (l2_lambda / m) * self.parameters[f'W{L}'] if l2_lambda > 0 else dW_L
        grads[f"db{L}"] = db_L # No regularization for bias terms
        # ----------------------------------------------

        # --- Hidden layers (loop backwards) ---
        for l in reversed(range(L - 1)): # Loop from L-2 down to 0
            # Grads for layer l+1 (parameters W{l+1}, b{l+1})
            current_cache = caches[l]
            linear_cache, activation_cache, dropout_mask = current_cache # Unpack cache

            dA_input = grads[f"dA{l + 1}"] # Gradient from the next layer forward

            # Apply dropout mask (if used in forward prop)
            if dropout_mask is not None:
                # print(f"  DEBUG [backward_prop]: Applying dropout mask to dA{l+1}", file=sys.stderr)
                dA_input = dA_input * dropout_mask  # Apply mask
                dA_input = dA_input / keep_prob     # Scale (inverted dropout)

            # Get grads dAL_prev, dW_temp, db_temp using activation backward
            dA_prev_temp, dW_temp, db_temp = self._activation_backward(dA_input, (linear_cache, activation_cache), activation=hidden_activation)

            # Store gradients for layer l+1
            grads[f"dA{l}"] = dA_prev_temp
            # Add L2 regularization gradient to dW_temp
            grads[f"dW{l + 1}"] = dW_temp + (l2_lambda / m) * self.parameters[f'W{l+1}'] if l2_lambda > 0 else dW_temp
            grads[f"db{l + 1}"] = db_temp # No regularization for bias terms
        # ------------------------------------

        # Return only dW and db gradients
        return {key: val for key, val in grads.items() if not key.startswith('dA')}

    def _update_params_gd(self, grads: Dict[str, np.ndarray], alpha: float):
        """Standard Gradient Descent update rule."""
        L = len(self.parameters) // 2
        for l in range(1, L + 1):
            self.parameters[f"W{l}"] -= alpha * grads[f"dW{l}"]
            self.parameters[f"b{l}"] -= alpha * grads[f"db{l}"]

    def _compute_loss(self, AL: np.ndarray, Y: np.ndarray, l2_lambda: float = 0.0) -> float:
        """Computes cross-entropy loss with optional L2 regularization."""
        # Log shapes upon entry
        self.log_callback(f"  DEBUG [_compute_loss]: AL shape={AL.shape}, Y shape={Y.shape}")

        m = Y.shape[0] # Number of examples
        # Ensure Y has the correct shape for one-hot encoding if needed
        if Y.ndim == 1:
            Y = Y.reshape(1, m)
        elif Y.shape[0] != 1 and Y.shape[1] == 1: # Handle (m, 1) column vector
            Y = Y.T # Transpose to (1, m)
        elif Y.shape[0] != 1: # Handle unexpected shapes
             self.log_callback(f"  WARNING [_compute_loss]: Unexpected Y shape {Y.shape}. Attempting reshape.")
             # This case is problematic, might indicate earlier error
             # Try flattening and reshaping, but this is risky
             Y = Y.flatten().reshape(1, -1)
             m = Y.shape[1] # Recalculate m if shape changed

        # One-hot encode Y
        # Assuming AL shape is (num_classes, num_samples)
        num_classes = AL.shape[0]
        self.log_callback(f"  DEBUG [_compute_loss]: Calling _one_hot with Y shape={Y.shape}, num_classes={num_classes}") # Log before one_hot
        Y_one_hot = self._one_hot(Y, num_classes)
        self.log_callback(f"  DEBUG [_compute_loss]: Y_one_hot shape={Y_one_hot.shape}") # Log after one_hot

        # Compute cross-entropy cost
        # Clip AL to prevent log(0). Softmax output should already be positive.
        epsilon = 1e-9
        logprobs = np.log(np.clip(AL, epsilon, 1.0))
        cross_entropy_cost = - (1./m) * np.sum(Y_one_hot * logprobs)

        # L2 Regularization Cost (if lambda > 0)
        l2_regularization_cost = 0.0
        if l2_lambda > 0:
            L = len(self.parameters) // 2
            l2_norm_sum = sum(np.sum(np.square(self.parameters[f'W{l}'])) for l in range(1, L + 1))
            l2_regularization_cost = (l2_lambda / (2 * m)) * l2_norm_sum

        cost = np.squeeze(cross_entropy_cost + l2_regularization_cost)
        # Handle potential NaN/Inf gracefully
        if np.isnan(cost) or np.isinf(cost):
            print(f"ERROR [_compute_loss]: Loss became NaN or Inf. CrossEntropy={cross_entropy_cost}, L2Reg={l2_regularization_cost}", file=sys.stderr)
            return np.inf
        return float(cost)


    # --- Public Methods --- #

    def train(self, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray,
              alpha: float, epochs: int, patience: int = 0,
              hidden_activation: str = "relu", optimizer_name: str = "GradientDescent",
              l2_lambda: float = 0.0, dropout_keep_prob: float = 1.0,
              progress_callback: Optional[Callable[[int, int, float, float], bool]] = None) -> Optional[Tuple[Dict[str, np.ndarray], List[float], List[float]]]:
        """Trains the model using specified data and hyperparameters."""
        self.stop_training_flag = False # Reset stop flag at beginning
        print(f"--- Starting Model Training (Class: {self.__class__.__name__}) ---", file=sys.stderr)
        print(f"    Hyperparameters: α={alpha}, epochs={epochs}, patience={patience}, activation={hidden_activation}, optimizer={optimizer_name}, L2λ={l2_lambda}, DropoutKeep={dropout_keep_prob}", file=sys.stderr)
        if l2_lambda > 0.0: print(f"    L2 Regularization enabled (λ={l2_lambda})", file=sys.stderr)
        if dropout_keep_prob < 1.0: print(f"    Dropout enabled (keep_prob={dropout_keep_prob})", file=sys.stderr)

        loss_history = []
        val_accuracy_history = []
        best_val_acc = -1.0
        epochs_no_improve = 0
        # Store a deep copy of initial params in case of early stopping
        best_params = {k: v.copy() for k, v in self.parameters.items()}

        # --- Adam Optimizer Initialization ---
        v: Dict[str, np.ndarray] = {}
        s: Dict[str, np.ndarray] = {}
        t: int = 0 # Timestep for bias correction
        beta1: float = 0.9
        beta2: float = 0.999
        epsilon: float = 1e-8
        L = len(self.parameters) // 2
        if optimizer_name.lower() == 'adam':
            print("    Initializing Adam parameters (v, s)...", file=sys.stderr)
            for l in range(1, L + 1):
                v[f"dW{l}"] = np.zeros_like(self.parameters[f"W{l}"])
                v[f"db{l}"] = np.zeros_like(self.parameters[f"b{l}"])
                s[f"dW{l}"] = np.zeros_like(self.parameters[f"W{l}"])
                s[f"db{l}"] = np.zeros_like(self.parameters[f"b{l}"])
        # ------------------------------------

        Y_dev_flat = Y_dev.flatten() if Y_dev.ndim > 1 else Y_dev # For accuracy calculation

        # --- Training Loop ---
        for i in range(epochs):
            if self.stop_training_flag:
                print(f"INFO: Training stopped at epoch {i} by flag.", file=sys.stderr)
                self.stop_training_flag = False # Reset flag
                return None # Indicate premature stop

            # Forward prop (with dropout enabled if keep_prob < 1.0)
            AL_train, caches, status_train = self._forward_prop(X_train, dropout_keep_prob, hidden_activation)
            if not status_train: print(f"ERROR: Forward prop failed (train) epoch {i}"); return None

            # Compute loss
            train_loss = self._compute_loss(AL_train, Y_train, l2_lambda)
            loss_history.append(train_loss)
            if np.isnan(train_loss) or np.isinf(train_loss): print(f"ERROR: Loss NaN/Inf epoch {i}"); return None

            # Backward prop
            grads = self._backward_prop(AL_train, Y_train, caches, hidden_activation, l2_lambda, dropout_keep_prob)
            # Check gradients for NaN/Inf
            if any(np.isnan(g).any() or np.isinf(g).any() for g in grads.values()): print(f"ERROR: Grads NaN/Inf epoch {i}"); return None

            # --- Parameter update ---
            if optimizer_name.lower() == 'adam':
                t += 1 # Increment timestep
                for l in range(1, L + 1):
                    # Momentum update (biased)
                    v[f"dW{l}"] = beta1 * v[f"dW{l}"] + (1 - beta1) * grads[f"dW{l}"]
                    v[f"db{l}"] = beta1 * v[f"db{l}"] + (1 - beta1) * grads[f"db{l}"]
                    # RMSprop update (biased)
                    s[f"dW{l}"] = beta2 * s[f"dW{l}"] + (1 - beta2) * np.square(grads[f"dW{l}"])
                    s[f"db{l}"] = beta2 * s[f"db{l}"] + (1 - beta2) * np.square(grads[f"db{l}"])
                    # Bias correction
                    v_corr_dW = v[f"dW{l}"] / (1 - beta1**t)
                    v_corr_db = v[f"db{l}"] / (1 - beta1**t)
                    s_corr_dW = s[f"dW{l}"] / (1 - beta2**t)
                    s_corr_db = s[f"db{l}"] / (1 - beta2**t)
                    # Update parameters
                    self.parameters[f"W{l}"] -= alpha * v_corr_dW / (np.sqrt(s_corr_dW) + epsilon)
                    self.parameters[f"b{l}"] -= alpha * v_corr_db / (np.sqrt(s_corr_db) + epsilon)
            else: # Standard Gradient Descent
                self._update_params_gd(grads, alpha)
            # Check parameters for NaN/Inf after update
            if any(np.isnan(p).any() or np.isinf(p).any() for p_arr in self.parameters.values() for p in [p_arr]):
                 print(f"ERROR: Params NaN/Inf after update epoch {i}"); return None
            # -------------------------

            # --- Validation --- (Dropout is OFF - keep_prob=1.0)
            AL_dev, _, status_dev = self._forward_prop(X_dev, keep_prob=1.0, hidden_activation=hidden_activation)
            val_acc = 0.0
            if status_dev:
                predictions_dev = np.argmax(AL_dev, axis=0) # Get class indices
                val_acc = np.mean(predictions_dev == Y_dev_flat) # Compare with flattened true labels
            else:
                 print(f"WARN: Forward prop failed (dev) epoch {i}", file=sys.stderr)
            val_accuracy_history.append(val_acc)
            # ------------------

            # --- Log & Callback ---
            print(f"Epoch {i+1}/{epochs} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}", file=sys.stderr)
            if progress_callback:
                if not progress_callback(i + 1, epochs, train_loss, val_acc):
                    print(f"INFO: Training stopped at epoch {i+1} by callback.", file=sys.stderr)
                    return None # Indicate stop via callback
            # ----------------------

            # --- Early Stopping ---
            if patience > 0:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    epochs_no_improve = 0
                    best_params = {k: v.copy() for k, v in self.parameters.items()} # Deep copy best parameters
                    # print(f"  INFO: New best validation accuracy: {best_val_acc:.4f}", file=sys.stderr) # Optional log
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"INFO: Early stopping triggered at epoch {i+1}. Restoring best parameters from epoch {i+1 - patience}.", file=sys.stderr)
                    self.parameters = best_params # Restore best parameters to the model instance
                    # Return best params and history up to this point
                    return self.parameters, loss_history, val_accuracy_history
            # --------------------

        # --- End of Training Loop ---
        print("--- Training Finished Normally ---", file=sys.stderr)
        # If early stopping was enabled but didn't trigger, best_params holds the last improvement
        # Otherwise, self.parameters holds the params from the final epoch
        final_params = best_params if patience > 0 and best_val_acc >= 0 else self.parameters
        self.parameters = final_params # Ensure model instance has the final parameters used
        # Return the final parameters and full history
        return self.parameters, loss_history, val_accuracy_history


    def predict(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Makes predictions using the learned parameters (returns probabilities).

        Args:
            X (np.ndarray): Input data (shape: input_size, num_examples).

        Returns:
            Optional[np.ndarray]: Predicted probabilities (shape: num_classes, num_examples) or None if prediction failed.
        """
        # Determine hidden activation from model state if needed, or assume default?
        # For simplicity, assume default used during training is sufficient here.
        # A more robust approach might store activation with parameters.
        # TODO: Infer hidden_activation from model state if possible or store it. Using 'relu' for now.
        hidden_activation = 'relu'
        AL, _, status = self._forward_prop(X, keep_prob=1.0, hidden_activation=hidden_activation) # Dropout OFF
        if not status:
            print("ERROR: Forward propagation failed during prediction.", file=sys.stderr)
            return None
        # Return the probabilities (output of softmax)
        return AL

    def get_params(self) -> Dict[str, np.ndarray]:
        """Returns the current model parameters."""
        # Return a copy to prevent external modification? Or return direct reference?
        # Returning direct reference for now.
        return self.parameters

    def load_params(self, parameters: Dict[str, np.ndarray]):
        """Loads parameters into the model. Performs basic shape checks."""
        print("Attempting to load parameters into model instance...", file=sys.stderr)
        # Check if keys match expected structure based on self.layer_dims
        expected_keys = set()
        L = len(self.layer_dims)
        for l in range(1, L):
            expected_keys.add(f'W{l}')
            expected_keys.add(f'b{l}')

        if set(parameters.keys()) != expected_keys:
             print(f"WARN: Loading parameters with keys {set(parameters.keys())} which differ from expected keys {expected_keys} based on model's layer_dims {self.layer_dims}.", file=sys.stderr)
             # Decide whether to proceed or raise error. Proceeding cautiously.

        # Check shapes
        dimensions_match = True
        for l in range(1, L):
            key_W = f'W{l}'; key_b = f'b{l}'
            expected_W_shape = (self.layer_dims[l], self.layer_dims[l-1])
            expected_b_shape = (self.layer_dims[l], 1)
            if key_W not in parameters:
                print(f"ERROR: Missing key '{key_W}' in loaded parameters.", file=sys.stderr); dimensions_match = False; break
            if parameters[key_W].shape != expected_W_shape:
                print(f"ERROR: Shape mismatch for key '{key_W}'. Expected {expected_W_shape}, got {parameters[key_W].shape}.", file=sys.stderr); dimensions_match = False; break
            if key_b not in parameters:
                print(f"ERROR: Missing key '{key_b}' in loaded parameters.", file=sys.stderr); dimensions_match = False; break
            if parameters[key_b].shape != expected_b_shape:
                 print(f"ERROR: Shape mismatch for key '{key_b}'. Expected {expected_b_shape}, got {parameters[key_b].shape}.", file=sys.stderr); dimensions_match = False; break

        if not dimensions_match:
             print("ERROR: Cannot load parameters due to shape or key mismatch with model architecture. Model parameters not updated.", file=sys.stderr)
             return # Stop loading

        # If checks pass
        self.parameters = parameters
        print("Model parameters loaded successfully into instance.", file=sys.stderr)

# --- End of Class ---
