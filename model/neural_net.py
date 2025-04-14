import numpy as np
import sys
from typing import List, Tuple, Dict, Any, Optional, Callable

# --- Activation Functions ---

def ReLU(Z: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit activation."""
    return np.maximum(0, Z)

def deriv_ReLU(Z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU."""
    return Z > 0

def Sigmoid(Z: np.ndarray) -> np.ndarray:
    """Sigmoid activation."""
    # Add clipping for numerical stability
    Z_clipped = np.clip(Z, -500, 500)
    return 1 / (1 + np.exp(-Z_clipped))

def deriv_Sigmoid(A: np.ndarray) -> np.ndarray: # Takes activation A, not Z
    """Derivative of Sigmoid w.r.t. Z, expressed using A."""
    # A = Sigmoid(Z)
    return A * (1 - A)

def Tanh(Z: np.ndarray) -> np.ndarray:
    """Hyperbolic Tangent activation."""
    return np.tanh(Z)

def deriv_Tanh(A: np.ndarray) -> np.ndarray: # Takes activation A, not Z
    """Derivative of Tanh w.r.t. Z, expressed using A."""
    # A = Tanh(Z)
    return 1 - np.power(A, 2)

def softmax(Z: np.ndarray) -> np.ndarray:
    """Softmax activation for the output layer."""
    # Shift Z by subtracting max for numerical stability
    shift_Z = Z - np.max(Z, axis=0, keepdims=True)
    A = np.exp(shift_Z) / np.sum(np.exp(shift_Z), axis=0, keepdims=True)
    # Add a small epsilon to prevent log(0) in loss calculation, clip values
    epsilon = 1e-10
    A = np.clip(A, epsilon, 1. - epsilon)
    return A

# --- Parameter Initialization ---

def init_params(layer_dims: List[int]) -> Dict[str, np.ndarray]:
    """Initializes parameters for a multi-layer network.

    Args:
        layer_dims (List[int]): List containing the size of each layer,
                                 e.g., [input_size, hidden1_size, ..., output_size].

    Returns:
        Dict[str, np.ndarray]: Dictionary containing initialized parameters
                                (W1, b1, W2, b2, ...).
    """
    np.random.seed(1) # For consistency
    parameters: Dict[str, np.ndarray] = {}
    L = len(layer_dims) # Number of layers including input
    print(f"  DEBUG [init_params]: Received layer_dims: {layer_dims}, Calculated L={L}", file=sys.stderr)

    for l in range(1, L):
        print(f"  DEBUG [init_params]: Inside loop, l = {l}", file=sys.stderr)
        # Use He initialization for ReLU, Xavier/Glorot for Tanh/Sigmoid might be better if chosen
        # For simplicity, using He for all now, could adapt later.
        parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2. / layer_dims[l-1])
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))

        assert(parameters[f'W{l}'].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters[f'b{l}'].shape == (layer_dims[l], 1))

    # --- Debug: Log created parameter keys --- #
    print(f"  DEBUG [init_params]: Created parameter keys: {list(parameters.keys())}", file=sys.stderr)
    # --------------------------------------- #
    return parameters

# --- Forward Propagation ---

def linear_forward(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Implements the linear part of a layer's forward propagation."""
    Z = W.dot(A_prev) + b
    cache = (A_prev, W, b) # Cache values needed for backward prop
    assert(Z.shape == (W.shape[0], A_prev.shape[1]))
    return Z, cache

def activation_forward(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, activation: str) -> Tuple[np.ndarray, Tuple[Any, np.ndarray]]:
    """Implements forward propagation for one layer (LINEAR -> ACTIVATION)."""
    Z, linear_cache = linear_forward(A_prev, W, b)
    activation_cache = Z # Cache Z needed for backprop

    if activation.lower() == "relu":
        A = ReLU(Z)
    elif activation.lower() == "sigmoid":
        A = Sigmoid(Z)
        activation_cache = A # Sigmoid derivative uses A
    elif activation.lower() == "tanh":
        A = Tanh(Z)
        activation_cache = A # Tanh derivative uses A
    elif activation.lower() == "softmax":
        A = softmax(Z)
        # Softmax backprop calculation is simpler combined with cross-entropy loss,
        # so we still cache Z here, but backprop uses (A - Y) directly as dZ.
    else:
        raise ValueError(f"Unknown activation function: {activation}")

    cache = (linear_cache, activation_cache) # linear_cache = (A_prev, W, b), activation_cache = Z or A
    return A, cache

def forward_prop(X: np.ndarray, parameters: Dict[str, np.ndarray], hidden_activation: str = "relu") -> Tuple[np.ndarray, List[Tuple[Any, Any]], bool]:
    """Implements forward propagation for the [LINEAR->ACTIVATION]*(L-1)->LINEAR->SOFTMAX model.

    Args:
        X (np.ndarray): Input data (shape: input_size, num_examples).
        parameters (Dict[str, np.ndarray]): Output of init_params().
        hidden_activation (str): Activation function for hidden layers ('relu', 'sigmoid', 'tanh').

    Returns:
        Tuple containing:
            AL (np.ndarray): Last post-activation value (output layer).
            caches (List): List of caches containing (linear_cache, activation_cache) for each layer.
            status (bool): True if successful, False if NaN/inf detected.
    """
    caches: List[Tuple[Any, Any]] = []
    A = X
    L = len(parameters) // 2 # Number of layers with parameters
    print(f"  DEBUG [forward_prop]: Number of layers (L) = {L}, Hidden Activation = {hidden_activation}", file=sys.stderr)

    try:
        # Implement [LINEAR -> hidden_activation] L-1 times
        for l in range(1, L):
            A_prev = A
            Wl = parameters[f'W{l}']
            bl = parameters[f'b{l}']
            # Use the specified hidden_activation for hidden layers
            A, cache = activation_forward(A_prev, Wl, bl, activation=hidden_activation)
            caches.append(cache)
            if np.isnan(A).any() or np.isinf(A).any():
                print(f"ERROR: NaN or Inf detected in activations at layer {l} ({hidden_activation}).", file=sys.stderr)
                return A, caches, False # Indicate failure

        # Implement LINEAR -> SOFTMAX for the last layer
        WL = parameters[f'W{L}']
        bL = parameters[f'b{L}']
        # Output layer always uses softmax
        AL, cache = activation_forward(A, WL, bL, activation="softmax")
        caches.append(cache)
        if np.isnan(AL).any() or np.isinf(AL).any():
            print("ERROR: NaN or Inf detected in output layer activations (softmax).", file=sys.stderr)
            return AL, caches, False # Indicate failure

        assert(AL.shape[1] == X.shape[1])
        return AL, caches, True # Indicate success

    except Exception as e:
        print(f"ERROR during forward propagation: {e}", file=sys.stderr)
        # Determine expected output shape
        L = len(parameters) // 2
        output_size = parameters[f'W{L}'].shape[0]
        dummy_AL = np.zeros((output_size, X.shape[1]))
        return dummy_AL, caches, False

# --- Backward Propagation ---

def one_hot(Y: np.ndarray, num_classes: int) -> np.ndarray:
    """Converts an array of integer labels to one-hot encoding."""
    m = Y.size
    # Ensure Y is 1D integer array
    Y_flat = Y.flatten().astype(int)
    one_hot_Y = np.zeros((num_classes, m))
    # Handle potential out-of-bounds labels gracefully
    valid_indices_mask = (Y_flat >= 0) & (Y_flat < num_classes)
    valid_labels = Y_flat[valid_indices_mask]
    valid_positions = np.arange(m)[valid_indices_mask]

    # Apply one-hot encoding only for valid labels/positions
    if valid_labels.size > 0:
        one_hot_Y[valid_labels, valid_positions] = 1

    # Log if any labels were out of bounds
    num_invalid = m - valid_labels.size
    if num_invalid > 0:
        print(f"  Warning [one_hot]: {num_invalid}/{m} labels were outside the expected range [0, {num_classes-1}].", file=sys.stderr)

    return one_hot_Y

def linear_backward(dZ: np.ndarray, linear_cache: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Implements the linear portion of backward propagation for a single layer."""
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ, A_prev.T)
    db = 1./m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

def activation_backward(dA: np.ndarray, cache: Tuple[Any, Any], activation: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Implements the backward propagation for the LINEAR->ACTIVATION layer."""
    linear_cache, activation_cache = cache # activation_cache is Z for ReLU/Softmax, A for Sigmoid/Tanh

    if activation.lower() == "relu":
        Z = activation_cache
        dZ = dA * deriv_ReLU(Z) # Element-wise product
    elif activation.lower() == "sigmoid":
        A = activation_cache
        dZ = dA * deriv_Sigmoid(A) # Derivative calculated using A
    elif activation.lower() == "tanh":
        A = activation_cache
        dZ = dA * deriv_Tanh(A) # Derivative calculated using A
    elif activation.lower() == "softmax":
        # The dZ for softmax+cross-entropy is simply AL - Y_one_hot.
        # This calculation happens *before* calling activation_backward for the last layer.
        dZ = dA # In this case, dA passed in *is* (AL - Y_one_hot)
    else:
        raise ValueError(f"Unknown activation function: {activation}")

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def backward_prop(AL: np.ndarray, Y: np.ndarray, caches: List[Tuple[Any, Any]], hidden_activation: str = "relu") -> Dict[str, np.ndarray]:
    """Implement the backward propagation for the [LINEAR->ACTIVATION] * (L-1) -> LINEAR -> SOFTMAX model.

    Args:
        AL (np.ndarray): Probability vector, output of the forward propagation.
        Y (np.ndarray): True "label" vector (shape: 1, num_examples or num_examples,).
        caches (List): List of caches from forward_prop.
        hidden_activation (str): Activation function used in hidden layers ('relu', 'sigmoid', 'tanh').

    Returns:
        Dict[str, np.ndarray]: A dictionary with the gradients.
    """
    grads: Dict[str, np.ndarray] = {}
    L = len(caches) # The number of layers
    m = AL.shape[1]
    # Ensure Y is shape (1, m)
    Y = Y.reshape(1, m)

    # --- Initializing the backpropagation ---
    # Calculate dZL for the output layer (Softmax + Cross-Entropy)
    num_classes = AL.shape[0]
    Y_one_hot = one_hot(Y.flatten(), num_classes) # one_hot expects 1D Y
    dZL = AL - Y_one_hot # Derivative of CrossEntropyLoss w.r.t Z for Softmax output

    # Gradients for the last layer (Layer L)
    current_cache = caches[L-1] # Cache for the last layer (Linear + Softmax)
    # The activation_backward for softmax expects dZL directly
    grads[f"dA{L-1}"], grads[f"dW{L}"], grads[f"db{L}"] = activation_backward(dZL, current_cache, activation="softmax")

    # --- Loop from l=L-2 down to 0 ---
    # Iterate through the hidden layers backwards
    for l in reversed(range(L - 1)):
        # Gradients for layer l: LINEAR -> hidden_activation
        current_cache = caches[l] # Cache for layer l
        dA_input = grads[f"dA{l + 1}"] # Get dA from the *next* layer (closer to output)

        # Use the specified hidden_activation for backward step
        dA_prev_temp, dW_temp, db_temp = activation_backward(dA_input, current_cache, activation=hidden_activation)

        # Store gradients for the current layer l (parameters W{l+1}, b{l+1})
        grads[f"dA{l}"] = dA_prev_temp
        grads[f"dW{l + 1}"] = dW_temp
        grads[f"db{l + 1}"] = db_temp

    # Clean up intermediate dA gradients (only need dW and db for updates)
    grads_final = {key: val for key, val in grads.items() if not key.startswith('dA')}
    return grads_final


# --- Parameter Update ---

def update_params(parameters: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], alpha: float) -> Dict[str, np.ndarray]:
    """Updates parameters using the gradient descent update rule."""
    L = len(parameters) // 2 # Number of layers
    for l in range(1, L + 1):
        parameters[f"W{l}"] = parameters[f"W{l}"] - alpha * grads[f"dW{l}"]
        parameters[f"b{l}"] = parameters[f"b{l}"] - alpha * grads[f"db{l}"]
    return parameters

# --- Prediction & Accuracy ---

def get_predictions(AL: np.ndarray) -> np.ndarray:
    """Gets the index of the max probability for each example."""
    return np.argmax(AL, axis=0)

def get_accuracy(predictions: np.ndarray, Y: np.ndarray) -> float:
    """Calculates the accuracy of predictions."""
    # Ensure Y is flattened
    Y_flat = Y.flatten()
    predictions_flat = predictions.flatten()
    if predictions_flat.shape != Y_flat.shape:
         print(f"WARN [get_accuracy]: Shape mismatch - Predictions: {predictions_flat.shape}, Y: {Y_flat.shape}", file=sys.stderr)
         return 0.0 # Or handle error appropriately
    print(f"  DEBUG [get_accuracy]: Comparing {np.sum(predictions_flat == Y_flat)} correct out of {Y_flat.size} examples", file=sys.stderr)
    return np.sum(predictions_flat == Y_flat) / Y_flat.size

# --- Loss Function ---

def compute_loss(AL: np.ndarray, Y: np.ndarray) -> float:
    """Computes the cross-entropy loss.

    Args:
        AL (np.ndarray): Probabilities from softmax output layer (num_classes, num_examples).
        Y (np.ndarray): True labels, vector of shape (1, num_examples) or (num_examples,).

    Returns:
        float: Cross-entropy cost.
    """
    m = Y.shape[1] if Y.ndim > 1 else Y.shape[0]
    num_classes = AL.shape[0]

    # Ensure Y is shape (1, m)
    Y = Y.reshape(1, m)
    Y_flat = Y.flatten().astype(int)

    # Check if one-hot encoding is needed (if Y contains class indices)
    if Y.shape[0] == 1 and num_classes > 1: # Assume Y needs one-hot encoding
        Y_one_hot = one_hot(Y_flat, num_classes)
    elif Y.shape[0] == num_classes: # Assume Y is already one-hot
        Y_one_hot = Y
    else:
        raise ValueError(f"Y shape {Y.shape} is not compatible with AL shape {AL.shape} for loss calculation.")

    # Cross-entropy loss: - (1/m) * sum(Y_one_hot * log(AL))
    # Clip AL to prevent log(0) - already done in softmax, but double-check
    epsilon = 1e-10
    AL_clipped = np.clip(AL, epsilon, 1. - epsilon)
    cost = - (1./m) * np.sum(Y_one_hot * np.log(AL_clipped))

    # Ensure cost is a scalar float
    cost = np.squeeze(cost)
    if np.isnan(cost) or np.isinf(cost):
        print(f"ERROR [compute_loss]: Loss is NaN or Inf. AL min/max: {np.min(AL)}, {np.max(AL)}", file=sys.stderr)
        # Optionally return a large number or raise an error
        return np.inf
    return float(cost)


# --- Gradient Descent ---
stop_training_flag = False # Global flag to signal stopping

def gradient_descent(X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray,
                     alpha: float, iterations: int,
                     parameters: Dict[str, np.ndarray],
                     hidden_activation: str = "relu", # Add activation function parameter
                     progress_callback: Optional[Callable[[int, int, float, float], bool]] = None, # iter, total, loss, val_acc -> continue?
                     patience: int = 0) -> Optional[Tuple[Dict[str, np.ndarray], List[float], List[float]]]:
    """Optimizes parameters using gradient descent.

    Args:
        X_train, Y_train: Training data and labels.
        X_dev, Y_dev: Development (validation) data and labels.
        alpha (float): Learning rate.
        iterations (int): Number of iterations (epochs).
        parameters (Dict[str, np.ndarray]): Initialized parameters.
        hidden_activation (str): Activation for hidden layers ('relu', 'sigmoid', 'tanh').
        progress_callback (Optional[Callable]): Function called each iteration
                                                with (iter_num(1-based), total_iters, train_loss, val_acc).
                                                Should return False to stop training.
        patience (int): Number of iterations to wait for validation accuracy improvement
                        before stopping early (0 to disable).

    Returns:
        Optional[Tuple[Dict[str, np.ndarray], List[float], List[float]]]:
            A tuple containing:
            - final_parameters (Dict): The learned parameters.
            - loss_history (List): List of training loss per iteration.
            - val_accuracy_history (List): List of validation accuracy per iteration.
            Returns None if training was stopped prematurely or an error occurred.
    """
    global stop_training_flag
    print(f"--- Starting Gradient Descent (Hidden Activation: {hidden_activation}) ---", file=sys.stderr)
    loss_history = []
    val_accuracy_history = []
    best_val_acc = -1.0
    epochs_no_improve = 0
    best_params = parameters # Store initial params as best initially

    # --- Get Number of Classes --- #
    L = len(parameters) // 2
    num_classes = parameters[f'W{L}'].shape[0]
    print(f"  DEBUG [gradient_descent]: Inferred num_classes = {num_classes}", file=sys.stderr)
    # ----------------------------- #

    # Ensure Y_dev is 1D for accuracy calculation
    Y_dev_flat = Y_dev.flatten() if Y_dev.ndim > 1 else Y_dev

    for i in range(iterations):
        if stop_training_flag:
            print(f"INFO: Training stopped at iteration {i} by flag.", file=sys.stderr)
            stop_training_flag = False # Reset flag
            return None # Indicate premature stop

        # Forward propagation on training data
        AL_train, caches, forward_status_train = forward_prop(X_train, parameters, hidden_activation=hidden_activation)
        if not forward_status_train:
            print(f"ERROR: Forward propagation failed on training set at iteration {i}. Stopping training.", file=sys.stderr)
            return None

        # Compute training loss
        train_loss = compute_loss(AL_train, Y_train)
        loss_history.append(train_loss)
        if np.isnan(train_loss) or np.isinf(train_loss):
             print(f"ERROR: Training loss is NaN/Inf at iteration {i}. Stopping training.", file=sys.stderr)
             return None

        # Backward propagation
        grads = backward_prop(AL_train, Y_train, caches, hidden_activation=hidden_activation)

        # Check gradients for NaN/Inf
        for key, grad in grads.items():
             if np.isnan(grad).any() or np.isinf(grad).any():
                  print(f"ERROR: Stopping training at iteration {i} due to NaN/inf in gradient '{key}'.", file=sys.stderr)
                  return None

        # Update parameters
        parameters = update_params(parameters, grads, alpha)

        # Check parameters for NaN/Inf
        for key, param in parameters.items():
            if np.isnan(param).any() or np.isinf(param).any():
                 print(f"ERROR: Stopping training at iteration {i} due to NaN/inf in updated parameter '{key}'.", file=sys.stderr)
                 return None

        # --- Calculate Validation Accuracy ---
        # Predict on development set using current parameters
        AL_dev, _, forward_status_dev = forward_prop(X_dev, parameters, hidden_activation=hidden_activation)
        if not forward_status_dev:
            print(f"WARN: Forward propagation failed on dev set at iteration {i}. Skipping validation.", file=sys.stderr)
            val_acc = 0.0 # Assign a default or handle as appropriate
        else:
            predictions_dev = get_predictions(AL_dev)
            val_acc = get_accuracy(predictions_dev, Y_dev_flat) # Use flattened Y_dev
        val_accuracy_history.append(val_acc)
        # -----------------------------------\n
        # --- Log Progress & Call Callback ---
        print(f"Iter: {i+1}/{iterations} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}", file=sys.stderr)
        if progress_callback:
            # Pass iteration (1-based), total, loss, accuracy
            should_continue = progress_callback(i + 1, iterations, train_loss, val_acc)
            if not should_continue:
                print(f"INFO: Training stopped at iteration {i+1} by callback.", file=sys.stderr)
                return None # Indicate stop via callback
        # -----------------------------------

        # --- Early Stopping Check ---
        if patience > 0:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                best_params = parameters.copy() # Save the best parameters found so far
                print(f"  INFO: New best validation accuracy: {best_val_acc:.4f} at epoch {i+1}", file=sys.stderr)
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"INFO: Early stopping triggered at iteration {i+1} after {patience} epochs without validation accuracy improvement.", file=sys.stderr)
                print(f"INFO: Returning parameters from epoch {i+1-patience} with best validation accuracy: {best_val_acc:.4f}", file=sys.stderr)
                # Return the *best* parameters found and history up to this point
                return best_params, loss_history, val_accuracy_history
        # ---------------------------

    print("--- Gradient Descent Finished ---", file=sys.stderr)
    # If early stopping was enabled, return best_params, otherwise return final params
    final_params = best_params if patience > 0 else parameters
    return final_params, loss_history, val_accuracy_history

# --- Prediction Function ---

def make_predictions(X: np.ndarray, parameters: Dict[str, np.ndarray], hidden_activation: str = "relu") -> np.ndarray:
    """Makes predictions using the learned parameters."""
    AL, _, status = forward_prop(X, parameters, hidden_activation=hidden_activation)
    if not status:
        print("ERROR: Forward propagation failed during prediction.", file=sys.stderr)
        # Return default predictions (e.g., zeros) or raise error
        num_classes = parameters[f"W{len(parameters)//2}"].shape[0]
        return np.zeros(X.shape[1], dtype=int)
    predictions = get_predictions(AL)
    return predictions
