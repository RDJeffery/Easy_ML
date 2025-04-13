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

def softmax(Z: np.ndarray) -> np.ndarray:
    """Softmax activation for the output layer."""
    # Shift Z by subtracting max for numerical stability
    shift_Z = Z - np.max(Z, axis=0, keepdims=True)
    A = np.exp(shift_Z) / np.sum(np.exp(shift_Z), axis=0, keepdims=True)
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

    for l in range(1, L):
        # Xavier/He initialization for weights
        parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2. / layer_dims[l-1])
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))

        assert(parameters[f'W{l}'].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters[f'b{l}'].shape == (layer_dims[l], 1))

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

    if activation == "relu":
        A = ReLU(Z)
    elif activation == "softmax":
        A = softmax(Z)
    else:
        raise ValueError(f"Unknown activation: {activation}")

    activation_cache = Z # Cache Z for ReLU derivative or softmax backprop
    cache = (linear_cache, activation_cache)
    return A, cache

def forward_prop(X: np.ndarray, parameters: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[Tuple[Any, Any]], bool]:
    """Implements forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX model.

    Args:
        X (np.ndarray): Input data (shape: input_size, num_examples).
        parameters (Dict[str, np.ndarray]): Output of init_params().

    Returns:
        Tuple containing:
            AL (np.ndarray): Last post-activation value (output layer).
            caches (List): List of caches containing (linear_cache, activation_cache) for each layer.
            status (bool): True if successful, False if NaN/inf detected.
    """
    caches: List[Tuple[Any, Any]] = []
    A = X
    L = len(parameters) // 2 # Number of layers with parameters

    try:
        # Implement [LINEAR -> RELU] L-1 times
        for l in range(1, L):
            A_prev = A
            Wl = parameters[f'W{l}']
            bl = parameters[f'b{l}']
            A, cache = activation_forward(A_prev, Wl, bl, activation="relu")
            caches.append(cache)
            if np.isnan(A).any() or np.isinf(A).any():
                print(f"ERROR: NaN or Inf detected in activations at layer {l}.", file=sys.stderr)
                return A, caches, False # Indicate failure

        # Implement LINEAR -> SOFTMAX for the last layer
        WL = parameters[f'W{L}']
        bL = parameters[f'b{L}']
        AL, cache = activation_forward(A, WL, bL, activation="softmax")
        caches.append(cache)
        if np.isnan(AL).any() or np.isinf(AL).any():
            print("ERROR: NaN or Inf detected in output layer activations.", file=sys.stderr)
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
    one_hot_Y = np.zeros((num_classes, m))
    # Handle potential out-of-bounds labels gracefully
    valid_indices = (Y >= 0) & (Y < num_classes)
    one_hot_Y[Y[valid_indices], np.arange(m)[valid_indices]] = 1
    if not np.all(valid_indices):
        print(f"Warning: Some labels were outside the expected range [0, {num_classes-1}].", file=sys.stderr)
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
    linear_cache, activation_cache = cache
    Z = activation_cache

    if activation == "relu":
        dZ = dA * deriv_ReLU(Z) # Element-wise product
    elif activation == "softmax":
        # This is slightly simplified for cross-entropy loss.
        # The dZ for softmax+cross-entropy is simply A - Y_one_hot
        # This calculation happens *before* calling activation_backward for the last layer.
        dZ = dA # In this case, dA passed in *is* (A - Y_one_hot)
    else:
        raise ValueError(f"Unknown activation: {activation}")

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def backward_prop(AL: np.ndarray, Y: np.ndarray, caches: List[Tuple[Any, Any]]) -> Dict[str, np.ndarray]:
    """Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SOFTMAX model.

    Args:
        AL (np.ndarray): Probability vector, output of the forward propagation.
        Y (np.ndarray): True "label" vector.
        caches (List): List of caches containing linear and activation caches for each layer.

    Returns:
        Dict[str, np.ndarray]: A dictionary with the gradients with respect to each parameter.
    """
    grads: Dict[str, np.ndarray] = {}
    L = len(caches) # The number of layers
    m = AL.shape[1]
    # Ensure Y is 1D array
    Y = Y.reshape(1, m) if Y.ndim == 1 else Y

    # --- Initializing the backpropagation --- 
    # Calculate dZL for the output layer (Softmax + Cross-Entropy)
    num_classes = AL.shape[0]
    Y_one_hot = one_hot(Y.flatten(), num_classes) # one_hot expects 1D Y
    dZL = AL - Y_one_hot

    # Gradients for the last layer (LINEAR part of Softmax layer)
    current_cache = caches[L-1]
    linear_cache_L, _ = current_cache # Softmax activation cache (Z) not used directly here
    grads[f"dA{L-1}"], grads[f"dW{L}"], grads[f"db{L}"] = linear_backward(dZL, linear_cache_L)

    # --- Loop from l=L-2 to l=0 --- 
    for l in reversed(range(L - 1)):
        # Gradients for layer l: LINEAR -> RELU
        current_cache = caches[l]
        # Pass dA from the previous layer (l+1)
        dA_prev_temp, dW_temp, db_temp = activation_backward(grads[f"dA{l + 1}"], current_cache, activation="relu")
        grads[f"dA{l}"] = dA_prev_temp
        grads[f"dW{l + 1}"] = dW_temp
        grads[f"db{l + 1}"] = db_temp

    # Clean up intermediate dA gradients (only need dW and db)
    grads = {key: val for key, val in grads.items() if not key.startswith('dA')}
    return grads

# --- Parameter Update ---

def update_params(parameters: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], alpha: float) -> Dict[str, np.ndarray]:
    """Updates parameters using gradient descent.

    Args:
        parameters (Dict): Dictionary containing parameters.
        grads (Dict): Dictionary containing gradients.
        alpha (float): Learning rate.

    Returns:
        Dict: Dictionary containing updated parameters.
    """
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter
    for l in range(1, L + 1):
        parameters[f"W{l}"] = parameters[f"W{l}"] - alpha * grads[f"dW{l}"]
        parameters[f"b{l}"] = parameters[f"b{l}"] - alpha * grads[f"db{l}"]

    return parameters

# --- Prediction & Evaluation ---

def get_predictions(AL: np.ndarray) -> np.ndarray:
    """Gets the predicted class index from the output activations."""
    return np.argmax(AL, 0)

def get_accuracy(predictions: np.ndarray, Y: np.ndarray) -> float:
    """Calculates the accuracy between predictions and true labels."""
    # Ensure Y is 1D for comparison
    Y = Y.flatten()
    predictions = predictions.flatten()
    if Y.size != predictions.size:
         print(f"Warning: Label count ({Y.size}) != Prediction count ({predictions.size}) in get_accuracy.", file=sys.stderr)
         return 0.0 # Or handle error as appropriate
    if Y.size == 0:
        return 0.0 # Avoid division by zero
    return np.sum(predictions == Y) / Y.size

def compute_loss(A2: np.ndarray, Y: np.ndarray) -> float:
    """Computes the cross-entropy loss.

    Handles potential log(0) using epsilon.
    Determines num_classes from the network output A2.
    """
    m: int = Y.size
    num_classes: int = A2.shape[0] # Get num_classes from network output shape
    one_hot_Y: np.ndarray = one_hot(Y.flatten(), num_classes) # Ensure Y is 1D
    # Add epsilon to avoid log(0)
    eps: float = 1e-10
    # Ensure shapes match before calculation
    if one_hot_Y.shape[0] != A2.shape[0] or one_hot_Y.shape[1] != A2.shape[1]:
        print(f"ERROR: Shape mismatch in compute_loss! one_hot_Y: {one_hot_Y.shape}, A2: {A2.shape}", file=sys.stderr)
        return np.nan
    loss: float = -1 / m * np.sum(one_hot_Y * np.log(A2 + eps))
    return loss

# --- Gradient Descent Loop ---

# Global flag to allow external stop request (used by TrainingWorker)
stop_training_flag = False

def gradient_descent(X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray,
                     alpha: float, iterations: int,
                     parameters: Dict[str, np.ndarray],
                     progress_callback: Optional[Callable[[int, int], None]] = None,
                     patience: int = 0) -> Tuple[Dict[str, np.ndarray], List[float], List[float]]:
    """Performs gradient descent and returns trained parameters and history.

    Args:
        X_train, Y_train: Training data and labels.
        X_dev, Y_dev: Validation data and labels.
        alpha (float): Learning rate.
        iterations (int): Number of training iterations.
        parameters (Dict): Initial model parameters.
        progress_callback (Optional): Callback for progress updates (iter, total_iters).
        patience (int): Patience for early stopping (0 to disable).

    Returns:
        tuple: (parameters, train_loss_history, val_accuracy_history)
               Final parameters and lists of training loss and validation accuracy.
    """
    global stop_training_flag
    stop_training_flag = False # Reset flag at start

    train_loss_history: List[float] = []
    val_accuracy_history: List[float] = []

    # Early Stopping Variables
    best_val_accuracy: float = -1.0
    patience_counter: int = 0
    log_interval: int = 10

    for i in range(iterations):
        if stop_training_flag:
            print("--- STOP SIGNALLED via global flag ---", file=sys.stderr)
            break

        # Forward propagation
        AL, caches, status = forward_prop(X_train, parameters)
        if not status:
            print(f"ERROR: Stopping training at iteration {i+1} due to NaN/inf in forward propagation.", file=sys.stderr)
            break

        # Backward propagation
        grads = backward_prop(AL, Y_train, caches)

        # Check for NaN/inf in gradients
        for key, grad in grads.items():
            if np.isnan(grad).any() or np.isinf(grad).any():
                 print(f"ERROR: Stopping training at iteration {i+1} due to NaN/inf in gradient '{key}'.", file=sys.stderr)
                 stop_training_flag = True # Use flag to break outer loop cleanly
                 break
        if stop_training_flag:
            break

        # Update parameters
        parameters = update_params(parameters, grads, alpha)

        # Check for NaN/inf in updated parameters
        for key, param in parameters.items():
            if np.isnan(param).any() or np.isinf(param).any():
                 print(f"ERROR: Stopping training at iteration {i+1} due to NaN/inf in updated parameter '{key}'.", file=sys.stderr)
                 stop_training_flag = True
                 break
        if stop_training_flag:
            break

        # Call progress callback
        if progress_callback:
            should_continue = progress_callback(i + 1, iterations)
            if not should_continue:
                print(f"\n--- STOP SIGNALLED via callback at iteration {i+1} ---", file=sys.stderr)
                break

        # Log progress and validation accuracy periodically
        if i % log_interval == 0 or i == iterations - 1:
            train_loss = compute_loss(AL, Y_train)
            train_loss_history.append(train_loss)

            val_predictions = make_predictions(X_dev, parameters)
            val_accuracy = get_accuracy(val_predictions, Y_dev)
            val_accuracy_history.append(val_accuracy)

            # Early Stopping Check
            if patience > 0:
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n--- EARLY STOPPING TRIGGERED at iteration {i+1} ---", file=sys.stderr)
                    break

            print(f"Iter: {i+1}/{iterations} | Train Loss: {train_loss:.4f} | Val Acc: {val_accuracy:.4f}")

    return parameters, train_loss_history, val_accuracy_history

# --- Prediction Function ---

def make_predictions(X: np.ndarray, parameters: Dict[str, np.ndarray]) -> np.ndarray:
    """Makes predictions using the trained multi-layer model."""
    AL, _, status = forward_prop(X, parameters)
    if not status:
        print("Warning: Forward propagation failed during prediction. Returning empty predictions.", file=sys.stderr)
        return np.array([])
    predictions: np.ndarray = get_predictions(AL)
    return predictions
