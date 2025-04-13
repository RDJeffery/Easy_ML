import numpy as np
import sys
from typing import Tuple, Callable, Optional, List

def init_params(num_classes: int = 10, hidden_layer_size: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initializes weights and biases for a 2-layer neural network.

    Args:
        num_classes (int): The number of output classes (neurons in the output layer). Defaults to 10.
        hidden_layer_size (int): The number of neurons in the hidden layer. Defaults to 10.

    Returns:
        tuple: (W1, b1, W2, b2) initialized parameters as NumPy arrays.
    """
    # Input layer size (784 for 28x28 images)
    n_x: int = 784
    # Hidden layer size
    n_h: int = hidden_layer_size
    # Output layer size (number of classes)
    n_y: int = num_classes

    # Initialize weights with small random values and biases to zero
    W1: np.ndarray = np.random.rand(n_h, n_x) - 0.5
    b1: np.ndarray = np.zeros((n_h, 1))
    W2: np.ndarray = np.random.rand(n_y, n_h) - 0.5
    b2: np.ndarray = np.zeros((n_y, 1))
    return W1, b1, W2, b2

def ReLU(Z: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit activation function."""
    return np.maximum(Z, 0)

def softmax(Z: np.ndarray) -> np.ndarray:
    """Softmax activation function for output layer.

    Adds epsilon for numerical stability.
    """
    eps: float = 1e-10  # Add epsilon for numerical stability
    # Subtract max for numerical stability before exponentiation
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A: np.ndarray = exp_Z / (np.sum(exp_Z, axis=0, keepdims=True) + eps)
    return A

def forward_prop(W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Performs forward propagation for a 2-layer network.

    Args:
        W1, b1: Parameters for the first layer.
        W2, b2: Parameters for the second layer.
        X: Input data (shape: n_x, m).

    Returns:
        tuple: (Z1, A1, Z2, A2, status)
               Z1, A1: Linear and activation results for layer 1.
               Z2, A2: Linear and activation results for layer 2 (output).
               status is True if successful, False if NaN/inf detected.
    """
    try:
        Z1 = W1.dot(X) + b1
        A1 = ReLU(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = softmax(Z2)

        # Check for NaN/inf in activations
        if np.isnan(A1).any() or np.isinf(A1).any() or \
           np.isnan(A2).any() or np.isinf(A2).any():
            print("ERROR: NaN or Inf detected in activations during forward propagation.", file=sys.stderr)
            return Z1, A1, Z2, A2, False # Indicate failure

        return Z1, A1, Z2, A2, True # Indicate success
    except Exception as e:
        print(f"ERROR during forward propagation: {e}", file=sys.stderr)
        # Return dummy values and failure status if dot products etc fail
        # Shapes might be wrong depending on where error occurred
        dummy_Z1 = np.zeros_like(b1)
        dummy_A1 = np.zeros_like(b1)
        dummy_Z2 = np.zeros_like(b2)
        dummy_A2 = np.zeros((W2.shape[0], X.shape[1]))
        return dummy_Z1, dummy_A1, dummy_Z2, dummy_A2, False

def one_hot(Y: np.ndarray, num_classes: int) -> np.ndarray:
    """Converts an array of integer labels to one-hot encoding.

    Args:
        Y (np.ndarray): 1D array of integer labels (shape: (m,)).
        num_classes (int): The total number of classes.

    Returns:
        np.ndarray: One-hot encoded labels (shape: (num_classes, m)).
    """
    m = Y.size
    one_hot_Y = np.zeros((num_classes, m))
    one_hot_Y[Y, np.arange(m)] = 1
    return one_hot_Y

def deriv_ReLU(Z: np.ndarray) -> np.ndarray:
    """Derivative of the ReLU activation function."""
    return Z > 0

def backward_prop(Z1: np.ndarray, A1: np.ndarray, Z2: np.ndarray, A2: np.ndarray,
                  W1: np.ndarray, W2: np.ndarray, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Performs backward propagation to calculate gradients.

    Args:
        Z1, A1, Z2, A2: Outputs from forward propagation.
        W1, W2: Weights used in forward propagation.
        X: Input data.
        Y: True labels.

    Returns:
        tuple: (dW1, db1, dW2, db2) gradients for the parameters.
    """
    m: int = Y.size # Get batch size
    num_classes: int = W2.shape[0]
    Y_one_hot: np.ndarray = one_hot(Y, num_classes)
    dZ2: np.ndarray = A2 - Y_one_hot
    dW2: np.ndarray = 1 / m * dZ2.dot(A1.T)
    db2: np.ndarray = 1 / m * np.sum(dZ2, axis=1, keepdims=True) # Sum over batch dimension
    dZ1: np.ndarray = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1: np.ndarray = 1 / m * dZ1.dot(X.T)
    db1: np.ndarray = 1 / m * np.sum(dZ1, axis=1, keepdims=True) # Sum over batch dimension
    return dW1, db1, dW2, db2

def update_params(W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray,
                  dW1: np.ndarray, db1: np.ndarray, dW2: np.ndarray, db2: np.ndarray,
                  alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Updates parameters using gradient descent rule."""
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2: np.ndarray) -> np.ndarray:
    """Gets the predicted class index from the output activations."""
    return np.argmax(A2, 0)

def get_accuracy(predictions: np.ndarray, Y: np.ndarray) -> float:
    """Calculates the accuracy between predictions and true labels."""
    return np.sum(predictions == Y) / Y.size

# Function to compute Cross-Entropy Loss
def compute_loss(A2: np.ndarray, Y: np.ndarray) -> float:
    """Computes the cross-entropy loss.

    Handles potential log(0) using epsilon.
    Determines num_classes from the network output A2.
    """
    m: int = Y.size
    num_classes: int = A2.shape[0] # Get num_classes from network output shape
    one_hot_Y: np.ndarray = one_hot(Y, num_classes) # Use correct num_classes
    # Add epsilon to avoid log(0)
    eps: float = 1e-10
    # Ensure shapes match before calculation (add check?)
    if one_hot_Y.shape[0] != A2.shape[0]:
        print(f"ERROR: Shape mismatch in compute_loss! one_hot_Y: {one_hot_Y.shape}, A2: {A2.shape}", file=sys.stderr)
        # Handle error appropriately, e.g., return a high loss or NaN
        return np.nan 
    loss: float = -1 / m * np.sum(one_hot_Y * np.log(A2 + eps))
    return loss

def gradient_descent(X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray,
                     alpha: float, iterations: int,
                     W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray,
                     progress_callback: Optional[Callable[[int, int], None]] = None,
                     patience: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[float], List[float]]:
    """Performs gradient descent and returns trained parameters and history.

    Args:
        X_train (np.ndarray): Training features (shape: n_x, m_train).
        Y_train (np.ndarray): Training labels (shape: m_train,).
        X_dev (np.ndarray): Validation features (shape: n_x, m_dev).
        Y_dev (np.ndarray): Validation labels (shape: m_dev,).
        alpha (float): Learning rate.
        iterations (int): Number of training iterations.
        W1, b1, W2, b2 (np.ndarray): Initial model parameters.
        progress_callback (Optional[Callable[[int, int], None]]): Callback for progress updates.
        patience (int): Patience for early stopping (0 to disable).

    Returns:
        tuple: (W1, b1, W2, b2, train_loss_history, val_accuracy_history)
               Final parameters and lists of training loss and validation accuracy.
    """
    train_loss_history: List[float] = []
    val_accuracy_history: List[float] = []

    # Early Stopping Variables
    best_val_accuracy: float = -1.0
    patience_counter: int = 0
    # Use the same interval for checking accuracy as for logging
    log_interval: int = 10

    for i in range(iterations):
        # Training step
        Z1, A1, Z2, A2, status = forward_prop(W1, b1, W2, b2, X_train)
        if not status:
            print(f"ERROR: Stopping training at iteration {i+1} due to NaN/inf in forward propagation.", file=sys.stderr)
            break # Stop training if forward prop failed

        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X_train, Y_train)

        # --- Check for NaN/inf in gradients ---
        if np.isnan(dW1).any() or np.isinf(dW1).any() or \
           np.isnan(db1).any() or np.isinf(db1).any() or \
           np.isnan(dW2).any() or np.isinf(dW2).any() or \
           np.isnan(db2).any() or np.isinf(db2).any():
            print(f"ERROR: Stopping training at iteration {i+1} due to NaN/inf in gradients.", file=sys.stderr)
            break # Stop training
        # ------------------------------------

        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        # --- Check for NaN/inf in updated parameters ---
        if np.isnan(W1).any() or np.isinf(W1).any() or \
           np.isnan(b1).any() or np.isinf(b1).any() or \
           np.isnan(W2).any() or np.isinf(W2).any() or \
           np.isnan(b2).any() or np.isinf(b2).any():
            print(f"ERROR: Stopping training at iteration {i+1} due to NaN/inf in updated weights/biases.", file=sys.stderr)
            break # Stop training
        # -----------------------------------------

        # Call progress callback if provided (call every iteration for smooth bar)
        if progress_callback:
            # Check the return value to allow early stopping via callback
            should_continue = progress_callback(i + 1, iterations)
            if not should_continue:
                print(f"\n--- STOP SIGNALLED via callback at iteration {i+1} ---", file=sys.stderr)
                break # Exit the training loop

        # Log progress and validation accuracy every N iterations (e.g., 10)
        if i % log_interval == 0 or i == iterations - 1:
            # --- Calculate Training Loss ---
            train_loss = compute_loss(A2, Y_train)
            train_loss_history.append(train_loss)

            # --- Calculate Validation Accuracy ---
            val_predictions = make_predictions(X_dev, W1, b1, W2, b2)
            val_accuracy = get_accuracy(val_predictions, Y_dev)
            val_accuracy_history.append(val_accuracy)

            # --- Early Stopping Check ---
            if patience > 0: # Only check if patience is enabled
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0 # Reset counter on improvement
                    # Optionally save the best weights here if needed later
                else:
                    patience_counter += 1 # Increment counter if no improvement
                    print(f"    (Patience check: {patience_counter}/{patience})") # Optional debug log
                if patience_counter >= patience:
                    print(f"\n--- EARLY STOPPING TRIGGERED at iteration {i+1} ---")
                    print(f"Validation accuracy did not improve for {patience} checks.")
                    print(f"Best validation accuracy achieved: {best_val_accuracy:.4f}")
                    break # Exit the training loop
            # --------------------------

            # Print to stdout (which is redirected to UI log)
            print(f"Iter: {i+1}/{iterations} | Train Loss: {train_loss:.4f} | Val Acc: {val_accuracy:.4f}")

    return W1, b1, W2, b2, train_loss_history, val_accuracy_history

def make_predictions(X: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Makes predictions using the trained model.

    Args:
        X (np.ndarray): Input data (shape: n_x, m).
        W1, b1, W2, b2 (np.ndarray): Trained model parameters.

    Returns:
        np.ndarray: Predicted class indices (shape: m,).
    """
    _, _, _, A2, status = forward_prop(W1, b1, W2, b2, X)
    if not status:
        print("Warning: Forward propagation failed during prediction. Returning empty predictions.", file=sys.stderr)
        # Return an empty array or handle error as appropriate
        return np.array([]) # Or raise an exception
    predictions: np.ndarray = get_predictions(A2)
    return predictions
