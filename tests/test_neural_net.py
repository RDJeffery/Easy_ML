# tests/test_neural_net.py

import unittest
import numpy as np
import sys
import os

# Adjust path to import from the parent directory's 'model' module
# This assumes the tests are run from the project root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import neural_net

class TestNeuralNetCore(unittest.TestCase):

    def setUp(self):
        """Set up common parameters and data for tests."""
        self.n_x = 50  # Input features
        self.n_h = 5   # Hidden units
        self.n_y = 3   # Output classes
        self.m = 10    # Batch size (number of samples)

        # Initialize parameters using the function we tested
        self.W1, self.b1, self.W2, self.b2 = neural_net.init_params(
            num_classes=self.n_y, hidden_layer_size=self.n_h
        )
        # Override W1 shape to match self.n_x
        self.W1 = np.random.rand(self.n_h, self.n_x) - 0.5

        # Create dummy data
        self.X = np.random.rand(self.n_x, self.m)
        self.Y = np.random.randint(0, self.n_y, size=self.m)

    def test_init_params_default(self):
        """Test init_params with default arguments."""
        W1, b1, W2, b2 = neural_net.init_params()
        self.assertEqual(W1.shape, (10, 784), "W1 shape mismatch (default)")
        self.assertEqual(b1.shape, (10, 1), "b1 shape mismatch (default)")
        self.assertEqual(W2.shape, (10, 10), "W2 shape mismatch (default)")
        self.assertEqual(b2.shape, (10, 1), "b2 shape mismatch (default)")
        # Check biases are zero
        self.assertTrue(np.all(b1 == 0), "b1 not initialized to zero")
        self.assertTrue(np.all(b2 == 0), "b2 not initialized to zero")
        # Check weights are roughly centered around 0
        self.assertAlmostEqual(np.mean(W1), 0.0, delta=0.1, msg="W1 mean too far from 0")
        self.assertAlmostEqual(np.mean(W2), 0.0, delta=0.1, msg="W2 mean too far from 0")

    def test_init_params_custom(self):
        """Test init_params with custom arguments."""
        num_classes = 5
        hidden_size = 20
        W1, b1, W2, b2 = neural_net.init_params(num_classes=num_classes, hidden_layer_size=hidden_size)
        self.assertEqual(W1.shape, (hidden_size, 784), "W1 shape mismatch (custom)")
        self.assertEqual(b1.shape, (hidden_size, 1), "b1 shape mismatch (custom)")
        self.assertEqual(W2.shape, (num_classes, hidden_size), "W2 shape mismatch (custom)")
        self.assertEqual(b2.shape, (num_classes, 1), "b2 shape mismatch (custom)")
        self.assertTrue(np.all(b1 == 0), "b1 not initialized to zero (custom)")
        self.assertTrue(np.all(b2 == 0), "b2 not initialized to zero (custom)")

    def test_relu(self):
        """Test the ReLU activation function."""
        Z = np.array([[-1, 0, 1], [2, -2, 0.5]])
        A = neural_net.ReLU(Z)
        expected_A = np.array([[0, 0, 1], [2, 0, 0.5]])
        np.testing.assert_array_equal(A, expected_A, "ReLU output incorrect")

    def test_deriv_relu(self):
        """Test the derivative of the ReLU function."""
        Z = np.array([[-1, 0, 1], [2, -2, 0.0]]) # Test edge case 0
        deriv = neural_net.deriv_ReLU(Z)
        expected_deriv = np.array([[0, 0, 1], [1, 0, 0]]) # Z > 0
        np.testing.assert_array_equal(deriv, expected_deriv, "deriv_ReLU output incorrect")

    def test_softmax(self):
        """Test the softmax activation function."""
        # Test single vector
        Z1 = np.array([[1], [2], [3]])
        A1 = neural_net.softmax(Z1)
        self.assertAlmostEqual(np.sum(A1), 1.0, msg="Softmax output should sum to 1 (single vector)")
        self.assertTrue(np.all(A1 > 0), "Softmax probabilities should be positive")
        self.assertEqual(np.argmax(A1), 2, "Softmax highest probability mismatch")

        # Test multiple columns (batch)
        Z_batch = np.array([[1, 0], [2, 1], [3, 2]]) # Shape (3, 2)
        A_batch = neural_net.softmax(Z_batch)
        self.assertEqual(A_batch.shape, Z_batch.shape, "Softmax output shape mismatch (batch)")
        # Check that each column sums to 1
        col_sums = np.sum(A_batch, axis=0)
        np.testing.assert_allclose(col_sums, [1.0, 1.0], rtol=1e-6, err_msg="Softmax columns should sum to 1 (batch)")
        # Check argmax for each column
        np.testing.assert_array_equal(np.argmax(A_batch, axis=0), [2, 2], "Softmax argmax mismatch (batch)")

    def test_one_hot(self):
        """Test the one-hot encoding function."""
        Y = np.array([0, 2, 1, 0]) # Labels for 4 samples
        num_classes = 3
        one_hot_Y = neural_net.one_hot(Y, num_classes)
        expected_one_hot = np.array([
            [1, 0, 0, 1], # Class 0
            [0, 0, 1, 0], # Class 1
            [0, 1, 0, 0]  # Class 2
        ])
        self.assertEqual(one_hot_Y.shape, (num_classes, Y.size), "one_hot shape mismatch")
        np.testing.assert_array_equal(one_hot_Y, expected_one_hot, "one_hot encoding incorrect")

        # Test with single label
        Y_single = np.array([1])
        one_hot_single = neural_net.one_hot(Y_single, num_classes)
        expected_single = np.array([[0], [1], [0]])
        self.assertEqual(one_hot_single.shape, (num_classes, 1), "one_hot shape mismatch (single)")
        np.testing.assert_array_equal(one_hot_single, expected_single, "one_hot encoding incorrect (single)")

    def test_forward_prop_shapes_status(self):
        """Test forward_prop output shapes and status flag."""
        Z1, A1, Z2, A2, status = neural_net.forward_prop(self.W1, self.b1, self.W2, self.b2, self.X)

        self.assertTrue(status, "Forward prop status should be True for valid inputs")
        # Check shapes (A=Activations, Z=Linear part)
        self.assertEqual(Z1.shape, (self.n_h, self.m), "Z1 shape mismatch")
        self.assertEqual(A1.shape, (self.n_h, self.m), "A1 shape mismatch")
        self.assertEqual(Z2.shape, (self.n_y, self.m), "Z2 shape mismatch")
        self.assertEqual(A2.shape, (self.n_y, self.m), "A2 shape mismatch")

    def test_backward_prop_shapes(self):
        """Test backward_prop output shapes."""
        # Need results from forward prop first
        Z1, A1, Z2, A2, status = neural_net.forward_prop(self.W1, self.b1, self.W2, self.b2, self.X)
        self.assertTrue(status, "Forward prop failed, cannot test backward prop")

        # Perform backward prop
        dW1, db1, dW2, db2 = neural_net.backward_prop(Z1, A1, Z2, A2, self.W1, self.W2, self.X, self.Y)

        # Check gradient shapes match parameter shapes
        self.assertEqual(dW1.shape, self.W1.shape, "dW1 shape mismatch")
        self.assertEqual(db1.shape, self.b1.shape, "db1 shape mismatch")
        self.assertEqual(dW2.shape, self.W2.shape, "dW2 shape mismatch")
        self.assertEqual(db2.shape, self.b2.shape, "db2 shape mismatch")

    def test_update_params(self):
        """Test the update_params function."""
        # Get gradients
        Z1, A1, Z2, A2, f_status = neural_net.forward_prop(self.W1, self.b1, self.W2, self.b2, self.X)
        self.assertTrue(f_status, "Forward prop failed, cannot test update_params")
        dW1, db1, dW2, db2 = neural_net.backward_prop(Z1, A1, Z2, A2, self.W1, self.W2, self.X, self.Y)

        # Store original params (make copies)
        W1_orig, b1_orig, W2_orig, b2_orig = self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy()
        alpha = 0.1

        # Update params
        W1_new, b1_new, W2_new, b2_new = neural_net.update_params(
            self.W1, self.b1, self.W2, self.b2, dW1, db1, dW2, db2, alpha
        )

        # Check shapes are maintained
        self.assertEqual(W1_new.shape, W1_orig.shape, "W1 shape changed after update")
        self.assertEqual(b1_new.shape, b1_orig.shape, "b1 shape changed after update")
        self.assertEqual(W2_new.shape, W2_orig.shape, "W2 shape changed after update")
        self.assertEqual(b2_new.shape, b2_orig.shape, "b2 shape changed after update")

        # Check params have changed (assuming non-zero gradients and alpha)
        self.assertFalse(np.allclose(W1_new, W1_orig), "W1 did not change after update")
        self.assertFalse(np.allclose(b1_new, b1_orig), "b1 did not change after update")
        self.assertFalse(np.allclose(W2_new, W2_orig), "W2 did not change after update")
        self.assertFalse(np.allclose(b2_new, b2_orig), "b2 did not change after update")

    def test_compute_loss(self):
        """Test the compute_loss function."""
        # Get activations A2
        _, _, _, A2, status = neural_net.forward_prop(self.W1, self.b1, self.W2, self.b2, self.X)
        self.assertTrue(status, "Forward prop failed, cannot test compute_loss")

        # Ensure A2 has valid probabilities (avoid numerical issues in test itself)
        A2 = np.clip(A2, 1e-10, 1.0) # Clip values
        A2 /= np.sum(A2, axis=0, keepdims=True) # Re-normalize

        loss = neural_net.compute_loss(A2, self.Y)

        self.assertIsInstance(loss, float, "Loss should be a float")
        self.assertGreaterEqual(loss, 0.0, "Loss should be non-negative")

    def test_get_accuracy(self):
        """Test the get_accuracy function."""
        predictions = np.array([0, 1, 2, 0, 1])
        Y_true      = np.array([0, 1, 1, 0, 1]) # 4 out of 5 correct
        accuracy = neural_net.get_accuracy(predictions, Y_true)
        self.assertAlmostEqual(accuracy, 0.8, msg="Accuracy calculation incorrect")

        predictions_none = np.array([0, 0, 0])
        Y_true_none      = np.array([1, 1, 1]) # 0 correct
        accuracy_none = neural_net.get_accuracy(predictions_none, Y_true_none)
        self.assertAlmostEqual(accuracy_none, 0.0, msg="Accuracy calculation incorrect (0% case)")

        predictions_all = np.array([2, 1, 0])
        Y_true_all      = np.array([2, 1, 0]) # All correct
        accuracy_all = neural_net.get_accuracy(predictions_all, Y_true_all)
        self.assertAlmostEqual(accuracy_all, 1.0, msg="Accuracy calculation incorrect (100% case)")

    def test_make_predictions(self):
        """Test the make_predictions function."""
        predictions = neural_net.make_predictions(self.X, self.W1, self.b1, self.W2, self.b2)

        # Check shape (should be 1D array with length m)
        self.assertEqual(predictions.shape, (self.m,), "Predictions shape mismatch")

        # Check values are valid class indices
        self.assertTrue(np.all(predictions >= 0), "Predictions contain negative indices")
        self.assertTrue(np.all(predictions < self.n_y), "Predictions contain indices >= num_classes")
        # Check dtype is integer
        self.assertTrue(issubclass(predictions.dtype.type, np.integer), "Predictions dtype is not integer")


if __name__ == '__main__':
    unittest.main() 