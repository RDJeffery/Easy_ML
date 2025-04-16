# ui/tabs/train_tab.py

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout,
    QProgressBar
)
from PyQt5.QtCore import Qt

# --- Add project root to sys.path for robust model imports --- #
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------------------------------- #

# Import model classes for type checking (optional but good practice)
# Assuming they are available via sys.path from the main execution context
try:
    from model.neural_net import SimpleNeuralNetwork
except ImportError:
    SimpleNeuralNetwork = None
try:
    from model.cnn_model import CNNModel
except ImportError:
    CNNModel = None

class TrainTab(QWidget):
    """Widget defining the UI components for the Train Tab."""
    def __init__(self, parent_window=None):
        super().__init__(parent=parent_window)
        self.parent_window = parent_window

        main_layout = QVBoxLayout(self)

        # Create and add the main training group
        # Note: We store the group itself on the parent for enabling/disabling
        self.parent_window.training_group = self._create_training_group()
        main_layout.addWidget(self.parent_window.training_group)

        # Create and add the model management group
        model_mgmt_group = self._create_model_mgmt_group()
        main_layout.addWidget(model_mgmt_group)

        # Create and add the expand plot button layout
        plot_info_layout = self._create_plot_info_layout()
        main_layout.addLayout(plot_info_layout)

        main_layout.addStretch() # Push content up

    def _create_training_group(self):
        """Creates the GroupBox for training configuration widgets."""
        training_group = QGroupBox("Training Controls (No dataset loaded)")
        main_layout = QVBoxLayout()
        form_layout = QFormLayout()

        # --- Model Selection --- #
        self.parent_window.model_type_combo = QComboBox()
        self.parent_window.model_type_combo.setToolTip("Select the type of model to train.")
        model_types = []
        if SimpleNeuralNetwork: model_types.append("Simple NN")
        if CNNModel: model_types.append("CNN")

        if not model_types:
            self.parent_window.model_type_combo.addItem("No Models Found")
            self.parent_window.model_type_combo.setEnabled(False)
        else:
            self.parent_window.model_type_combo.addItems(model_types)
            # Set initial based on MainWindow's current_model_type attribute
            initial_model_type = getattr(self.parent_window, 'current_model_type', None)
            if initial_model_type and initial_model_type in model_types:
                 self.parent_window.model_type_combo.setCurrentText(initial_model_type)
            elif "CNN" in model_types: # Fallback preference
                self.parent_window.model_type_combo.setCurrentText("CNN")
            elif "Simple NN" in model_types:
                 self.parent_window.model_type_combo.setCurrentText("Simple NN")

        self.parent_window.model_type_combo.currentTextChanged.connect(self.parent_window._on_model_type_changed)
        form_layout.addRow("Model Type:", self.parent_window.model_type_combo)
        main_layout.addLayout(form_layout)
        # ------------------------ #

        # --- Simple NN Group --- #
        self.parent_window.layer_sizes_group = QGroupBox("Simple NN: Specific Parameters")
        layer_sizes_form_layout = QFormLayout()
        self.parent_window.hidden_layers_input = QLineEdit("100")
        self.parent_window.hidden_layers_input.setPlaceholderText("e.g., 128, 64, 32")
        self.parent_window.hidden_layers_input.setToolTip("Enter comma-separated sizes for hidden layers")
        layer_sizes_form_layout.addRow("Hidden Layers:", self.parent_window.hidden_layers_input)
        self.parent_window.activation_combo = QComboBox()
        self.parent_window.activation_combo.addItems(["ReLU", "Sigmoid", "Tanh"])
        self.parent_window.activation_combo.setToolTip("Activation function for hidden layers (ReLU recommended)")
        layer_sizes_form_layout.addRow("Activation:", self.parent_window.activation_combo)
        self.parent_window.optimizer_combo = QComboBox()
        self.parent_window.optimizer_combo.addItems(["Adam", "GradientDescent"])
        self.parent_window.optimizer_combo.setToolTip("Optimization algorithm (Adam recommended)")
        layer_sizes_form_layout.addRow("Optimizer:", self.parent_window.optimizer_combo)
        self.parent_window.l2_lambda_input = QDoubleSpinBox()
        self.parent_window.l2_lambda_input.setRange(0.0, 10.0); self.parent_window.l2_lambda_input.setDecimals(5); self.parent_window.l2_lambda_input.setSingleStep(0.001); self.parent_window.l2_lambda_input.setValue(0.0)
        self.parent_window.l2_lambda_input.setToolTip("L2 regularization strength (lambda). 0.0 = disabled.")
        layer_sizes_form_layout.addRow("L2 λ:", self.parent_window.l2_lambda_input)
        self.parent_window.dropout_keep_prob_input = QDoubleSpinBox()
        self.parent_window.dropout_keep_prob_input.setRange(0.1, 1.0); self.parent_window.dropout_keep_prob_input.setDecimals(2); self.parent_window.dropout_keep_prob_input.setSingleStep(0.1); self.parent_window.dropout_keep_prob_input.setValue(1.0)
        self.parent_window.dropout_keep_prob_input.setToolTip("Probability of keeping a neuron during training (Dropout). 1.0 = disabled.")
        layer_sizes_form_layout.addRow("Dropout Keep Prob:", self.parent_window.dropout_keep_prob_input)
        self.parent_window.layer_sizes_group.setLayout(layer_sizes_form_layout)
        main_layout.addWidget(self.parent_window.layer_sizes_group)
        # ---------------------- #

        # --- Common Parameters Group --- #
        self.parent_window.common_params_group = QGroupBox("Common Training Parameters")
        common_form_layout = QFormLayout()
        self.parent_window.epochs_input = QSpinBox()
        self.parent_window.epochs_input.setRange(1, 10000); self.parent_window.epochs_input.setValue(100)
        self.parent_window.epochs_input.setToolTip("Number of training iterations through the entire dataset")
        common_form_layout.addRow("Epochs:", self.parent_window.epochs_input)
        self.parent_window.learning_rate_input = QDoubleSpinBox()
        self.parent_window.learning_rate_input.setRange(0.00001, 1.0); self.parent_window.learning_rate_input.setDecimals(5); self.parent_window.learning_rate_input.setSingleStep(0.0001); self.parent_window.learning_rate_input.setValue(0.001)
        self.parent_window.learning_rate_input.setToolTip("Controls how much model weights are adjusted (learning rate / alpha)")
        common_form_layout.addRow("Learn Rate (α):", self.parent_window.learning_rate_input)
        self.parent_window.patience_input = QSpinBox()
        self.parent_window.patience_input.setRange(0, 1000); self.parent_window.patience_input.setValue(10)
        self.parent_window.patience_input.setToolTip("Epochs to wait for validation improvement before stopping early (0=disabled)")
        common_form_layout.addRow("Patience (Early Stop):", self.parent_window.patience_input)
        self.parent_window.batch_size_input = QSpinBox()
        self.parent_window.batch_size_input.setRange(1, 2048); self.parent_window.batch_size_input.setValue(64)
        self.parent_window.batch_size_input.setToolTip("Number of samples per gradient update (powers of 2 common)")
        common_form_layout.addRow("Batch Size:", self.parent_window.batch_size_input)
        self.parent_window.common_params_group.setLayout(common_form_layout)
        main_layout.addWidget(self.parent_window.common_params_group)
        # ---------------------------- #

        # --- CNN Parameters Group --- #
        self.parent_window.cnn_params_group = QGroupBox("CNN: Specific Parameters")
        cnn_form_layout = QFormLayout()
        cnn_form_layout.addRow(QLabel("Filters, Kernel Size, etc. TBD..."))
        self.parent_window.cnn_params_group.setLayout(cnn_form_layout)
        main_layout.addWidget(self.parent_window.cnn_params_group)
        # -------------------------- #

        # --- Training Actions --- #
        action_layout = QHBoxLayout()
        self.parent_window.start_button = QPushButton("🚀 Start Training")
        self.parent_window.start_button.clicked.connect(self.parent_window.start_training)
        self.parent_window.start_button.setEnabled(False)
        self.parent_window.start_button.setToolTip("Begin the model training process (requires loaded data)")
        action_layout.addWidget(self.parent_window.start_button)
        self.parent_window.stop_button = QPushButton("🛑 Stop Training")
        self.parent_window.stop_button.clicked.connect(self.parent_window._stop_training)
        self.parent_window.stop_button.setEnabled(False)
        self.parent_window.stop_button.setToolTip("Interrupt the currently running training process")
        action_layout.addWidget(self.parent_window.stop_button)
        main_layout.addLayout(action_layout)
        # ----------------------- #

        # --- Progress Bar --- #
        self.parent_window.progress_bar = QProgressBar()
        self.parent_window.progress_bar.setTextVisible(True)
        self.parent_window.progress_bar.setValue(0)
        self.parent_window.progress_bar.setToolTip("Shows the progress of the current training run")
        main_layout.addWidget(self.parent_window.progress_bar)
        # ------------------- #

        # --- Accuracy Display --- #
        self.parent_window.accuracy_label = QLabel("Final Validation Accuracy: --")
        self.parent_window.accuracy_label.setToolTip("Accuracy achieved on the validation set after training completes")
        main_layout.addWidget(self.parent_window.accuracy_label)
        # ---------------------- #

        training_group.setLayout(main_layout)
        training_group.setEnabled(False) # Disabled until data is loaded (controlled by MainWindow)

        # Set initial visibility of model-specific groups
        # This requires _update_hyperparameter_visibility to be callable on parent
        if hasattr(self.parent_window, '_update_hyperparameter_visibility'):
             self.parent_window._update_hyperparameter_visibility()

        return training_group

    def _create_model_mgmt_group(self):
        """Creates the GroupBox for saving and loading model weights."""
        mgmt_group = QGroupBox("Model Management")
        layout = QHBoxLayout()
        self.parent_window.save_button = QPushButton("💾 Save Weights")
        self.parent_window.save_button.setToolTip("Save the current model weights and biases")
        self.parent_window.save_button.clicked.connect(self.parent_window.save_weights)
        self.parent_window.save_button.setEnabled(False)
        layout.addWidget(self.parent_window.save_button)
        self.parent_window.load_button = QPushButton("📂 Load Weights")
        self.parent_window.load_button.setToolTip("Load previously saved model weights and biases")
        self.parent_window.load_button.clicked.connect(self.parent_window.load_weights)
        # Load button enabling is handled in MainWindow._post_load_update
        layout.addWidget(self.parent_window.load_button)
        layout.addStretch()
        mgmt_group.setLayout(layout)
        return mgmt_group

    def _create_plot_info_layout(self):
        """Creates the layout for the expand plot button."""
        plot_info_layout = QHBoxLayout()
        expand_plot_button = QPushButton("🔎 Expand Plot")
        expand_plot_button.clicked.connect(self.parent_window._show_expanded_plot)
        plot_info_layout.addWidget(expand_plot_button)
        plot_info_label = QLabel("<- View training history in separate window")
        plot_info_label.setStyleSheet("font-style: italic; color: grey;")
        plot_info_layout.addWidget(plot_info_label)
        plot_info_layout.addStretch()
        return plot_info_layout 