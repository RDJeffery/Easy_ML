# ui/tabs/test_tab.py

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QGroupBox, QSplitter
)
from PyQt5.QtCore import Qt

# --- Add project root to sys.path for robust widget imports --- #
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------------------------------- #

# Import custom widgets used on this tab
# Assuming they are available via sys.path from the main execution context
try:
    from ui.drawing_canvas import DrawingCanvas
except ImportError as e:
    print(f"ERROR: Could not import DrawingCanvas in TestTab: {e}")
    class DrawingCanvas(QWidget): pass # Dummy

try:
    from ui.probability_bar_graph import PredictionVisualizer
except ImportError as e:
    print(f"ERROR: Could not import PredictionVisualizer in TestTab: {e}")
    class PredictionVisualizer(QWidget):
         def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs); QLabel("Error", self)
         def clear_graph(self): pass
         def set_probabilities(self, *args, **kwargs): pass
         def get_predicted_class_name(self): return None

class TestTab(QWidget):
    """Widget defining the UI components for the Test/Inference Tab."""
    def __init__(self, parent_window=None):
        super().__init__(parent=parent_window)
        self.parent_window = parent_window

        layout = QVBoxLayout(self)

        # --- Create UI Elements (Store references locally on self) --- #
        self._create_ui_elements()

        # --- Assemble Layout --- #
        inference_group = self._assemble_layout()
        layout.addWidget(inference_group)
        layout.addStretch() # Push content up

        # --- Connect Signals --- #
        self._connect_signals()

        # --- Initial State --- #
        self._set_initial_state()

    def _create_ui_elements(self):
        """Creates the individual UI widgets for the tab."""
        # File Prediction
        self.predict_file_button = QPushButton("📂 Select & Predict File")
        self.predict_file_button.setToolTip("Load an image file (e.g., PNG, JPG, JPEG) for prediction")

        # Drawing Canvas & Buttons
        self.drawing_canvas = DrawingCanvas(width=140, height=140, parent=self) # Parent is now TestTab
        self.predict_drawing_button = QPushButton("Predict Drawing")
        self.predict_drawing_button.setToolTip("Predict the digit currently drawn on the canvas")
        self.clear_drawing_button = QPushButton("Clear Canvas")

        # Results Display
        self.image_preview_label = QLabel("Image Preview Here")
        self.image_preview_label.setAlignment(Qt.AlignCenter)
        self.image_preview_label.setMinimumSize(100, 100)
        self.image_preview_label.setStyleSheet("border: 1px solid gray; background-color: white;")

        self.prediction_visualizer = PredictionVisualizer()

        self.feedback_label = QLabel("Prediction: N/A")
        self.feedback_label.setAlignment(Qt.AlignCenter)
        font = self.feedback_label.font()
        font.setPointSize(11); font.setBold(True) # Slightly smaller than old label
        self.feedback_label.setFont(font)
        self.feedback_label.setToolTip("Shows the predicted class")

        self.yes_button = QPushButton("👍 Yes")
        self.no_button = QPushButton("👎 No")

    def _assemble_layout(self):
        """Creates layouts and assembles the widgets into the main group box."""
        inference_group = QGroupBox("Model Testing (Inference)")
        main_group_layout = QVBoxLayout(inference_group) # Layout for the group box

        # Splitter
        splitter = QSplitter(Qt.Horizontal)

        # --- Left Side: Input --- #
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)

        # File Input Row
        file_input_layout = QHBoxLayout()
        file_input_layout.addWidget(QLabel("Test with Image File:"))
        file_input_layout.addWidget(self.predict_file_button)
        file_input_layout.addStretch()
        input_layout.addLayout(file_input_layout)

        # Drawing Input Group
        drawing_group = QGroupBox("Draw Input")
        drawing_layout = QVBoxLayout(drawing_group)
        drawing_layout.addWidget(self.drawing_canvas)
        drawing_buttons_layout = QHBoxLayout()
        drawing_buttons_layout.addWidget(self.predict_drawing_button)
        drawing_buttons_layout.addWidget(self.clear_drawing_button)
        drawing_layout.addLayout(drawing_buttons_layout)
        input_layout.addWidget(drawing_group)
        input_layout.addStretch()

        # --- Right Side: Results --- #
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)

        results_layout.addWidget(self.image_preview_label)
        results_layout.addWidget(self.prediction_visualizer)
        results_layout.addWidget(self.feedback_label)

        feedback_buttons_layout = QHBoxLayout()
        feedback_buttons_layout.addStretch()
        feedback_buttons_layout.addWidget(self.yes_button)
        feedback_buttons_layout.addWidget(self.no_button)
        feedback_buttons_layout.addStretch()
        results_layout.addLayout(feedback_buttons_layout)
        results_layout.addStretch()

        # --- Add to Splitter --- #
        splitter.addWidget(input_widget)
        splitter.addWidget(results_widget)
        splitter.setStretchFactor(0, 1) # Adjust ratios as needed
        splitter.setStretchFactor(1, 2)

        main_group_layout.addWidget(splitter)
        return inference_group

    def _connect_signals(self):
        """Connects signals for widgets owned by this tab."""
        # Connect to methods in the parent window (MainWindow)
        self.predict_file_button.clicked.connect(self.parent_window._predict_image_file)
        self.predict_drawing_button.clicked.connect(self.parent_window._predict_drawing)
        self.clear_drawing_button.clicked.connect(self.drawing_canvas.clearCanvas)
        # Yes/No buttons are connected in MainWindow __init__

    def _set_initial_state(self):
        """Sets the initial enabled/disabled state of widgets."""
        self.predict_file_button.setEnabled(False)
        self.predict_drawing_button.setEnabled(False)
        self.yes_button.setEnabled(False)
        self.no_button.setEnabled(False)
        # clear_drawing_button is always enabled 