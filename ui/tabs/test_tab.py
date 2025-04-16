# ui/tabs/test_tab.py

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QGroupBox
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
    from ui.probability_bar_graph import ProbabilityBarGraph
except ImportError as e:
    print(f"ERROR: Could not import ProbabilityBarGraph in TestTab: {e}")
    class ProbabilityBarGraph(QWidget): pass # Dummy

class TestTab(QWidget):
    """Widget defining the UI components for the Test/Inference Tab."""
    def __init__(self, parent_window=None):
        super().__init__(parent=parent_window)
        self.parent_window = parent_window

        layout = QVBoxLayout(self)
        inference_group = self._create_inference_group()
        layout.addWidget(inference_group)
        layout.addStretch() # Push content up

    def _create_inference_group(self):
        """Creates the GroupBox containing widgets for testing the model (inference)."""
        inference_group = QGroupBox("Model Testing (Inference)")
        layout = QVBoxLayout()

        info_label = QLabel(
            "Use this section to test the currently trained or loaded model on new data. "
            "You can either select an image file or draw a digit yourself."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-style: italic; padding-bottom: 10px;")
        layout.addWidget(info_label)

        # --- Test with File Section ---
        file_test_layout = QHBoxLayout()
        file_label = QLabel("Test with Image File:")
        file_test_layout.addWidget(file_label)
        # Store button reference on parent window
        self.parent_window.predict_file_button = QPushButton("Select & Predict File")
        self.parent_window.predict_file_button.setToolTip("Load an image file (e.g., PNG, JPG, JPEG) for prediction")
        self.parent_window.predict_file_button.clicked.connect(self.parent_window._predict_image_file)
        file_test_layout.addWidget(self.parent_window.predict_file_button)
        file_test_layout.addStretch()
        layout.addLayout(file_test_layout)

        # --- Test with Drawing Section ---
        drawing_test_layout = QHBoxLayout()
        # Instantiate DrawingCanvas, store reference on parent
        self.parent_window.drawing_canvas = DrawingCanvas(width=140, height=140, parent=self.parent_window) # Pass main window as parent
        drawing_test_layout.addWidget(self.parent_window.drawing_canvas)

        drawing_buttons_layout = QVBoxLayout()
        # Store button reference on parent window
        self.parent_window.predict_drawing_button = QPushButton("Predict Drawing")
        self.parent_window.predict_drawing_button.setToolTip("Predict the digit currently drawn on the canvas")
        self.parent_window.predict_drawing_button.clicked.connect(self.parent_window._predict_drawing)
        drawing_buttons_layout.addWidget(self.parent_window.predict_drawing_button)

        clear_button = QPushButton("Clear Canvas")
        # Connect directly to the canvas stored on the parent window
        clear_button.clicked.connect(self.parent_window.drawing_canvas.clearCanvas)
        drawing_buttons_layout.addWidget(clear_button)
        drawing_buttons_layout.addStretch()
        drawing_test_layout.addLayout(drawing_buttons_layout)
        layout.addLayout(drawing_test_layout)

        # --- Results Display Section ---
        results_layout = QHBoxLayout()
        # Store label references on parent window
        self.parent_window.image_preview_label = QLabel("Image Preview Here")
        self.parent_window.image_preview_label.setAlignment(Qt.AlignCenter)
        self.parent_window.image_preview_label.setMinimumSize(100, 100)
        self.parent_window.image_preview_label.setStyleSheet("border: 1px solid gray; background-color: white;")
        results_layout.addWidget(self.parent_window.image_preview_label)

        results_display_vlayout = QVBoxLayout()
        self.parent_window.prediction_result_label = QLabel("Prediction: N/A")
        font = self.parent_window.prediction_result_label.font()
        font.setPointSize(12); font.setBold(True)
        self.parent_window.prediction_result_label.setFont(font)
        self.parent_window.prediction_result_label.setAlignment(Qt.AlignCenter)
        self.parent_window.prediction_result_label.setToolTip("Shows the predicted class and confidence")
        results_display_vlayout.addWidget(self.parent_window.prediction_result_label)

        # Instantiate ProbabilityBarGraph, store reference on parent
        self.parent_window.probability_graph = ProbabilityBarGraph()
        results_display_vlayout.addWidget(self.parent_window.probability_graph)
        results_layout.addLayout(results_display_vlayout)
        layout.addLayout(results_layout)

        inference_group.setLayout(layout)
        # Initial enabling/disabling is handled by MainWindow
        self.parent_window.predict_file_button.setEnabled(False)
        self.parent_window.predict_drawing_button.setEnabled(False)

        return inference_group 