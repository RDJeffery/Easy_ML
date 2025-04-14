# ui/probability_bar_graph.py

from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtGui import QPainter, QColor, QBrush, QFontMetrics, QPen
from PyQt5.QtCore import Qt, QRectF
import numpy as np

class ProbabilityBarGraph(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.probabilities = []
        self.predicted_class = -1 # Can store index (int) or name (str)
        self.class_names = None   # Optional list of class names
        self.setMinimumHeight(100) # Ensure it has some height
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred) # Allow horizontal expansion

    def set_probabilities(self, probs, predicted_class_or_name=-1, class_names=None):
        # Expects a 1D numpy array or list
        self.probabilities = np.asarray(probs)
        self.predicted_class = predicted_class_or_name # Store index or name
        # Store class names if provided and length matches probabilities
        if class_names and len(class_names) == len(self.probabilities):
            self.class_names = list(class_names)
        else:
            self.class_names = None # Reset if not provided or length mismatch
        self.update() # Trigger a repaint

    def clear_graph(self):
        self.probabilities = []
        self.predicted_class = -1
        self.class_names = None
        self.update()

    def paintEvent(self, event):
        # Check if the probabilities container is empty (works for list or numpy array)
        if len(self.probabilities) == 0:
            return # Don't draw if no data

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        num_classes = len(self.probabilities)
        if num_classes == 0:
            return

        width = self.width()
        height = self.height()
        padding = 5 # Padding around the graph
        label_width = 40 # Space for labels like "0: 0.95"
        graph_area_width = width - padding * 2 - label_width
        # Calculate bar height ensuring at least 1 pixel per bar + padding
        bar_height = max(1.0, (height - padding * (num_classes + 1)) / num_classes)


        if graph_area_width <= 0: return # Not enough space

        # Font for labels
        font = painter.font()
        fm = QFontMetrics(font)
        text_height = fm.height()

        # --- Determine Label Width Dynamically --- #
        max_label_width = 0
        temp_labels = []
        for i, prob in enumerate(self.probabilities):
            if self.class_names:
                label_text = f"{self.class_names[i]}: {prob:.2f}"
            else:
                label_text = f"{i}: {prob:.2f}"
            temp_labels.append(label_text)
            max_label_width = max(max_label_width, fm.horizontalAdvance(label_text))

        label_width = max(40, max_label_width + padding) # Use calculated width, min 40
        graph_area_width = width - padding * 2 - label_width # Recalculate graph area
        if graph_area_width <= 0: return # Check space again after recalculating
        # ----------------------------------------- #

        # Define colors
        bar_color = QColor("#3498db") # Blue
        highlight_color = QColor("#e74c3c") # Red
        text_color = QColor(Qt.black)
        painter.setPen(Qt.NoPen) # No outline for bars by default

        for i, prob in enumerate(self.probabilities):
            bar_y = padding + i * (bar_height + padding)

            # Check if bar fits vertically
            if bar_y + bar_height > height - padding:
                 break # Stop drawing if we run out of vertical space

            bar_width = max(0, prob * graph_area_width) # Ensure non-negative width

            # --- Choose color (Handle name or index prediction) --- #
            highlight = False
            if self.class_names:
                # Compare predicted name with current class name
                if isinstance(self.predicted_class, str) and self.predicted_class == self.class_names[i]:
                    highlight = True
            elif isinstance(self.predicted_class, int) and i == self.predicted_class:
                 # Compare predicted index with current index
                 highlight = True

            if highlight:
                 painter.setBrush(QBrush(highlight_color))
            else:
                 painter.setBrush(QBrush(bar_color))
            # ------------------------------------------------------ #

            # Draw bar
            painter.drawRect(QRectF(padding + label_width, bar_y, bar_width, bar_height))

            # Draw label (Use pre-calculated label text from temp_labels)
            label_text = temp_labels[i]
            text_x = padding
            # Center text vertically in the bar space
            text_y = bar_y + bar_height / 2 + text_height / 4 # Approximation for vertical centering

            painter.setPen(text_color) # Set pen for text
            painter.drawText(int(text_x), int(text_y), label_text)
            painter.setPen(Qt.NoPen) # Reset pen