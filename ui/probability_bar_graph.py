# ui/probability_bar_graph.py

from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtGui import QPainter, QColor, QBrush, QFontMetrics, QPen
from PyQt5.QtCore import Qt, QRectF
import numpy as np

class ProbabilityBarGraph(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.probabilities = []
        self.predicted_class = -1
        self.setMinimumHeight(100) # Ensure it has some height
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred) # Allow horizontal expansion

    def set_probabilities(self, probs, predicted_class=-1):
        # Expects a 1D numpy array or list
        self.probabilities = np.asarray(probs)
        self.predicted_class = predicted_class
        self.update() # Trigger a repaint

    def clear_graph(self):
        self.probabilities = []
        self.predicted_class = -1
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

            # Choose color
            if i == self.predicted_class:
                 painter.setBrush(QBrush(highlight_color))
            else:
                 painter.setBrush(QBrush(bar_color))

            # Draw bar
            painter.drawRect(QRectF(padding + label_width, bar_y, bar_width, bar_height))

            # Draw label (Class Index: Probability)
            label_text = f"{i}: {prob:.2f}"
            text_x = padding
            # Center text vertically in the bar space
            text_y = bar_y + bar_height / 2 + text_height / 4 # Approximation for vertical centering

            painter.setPen(text_color) # Set pen for text
            painter.drawText(int(text_x), int(text_y), label_text)
            painter.setPen(Qt.NoPen) # Reset pen