# ui/probability_bar_graph.py

from PyQt5.QtWidgets import QWidget, QSizePolicy, QToolTip
from PyQt5.QtGui import QPainter, QColor, QBrush, QFontMetrics, QPen
from PyQt5.QtCore import Qt, QRectF, QPoint
import numpy as np
import math

# Define a list of distinct colors for pie slices
# (Add more colors if you expect more than 10-12 classes frequently)
PREDEFINED_COLORS = [
    QColor("#3498db"), QColor("#e74c3c"), QColor("#2ecc71"), QColor("#f1c40f"),
    QColor("#9b59b6"), QColor("#34495e"), QColor("#1abc9c"), QColor("#e67e22"),
    QColor("#7f8c8d"), QColor("#c0392b"), QColor("#2980b9"), QColor("#27ae60")
]

class PredictionVisualizer(QWidget): # Renamed class
    def __init__(self, parent=None):
        super().__init__(parent)
        self.probabilities = np.array([]) # Initialize as numpy array
        self.predicted_class_index = -1 # Store index of predicted class
        self.predicted_class_name = None # Store name of predicted class
        self.class_names = None   # Optional list of class names
        self.setMinimumSize(150, 150) # Ensure it has some size
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMouseTracking(True) # Enable mouse tracking for tooltips
        self._hovered_slice_index = -1 # Track which slice is hovered

    def set_probabilities(self, probs, predicted_class_or_name=-1, class_names=None):
        # Keep predicted_class_or_name argument for compatibility for now, but ignore it
        self.probabilities = np.asarray(probs)
        self.class_names = list(class_names) if class_names and len(class_names) == len(self.probabilities) else None

        # Determine predicted index and name from probabilities
        if len(self.probabilities) > 0:
            # Handle potential NaN or inf values before argmax
            valid_probs = np.nan_to_num(self.probabilities, nan=-np.inf)
            if np.any(valid_probs > -np.inf): # Check if there are any valid numbers
                 self.predicted_class_index = np.argmax(valid_probs)
                 if self.class_names and 0 <= self.predicted_class_index < len(self.class_names):
                     self.predicted_class_name = self.class_names[self.predicted_class_index]
                 else:
                     # Fallback if class names aren't available or index is out of bounds
                     self.predicted_class_name = f"Class {self.predicted_class_index}"
            else: # All probabilities were NaN or -inf
                 self.predicted_class_index = -1
                 self.predicted_class_name = "Invalid Probabilities"

        else: # No probabilities provided
            self.predicted_class_index = -1
            self.predicted_class_name = None

        # Reset hover state when probabilities change
        self._hovered_slice_index = -1
        self.update() # Trigger a repaint

    def clear_graph(self):
        self.probabilities = np.array([])
        self.predicted_class_index = -1
        self.predicted_class_name = None
        self.class_names = None
        self._hovered_slice_index = -1
        self.update()

    def get_predicted_class_name(self) -> str | None:
        """Returns the name of the predicted class."""
        return self.predicted_class_name

    # --- Tooltip Handling --- #
    def mouseMoveEvent(self, event):
        if len(self.probabilities) == 0:
            QToolTip.hideText()
            self._hovered_slice_index = -1
            self.update()
            return

        pos = event.pos()
        width = self.width()
        height = self.height()
        diameter = min(width, height) * 0.8 # Use 80% of the smaller dimension
        radius = diameter / 2
        center = QPoint(width // 2, height // 2)

        dx = pos.x() - center.x()
        dy = pos.y() - center.y()
        distance_sq = dx**2 + dy**2

        if distance_sq > radius**2:
            # Mouse is outside the pie chart
            if self._hovered_slice_index != -1:
                 self._hovered_slice_index = -1
                 QToolTip.hideText()
                 self.update() # Trigger repaint to remove highlight
            return

        # Calculate angle (adjusting for QPainter's coordinate system and 0 degrees at 12 o'clock)
        angle_rad = math.atan2(-dy, dx) # Y is inverted in Qt coordinates
        angle_deg = math.degrees(angle_rad)
        angle_deg = (90 - angle_deg + 360) % 360 # Convert to 0 deg at 12 o'clock, positive clockwise, ensure positive

        # Map angle to probability slice
        current_angle_start_deg = 0.0 # Use float
        found_slice = -1
        prob_sum_for_angle = np.sum(np.nan_to_num(self.probabilities)) # Handle potential NaNs
        if prob_sum_for_angle <= 1e-9: # Use tolerance for float comparison
            if self._hovered_slice_index != -1:
                 self._hovered_slice_index = -1
                 QToolTip.hideText()
                 self.update()
            return

        # --- Revised Slice Check Logic --- #
        for i, prob in enumerate(np.nan_to_num(self.probabilities)): # Use cleaned probabilities
            if prob < 0: prob = 0 # Ignore negative probabilities for angle calculation

            span_angle_deg = (prob / prob_sum_for_angle) * 360.0 # Use float
            # Avoid issues with extremely small spans close to zero
            if span_angle_deg < 1e-6: continue

            current_angle_end_deg = current_angle_start_deg + span_angle_deg

            # Normalize angles for robust comparison
            norm_angle = angle_deg
            norm_start = current_angle_start_deg % 360.0
            norm_end = current_angle_end_deg % 360.0

            # Add a small epsilon for boundary checks
            epsilon = 1e-6

            match_found = False
            if norm_start < norm_end: # Normal case (e.g., start=10, end=40)
                if norm_start - epsilon <= norm_angle < norm_end - epsilon:
                    match_found = True
            else: # Wrap-around case (e.g., start=350, end=20)
                if norm_start - epsilon <= norm_angle < 360 or 0 <= norm_angle < norm_end - epsilon:
                    match_found = True

            if match_found:
                found_slice = i
                break

            current_angle_start_deg = current_angle_end_deg # Keep cumulative angle
        # --- End Revised Slice Check Logic --- #

        # Update tooltip and highlight if the hovered slice changed
        if found_slice != -1 and found_slice != self._hovered_slice_index:
            self._hovered_slice_index = found_slice
            class_label = self.class_names[found_slice] if self.class_names else f"Class {found_slice}"
            probability = self.probabilities[found_slice] # Get original probability
            tooltip_text = f"{class_label}: {probability:.3f}"
            QToolTip.showText(event.globalPos(), tooltip_text, self)
            self.update() # Trigger repaint for highlight
        elif found_slice == -1 and self._hovered_slice_index != -1:
            # Moved out of all slices
            self._hovered_slice_index = -1
            QToolTip.hideText()
            self.update()

    def leaveEvent(self, event):
        # Hide tooltip when mouse leaves the widget
        QToolTip.hideText()
        if self._hovered_slice_index != -1:
            self._hovered_slice_index = -1
            self.update()

    # --- Drawing --- #
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Check for valid probabilities
        valid_probabilities = np.nan_to_num(self.probabilities)
        prob_sum = np.sum(valid_probabilities)

        if len(valid_probabilities) == 0 or prob_sum <= 0:
            painter.drawText(self.rect(), Qt.AlignCenter, "No prediction data")
            return

        width = self.width()
        height = self.height()
        side = min(width, height)
        padding = 10 # Padding around the pie
        diameter = side - 2 * padding

        if diameter <= 0: return # Not enough space to draw

        # Calculate bounding rectangle for the pie chart, centered in the widget
        rect_x = (width - diameter) / 2
        rect_y = (height - diameter) / 2
        bounding_rect = QRectF(rect_x, rect_y, diameter, diameter)

        # --- Draw Pie Slices --- #
        current_angle_start_degrees = 90.0 # Start at 12 o'clock
        current_angle_start_qt = int(current_angle_start_degrees * 16) # Convert for QPainter

        for i, prob in enumerate(valid_probabilities):
            if prob < 0: prob = 0 # Treat negative probabilities as zero for drawing

            span_angle_degrees = (prob / prob_sum) * 360.0
            span_angle_qt = int(round(span_angle_degrees * 16))

            # Assign color (cycle through predefined colors)
            color = PREDEFINED_COLORS[i % len(PREDEFINED_COLORS)]
            painter.setBrush(QBrush(color))

            # Highlight predicted or hovered slice
            is_predicted = (i == self.predicted_class_index)
            is_hovered = (i == self._hovered_slice_index)

            # Use a slightly thicker black pen for the predicted slice outline
            # Use a white pen for the hovered slice outline
            # Use a thin gray pen for other slices
            if is_hovered:
                current_pen = QPen(Qt.white, 1.5)
                painter.setBrush(QBrush(color.darker(130))) # Darken hovered slice slightly
            elif is_predicted:
                current_pen = QPen(Qt.black, 1.2)
            else:
                current_pen = QPen(Qt.gray, 0.5)

            painter.setPen(current_pen)

            # Draw the slice
            if span_angle_qt > 0:
                 painter.drawPie(bounding_rect, current_angle_start_qt, span_angle_qt)

            # Update start angle for the next slice (use degrees for accuracy)
            current_angle_start_degrees += span_angle_degrees
            current_angle_start_qt = int(round(current_angle_start_degrees * 16))