# ui/plot_widget.py

import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout,QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt # Use pyplot for style setting maybe
import numpy as np

class PlotWidget(QWidget):
    """A custom widget to display a Matplotlib figure."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__(parent)

        # Create a Matplotlib figure and canvas
        # constrained_layout attempts to prevent labels overlapping
        self.figure = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        # Expand horizontally, fixed vertically (or Preferred?)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        # Create a navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Initialize axes
        self.ax_loss = None
        self.ax_acc = None
        self._create_initial_axes()

    def _create_initial_axes(self):
        """Create initial empty axes for loss and accuracy."""
        self.figure.clear()
        self.ax_loss = self.figure.add_subplot(111)
        self.ax_acc = self.ax_loss.twinx() # Share x-axis
        self.ax_loss.set_title('Training Progress (No Data Yet)')
        self.ax_loss.set_xlabel('Iterations')
        self.ax_loss.set_ylabel('Training Loss', color='tab:blue')
        self.ax_acc.set_ylabel('Validation Accuracy', color='tab:red')
        self.ax_loss.tick_params(axis='y', labelcolor='tab:blue')
        self.ax_acc.tick_params(axis='y', labelcolor='tab:red')
        self.figure.tight_layout() # Adjust layout
        self.canvas.draw()

    def update_plot(self, loss_history, accuracy_history, interval=10):
        """Updates the plot with new training history data."""
        if not loss_history or not accuracy_history:
            self._create_initial_axes() # Reset if no data
            return

        # Clear previous plots
        self.ax_loss.cla()
        self.ax_acc.cla()

        iterations = len(loss_history)
        epochs_per_log = interval
        x_values = np.arange(0, iterations * epochs_per_log, epochs_per_log)

        # Plot Training Loss
        self.ax_loss.plot(x_values, loss_history, label='Training Loss', color='tab:blue')
        self.ax_loss.set_xlabel(f'Iterations (Logged every {interval})')
        self.ax_loss.set_ylabel('Training Loss', color='tab:blue')
        self.ax_loss.tick_params(axis='y', labelcolor='tab:blue')
        self.ax_loss.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Plot Validation Accuracy on the twin Axes
        self.ax_acc.plot(x_values, accuracy_history, label='Validation Accuracy', color='tab:red', linestyle='--')
        self.ax_acc.set_ylabel('Validation Accuracy', color='tab:red')
        self.ax_acc.tick_params(axis='y', labelcolor='tab:red')
        self.ax_acc.set_ylim(0, 1.05) # Accuracy typically between 0 and 1

        # Add title and legend
        self.ax_loss.set_title('Training Loss and Validation Accuracy')
        # Combine legends from both axes
        lines_loss, labels_loss = self.ax_loss.get_legend_handles_labels()
        lines_acc, labels_acc = self.ax_acc.get_legend_handles_labels()
        self.ax_acc.legend(lines_loss + lines_acc, labels_loss + labels_acc, loc='best')

        self.figure.tight_layout()
        self.canvas.draw()

# Example usage (for testing the widget independently)
if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication, QMainWindow
    app = QApplication(sys.argv)
    window = QMainWindow()
    plot_widget = PlotWidget(window)
    window.setCentralWidget(plot_widget)

    # Example data
    loss = np.random.rand(50) * 2
    acc = np.linspace(0.5, 0.95, 50) + np.random.rand(50) * 0.05
    plot_widget.update_plot(loss, acc, interval=10)

    window.setWindowTitle('Plot Widget Test')
    window.show()
    sys.exit(app.exec_()) 