# ui/drawing_canvas.py

from PyQt5.QtWidgets import QWidget, QApplication, QPushButton, QVBoxLayout, QLabel
from PyQt5.QtGui import QPainter, QImage, QPen, QPixmap, QColor
from PyQt5.QtCore import Qt, QPoint, QSize
import numpy as np
from PIL import Image as PILImage
import sys

class DrawingCanvas(QWidget):
    """A simple widget for drawing with the mouse."""
    def __init__(self, width=200, height=200, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.black)
        self.drawing = False
        self.last_point = QPoint()
        # Make pen thinner to better match QuickDraw data style
        self.pen = QPen(Qt.white, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.drawing:
            painter = QPainter(self.image)
            painter.setPen(self.pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update() # Trigger repaint

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(self.rect(), self.image, self.image.rect())

    def clearCanvas(self):
        self.image.fill(Qt.black)
        self.update()

    def getDrawingArray(self, target_size=(28, 28)) -> np.ndarray | None:
        """Gets the drawing, converts to grayscale, resizes, normalizes, and flattens."""
        try:
            # Convert QImage to PIL Image
            qimage_copy = self.image.copy() # Work on a copy

            # Manual conversion QImage -> NumPy -> PIL Image to bypass Qt bindings issue
            qimage_copy = qimage_copy.convertToFormat(QImage.Format_ARGB32) # Ensure 4 channels
            ptr = qimage_copy.constBits()
            ptr.setsize(qimage_copy.byteCount())
            # Create NumPy array (h, w, 4 channels)
            arr = np.array(ptr).reshape(qimage_copy.height(), qimage_copy.width(), 4)
            # Create PIL Image from array (use RGB from ARGB)
            pil_img = PILImage.fromarray(arr[:, :, :3], 'RGB')

            # Convert to grayscale and resize
            pil_img_gray = pil_img.convert("L")
            pil_img_resized = pil_img_gray.resize(target_size, PILImage.Resampling.LANCZOS)

            # Convert to numpy array and normalize
            img_array = np.array(pil_img_resized)

            # Check if the image is mostly black (or empty)
            if np.mean(img_array) < 5:
                print("Warning: Drawing canvas appears empty.")
                return None

            # Return the raw 2D numpy array (uint8)
            # Normalization and reshaping should happen in the caller
            return img_array
        except Exception as e:
            print(f"Error processing drawing: {e}")
            return None

    def getPreviewPixmap(self, target_size=(28, 28)) -> QPixmap | None:
        """Gets the drawing as a 28x28 QPixmap for preview."""
        try:
            qimage_copy = self.image.copy()

            # Manual conversion QImage -> NumPy -> PIL Image to bypass Qt bindings issue
            qimage_copy = qimage_copy.convertToFormat(QImage.Format_ARGB32) # Ensure 4 channels
            ptr = qimage_copy.constBits()
            ptr.setsize(qimage_copy.byteCount())
            # Create NumPy array (h, w, 4 channels)
            arr = np.array(ptr).reshape(qimage_copy.height(), qimage_copy.width(), 4)
            # Create PIL Image from array (use RGB from ARGB)
            pil_img = PILImage.fromarray(arr[:, :, :3], 'RGB')

            pil_img_gray = pil_img.convert("L")
            pil_img_resized = pil_img_gray.resize(target_size, PILImage.Resampling.LANCZOS)

            # Convert PIL (grayscale) back to QImage manually
            bytes_data = pil_img_resized.tobytes()
            # For grayscale, the stride (bytes per line) is just the width
            qimage_preview = QImage(bytes_data, pil_img_resized.width, pil_img_resized.height, pil_img_resized.width, QImage.Format_Grayscale8)

            # Important: Need to copy the QImage data, otherwise it might point to released memory
            # Create the pixmap *from* the QImage
            pixmap = QPixmap.fromImage(qimage_preview)
            return pixmap

        except Exception as e:
            print(f"Error creating preview pixmap: {e}")
            return None

# Example usage (for testing the widget independently)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = QWidget()
    layout = QVBoxLayout()

    canvas = DrawingCanvas(280, 280)
    layout.addWidget(canvas)

    clear_button = QPushButton("Clear")
    clear_button.clicked.connect(canvas.clearCanvas)
    layout.addWidget(clear_button)

    def show_array():
        arr = canvas.getDrawingArray()
        if arr is not None:
            print("Got array:", arr.shape, arr.dtype, np.min(arr), np.max(arr))
            # Optional: Display the 28x28 version
            import matplotlib.pyplot as plt
            plt.imshow(arr.reshape(28, 28), cmap='gray')
            plt.show()
        else:
            print("Array is None (canvas likely empty)")

    get_button = QPushButton("Get Array")
    get_button.clicked.connect(show_array)
    layout.addWidget(get_button)

    main_window.setLayout(layout)
    main_window.setWindowTitle('Drawing Canvas Test')
    main_window.show()
    sys.exit(app.exec_()) 