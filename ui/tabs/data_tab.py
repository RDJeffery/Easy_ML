import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QComboBox, QLineEdit, QSpinBox, QGroupBox, QFormLayout,
    QFrame, QListWidget, QAbstractItemView
)
from PyQt5.QtCore import Qt

class DataTab(QWidget):
    """Widget defining the UI components for the Data Tab."""
    def __init__(self, parent_window=None):
        super().__init__(parent=parent_window) # Correct parent passing
        self.parent_window = parent_window

        layout = QVBoxLayout(self)
        dataset_group = self._create_dataset_group()
        layout.addWidget(dataset_group)
        layout.addStretch()

    def _create_dataset_group(self):
        """Creates the GroupBox containing dataset selection and loading widgets."""
        dataset_group = QGroupBox("Dataset Loading")
        form_layout = QFormLayout()

        # --- Dataset Selection ---
        # Create widgets and store them on the parent window instance
        self.parent_window.dataset_dropdown = QComboBox()
        self.parent_window.dataset_dropdown.setToolTip("Select a dataset automatically found in the 'data' directory")
        self.parent_window.dataset_dropdown.currentIndexChanged.connect(self.parent_window._on_dataset_selected)
        form_layout.addRow("Select Dataset:", self.parent_window.dataset_dropdown)

        self.parent_window.load_dataset_button = QPushButton("Load Selected")
        self.parent_window.load_dataset_button.setToolTip("Load the dataset chosen in the dropdown above")
        self.parent_window.load_dataset_button.clicked.connect(self.parent_window.load_selected_dataset)
        form_layout.addRow(self.parent_window.load_dataset_button)

        # --- QuickDraw Selection Widgets ---
        self.parent_window.quickdraw_select_label = QLabel("Select QuickDraw Categories (Ctrl/Shift+Click):")
        self.parent_window.quickdraw_list_widget = QListWidget()
        self.parent_window.quickdraw_list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.parent_window.quickdraw_list_widget.setMinimumHeight(100)
        self.parent_window.quickdraw_list_widget.setToolTip("Select specific categories to load, or use the random option below.")

        self.parent_window.quickdraw_random_label = QLabel("Or Load Random:")
        self.parent_window.quickdraw_random_count = QSpinBox()
        self.parent_window.quickdraw_random_count.setRange(0, 100)
        self.parent_window.quickdraw_random_count.setValue(5)
        self.parent_window.quickdraw_random_count.setToolTip("Specify the number of categories to load randomly (ignores list selection). Set to 0 to use list selection.")

        form_layout.addRow(self.parent_window.quickdraw_select_label)
        form_layout.addRow(self.parent_window.quickdraw_list_widget)
        form_layout.addRow(self.parent_window.quickdraw_random_label, self.parent_window.quickdraw_random_count)

        self.parent_window.quickdraw_select_label.setVisible(False)
        self.parent_window.quickdraw_list_widget.setVisible(False)
        self.parent_window.quickdraw_random_label.setVisible(False)
        self.parent_window.quickdraw_random_count.setVisible(False)

        # --- Custom CSV Upload ---
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        form_layout.addRow(separator)

        upload_button = QPushButton("Upload Custom CSV")
        upload_button.setToolTip("Upload your own CSV (pixels, base64 images, or image paths)")
        upload_button.clicked.connect(self.parent_window.upload_csv_dataset)
        form_layout.addRow("Load Custom:", upload_button)

        self.parent_window.label_col_input = QSpinBox()
        self.parent_window.label_col_input.setRange(0, 1000)
        self.parent_window.label_col_input.setValue(0)
        self.parent_window.label_col_input.setToolTip("Index (0-based) of the column containing the labels in your CSV")
        form_layout.addRow("Label Col Idx:", self.parent_window.label_col_input)

        self.parent_window.image_col_input = QSpinBox()
        self.parent_window.image_col_input.setRange(-1, 1000)
        self.parent_window.image_col_input.setValue(-1)
        self.parent_window.image_col_input.setToolTip("Index of image data column (base64/path). Set to -1 if columns contain raw pixel values.")
        form_layout.addRow("Image Col Idx:", self.parent_window.image_col_input)

        self.parent_window.image_type_combo = QComboBox()
        self.parent_window.image_type_combo.addItems(["(Not Applicable)", "base64", "path"])
        self.parent_window.image_type_combo.setToolTip("Select 'base64' or 'path' if Image Col Idx is >= 0")
        self.parent_window.image_type_combo.setEnabled(False)
        form_layout.addRow("Type:", self.parent_window.image_type_combo)

        # Connect valueChanged signal
        if hasattr(self.parent_window, 'image_col_input') and hasattr(self.parent_window, '_update_image_col_type_state'):
             self.parent_window.image_col_input.valueChanged.connect(self.parent_window._update_image_col_type_state)

        dataset_group.setLayout(form_layout)
        return dataset_group 