import sys
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QDialogButtonBox, QSpacerItem,
    QSizePolicy
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

class AboutDialog(QDialog):
    """A simple dialog to display application information."""
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("About EasyML")
        self.setMinimumWidth(350)

        # Layouts
        main_layout = QVBoxLayout(self)
        content_layout = QHBoxLayout() # Icon on left, text on right
        button_layout = QHBoxLayout()

        # --- Content --- #
        # Icon
        icon_label = QLabel()
        try:
            pixmap = QPixmap('assets/icon.png').scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            icon_label.setPixmap(pixmap)
        except Exception as e:
            print(f"WARN: Could not load about icon: {e}") # Non-critical
            icon_label.setText("Icon") # Fallback text
        content_layout.addWidget(icon_label)

        # Text content
        text_layout = QVBoxLayout()
        title_label = QLabel("EasyML v2.0.0")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title_label.setFont(title_font)
        text_layout.addWidget(title_label)

        subtitle_label = QLabel("A Neural Net & CNN Playground")
        text_layout.addWidget(subtitle_label)

        # Spacer
        text_layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))

        builder_label = QLabel("Built by Pixel Alchemy")
        text_layout.addWidget(builder_label)

        content_layout.addLayout(text_layout)
        main_layout.addLayout(content_layout)

        # --- Buttons --- #
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        # Center the button
        button_layout.addStretch()
        button_layout.addWidget(button_box)
        button_layout.addStretch()

        main_layout.addLayout(button_layout)

        self.setLayout(main_layout) 