"""Pop-up windows opened from the main window."""

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
)

from hackaton import config
from hackaton.ui import styles


class GraphWindow(QWidget):
    """Displays the pre-rendered statistics graph."""

    def __init__(self, image_path=None):
        super().__init__()
        self.setWindowTitle("Graph Display")
        self.setFixedSize(1520, 320)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel()
        pixmap = QPixmap(str(image_path or config.GRAPH_IMAGE))
        label.setPixmap(pixmap.scaled(1510, 310, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        self.setLayout(layout)


class DataTableWindow(QWidget):
    """Displays every advertisement of one category in a table."""

    def __init__(self, df: pd.DataFrame, category_label: str):
        super().__init__()
        self.setWindowTitle(f"Data: {category_label}")
        self.setFixedSize(800, 600)

        layout = QVBoxLayout()

        entries = df.to_dict('records')
        if entries:
            headers = list(entries[0].keys())
            table = QTableWidget(len(entries), len(headers))
            table.setHorizontalHeaderLabels(headers)
            table.setStyleSheet(styles.DATA_TABLE)
            # Populate the table with values from each entry
            for row_idx, entry in enumerate(entries):
                for col_idx, key in enumerate(headers):
                    value = str(entry.get(key, ""))
                    table.setItem(row_idx, col_idx, QTableWidgetItem(value))
            layout.addWidget(table)
        else:
            # Show a friendly message if there's no data
            no_label = QLabel("No data available for this category.")
            no_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(no_label)

        self.setLayout(layout)


class QRCodeWindow(QWidget):
    """Displays the QR code linking to the advertisements."""

    def __init__(self, image_path=None):
        super().__init__()
        self.setWindowTitle("QRCode")
        self.setFixedSize(260, 260)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setAlignment(Qt.AlignCenter)

        label = QLabel()
        pixmap = QPixmap(str(image_path or config.QRCODE_IMAGE)) \
            .scaled(240, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)

        layout.addWidget(label)
        self.setLayout(layout)
