from PySide6.QtWidgets import QApplication, QMainWindow, QHBoxLayout,QPushButton, QWidget
from wgpu.gui.qt import WgpuCanvas
import numpy as np

import fastplotlib
import fastplotlib.graphics.image
import sys

from cd_CropperCentralWidget import CropperCentralWidget, ImageViewerWidget


class CropperMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video cropper")
        self.setMinimumWidth(800)
        self.setMinimumHeight(500)

        self.setCentralWidget(CropperCentralWidget())

#app = QApplication(sys.argv)
if not QApplication.instance():
    app = QApplication(sys.argv)
else:
    app = QApplication.instance()

window=CropperMainWindow()
window.show()
app.exec()