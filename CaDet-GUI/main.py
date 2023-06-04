from PySide6.QtWidgets import QApplication, QMainWindow, QHBoxLayout,QPushButton, QWidget, QTabWidget
from PySide6.QtGui import QAction,QKeySequence
from wgpu.gui.qt import WgpuCanvas
import numpy as np

import fastplotlib
import fastplotlib.graphics.image
import sys

from CropperCentralWidget import CropperCentralWidget, ImageViewerWidget
#from MotionCorrectionCentralWidget import MotionCorrectionCentralWidget
from CaimanViewerWidget import CaimanViewerWidget

class GUIMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Caiman GUI")
        self.setMinimumWidth(1200)
        self.setMinimumHeight(800)

        menuBar = self.menuBar()
       # menuBar.setNativeMenuBar(False)
        openAction = QAction("Open",self)
        openAction.setShortcut(QKeySequence("Ctrl+o"))
        openAction.triggered.connect(self.load_file)



        saveAction = QAction("Save",self)
        saveAction.setShortcut(QKeySequence("Ctrl+s"))
        saveAction.triggered.connect(self.save_file)

       # openAction.setMenuRole(QAction.MenuRole.NoRole)

        fileMenu = menuBar.addMenu('&File')
        fileMenu.addAction(openAction)
        fileMenu.addAction(saveAction)

        self.tabs = QTabWidget()
        self.cropperWidget= CropperCentralWidget()
        self.viewerWidget = CaimanViewerWidget()

        self.tabs.addTab(self.cropperWidget, "Cropper")
        self.tabs.addTab(self.viewerWidget, "View output")
        self.setCentralWidget(self.tabs)

    def load_file(self):
        self.tabs.currentWidget().load()

    def save_file(self):
        self.tabs.currentWidget().save()



app = QApplication(sys.argv)
window=GUIMainWindow()
window.show()
app.exec()