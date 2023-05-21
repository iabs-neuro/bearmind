import caiman

from PySide6.QtWidgets import QMainWindow,QSlider, QHBoxLayout,QVBoxLayout,QPushButton,QLabel, QWidget
from PySide6.QtWidgets import QFileDialog, QLineEdit,QSpinBox,QGridLayout, QTabWidget

from wgpu.gui.qt import WgpuCanvas
import numpy as np
from pathlib import Path
import pickle
from PySide6.QtCore import Qt
import fastplotlib
import fastplotlib.graphics.image
import sys
import cv2
import os
import pygfx as gfx

class MotionCorrectionCentralWidget(QWidget):

    def __init__(self):
        super().__init__()

        layout = QHBoxLayout()
        button = QPushButton()
        button.setText("Apply motion correction")

        layout.addWidget(button)

        self.setLayout(layout)

