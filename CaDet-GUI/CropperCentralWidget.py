from PySide6.QtWidgets import QMainWindow,QSlider, QHBoxLayout,QVBoxLayout,QPushButton,QLabel, QWidget
from PySide6.QtWidgets import QFileDialog, QLineEdit,QSpinBox,QGridLayout, QTabWidget
from PySide6.QtGui import QIntValidator
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
import platform


def load_video_from_list_of_files_cv2(paths):
    # This function loads videos using cv2
    frames = []
    for path in paths:
        cap = cv2.VideoCapture(str(path))
        ret = True
        while ret:
            ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
            if ret:
                frames.append(img[:,:,0])
    video = np.stack(frames, axis=0) # dimensions (T, H, W, C)
    return video
    

def load_video_from_list_of_files(paths):
    # This function loads videos using moviepy
    video = []
    from moviepy.editor import VideoFileClip
    for path in paths:
        clip = VideoFileClip(str(path))
        for frame in clip.iter_frames():
            video.append(frame[:,:,0])
        del video[-1] # To remove last duplicated frame
    return np.asarray(video)


        


class CropperRectangle(gfx.Mesh):
    def __init__(self, image_shape, size=0, pos="DOWN", color="#454545"):
        '''
            A single cropping rectrange represented as a pygfx.WorldObject
        '''

        super().__init__()
        self.size = size # Initial cropping size (=width for LEFT & RIGHT; =height for UP & DOWN croppers) 
        self.image_shape = image_shape # Shape of the image to adjust the size
        self.pos = pos # Position (UP, DOWN, LEFT, RIGHT)
        self.color = color
        self.Zindex = 10
        self.set_material()
        self.adjust_size_and_position()

    def set_material(self):
        self.material = gfx.MeshBasicMaterial(self.color)

    def adjust_size_and_position(self):
        
        if self.pos == "DOWN" or self.pos == "UP":
            width = self.image_shape[0]
            height = self.size
            if self.pos=="UP":
                self.position.set(self.image_shape[0]/2 -0.5 , self.image_shape[1] - self.size/2 -0.5, self.Zindex)
            if self.pos=="DOWN":
                self.position.set(self.image_shape[0]/2 -0.5, self.size/2 -0.5, self.Zindex)

        if self.pos == "LEFT" or self.pos == "RIGHT":
            width = self.size
            height = self.image_shape[1]
            if self.pos=="LEFT":
                self.position.set(self.size/2 -0.5, self.image_shape[1]/2 -0.5, self.Zindex)
            if self.pos=="RIGHT":
                self.position.set(self.image_shape[0] - self.size/2 -0.5, self.image_shape[1]/2 -0.5, self.Zindex)

        self.geometry = gfx.geometries.plane_geometry(width, height)
       

    
class ImageViewerWidget(QWidget):

    def __init__(self):
        '''
            Widget holding the WGPUCanvas and the time-slider for scrolling
        '''
        super().__init__()
    
        layout = QVBoxLayout()
        self.canvas = WgpuCanvas()
        layout.addWidget(self.canvas)
    
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.change_time_value)
        
        self.spinBox = QSpinBox()
        self.spinBox.valueChanged.connect(self.change_time_value)

        

        # Slider with SpinBox horizontal container
        sliderWithSpinbox = QWidget()
        sliderWithSpinbox_layout = QHBoxLayout()
        sliderWithSpinbox_layout.addWidget(self.slider)
        sliderWithSpinbox_layout.addWidget(self.spinBox)
        sliderWithSpinbox.setLayout(sliderWithSpinbox_layout)
        sliderWithSpinbox.setMaximumHeight(50)


        layout.addWidget(sliderWithSpinbox)
        self.setLayout(layout)

    def load_data_from_directory(self,directory_name):
        
        self.directory_path =Path(directory_name)

        avi_list = list(self.directory_path.glob("*.avi"))
        print(f"Found {len(avi_list)} avi files:",avi_list)

        
        data = load_video_from_list_of_files(avi_list)

        print("Data shape: {}".format(data.shape))
        if len(data.shape)==4: # if RGB image
            self.set_image_data(data[:,:,:,0]) # Selecting only red channel
        else:
            self.set_image_data(data)

        
    def set_image_data(self, data):
        self.image_data = data
        self.slider.setMaximum(data.shape[0]-1)
        self.spinBox.setMaximum(data.shape[0]-1)

        self.spinBox.setMinimum(0)
        self.slider.setMinimum(0)

    def setup_plot(self):
        self.plot = fastplotlib.Plot(canvas=self.canvas)
        self.im = fastplotlib.graphics.image.ImageGraphic(self.image_data[0,:,:], cmap = 'gray', vmin=np.min(self.image_data), vmax = np.max(self.image_data))
        self.plot.add_graphic(self.im)
        self.plot.show()

    def setup_croppers(self):
        '''Setting up cropper Rectangle Meshes'''
        self.croppers = dict()
        self.croppers["UP"] = CropperRectangle(self.image_data.shape[1::],0,pos="UP")
        self.croppers["DOWN"] = CropperRectangle(self.image_data.shape[1::],0,pos="DOWN")
        self.croppers["LEFT"] = CropperRectangle(self.image_data.shape[1::],0,pos="LEFT")
        self.croppers["RIGHT"] = CropperRectangle(self.image_data.shape[1::],0,pos="RIGHT")

        for cropper in self.croppers.values():
            self.plot.scene.add(cropper)

    def change_time_value(self, data):
        self.slider.value = int(data)
        self.slider.sliderPosition = int(data)
        self.slider.update()
        self.slider.repaint()
    
        self.spinBox.setValue(int(data))
        self.im.data = self.image_data[int(data),:,:]





class SpinBoxesPanel(QWidget):
    
    def __init__(self):
        super().__init__()

        self.SpinBoxes = dict()
        self.Captions = dict()

        self.SpinBoxes["LEFT"] = QSpinBox()
        self.Captions["LEFT"] = QLabel()
        self.Captions["LEFT"].setText("Left crop")


        self.SpinBoxes["RIGHT"] = QSpinBox()
        self.Captions["RIGHT"] = QLabel()
        self.Captions["RIGHT"].setText("Right crop")


        self.SpinBoxes["UP"] = QSpinBox()
        self.Captions["UP"] = QLabel()
        self.Captions["UP"].setText("Top crop")

        self.SpinBoxes["DOWN"] = QSpinBox()
        self.Captions["DOWN"] = QLabel()
        self.Captions["DOWN"].setText("Bottom crop")


        layout = QGridLayout()
        layout.addWidget(self.SpinBoxes["LEFT"], 0,0)
        layout.addWidget(self.Captions["LEFT"], 0,1)

        layout.addWidget(self.SpinBoxes["RIGHT"], 1,0)
        layout.addWidget(self.Captions["RIGHT"], 1,1)

        layout.addWidget(self.SpinBoxes["UP"], 2,0)
        layout.addWidget(self.Captions["UP"], 2,1)

        layout.addWidget(self.SpinBoxes["DOWN"], 3,0)
        layout.addWidget(self.Captions["DOWN"], 3,1)


        self.setLayout(layout)


class ControlPanelWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # Load Button
        load_button = QPushButton()
        load_button.setText("Open movie directory")
        layout.addWidget(load_button)
        self.load_button = load_button
        


        # Cropper spin boxes
        self.spinBoxesPanel = SpinBoxesPanel()
        layout.addWidget(self.spinBoxesPanel)


        # Save button
        self.save_button = QPushButton()
        self.save_button.setText("Save crops")
        layout.addWidget(self.save_button)


        #layout.addWidget(self.paired_slider)
        self.setLayout(layout)


class CropperCentralWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.im_viewer = ImageViewerWidget()
        self.control_panel = ControlPanelWidget()
        self.control_panel.load_button.clicked.connect(self.load)
        self.control_panel.save_button.clicked.connect(self.save)

        layout = QHBoxLayout()
        layout.addWidget(self.im_viewer)
        layout.addWidget(self.control_panel)
        self.connect_spinboxes_to_croppers()
        self.setLayout(layout)


    def connect_spinboxes_to_croppers(self):
        self.control_panel.spinBoxesPanel.SpinBoxes["LEFT"].valueChanged.connect(self.adjust_left_cropper_size)
        self.control_panel.spinBoxesPanel.SpinBoxes["RIGHT"].valueChanged.connect(self.adjust_right_cropper_size)
        self.control_panel.spinBoxesPanel.SpinBoxes["UP"].valueChanged.connect(self.adjust_up_cropper_size)
        self.control_panel.spinBoxesPanel.SpinBoxes["DOWN"].valueChanged.connect(self.adjust_down_cropper_size)

    def load(self):
        directory = QFileDialog.getExistingDirectory(self, "Open Directory")

        self.im_viewer.load_data_from_directory(directory)
       # self.im_viewer.set_image_data(np.random.rand(100,500,500))
        self.im_viewer.setup_plot()
        self.im_viewer.setup_croppers()
        self.adjust_SpinBoxes_limits()
        pass


    def adjust_SpinBoxes_limits(self):

        self.control_panel.spinBoxesPanel.SpinBoxes["LEFT"].setRange(0,self.im_viewer.image_data.shape[1]//2)
        self.control_panel.spinBoxesPanel.SpinBoxes["RIGHT"].setRange(0,self.im_viewer.image_data.shape[1]//2)
        self.control_panel.spinBoxesPanel.SpinBoxes["UP"].setRange(0,self.im_viewer.image_data.shape[2]//2)
        self.control_panel.spinBoxesPanel.SpinBoxes["DOWN"].setRange(0,self.im_viewer.image_data.shape[2]//2)

        for s in self.control_panel.spinBoxesPanel.SpinBoxes.values():
            s.setSingleStep(10)
        pass


    def adjust_cropper_size(self,pos,size):
        self.im_viewer.croppers[pos].size=size
        self.im_viewer.croppers[pos].adjust_size_and_position() 

    def adjust_left_cropper_size(self, size):
        self.adjust_cropper_size("LEFT", size)

    def adjust_right_cropper_size(self, size):
        self.adjust_cropper_size("RIGHT", size)

    def adjust_up_cropper_size(self, size):
        self.adjust_cropper_size("UP", size)

    def adjust_down_cropper_size(self, size):
        self.adjust_cropper_size("DOWN", size)
       

    def save(self):
        save_dir = self.im_viewer.directory_path
        save_name = str(self.im_viewer.directory_path.name)+ "_cropping.pickle"
        save_path = Path(save_dir).joinpath(save_name)

        cropping_dict = {
            "IMAGE_SHAPE":self.im_viewer.image_data.shape,
            "LEFT":self.im_viewer.croppers["LEFT"].size,
            "RIGHT":self.im_viewer.croppers["RIGHT"].size,
            "UP":self.im_viewer.croppers["UP"].size,
            "DOWN":self.im_viewer.croppers["DOWN"].size
        }
        with open(save_path, "wb") as f:
            pickle.dump(cropping_dict, f)
        print(cropping_dict)
        print("Saving croppings to {}".format(save_path))
    
