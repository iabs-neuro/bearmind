from PySide6.QtCore import Qt, Signal,Slot

from PySide6.QtWidgets import QMainWindow,QSlider, QHBoxLayout,QVBoxLayout,QPushButton,QLabel, QWidget
from PySide6.QtWidgets import QFileDialog, QLineEdit,QSpinBox,QGridLayout, QTabWidget, QSizePolicy,QTextEdit


import numpy as np
import matplotlib.pyplot as plt

import pyqtgraph as pg
import fastplotlib
from sortedcontainers import SortedSet

import pickle
from wgpu.gui.qt import WgpuCanvas

from CaimanDataManager import CaimanDataManager


def filter_array_by_many_masks(source_arr, masks):
    compound_mask = np.ones_like(masks[0])
    for mask in masks:
        compound_mask = compound_mask & mask
    return source_arr[compound_mask]



#### --------------------------------------------------------------------------------------------------------------------------------------

class SpatialComponentViewerWidget(QWidget):
    ToggleComponentSelectionSignal = Signal(int) # The signal that is emitted every time we toggle the selection of any component


    def __init__(self, *args, **kwargs):
        '''
            A container widget for displaying spatial components. Holds WgpuCanvas and associated widgets
        '''

        super().__init__(*args, **kwargs)

        self.CaimanDataManager = None
      
        self.setup_canvas()

    def get_component_color(self,k):
        return self.CaimanDataManager.get_component_color(k)
    
    def setup_canvas(self):
        layout = QVBoxLayout()
        self.canvas =  WgpuCanvas() 
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.components_plot = fastplotlib.Plot(canvas=self.canvas)
        
    def onClick_image(self, event):
        '''Called every time the underlying component Image is double-clicked'''
        x,y = event.pick_info["index"]
        component_ids = self.CaimanDataManager.get_components_from_point(x,y)
        for component_id in component_ids:
            if not self.CaimanDataManager.discarded_mask[component_id]:
                self.ToggleComponentSelectionSignal.emit(component_id)

    def onClick_label(self,event):
        '''Called every time the text label is double-clicked'''
        component_id = event.pick_info["world_object"].component_id
        if not self.CaimanDataManager.discarded_mask[component_id]:
            self.ToggleComponentSelectionSignal.emit(component_id)

    def setup_contours(self):
        '''Plots component contours and labels and stores the resulting Graphics objects in self'''
        self.component_contours = dict()
        self.component_labels = dict()

        for contour_id in self.CaimanDataManager.good_component_ids:
            color=self.get_component_color(contour_id)
            contour = self.CaimanDataManager.get_spatial_contour(contour_id)

            contour_line = fastplotlib.graphics.LineGraphic(contour, colors=color)
            self.component_contours[contour_id] = contour_line

            self.components_plot.add_graphic(contour_line)

            
            label_pos = np.append(np.mean(contour,axis=0),0)
            label_graphics = fastplotlib.graphics.TextGraphic(str(contour_id), position=label_pos,face_color=color, size=5)
            label_graphics.world_object.add_event_handler(self.onClick_label, "double_click")
            label_graphics.world_object.component_id = contour_id
            self.component_labels[contour_id] = label_graphics
            self.components_plot.add_graphic(label_graphics)

        self.components_plot.show()
        self.update_displayed_contours_and_labels()

    def plot_component_matrix(self, which="good"):
        squished_spatial_matrix = self.CaimanDataManager.get_spatial_matrix(which).T
        im = fastplotlib.graphics.image.ImageGraphic(squished_spatial_matrix, cmap="binary_r")
        im.world_object.add_event_handler(self.onClick_image, "double_click")
        self.components_plot.add_graphic(im)
        self.components_plot.show()

    def recalculate_contours(self, component_ids):
        '''
            Updates the geometry of spatial contours from CaimanDataManager. Used typically after merging components
        '''
        for component_id in component_ids:
            contour = self.CaimanDataManager.get_spatial_contour(component_id)
            color=self.get_component_color(component_id)
            contour_line = fastplotlib.graphics.LineGraphic(contour, colors=color)


            self.components_plot.scene.remove(self.component_contours[component_id].world_object)
            self.component_contours[component_id] = contour_line
            self.components_plot.add_graphic(contour_line,center=False)


    def update_displayed_contours_and_labels(self):
        '''
            Updates displayed contours and labels based on selected components and selected mode
        '''
        selected_components = self.CaimanDataManager.get_selected_components()
        unselected_components = self.CaimanDataManager.get_unselected_components()
        discarded_components = self.CaimanDataManager.get_discarded_components()

        for component_id in selected_components:
            self.component_labels[component_id].update_face_color("#ffffff")
            self.component_contours[component_id].colors = "#ffffff"
            self.component_contours[component_id].thickness = 2
            self.component_labels[component_id].update_size(5)

        for component_id in unselected_components:
            if component_id in self.component_labels.keys():

                self.component_labels[component_id].update_face_color(self.CaimanDataManager.get_component_color(component_id))
                self.component_contours[component_id].colors = self.CaimanDataManager.get_component_color(component_id)

                if self.CaimanDataManager.selectedModeActive:
                    self.component_contours[component_id].thickness = 0
                    self.component_labels[component_id].update_size(0)
                    
                else:
                    self.component_contours[component_id].thickness = 1.5
                    self.component_labels[component_id].update_size(5)

        for component_id in discarded_components:
            self.component_contours[component_id].world_object.visible=False
            self.component_labels[component_id].world_object.visible=False

           
#### --------------------------------------------------------------------------------------------------------------------------------------

class TemporalComponentsWidget(QWidget):
    ToggleComponentSelectionSignal = Signal(int)   # The signal that is emitted every time we toggle the selection of any component

    def __init__(self, N_plots_per_page=5):
        super().__init__()

        self.CaimanDataManager = None
        self.N_plots_per_page = N_plots_per_page

        self.xlim = (0,100) # Dynamically updating x-boundary

        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.scene().sigMouseClicked.connect(self.mouse_clicked)

        self.plotDataItems = dict() # This is where we are going to store temporal traces

        self.slider = QSlider(Qt.Vertical)
        self.slider.setInvertedAppearance(True)
        self.slider.valueChanged.connect(self.on_slider_scroll)


        layout = QHBoxLayout()
        layout.addWidget(self.graphics_widget)
        layout.addWidget(self.slider)
        self.setLayout(layout)

        self.create_plot_placeholders()
        self.hide_redundant_ticks()


    def get_component_ids_to_display(self):
        ''' Helper function which specifies which components need to be displayed depending on the view mode and the discarded components'''
        if self.CaimanDataManager.selectedModeActive:
            return filter_array_by_many_masks(self.CaimanDataManager.component_ids,
                                              masks=[
                                                  self.CaimanDataManager.good_components_mask,
                                                  self.CaimanDataManager.selected_mask,
                                                  np.invert(self.CaimanDataManager.discarded_mask)
                                              ])
        else:
             return filter_array_by_many_masks(self.CaimanDataManager.component_ids,
                                              masks=[
                                                  self.CaimanDataManager.good_components_mask,
                                                  np.invert(self.CaimanDataManager.discarded_mask)
                                              ])
        
        
    def mouse_clicked(self, event):
        ''' Every time one of the subplots is clicked'''
        if event.button() == Qt.LeftButton:
            items = self.graphics_widget.scene().items(event.scenePos())
            clicked_plot_item = [x for x in items if isinstance(x, pg.PlotItem)][0]
            clicked_data_item = clicked_plot_item.listDataItems()[0]
            self.ToggleComponentSelectionSignal.emit(clicked_data_item.component_id)

    def update_slider_range(self):
        ''' Dynamically updating slider range, depending on which components need to be displayed'''
        self.slider.setRange(0, max(0,len(self.displayed_components)-self.N_plots_per_page))

    def on_slider_scroll(self):
        self.update_displayed_plots()

    def add_component_temporal_trace(self, component_id):
        trace = self.CaimanDataManager.get_temporal_trace(component_id)
        color = self.CaimanDataManager.get_component_color(component_id)
        x = np.arange(len(trace))
        x_min, x_max = x[0], x[-1]
        self.xlim = (min(self.xlim[0], x_min), max(self.xlim[1], x_max)) # Updating common x-limits

        dataItem = pg.PlotDataItem(x, trace,pen=pg.mkPen(color))
        dataItem.component_id = component_id
        self.plotDataItems[component_id] = dataItem
        self.update_plot_limits()


    def update_plotDataItems(self):
        '''
            Updates the appearance of each PlotDataItem based on component selection
        '''
        for component_id in self.plotDataItems.keys():
            plotDataItem = self.plotDataItems[component_id]
            selected = self.CaimanDataManager.get_component_selection(component_id)

            if selected:
                plotDataItem.setPen(color="#ffffff")
            else:
                plotDataItem.setPen(color=self.CaimanDataManager.get_component_color(component_id))



    def recalculate_plotDataItems(self,component_ids):
        '''
            Updates the data of each PlotDataItem based on CaimanDataManager
        '''
        for component_id in component_ids:
            new_trace = self.CaimanDataManager.get_temporal_trace(component_id)
            self.plotDataItems[component_id].setData(np.arange(len(new_trace)), new_trace)
        self.update_plot_limits()



    def update_displayed_plots(self):
        '''
            Updates displayed PlotItems with corresponding PlotDataItem based on slider and selection mode
        '''

        self.displayed_components = self.get_component_ids_to_display()
        slider_offset = self.slider.value()

        for k, plot in enumerate(self.plotItems):  # Clearing all plots
            plot.titleLabel.setText("")
            plot.clear()

        for k, plot in enumerate(self.plotItems):
            if k+slider_offset >= len(self.displayed_components):
                return
            component_id = self.displayed_components[k+slider_offset]
            plot.setTitle(f"Component {component_id}")
            plot.addItem(self.plotDataItems[component_id])
            self.onZoom()



    def onZoom(self):
        '''Called every time any of the subplots is zoomed'''
        xlim, ylim = self.plotItems[0].viewRange()
        xmin,xmax = xlim[0], xlim[1]

        for k, plot in enumerate(self.plotItems):
            dataitems = plot.listDataItems()
            if len(dataitems)>0:
                ymin, ymax = dataitems[0].dataBounds(1, orthoRange=(xmin, xmax))
                plot.getViewBox().setYRange(ymin, ymax, padding=0.3)

    def update_plot_limits(self):
        for plot in self.plotItems:
            viewBox = plot.getViewBox()
            viewBox.setLimits(minXRange=self.xlim[0], xMin=self.xlim[0], xMax=self.xlim[1], maxXRange=self.xlim[1])

    def create_plot_placeholders(self):
        self.plotItems = []

        for k in range(self.N_plots_per_page):
            plot = self.graphics_widget.addPlot(col=0, row=k, title=f"Component {k}")
            plot.titleLabel.size=5
            plot.sigXRangeChanged.connect(self.onZoom)

            viewBox = plot.getViewBox()
            viewBox.setMouseEnabled(x=True, y=False)
            viewBox.setDefaultPadding(0.3)
            
            if k>0:
                plot.setXLink(self.plotItems[0])
            self.plotItems.append(plot)

    def hide_redundant_ticks(self):
        ''' Hides x-axis ticks for all subplots except the last one'''
        for k in range(len(self.plotItems)-1):
            self.plotItems[k].hideAxis('bottom')

#### --------------------------------------------------------------------------------------------------------------------------------------


class CaimanViewerWidget(QWidget):

    def __init__(self):
        super().__init__()
        
        self.CaimanDataManager = None

        self.temporal_plots_widget = TemporalComponentsWidget(5) 
        self.spatial_component_widget = SpatialComponentViewerWidget() # Spatial components widget 
        
        self.temporal_plots_widget.ToggleComponentSelectionSignal.connect(self.toggle_component_selection)
        self.spatial_component_widget.ToggleComponentSelectionSignal.connect(self.toggle_component_selection)


        # --------- Defining Layout
        mainLayout = QVBoxLayout()
        componentViewersLayout = QHBoxLayout()
        componentViewersLayout.addWidget(self.spatial_component_widget, 5)
        componentViewersLayout.addWidget(self.temporal_plots_widget, 5)


        self.controlStripLayout = QHBoxLayout()
        self.buttonRowLayout = QHBoxLayout()

        self.controlStripLayout.addLayout(self.buttonRowLayout,5)
        self.controlStripLayout.addSpacing(50)
        self.statusTextEdit = QTextEdit(text="No open file. Press Ctrl(Cmd) + O to open a directory with .avi files")
        self.statusTextEdit.setReadOnly(True)
        


        self.controlStripLayout.addWidget(self.statusTextEdit,3)

        # Buttons
        self.selectedModeButton = QPushButton("🔎 Show selected")
        self.selectedModeButton.setCheckable(True)
        self.selectedModeButton.clicked.connect(self.toggle_selected_mode)


        self.delectAllButton = QPushButton("📤 Clear selection")
        self.delectAllButton.clicked.connect(self.deselect_all_components)


        self.discardSelectedButton = QPushButton("❌ Discard selected")
        self.mergeSelectedButton = QPushButton("🪄 Merge selected")


        self.discardSelectedButton.clicked.connect(self.discard_selected_components)
        self.mergeSelectedButton.clicked.connect(self.merge_selected_components)

        self.buttonRowLayout.addWidget(self.selectedModeButton)
        self.buttonRowLayout.addWidget(self.delectAllButton)
        self.buttonRowLayout.addWidget(self.discardSelectedButton)
        self.buttonRowLayout.addWidget(self.mergeSelectedButton)

        mainLayout.addLayout(componentViewersLayout,9)
        mainLayout.addLayout(self.controlStripLayout,1)

        self.setLayout(mainLayout)

        self.disable_all_buttons()
   
         
    def enable_all_buttons(self):
        layout = self.buttonRowLayout
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if isinstance(item.widget(), QPushButton):
                item.widget().setEnabled(True)

    def disable_all_buttons(self):
        layout = self.buttonRowLayout
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if isinstance(item.widget(), QPushButton):
                item.widget().setEnabled(False)



    def load(self):
        '''
            Top-level function to open a .pickle file with CAIMAN outputs
        '''
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Pickle Files (*.pickle)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.exec()
        
        selected_files = file_dialog.selectedFiles()
        if selected_files:
            file_path = selected_files[0]

        if file_path:
            self.CaimanDataManager = CaimanDataManager(file_path)
            self.temporal_plots_widget.CaimanDataManager = self.CaimanDataManager
            self.spatial_component_widget.CaimanDataManager = self.CaimanDataManager


            self.plot_temporal_components()
            self.spatial_component_widget.plot_component_matrix()
            self.spatial_component_widget.setup_contours()
            self.enable_all_buttons()
            self.update_all_panels()


    def save(self):
        '''
            Top-level function to save the modified outputs as a .pickle
        '''
        
        save_name = QFileDialog.getSaveFileName(self, 'Save output', "", ".pickle")[0]
        print(f"Saving to {save_name}")

        with open(save_name, "wb") as f:
            pickle.dump(self.CaimanDataManager.estimates, f)


    def deselect_all_components(self):
        for component_id in self.CaimanDataManager.component_ids:
            self.CaimanDataManager.set_component_selection(component_id, False)
        self.CaimanDataManager.set_selected_mode(False)
        self.update_all_panels()
    
    def discard_selected_components(self):
        components_to_discard = self.CaimanDataManager.get_selected_components()
        self.CaimanDataManager.discard_components(
            components_to_discard
        )
        self.CaimanDataManager.set_component_selection(components_to_discard, False)


        self.update_all_panels()

    def merge_selected_components(self):
        components_to_merge = self.CaimanDataManager.get_selected_components()
        self.CaimanDataManager.merge_components(
            components_to_merge
        )
        
        self.spatial_component_widget.recalculate_contours([np.min(components_to_merge)])
        self.temporal_plots_widget.recalculate_plotDataItems([np.min(components_to_merge)])
        self.deselect_all_components()
        self.update_all_panels()


    def toggle_component_selection(self,component_id):
        self.CaimanDataManager.toggle_component_selection(component_id)
        self.update_all_panels()
        
    def update_all_panels(self):
        ''' Top-level function to be called every time selection or display mode changes. Takes case of redrawing temporal and spatial components'''

        self.spatial_component_widget.update_displayed_contours_and_labels()
        self.temporal_plots_widget.update_plotDataItems()
        self.temporal_plots_widget.update_displayed_plots()
        self.temporal_plots_widget.update_slider_range()
        self.update_status_text()
        self.selectedModeButton.setChecked(self.CaimanDataManager.selectedModeActive) 

    def update_status_text(self):
        selectedComponents = self.CaimanDataManager.get_selected_components()
        discardedComponents = self.CaimanDataManager.get_discarded_components()

        if len(selectedComponents)==0:
            selectedComponents_status_text = "–"
        else:
            selectedComponents_status_text = str(list(selectedComponents))
        
        if len(discardedComponents)==0:
            discardedComponents_status_text = "–"
        else:
            discardedComponents_status_text = str(list(discardedComponents))

        self.statusTextEdit.setText(
            f"Open file: {self.CaimanDataManager.estimates_filepath} \n\n"+
            f"Selected components: {selectedComponents_status_text}\nDiscarded components: {discardedComponents_status_text}")
        
    def toggle_selected_mode(self):
        self.CaimanDataManager.toggle_selected_mode()
        self.update_all_panels()
        
    def get_component_color(self,k):
        return self.CaimanDataManager.get_component_color(k)

    def plot_temporal_components(self):
    
        for k in self.CaimanDataManager.good_component_ids:
            self.temporal_plots_widget.add_component_temporal_trace(k)
        self.temporal_plots_widget.update_displayed_plots()
        self.temporal_plots_widget.update_slider_range()
