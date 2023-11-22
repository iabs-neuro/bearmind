import pickle
import cmasher
import numpy as np
import skimage.measure
from matplotlib.colors import to_hex
from scipy.sparse import csc_matrix

def filter_array_by_many_masks(source_arr, masks):
    compound_mask = np.ones_like(masks[0])
    for mask in masks:
        compound_mask = compound_mask & mask
    return source_arr[compound_mask]


class CaimanDataManager:
    def __init__(self, estimates_filepath, components_cmap = cmasher.guppy_r) -> None:
        '''
            Utility class providing an interface to use CAIMAN's Estimates output and keep track of component operations (select, discard, merge)
        '''
        self.estimates_filepath = estimates_filepath
        self.load_estimates()

        self.components_cmap = components_cmap
        self.spatial_contours = None
        self.construct_component_mapping_matrix()

        self.component_ids = np.arange(self.N_components_total,dtype=int)
        self.good_component_ids = self.estimates.idx_components
        self.good_components_mask = np.zeros(self.N_components_total, dtype=bool)  
        for component_id in self.good_component_ids:
            self.good_components_mask[component_id]=True

        self.selected_mask = np.zeros(self.N_components_total, dtype=bool)
        self.discarded_mask = np.zeros(self.N_components_total, dtype=bool) # Array of flags to keep track of discarded components
        self.selectedModeActive = False         # Flag to toggle between selected and all components

        
        self.find_contours()
        

    def load_estimates(self):
        with open(self.estimates_filepath,"rb") as f:
            self.estimates = pickle.load(f)
        self.N_components_total = len(self.estimates.C)
        
    def get_component_color(self, component_id):
        return to_hex(self.components_cmap(component_id/self.N_components_total))
    
    def get_temporal_trace(self, component_id):
        return self.estimates.C[component_id,:]
    
    def get_spatial_matrix(self, component_id="good"):
        '''
            Returns the spatial matrix corresponding to a particular component.
            Takes data from self.estimates.A
        '''
        if component_id=="good":
            return np.reshape(self.estimates.A[:,self.estimates.idx_components].sum(axis=1), self.estimates.dims, order='F')
        if component_id=="all":
            return np.reshape(self.estimates.A[:,:].sum(axis=1), self.estimates.dims, order='F')
    
        return np.reshape(self.estimates.A[:,component_id].toarray(), self.estimates.dims, order='F')

    def find_contours(self):
        ''' Find and store controus for all components'''
        self.spatial_contours = dict()
        for k in range(self.N_components_total):
            component_matrix = self.get_spatial_matrix(k)
            self.spatial_contours[k] = np.array(skimage.measure.find_contours(component_matrix>0,0)[0])

    def construct_component_mapping_matrix(self, which="good"):
        if which=="good":
            components_to_map = self.estimates.idx_components
        else:
            components_to_map = np.arange(self.N_components_total)

        self.component_mapping = np.zeros(shape=self.estimates.dims, dtype=object)
        for i in range(self.estimates.dims[0]):
            for j in range(self.estimates.dims[1]):
                self.component_mapping[i,j]=[]

        for component_id in components_to_map:
            mask = (self.get_spatial_matrix(component_id)>0)
            for x in self.component_mapping[mask]:
                x.append(component_id)
    
    def get_components_from_point(self, x,y):
        print(self.component_mapping[x,y])
        return self.component_mapping[x,y]

    def get_spatial_contour(self, component_id):
        ''' Get spatial contour of component'''
        if self.spatial_contours is None:
            self.find_contours()
        return self.spatial_contours[component_id]

    def merge_components(self, component_ids):
        '''
            Merges several components into one.
            This is done by:

                1) Setting the spatial contour to be the union of component contours
                2) Setting the temporal trace to be the average of component traces

            Modifications override the component with the lowest id, all the other components are simply discarded (removed from idx_components)
        '''
        print("Merging ",component_ids," components")

        unified_matrix = np.any([self.get_spatial_matrix(k) for k in component_ids],axis=0)
        average_contour = np.array(skimage.measure.find_contours(unified_matrix>0,0)[0])
        average_trace = np.mean([self.get_temporal_trace(k) for k in component_ids], axis=0)


        modified_component = np.min(component_ids)

        for i in range(self.estimates.dims[0]): # Without the loop numpy for some reason can't assign object values as elements of array :/
            for j in range(self.estimates.dims[1]):
                if unified_matrix[i,j]>0:
                    self.component_mapping[i,j] = [modified_component]


        self.estimates.A[:,modified_component] = csc_matrix(unified_matrix).reshape((np.prod(self.estimates.dims),1),order="F")
        self.estimates.C[modified_component,:] = average_trace
        self.spatial_contours[modified_component] = average_contour


    def toggle_selected_mode(self):
        self.selectedModeActive = (not self.selectedModeActive)

    def set_selected_mode(self, target_state):
        self.selectedModeActive = target_state

    def toggle_component_selection(self, component_id):
        self.selected_mask[component_id] = (not self.selected_mask[component_id])

    def set_component_selection(self, component_ids,target_state):
        self.selected_mask[component_ids]=target_state

    def get_component_selection(self, component_id):
        return self.selected_mask[component_id]

    def get_selected_components(self):
        return self.component_ids[self.selected_mask]
    
    def get_unselected_components(self):
        return self.component_ids[np.invert(self.selected_mask)]
    
    def discard_components(self, component_ids):
        self.discarded_mask[component_ids] = True

    def restore_components(self, component_ids):
        self.discarded_mask[component_ids] = False

    def get_discarded_components(self):
        return self.component_ids[self.discarded_mask]
    

    




