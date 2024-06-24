# BEARMiND
### A pipeline for Batch Examination & Analysis of Raw Miniscopic Neural Data
<img src="/bearmind_logo.png" align="left">
Processing of large amounts of miniscopic data often appears to be a time-keeping procedure, requiring user’s efforts for step-by-step launching of various procedures, including manual intervention and inspection of putative neural units. The aim of this NoRMCorre- and CaImAn-based full-Python pipeline is to optimize users' efforts and time for batch miniscopic data processing. Instead of one-by-one analysing of each imaging session, here you should spend some time at the beginning, examinating all raw videos, specifying field of view and other parameters, then launch the batch routines for motion correction and cnmf for all videos, which may take a while if the amount of data is significant, and then all you need is just to examine the results and make some corrections if you want! Also, you can use a special designed handy module for detection of significant calcium events based on scalable thresholding and trace approximation. The pipeline exists in the form of Jupyter notebook with subsequent cells, grouped in modules. Each module produces  third-party-compartible outputs and therefore can be run independently.
</br>

## Installation
First, you need to install CaImAn environment: https://github.com/flatironinstitute/CaImAn</br>
In brief, all you need for getting CaImAn installed is to type the following commands in your Anaconda (or miniconda) prompt:
</br></br>
install mamba in base environment: `conda install -n base -c conda-forge mamba` </br>
install caiman (enter desired venv name instead of <NEW_ENV_NAME>): `mamba create -n <NEW_ENV_NAME> -c conda-forge caiman` </br>
activate virtual environment: `conda activate caiman`  </br>
Install dependencies: `pip install  moviepy PySide6 wgpu glfw fastplotlib jupyter_rfb sidecar sortedcontainers cmasher opencv-python ssqueezepy`
</br>

Then, you need to clone this repo to your PC. You may do it by downloading .zip file (see the button above) and unpacking it, OR you may use your git client and type "git clone https://github.com/iabs-neuro/bearmind" in a command prompt.

### Alternative caiman env installation
If you have some trouble with mamba, you can try libmamba, which is a more conda-friendly solver, more information can be seen here:</br>
https://www.anaconda.com/blog/conda-is-fast-now</br>
https://conda.github.io/conda-libmamba-solver/user-guide/</br>
To do this, you may need first to update your conda distribution:</br>
`conda update -n base conda`</br>
Install the libmamba solver:</br>
`conda install -n base conda-libmamba-solver`</br>
You can set this solver as the default one by running </br>
`conda config --set solver libmamba`</br>
and then create the caiman environment:</br>
`conda create -n caiman -c conda-forge caiman`</br>
OR, you can use the libmamba solver just for this time:</br>
`conda create -n caiman -c conda-forge caiman --solver=libmamba`</br>
Finally, activate the caiman environment:</br>
`conda activate caiman` </br></br>

## Usage
Launch BEARMiND_demo.ipynb in a Jupyter Lab or in a Jupyter notebook, and follow the instructions. Typically you may want to duplicate the pipeline for each new user and/or experiment, but <s>bear</s>keep in mind that all .py files with this repo should present in the folder where you are launching the pipeline from. </br>
Here is a brief description of the main stages:<br/>
### Module 1. Initial inspection of miniscopic data
Here user can inspect raw miniscopic videos and define the optimal field of view. These parameters can be saved and copied for different imaging sessions.
INPUTS: Miniscopic calcium imaging data (.avi files)
OUTPUTS: Python archives (.pickle) with cropping parameters stored in the same folder with the .avi files.<br/>
<b>Batch cropping.</b> Here native miniscopic .avi files are cropped with respect to previously saved crops, concatenated and saved as .tif files in the working directory. Timestamps are copied as well.
</br>INPUTS: Natively stored miniscopic data and saved crop files 
</br>OUTPUTS: Cropped .tif files along with timestamps
### Module 2. Batch Motion Correction
Is based on the NoRMCorre piece-wise rigid motion correction routine [Pnevmatikakis & Giovanucci, 2017]. 
</br>INPUTS: cropped .tif files
</br>OUTPUTS: motion corrected .tif files
### Module 2.5. Setting of CNMF parameters
Here the user can load a limited amount of data and interactively adjust the key CNMF parameters: </br>
● gSig, the kernel size of the gaussian filter applied to the data for the proper segmentation of putative neurons</br>
● min_corr – minimal correlation value on the matrix for seeding a neuron  
● min_pnr – another threshold for seeding a neuron, minmal peak-to-noise ratio (PNR) of the time traces, corresponding to each pixel of the data.
</br>This module can be launched once for a batch of data from the same animal. 
</br>INPUTS: Motion corrected .tif files
</br>OUTPUTS: None
### Module 3. Batch CNMF
Is based on  the “vanilla” CaImAn routine, described in many details in [Giovanucci et al., 2019].
</br>INPUTS: Motion corrected .tif files
</br>OUTPUTS: CNMF results (estimates objects) saved as .pickle files
### Module 4. Examination of CNMF results
Here the user can load and inspect the results obtained by the CNMF routine in the module 3. The Bokeh-based interface supports simultaneous selection of neural contours along with their time traces. Both of the plots are panable and scalable. The user can manually delete or merge one or several selected components. In the latter case, the spatial component with the highest signal-to-noise ratio is kept, and the resulting trace is recalculated and this new neural unit is placed to the end of the list. The results of the analysis can be saved in human- readable format (.tif images and .csv tables).
</br>INPUTS: CNMF results (estimates objects) saved as .pickle files, obtained in Module 3; miniscopic timpestamp files
</br>OUTPUTS: the collection of .tif files with neural contours, stored in a separate folder; .csv table with time traces, the first column is a timestamp, the other are the traces numerated in the same order as the contours; .mat files with the array of contours for further matching of neurons between sessions which can be done by the CellReg [Sheintuch et al., 2017] routine. 
### Module 5. Calcium event detection (optional)
For the further analysis of  neural data along with animal’s behavior it’s often needed to deal with discrete events instead of continuous calcium traces. However, significant events should be separated from noise. Here, user is offered two different ways of event detection: by thresholding of local maxima with subsequent fit and by wavelet transform.
</br>INPUTS: timestamped .csv tables with traces
</br>OUTPUTS: .csv tables of the same size as traces with discrete (i.e., 0/1) event notation; separate .pickle files with all event parameters
</br></br>
## Troubleshooting
Here is a list of known bugs which is by no means not full. Please report bugs in 'Issues' (see the button above).
### Module 4
ERROR:bokeh.server.views.ws:Refusing websocket connection from Origin 'http://localhost:8891';                       use --allow-websocket-origin=localhost:8891 or set BOKEH_ALLOW_WS_ORIGIN=localhost:8891 to permit this; currently we allow origins {'localhost:8888'}
WARNING:tornado.access:403 GET /ws (::1) 0.00ms
</br></br>
To deal with it, run the cell below with a proper port number, which you can take from the address string of your browser. 


