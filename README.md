# BEARMiND
A pipeline for Batch Examination & Analysis of Raw Miniscopic Neural Data
</br>
</br>
The main idea of the project is to optimize user's efforts and time for batch miniscopic data processing. Instead of one-by-one analysing of each imaging session, here you should spend some time at the beginning, examinating all raw videos, specifying field of view and other parameters, then launch the batch routines for motion correction and cnmf for all videos, which may take a while if the amount of data is significant, and then all you need is just to examine the results and make some corrections if you want! Also, you can use a special designed handy module for detection of significant calcium events based on scalable thresholding and trace approximation.
</br>
</br>
NB!! This pipenine requires Jupyter Lab or Notebook 7 installed. If you don't want to update your Jupyter notebook, use old good CaDet pipeline, it moved to a separate folder CaDet, where you can run it as usual.

## Installation
First, you need to install CaImAn environment: https://github.com/flatironinstitute/CaImAn</br>
In brief, all you need for getting CaImAn installed is to type the following commands in your Anaconda (or miniconda) prompt:
</br></br>
conda install -n base -c conda-forge mamba   # install mamba in base environment</br>
mamba create -n caiman -c conda-forge caiman # install caiman</br>
conda activate caiman  # activate virtual environment</br>
</br>
Then, you need to clone this repo to your PC. You may do it by downloading .zip file (see the button above) and unpacking it, OR you may use your git client and type "git clone https://github.com/iabs-neuro/bearmind" in a command prompt.

## Usage
Launch BEARMiND_full_pipeline.ipynb in a Jupyter Lab or in a Jupyter notebook, and follow the instructions. Typically you may want to duplicate the pipeline for each new user and/or experiment, but keep in mind that .py files starting with bm_ should present in the folder where you are launching the pipeline from. 

## Troubleshooting
Here is a list of known bugs which is by no means not full. Please report bugs in 'Issues' (see the button above).
### Module 4
ERROR:bokeh.server.views.ws:Refusing websocket connection from Origin 'http://localhost:8891';                       use --allow-websocket-origin=localhost:8891 or set BOKEH_ALLOW_WS_ORIGIN=localhost:8891 to permit this; currently we allow origins {'localhost:8888'}
WARNING:tornado.access:403 GET /ws (::1) 0.00ms
</br></br>
To deal with it, run the cell below with a proper port number, which you can take from the address string of your browser. 


