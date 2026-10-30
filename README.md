# BEARMiND
### A pipeline for Batch Examination & Analysis of Raw Miniscopic Neural Data
<img src="/bearmind_logo.png" align="left">

Processing large amounts of miniscope data is usually a time-consuming, step-by-step
procedure requiring constant manual intervention and inspection of putative neural units.
BEARMiND is a NoRMCorre- and CaImAn-based full-Python pipeline that optimizes user effort
for **batch** miniscope data processing: you set parameters once at the beginning
(field of view, CNMF parameters), launch batch motion correction and CNMF over all sessions,
and then examine and curate the results. The pipeline is organized as a Jupyter notebook
with cells grouped into modules; each module produces third-party-compatible outputs and can
be run independently.

Beyond the original interactive workflow, this branch adds an **automated neuron quality
inspection** layer (machine-learning–based component filtering) and **wavelet-based calcium
event detection** (via the [DRIADA](https://github.com/iabs-neuro/driada) package), so that
large batches can be curated with far less manual work.

</br>

## Installation
First, install a CaImAn environment: https://github.com/flatironinstitute/CaImAn </br>
In brief, in your Anaconda/Miniconda prompt:
</br></br>
Install mamba in the base environment: `conda install -n base -c conda-forge mamba` </br>
Create the CaImAn env (replace `<ENV_NAME>`): `mamba create -n <ENV_NAME> -c conda-forge caiman` </br>
Activate it: `conda activate <ENV_NAME>` </br>
Install dependencies:
`pip install moviepy PySide6 wgpu glfw fastplotlib jupyter_rfb sidecar sortedcontainers cmasher opencv-python ssqueezepy bokeh natsort scikit-learn`
</br></br>
Install DRIADA (required for wavelet event detection and the auto-inspection metrics):
`pip install git+https://github.com/iabs-neuro/driada`
</br></br>
Then clone this repository:
`git clone https://github.com/iabs-neuro/bearmind`
</br>
(or download the .zip via the button above and unpack it).

### Alternative CaImAn env installation
If mamba gives you trouble, use the conda libmamba solver
(https://conda.github.io/conda-libmamba-solver/user-guide/):
</br>
`conda update -n base conda` </br>
`conda install -n base conda-libmamba-solver` </br>
`conda create -n caiman -c conda-forge caiman --solver=libmamba` </br>
`conda activate caiman` </br></br>

## Usage
Launch **`BEARMiND_full_pipeline.ipynb`** in Jupyter Lab / Notebook and follow the cells.
Typically you duplicate the pipeline per user and/or experiment; keep in mind that all `.py`
files from this repo must be present in the folder you launch the pipeline from.
</br>
Below is a brief description of the main stages.

### Module 1. Initial inspection of miniscope data
Inspect raw miniscope videos and define the optimal field of view. Crop parameters can be
saved and reused across sessions.
</br>INPUTS: miniscope calcium imaging data (`.avi` files)
</br>OUTPUTS: Python archives (`.pickle`) with cropping parameters, stored alongside the `.avi` files

**Batch cropping.** Native `.avi` files are cropped according to the saved crops, concatenated,
and written as `.tif` files in the working directory; timestamps are copied as well.
</br>INPUTS: natively stored miniscope data + saved crop files
</br>OUTPUTS: cropped `.tif` files with timestamps

### Module 2. Batch motion correction
Based on NoRMCorre piecewise-rigid motion correction [Pnevmatikakis & Giovannucci, 2017].
</br>INPUTS: cropped `.tif` files
</br>OUTPUTS: motion-corrected `.tif` files

### Module 2.5. Setting CNMF parameters
Load a limited amount of data and interactively tune the key CNMF parameters:
</br>● `gSig` – Gaussian filter kernel size for segmenting putative neurons
</br>● `min_corr` – minimal correlation for seeding a neuron
</br>● `min_pnr` – minimal peak-to-noise ratio of pixel time traces for seeding a neuron
</br>Usually run once per animal/batch.
</br>INPUTS: motion-corrected `.tif` files

### Module 3. Batch CNMF
Based on the CaImAn CNMF-E routine [Giovannucci et al., 2019].
</br>INPUTS: motion-corrected `.tif` files
</br>OUTPUTS: CNMF results (estimates objects) saved as `.pickle` files

### Module 4. Examination and curation of CNMF results
A Bokeh-based interface to load and inspect CNMF results. Neural contours and their time
traces are selectable, pannable and scalable; components can be deleted or merged (on merge,
the highest-SNR spatial component is kept and the trace is recalculated). Two operation modes
are supported:
</br>● **legacy** – classic manual curation by quality metrics (SNR, spatial correlation, etc.)
</br>● **capcan** – metrics enriched with reconstruction quality and a machine-learning
keep-probability, with on-tap per-neuron metric display and ML-probability coloring to speed
up curation.
</br>Results can be exported in human-readable form (`.tif` contour images, `.csv` trace tables)
and as `.mat` contour arrays for cross-session matching (e.g. CellReg [Sheintuch et al., 2017]).

### Module 5. Calcium event detection (optional)
For event-based analysis, significant calcium events are separated from noise via either
threshold-based detection (local-maxima thresholding + trace fitting) or **wavelet** detection
(DRIADA's generalized Morse wavelet ridge detection). Per-event statistics (kinetics, relative
amplitude dF/F₀, SNR) can be collected with `event_stats.py` / the `collect_*` scripts.
</br>INPUTS: timestamped `.csv` trace tables (or CaImAn estimates)
</br>OUTPUTS: `.csv` tables of 0/1 event notation; `.pickle` files with per-event parameters

## Automated quality inspection (auto-inspection)
For large batches, putative neurons can be filtered automatically instead of one-by-one.
`ae_launch.run_auto_inspection(...)` runs the auto-inspection pipeline on a CaImAn estimates
file: it computes per-neuron quality metrics (`auto_inspector.py`), detects edge/corner
artifacts (`corner_artifacts.py`), and applies a trained classifier (see `ml/`) to flag
components for deletion, returning a curated estimates object. Batch drivers
(`batch_autoinspect*.py`, `batch_export*.py`) apply this across many sessions. Models and the
training/evaluation scripts live under `ml/` (see `ml/README.md`).

## Troubleshooting
A non-exhaustive list of known issues — please report bugs in *Issues* (button above).

### Module 4 (Bokeh)
```
ERROR:bokeh.server.views.ws:Refusing websocket connection from Origin 'http://localhost:8891';
use --allow-websocket-origin=localhost:8891 or set BOKEH_ALLOW_WS_ORIGIN=localhost:8891 to permit this
```
Run the dedicated cell with the correct port number (taken from your browser's address bar).

## References
- Pnevmatikakis, E.A. & Giovannucci, A. (2017). NoRMCorre: An online algorithm for piecewise rigid motion correction of calcium imaging data. *J. Neurosci. Methods*.
- Giovannucci, A. et al. (2019). CaImAn: An open source tool for scalable calcium imaging data analysis. *eLife*.
- Sheintuch, L. et al. (2017). Tracking the same neurons across multiple days in Ca²⁺ imaging data (CellReg). *Cell Reports*.
