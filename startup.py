from tkinter.filedialog import askopenfilename, askopenfilenames, Tk
from bm_batch_routines import *
from bm_examinator import *
from bm_spike_detection import *
from wavelet_event_detection import *
from utils import *
from config import *

from glob import glob
import os
import tqdm
from jupyter_server import serverapp

# This is needed for the proper work of further manual file selection:
wnd = Tk()
wnd.wm_attributes('-topmost', 1)
response = wnd.withdraw()

folder_structure = set_folder_structure()

#global LOCAL_URLS
LOCAL_URLS = [server['url'] for server in list(serverapp.list_running_servers())]
os.environ["BOKEH_ALLOW_WS_ORIGIN"] = ','.join([origin[7:-1] for origin in LOCAL_URLS])

