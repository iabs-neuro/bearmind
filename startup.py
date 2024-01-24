from tkinter.filedialog import askopenfilename, askopenfilenames, Tk
from bm_batch_routines import *
from bm_examinator import *
from bm_spike_detection import *
from utils import *
from config import create_config, read_config, update_config, CONFIG

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
os.environ["BOKEH_ALLOW_WS_ORIGIN"] = LOCAL_URLS[0][7:-1]
