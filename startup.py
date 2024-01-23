from tkinter.filedialog import askopenfilename, askopenfilenames, Tk
from bm_batch_routines import *
from bm_examinator import *
from bm_spike_detection import *
from path_config import set_pathway

from glob import glob
import os
import tqdm
from jupyter_server import serverapp


global LOCAL_URLS
LOCAL_URLS = [server['url'] for server in list(serverapp.list_running_servers())]
os.environ["BOKEH_ALLOW_WS_ORIGIN"] = LOCAL_URLS[0][7:-1]

set_pathway()

# This is needed for the proper work of further manual file selection:
wnd = Tk()
wnd.wm_attributes('-topmost', 1)
response = wnd.withdraw()
