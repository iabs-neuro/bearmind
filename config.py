import json
import os
import multiprocessing
from pathlib import Path

import psutil

# The machine's main config lives next to this file, not in the folder Python started in: the batch's stages and
# the inspection server run from whatever folder they were started in and must read one file (autoinspection takes
# its joblib backend from it). Callers name it 'config.json', as before; mouse and session configs keep their paths.
MAIN_CONFIG = 'config.json'
CONFIG_PATH = Path(__file__).resolve().parent / MAIN_CONFIG

DATA_PATHWAYS = ('bonsai', 'legacy')

# Until update_config names the data folder, ROOT is the user's home folder, which exists on every machine.
# TEMP_PATHWAY is CaImAn's own default temp folder (~/caiman_data/temp).
DEFAULT_CONFIG = {
    'ROOT': str(Path.home()),
    'TEMP_PATHWAY': str(Path.home() / 'caiman_data' / 'temp'),
    'DATA_PATHWAY': 'bonsai',
    'CPUs': multiprocessing.cpu_count(),
    'RAM': int(psutil.virtual_memory().total/1024/1024/1024) + 1,
    # Parallelization backend for joblib: 'loky', 'threading', or 'multiprocessing'
    # Use 'threading' if you have GPU libraries (PyTorch/CuPy) that conflict with loky
    'PARALLELIZATION_BACKEND': 'loky'
}

DEFAULT_MOUSE_CONFIG = {
    'crop_params': {},
    'mc_params': {},
    'cnmf_params': {}
}


def _config_file(name):
    # CONFIG_PATH is read at call time, so a test can point the main config elsewhere
    return CONFIG_PATH if name == MAIN_CONFIG else Path(name)


def _is_main(name):
    return _config_file(name) == CONFIG_PATH


def create_config(content=DEFAULT_CONFIG, name=MAIN_CONFIG):
    content = dict(content)  # the defaults are shared; normalising ROOT must not change them
    if 'ROOT' in content.keys():
        content['ROOT'] = os.path.normpath(content['ROOT'])
    write_config(content, name=name)


def write_config(data, name=MAIN_CONFIG):
    with open(_config_file(name), 'w+') as fp:
        json.dump(data, fp)


def read_config(name=MAIN_CONFIG):
    """A config file as a dict; a machine without its main config yet gets a copy of DEFAULT_CONFIG (nothing is
    written: only write_config and update_config create files)."""
    path = _config_file(name)
    if _is_main(name) and not path.exists():
        return dict(DEFAULT_CONFIG)
    with open(path, 'r') as fp:
        config = json.load(fp)
        return config


def update_config(new_data, cpath=MAIN_CONFIG):
    old_config = read_config(name=cpath)
    if 'ROOT' in new_data.keys():
        new_data['ROOT'] = os.path.normpath(new_data['ROOT'])
    if 'TEMP_PATHWAY' in new_data:
        new_data['TEMP_PATHWAY'] = os.path.normpath(new_data['TEMP_PATHWAY'])
    old_config.update(new_data)
    if _is_main(cpath):  # add system info to main config
        system_info = {
            'CPUs': multiprocessing.cpu_count(),
            'RAM': int(psutil.virtual_memory().total/1024/1024/1024) + 1
        }
        old_config.update(system_info)

    write_config(old_config, name=cpath)

    if _is_main(cpath):
        # in place: bm_examinator and history.legacy_pipeline hold this dict from `from config import CONFIG`
        CONFIG.clear()
        CONFIG.update(read_config())


def create_mouse_configs(root=None):
    if root is None:
        root = CONFIG['ROOT']

    base_path = os.path.dirname(CONFIG['ROOT'])
    session_folders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f)) and 'CONFIGS' not in f]
    mouse_config_folder = os.path.join(base_path, 'MOUSE_CONFIGS')
    os.makedirs(mouse_config_folder, exist_ok=True)

    for sf in session_folders:
        ms_name = sf.split(sep='_')[1]
        mouse_config_path = os.path.join(base_path, mouse_config_folder, ms_name + '.json')
        if not os.path.exists(mouse_config_path):
            create_config(content=DEFAULT_MOUSE_CONFIG, name=mouse_config_path)


def create_session_configs(root=None):
    if root is None:
        root = CONFIG['ROOT']

    base_path = os.path.dirname(CONFIG['ROOT'])
    session_folders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f)) and 'CONFIGS' not in f]
    session_config_folder = os.path.join(base_path, 'SESSION_CONFIGS')
    os.makedirs(session_config_folder, exist_ok=True)

    for sf in session_folders:
        session_config_path = os.path.join(base_path, session_config_folder, sf + '.json')
        if not os.path.exists(session_config_path):
            create_config(content=DEFAULT_MOUSE_CONFIG, name=session_config_path)


# How many folders lie between a recording file and its session folder in each data layout:
# bonsai <session>/<file>; legacy (UCLA Miniscope) <session>/<date>/<HH_MM_SS>/Miniscope/<file>.
SESSION_FOLDER_DEPTH = {'bonsai': 0, 'legacy': 3}


def get_session_name_from_path(fname):
    pathway = CONFIG['DATA_PATHWAY']
    if pathway not in SESSION_FOLDER_DEPTH:
        raise ValueError(f"unknown DATA_PATHWAY: {pathway!r}; expected one of "
                         + ", ".join(repr(p) for p in DATA_PATHWAYS))
    return Path(os.path.normpath(fname)).parents[SESSION_FOLDER_DEPTH[pathway]].name


def get_mouse_config_path_from_fname(fname):
    session_name = get_session_name_from_path(fname)
    ms_name = session_name.split(sep='_')[1]  # Experiment_Mouse_Session
    cpath = get_mouse_config_path(ms_name)
    return cpath


def get_session_config_path_from_fname(fname):
    session_name = get_session_name_from_path(fname)
    cpath = get_session_config_path(session_name)
    return cpath


def get_mouse_config_path(ms_name):
    base_path = os.path.dirname(CONFIG['ROOT'])
    ms_config_name = os.path.join(base_path, 'MOUSE_CONFIGS', ms_name + '.json')
    return ms_config_name


def get_session_config_path(session):
    base_path = os.path.dirname(CONFIG['ROOT'])
    ms_config_name = os.path.join(base_path, 'SESSION_CONFIGS', session + '.json')
    return ms_config_name


def get_parallelization_backend():
    """Get the joblib parallelization backend.

    Automatically uses 'threading' when GPU wavelet backend is active
    to avoid conflicts with PyTorch/CuPy. Otherwise uses config setting
    (default: 'loky').

    Returns
    -------
    str
        'threading', 'loky', or 'multiprocessing'
    """
    # Import here to avoid circular imports
    from wavelet_backend import is_gpu_backend

    # Auto-switch to threading when GPU is active (avoids loky/CUDA conflicts)
    if is_gpu_backend():
        return 'threading'

    return CONFIG.get('PARALLELIZATION_BACKEND', 'loky')


CONFIG = read_config()
