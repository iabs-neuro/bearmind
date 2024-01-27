import json
import os
import multiprocessing
import psutil


DEFAULT_CONFIG = {
    'ROOT': 'C:\\Users\\1\\HM_NOF_4D',
    'DATA_PATHWAY': 'bonsai',
    'CPUs': multiprocessing.cpu_count(),
    'RAM': int(psutil.virtual_memory().total/1024/1024/1024) + 1
}

DEFAULT_MOUSE_CONFIG = {
    'crop_params': {},
    'mc_params': {},
    'cnmf_params': {}
}


def create_config(content=DEFAULT_CONFIG, name='config.json'):
    if 'ROOT' in content:
        content['ROOT'] = os.path.normpath(content['ROOT'])
    write_config(content, name=name)


def write_config(data, name='config.json'):
    with open(name, 'w+') as fp:
        json.dump(data, fp)


def read_config(name='config.json'):
    with open(name, 'r') as fp:
        config = json.load(fp)
        return config


def update_config(new_data, name='config.json'):
    old_config = read_config(name=name)
    if 'ROOT' in new_data:
        new_data['ROOT'] = os.path.normpath(new_data['ROOT'])
    old_config.update(new_data)

    if name == 'config.json':  # add system info to main config
        system_info = {
            'CPUs': multiprocessing.cpu_count(),
            'RAM': int(psutil.virtual_memory().total/1024/1024/1024) + 1
        }
        old_config.update(system_info)

    write_config(old_config, name=name)

    if name == 'config.json':
        # read new updated main config
        global CONFIG
        CONFIG = read_config()


def create_mouse_configs(root=None):
    if root is None:
        root = CONFIG['ROOT']

    mouse_folders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f)) and 'CONFIGS' not in f]
    mouse_config_folder = os.path.join(root, 'MOUSE_CONFIGS')
    os.makedirs(mouse_config_folder, exist_ok=True)

    for msf in mouse_folders:
        mouse_config_path = os.path.join(root, mouse_config_folder, msf + '.json')
        if not os.path.exists(mouse_config_path):
            create_config(content=DEFAULT_MOUSE_CONFIG, name=mouse_config_path)


def get_mouse_config_path(fname):
    splt_path = os.path.normpath(fname).split(os.sep)
    ms_name = splt_path[-2]
    ms_config_name = os.path.join(CONFIG['ROOT'], 'MOUSE_CONFIGS', ms_name + '.json')
    return ms_config_name


if not os.path.exists('config.json'):
    create_config()

CONFIG = read_config()
