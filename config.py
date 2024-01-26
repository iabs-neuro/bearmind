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


def create_config():
    if 'ROOT' in DEFAULT_CONFIG:
        DEFAULT_CONFIG['ROOT'] = os.path.normpath(DEFAULT_CONFIG['ROOT'])
    write_config(DEFAULT_CONFIG)


def write_config(data):
    with open('config.json', 'w+') as fp:
        json.dump(data, fp)


def read_config():
    with open('config.json', 'r') as fp:
        config = json.load(fp)
        return config


def update_config(new_data):
    old_config = read_config()
    if 'ROOT' in new_data:
        new_data['ROOT'] = os.path.normpath(new_data['ROOT'])
    old_config.update(new_data)

    system_info = {
        'CPUs': multiprocessing.cpu_count(),
        'RAM': int(psutil.virtual_memory().total/1024/1024/1024) + 1
    }
    old_config.update(system_info)

    write_config(old_config)

    # read new updated config
    global CONFIG
    CONFIG = read_config()


if not os.path.exists('config.json'):
    create_config()

CONFIG = read_config()
