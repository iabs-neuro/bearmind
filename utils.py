from config import DATA_PATHWAY


def set_folder_structure():

    global DATA_PATHWAY
    if DATA_PATHWAY == 'legacy':
        folder_structure = '*\\*\\*\\Miniscope\\'
    elif DATA_PATHWAY == 'bonsai':
        folder_structure = '*'
    else:
        raise ValueError('Wrong pathway!')

    return folder_structure


def write_to_config(root, pathway):
    config_content = f'ROOT = {root} \nDATA_PATHWAY = {pathway}'
    with open('config.py', 'w') as f:
        f.write(config_content)
