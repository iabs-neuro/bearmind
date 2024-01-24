from config import read_config


def set_folder_structure():
    config = read_config()
    pathway = config['DATA_PATHWAY']
    if pathway == 'legacy':
        folder_structure = '*\\*\\*\\Miniscope\\'
    elif pathway == 'bonsai':
        folder_structure = '*'
    else:
        raise ValueError('Wrong pathway!')

    return folder_structure

