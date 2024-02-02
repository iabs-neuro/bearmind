from config import read_config
import pytz
import datetime

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


def get_datetime():
    tz = pytz.timezone('Europe/Moscow')
    now = datetime.datetime.now(tz)

    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    return dt_string
