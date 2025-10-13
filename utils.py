from config import read_config
import pytz
import datetime
import numpy as np

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


def _plain_bfs(adj, source):
    '''
    adapted from networkx.algorithms.components.connected._plain_bfs

    Args:
        adj:
        source:

    Returns:

    '''

    n = adj.shape[0]
    seen = {source}
    nextlevel = [source]
    while nextlevel:
        thislevel = nextlevel
        nextlevel = []
        for v in thislevel:
            for w in get_neighbors_from_adj(adj, v):
                if w not in seen:
                    seen.add(w)
                    nextlevel.append(w)
            if len(seen) == n:
                return seen
    return seen


def get_neighbors_from_adj(a, node):
    inds = a[[node], :].nonzero()[1]
    return inds


def get_ccs_from_adj(adj):
    seen = set()
    for v in range(adj.shape[0]):
        if v not in seen:
            c = _plain_bfs(adj, v)
            seen.update(c)
            yield c


def calculate_polygon_area(coordinates):
    # Filter out points with NaN coordinates
    valid_coordinates = [point for point in coordinates if not (np.isnan(point[0]) or np.isnan(point[1]))]

    # Check if we have enough valid points to form a polygon
    if len(valid_coordinates) < 3:
        return 0  # Not enough points to form a polygon

    area = 0

    # Number of vertices
    n = len(valid_coordinates)

    # Calculate area using the Shoelace formula
    for i in range(n):
        j = (i + 1) % n
        area += valid_coordinates[i][0] * valid_coordinates[j][1]
        area -= valid_coordinates[j][0] * valid_coordinates[i][1]

    # Take absolute value and divide by 2
    area = abs(area) / 2

    return area