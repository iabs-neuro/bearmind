
def set_pathway():
    global DATA_PATHWAY
    global folder_structure
    DATA_PATHWAY = 'bonsai'
    # DATA_PATHWAY = 'legacy'

    global root
    root = "C:\\Users\\Public\\IABS_DATA\\HM_NOF_2D"

    if DATA_PATHWAY == 'legacy':
        folder_structure = '*\\*\\*\\Miniscope\\'
    elif DATA_PATHWAY == 'bonsai':
        folder_structure = '*'
    else:
        raise ValueError('Wrong pathway!')