"""
Config file for creating the data we need for our project.
"""

import os

RAW_DATA_PATH = os.path.join('data', 'raw', 'raw.pkl')
CLEAN_DATA_PATH = os.path.join('data', 'clean', 'modeling.pkl')
DATA_START_YEAR = 1970
DATA_END_YEAR = 2022
REMOVE_YEARS = [1981, 1994, 2020]
