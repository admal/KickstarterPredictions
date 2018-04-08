import pandas as pd
from config import *


def load_data():
    frames = []
    for file in DATA_FILES:
        data_frame = pd.read_csv(file, encoding='ISO-8859-1', usecols=DATA_COLUMNES)
        frames.append(data_frame)

    data = pd.concat(frames)
    return data
