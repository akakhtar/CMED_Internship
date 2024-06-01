import pickle
import pandas as pd
from .config import ENCODING_PATH, EXCEl_PATH


# loading of encoding to recognise the face
def load_encoding(path=ENCODING_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


# loading of Excel sheet on which result is to be written
def load_datasheet(path=EXCEl_PATH):
    return pd.read_excel(path)
