import os
from os.path import join, abspath, dirname
import numpy as np
import pandas as pd


def build_imgfile_to_labels_csv(data_path="annotations/", is_abspath=False):
    """ Builds a csv file that keeps all image file names and multi-hot labels as columns.

    Kwargs:
      - data_path: location of data-directory in format as it was provided by course
      - is_abspath: whether data_path is absolute or relative path

    Saves csv to data_path/file_to_labels_table.csv and Returns None """
    if not is_abspath:
        data_path = abspath(data_path)
    filenames = os.listdir(data_path)
    parent_dir = dirname(data_path)

    res = {"filename": [f"im{i}.jpg" for i in range(20001)]}
    for fname in filenames:
        _name = fname[:-4]
        data = open(join(data_path, fname), "r").read().splitlines()
        data = np.array(list(map(int, data)))
        np_vec = np.zeros(20001)
        np_vec[data] = 1
        res[_name] = np_vec

    df = pd.DataFrame(res)
    df.drop(0, axis=0, inplace=True)  # the first row is all zeroes as there doesn't exist file named im0.jpg
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(int)
    df.to_csv(join(parent_dir, "file_to_labels_table.csv"), index=False)
