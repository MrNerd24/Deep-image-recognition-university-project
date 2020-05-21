import os
from os.path import join, abspath, dirname
import numpy as np
import pandas as pd
import pickle


def save_pickle(dataset, path):
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


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


def get_train_val_indexes(df, split_prop, shuffle=False, stratified=False):
    """ Given index range and parameter settings, create indexes for train and validation sets.

    Args:
      - df: Pandas DataFrame with first column being image id, second .. n idx are binary class labels
      - split_prop: splitting proportion. 0.5 means 50/50 split, 0.8 means 80/20 split, 0.2 means 20/80, etc.
    Kwargs:
      - shuffle: whether to shuffle the index before splitting
      - stratified: try to ensure that each class has *at least* as extreme distribution as given in proportion
    returns train_index, validation_index """

    def shuffle_index(index):
        new_index = index.copy()
        np.random.shuffle(new_index)  # in-place, no parameter to set to False
        return new_index

    n_rows = df.shape[0]
    index = np.arange(n_rows)
    if stratified:
        train, val = [], []
        for name, value in df.sum(axis=0).iteritems():
            if name == "imageId":
                continue

            # create mask then subset full index
            mask = (df.loc[:, [name]] == 1).to_numpy()
            mask = mask.reshape(mask.shape[0])  # reduce to same dim with index 1d array
            sub_idx = index[mask]
            n_rows = len(sub_idx)
            split_point = int(n_rows*split_prop)
            if shuffle:
                sub_idx = shuffle_index(sub_idx)
            _train, _val = sub_idx[:split_point], sub_idx[split_point:]
            train.extend(_train)
            val.extend(_val)

        # As this is multilabel dataset, each inastance may have duplicates of index numbers
        train, val = np.array(list(set(train))), np.array(list(set(val)))
        # Multilabel nature may also cause train and val sets to have intersecting indexes
        intersect_elems = np.intersect1d(train, val)
        rm_from_train, rm_from_val = np.array_split(intersect_elems, 2)
        train = np.array([e for e in train if e not in rm_from_train])
        val = np.array([e for e in val if e not in rm_from_val])
        assert len(train)+len(val) == 10176, "All non all-zero instances should be covered in train, val sets..."

        # Now the all-zero rows has to be included with props resulted in previous steps
        row_label_counts = df.iloc[:, 1:].sum(axis=1).to_numpy()
        mask = (row_label_counts == 0)
        mask = mask.reshape(mask.shape[0])
        sub_idx = index[mask]
        if shuffle:
            sub_idx = shuffle_index(sub_idx)
        n_rows = len(sub_idx)
        tmp_split_prop = len(train)/(len(train)+len(val))
        split_point = int(n_rows*tmp_split_prop)
        _train, _val = sub_idx[:split_point], sub_idx[split_point:]
        assert len(_train)+len(_val) == 9824, "All of all-zero instances should be covered in train, val sets..."
        train = np.concatenate([train, _train])
        val = np.concatenate([val, _val])
        assert len(train)+len(val) == df.shape[0], "Data sampling resulted incorrect sample size"
    else:
        split_point = int(n_rows*split_prop)
        if shuffle:
            index = shuffle_index(index)
        train, val = index[:split_point], index[split_point:]

    return train, val
