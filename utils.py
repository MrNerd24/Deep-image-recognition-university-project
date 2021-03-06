import os
from os.path import join, abspath, dirname
from operator import eq, lt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import image_dataset
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

def get_indexes_for_mislabelled_images(dataset, y_trues, y_hats, mislabels_count = 1, true_labels_count = 0, y_hat_labels_count = 0, operators = [eq, eq], return_sample = True, sample_size = 12):
    """ Used for error analysis. Returns a list of indexes for the images where the number of mislabelings
        match the given parameters.

    Args:
      - dataset: the image dataset from where the indexes are searched from
      - y_trues: matrix of true labels for the image dataset
      - y_hats: matrix of labels given by a classifier algorithm
    Kwargs:
      - mislabels_count: number of mislabelings for each image
      - true_labels_count: number of labels that the images actually have
      - y_hat_labels_count: number of labels that the images have according to the classifier
      - operators: a list of two functions that determine whether the number of labels should be exatly as high or at least as high
      - return_sample: boolean for whether the function returns all the indexes that match the given parameter or just a random sample of them
      - sample_size: the size of the sample of indexes
    returns a list of indexes """
    losses = np.sum((np.array(y_hats) - np.array(y_trues))**2, axis = 1)
    mx_idx = np.argwhere(losses == mislabels_count)
    new_idx = []
    for idx in mx_idx:
        idx = int(idx)
        datum = dataset[idx]
        true_labels = datum["labelsString"]
        y_hat = np.array(y_hats[idx], dtype = 'bool')
        if (operators[0](len(true_labels), true_labels_count) and operators[1](np.count_nonzero(y_hat), y_hat_labels_count)):    
            new_idx.append(idx)
            
    if (return_sample and len(new_idx) > sample_size):
        sample = np.array(new_idx)
        arr = np.random.choice(len(sample), size = sample_size, replace = False)
        return(sample[arr])
    else:
        return new_idx

def plot_imgs_with_labels(dataset, y_hats, sample, pltsize = 6, rows = 4, cols = 3):
    """ Used for error analysis. Plots the images from a given index set along with the true labels and predicted labels
        of those images.
    Args:
      - dataset: the image dataset that has the images to be plotted
      - y_hats: matrix of labels given by a classifier algorithm
      - sample: the indexes of the images to be plotted
    Kwargs:
      - pltsize: the size of the plotted images
      - rows: the number of rows for the plotted image
      - cols: the number of columns for the plotted image
    plots an image and Returns None """
    labelList = np.array(pd.read_csv("file_to_labels_table.csv").columns[1:])
    pltsize = 6
    rows = 6
    cols = 3
    plt.figure(figsize=(3 * pltsize, 4 * pltsize))
    for i in range(len(sample)):
        i_idx = sample[i]
        datum = dataset[i_idx]
        plt.subplot(rows, cols, i + 1)
        plt.axis('off')
        plt.imshow(datum["imagePil"])
        plt.title("{} : {}".format(datum["labelsString"], labelList[np.array(y_hats[i_idx], dtype = 'bool')]))

def get_mislabels_count_for_each_label(y_hats, y_trues):
    """ Used for error analysis. Returns a list of mislabeling counts for each label.
    Args:
      - y_hats: matrix of labels given by a classifier algorithm
      - y_trues: matrix of true labels for the image dataset
    returns a list of mislabeling counts for each label """
    labelList = np.array(pd.read_csv("file_to_labels_table.csv").columns[1:])
    ll = (y_hats == y_trues)
    mislabels = np.zeros(15, dtype = 'int')
    for i in range(14):
        idx = np.argwhere(ll[:,i] == False)
        mislabels[i] = len(idx)
    return(list(zip(labelList, mislabels)))

def get_mislabels_count_distribution(y_hats, y_trues):
    """ Used for error analysis. Returns the distribution of mislabeling counts.
    Args:
      - y_hats: matrix of labels given by a classifier algorithm
      - y_trues: matrix of true labels for the image dataset
    returns the distribution of mislabeling counts """
    losses = np.sum((np.array(y_hats) - np.array(y_trues))**2, axis = 1)
    mislabel_counts = np.zeros(14)
    for i in range(14):
        idx = np.argwhere(losses == i)
        mislabel_counts[i] = idx.shape[0]
    return np.array(mislabel_counts, dtype = 'int')

def get_indexes_of_mislabeled_images_with_label(label_idx, y_trues, y_hats):
    """ Used for error analysis. Returns a list of indexes for images that had a specific label and were
        mislabeled by a classification algorithm.
    Args:
      - label_idx: the index of the label that each image should have
      - y_trues: matrix of true labels for the image dataset
      - y_hats: matrix of labels given by a classifier algorithm
    returns  img_with_the_label_idxs """
    labelList = np.array(pd.read_csv("file_to_labels_table.csv").columns[1:])
    print(labelList[label_idx])
    img_with_the_label_idxs = np.array(list(map(lambda x: int(x), np.argwhere(y_trues[:, label_idx] == 1))))
    return img_with_the_label_idxs[np.array(list(map(lambda x: int(x), np.argwhere(y_hats[img_with_the_label_idxs, label_idx] == 0))))]
