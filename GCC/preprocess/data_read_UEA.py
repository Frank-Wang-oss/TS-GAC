import numpy as np
import os
import pickle
from scipy.io.arff import loadarff
from sklearn.utils import Bunch
from urllib.request import urlretrieve
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from pyts import datasets
import torch

def _parse_relational_arff(data):
    X_data = np.asarray(data[0])
    n_samples = len(X_data)
    X, y = [], []

    if X_data[0][0].dtype.names is None:
        for i in range(n_samples):
            X_sample = np.asarray(
                [X_data[i][name] for name in X_data[i].dtype.names]
            )
            X.append(X_sample.T)
            y.append(X_data[i][1])
    else:
        for i in range(n_samples):
            X_sample = np.asarray(
                [X_data[i][0][name] for name in X_data[i][0].dtype.names]
            )
            X.append(X_sample.T)
            y.append(X_data[i][1])

    X = np.asarray(X).astype('float64')
    y = np.asarray(y)

    try:
        y = y.astype('float64').astype('int64')
    except ValueError:
        y = y.astype(str)

    return X, y

def str2value(label_train, label_test):
    class_ = set(np.concatenate([label_train, label_test], 0))
    class_ = list(class_)
    num_class = len(class_)

    class_dict = {}
    for i in range(num_class):
        class_dict[class_[i]] = i

    # print(class_dict)
    label_train_value = []
    for train_i in label_train:
        label_train_value.append(class_dict[train_i])
    label_train_value = np.stack(label_train_value, 0)

    label_test_value = []
    for test_i in label_test:
        label_test_value.append(class_dict[test_i])
    label_test_value = np.stack(label_test_value, 0)

    # print(label_train)
    # print(label_train_value)

    return label_train_value, label_test_value

def _load_uea_dataset(dataset, path):
    """Load a UEA data set from a local folder.

    Parameters
    ----------
    dataset : str
        Name of the dataset.

    path : str
        The path of the folder containing the cached data set.

    Returns
    -------
    data : Bunch
        Dictionary-like object, with attributes:

        data_train : array of floats
            The time series in the training set.
        data_test : array of floats
            The time series in the test set.
        target_train : array
            The classification labels in the training set.
        target_test : array
            The classification labels in the test set.
        DESCR : str
            The full description of the dataset.
        url : str
            The url of the dataset.

    Notes
    -----
    Missing values are represented as NaN's.

    """
    new_path = path + dataset + '/'
    try:
        description_file = [
            file for file in os.listdir(new_path)
            if ('Description.txt' in file
                or dataset + '.txt' in file)
        ][0]
    except IndexError:
        description_file = None

    if description_file is not None:
        try:
            with(open(new_path + description_file, encoding='utf-8')) as f:
                description = f.read()
        except UnicodeDecodeError:
            with(open(new_path + description_file,
                      encoding='ISO-8859-1')) as f:
                description = f.read()
    else:
        description = None

    data_train = loadarff(new_path + dataset + '_TRAIN.arff')
    X_train, y_train = _parse_relational_arff(data_train)

    data_test = loadarff(new_path + dataset + '_TEST.arff')
    X_test, y_test = _parse_relational_arff(data_test)

    bunch = Bunch(
        data_train=X_train, target_train=y_train,
        data_test=X_test, target_test=y_test,
        DESCR=description,
        url=("http://www.timeseriesclassification.com/"
             "description.php?Dataset={}".format(dataset))
    )

    data_train = bunch.data_train
    data_test = bunch.data_test
    target_train = bunch.target_train
    target_test = bunch.target_test


    target_train, target_test = str2value(target_train, target_test)

    return (data_train, data_test,
            target_train, target_test)



def load_ts_uea(files_name, root, train):
    if train:
        path = os.path.join(root,files_name,files_name+'_TRAIN.ts')
    else:
        path = os.path.join(root,files_name,files_name+'_TEST.ts')

    df = pd.read_csv(path, sep = '\t')
    num_row = df.shape[0]

    data_label = []
    for line in range(num_row):
        line_data = df.loc[line].values[0]
        data_label.append(line_data)
        if line_data == '@data':
            data_label = []
    data = []
    label = []

    for data_i in range(len(data_label)):
        split_data = data_label[data_i].split(':')
        num_sensors = len(split_data)-1
        sensors_signal = []
        for sensor_i in range(num_sensors):
            data_signal_sensor_i = []
            for i in split_data[sensor_i].split(','):
                data_signal_sensor_i.append(float(i))
            data_signal_sensor_i = np.stack(data_signal_sensor_i)
            sensors_signal.append(data_signal_sensor_i)
        label_signal = float(split_data[-1])
        sensors_signal = np.stack(sensors_signal, 0)
        data.append(sensors_signal)
        label.append(label_signal)
    data = np.stack(data, 0)
    label = np.stack(label, 0)
    # print(data.shape)

    data = data[:,:,1::8]
    # print(data_t.shape)


    return data, label


def load_ts_uea_sp(files_name, root, train):
    if train:
        path = os.path.join(root, files_name, files_name + 'Eq_TRAIN.ts')
    else:
        path = os.path.join(root, files_name, files_name + 'Eq_TEST.ts')

    df = pd.read_csv(path, sep='\t')
    num_row = df.shape[0]

    data_label = []
    for line in range(num_row):
        line_data = df.loc[line].values[0]
        data_label.append(line_data)
        if line_data == '@data':
            data_label = []
    data = []
    label = []

    for data_i in range(len(data_label)):
        split_data = data_label[data_i].split(':')
        num_sensors = len(split_data) - 1
        sensors_signal = []
        for sensor_i in range(num_sensors):
            data_signal_sensor_i = []
            for i in split_data[sensor_i].split(','):
                data_signal_sensor_i.append(float(i))
            data_signal_sensor_i = np.stack(data_signal_sensor_i)
            sensors_signal.append(data_signal_sensor_i)
        label_signal = float(split_data[-1])
        sensors_signal = np.stack(sensors_signal, 0)
        data.append(sensors_signal)
        label.append(label_signal)
    data = np.stack(data, 0)
    label = np.stack(label, 0)
    return data, label

def data_normalization(data):
    num_sensors = data.shape[1]
    data_nor = []
    for sensor_i in range(num_sensors):
        data_i = data[:, sensor_i, :]
        sensor_mean = np.mean(data_i)
        sensor_std = np.std(data_i)
        if sensor_std != 0:
            data_nor.append((data_i - sensor_mean) / (sensor_std))

    data_nor = np.stack(data_nor, 1)
    return data_nor



def data_loader(files_name, root):
    if files_name in arff_read_UEA:
        data_train, data_test, label_train, label_test = _load_uea_dataset(files_name, root)
        # print(label_train)
    else:
        data_train, label_train = load_ts_uea(files_name, root, True)
        data_test, label_test = load_ts_uea(files_name, root, False)
        label_train, label_test = str2value(label_train, label_test)


    data_train = data_normalization(data_train)
    data_test = data_normalization(data_test)

    root_saved_path = '../data'
    if not os.path.exists(os.path.join(root_saved_path,files_name)):
        os.mkdir(os.path.join(root_saved_path,files_name))

    data_train = torch.from_numpy(data_train)
    label_train = torch.from_numpy(label_train).long()
    data_test = torch.from_numpy(data_test)
    label_test = torch.from_numpy(label_test).long()


    if data_train.size(-1)%2 != 0:
        data_train = data_train[:,:,:-1]
        data_test = data_test[:,:,:-1]

    print(torch.min(label_train))


    torch.save({'samples':data_train, 'labels':label_train}, os.path.join(root_saved_path,files_name, 'train.pt'))
    torch.save({'samples':data_test, 'labels':label_test}, os.path.join(root_saved_path,files_name, 'test.pt'))


arff_read_UEA = ['ArticularyWordRecognition','FingerMovements']
ts_read_UEA = ['SpokenArabicDigitsEq']
