import os

import torchvision
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import MinMaxScaler, minmax_scale
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from datagen import densityDataGen
from data.readFile import read_file


def getData(dataset, dim=2, num=5000):
    dataset = dataset.lower()
    if dataset == "own":
        return NotImplementedError
        # return getOwn(dim, num)
    elif dataset == "ownhigh":
        return NotImplementedError
        # return getOwnHigh(dim, num)
    elif dataset == "blobs":
        return getBlobs(dim, num)
    elif dataset == "moons":
        return getMoons(num)
    elif dataset == "mnist" and dim == 2:
        return getMNIST_2D()
    elif dataset == "mnist":
        raise getMNIST()
    elif dataset == "kdd99" and dim == 2:
        return getKDD99_2D()
    elif dataset == "kdd99" and dim == 3:
        return getKDD99_3D()
    elif dataset == "kdd99":
        return getKDD99()
    elif dataset == "emnist_d" or dataset == "emnist_digits" or dataset == "digits":
        return getEMNIST_Digits()
    elif dataset == "emnist_l" or dataset == "emnist_letters" or dataset == "letters":
        return getEMNIST_Letters()
    elif dataset == "covertype":
        return getCovertype()
    elif dataset == "kmnist" and dim == 128:
        return getKMNIST_128D()
    elif dataset == "kmnist":
        return getKMNIST()
    elif dataset == "rbf" or dataset == "mrbf":
        return getRBF()
    elif dataset == "electricity" or dataset == "elec2":
        return getElectricity()
    elif dataset == "rh" or dataset == "rotatinghyperplane" or dataset == "hyperplane":
        return getRotatingHyperplane()
    elif dataset == "isolet":
        return getIsolet()
    else:
        return getStored(dataset)


# def getOwn(dim, num):
# 	clunum = 10
# 	datagen = densityDataGen(dim=dim, ratio_noise=0.1, max_retry=5,
# 	                         dens_factors=[1, 1, 0.5, 0.3, 2, 1.2, 0.9, 0.6, 1.4, 1.1], square=True,
# 	                         clunum=clunum, seed=1, core_num=200,
# 	                         momentum=[0.5, 0.75, 0.8, 0.3, 0.5, 0.4, 0.2, 0.6, 0.45, 0.7],
# 	                         branch=[0, 0.05, 0.1, 0, 0, 0.1, 0.02, 0, 0, 0.25],
# 	                         con_min_dist=0.8, verbose=False, safety=True, domain_size=20, random_start=False)
# 	maxall = 0
# 	minall = datagen.domain_size
#
# 	for d in range(dim):
# 		if maxall < datagen.maxs[d]:
# 			maxall = datagen.maxs[d]
# 		if minall > datagen.mins[d]:
# 			minall = datagen.mins[d]
# 	dspan = maxall - minall
#
# 	truemin = minall - 0.1 * dspan
# 	truedspan = 1.2 * dspan
# 	datagen.update_scaler(min_val=truemin, dspan=truedspan)
# 	cmd_string = "(0#1#2#3#4|0#2#4#6#8|5#6#7#8#9|1#3#5#7#9)"
# 	datagen.init_stream(command=cmd_string, default_duration=num)
# 	return datagen, dim, num, clunum
#
# def getOwnHigh(dim, num):
# 	clunum = 10
# 	datagen = densityDataGen(dim=dim, ratio_noise=0.1, max_retry=5,
# 	                           dens_factors=[1, 1, 0.5, 0.3, 2, 1.2, 0.9, 0.6, 1.4, 1.1], square=True,
# 	                           clunum=clunum, seed=6, core_num=200, momentum=0.8, step=1.5,
# 	                           branch=0.1, star=1, verbose=False, safety=False, domain_size=20, random_start=False)
# 	maxall = 0
# 	minall = datagen.domain_size
#
# 	for d in range(dim):
# 		if maxall < datagen.maxs[d]:
# 			maxall = datagen.maxs[d]
# 		if minall > datagen.mins[d]:
# 			minall = datagen.mins[d]
# 	dspan = maxall - minall
#
# 	truemin = minall - 0.1 * dspan
# 	truedspan = 1.2 * dspan
# 	cmd_string = "(0#1#2#3#4|0#2#4#6#8|5#6#7#8#9|1#3#5#7#9)"
# 	datagen.init_stream(command=cmd_string, default_duration=num)
# 	return datagen, dim, num, clunum

def getBlobs(dim, num):
    scaler = MinMaxScaler()
    clunum = 3
    data, labels = make_blobs(n_features=dim, n_samples=num, random_state=2, centers=clunum)  # 128
    data = scaler.fit_transform(data)
    # plt.scatter(data[:, 0], data[:, 1])
    # plt.show()
    return iter(zip(data, labels)), dim, num, clunum


def getMoons(num):
    scaler = MinMaxScaler()
    data, labels = make_moons(n_samples=num, random_state=1)
    data = scaler.fit_transform(data)
    return iter(zip(data, labels)), 2, num, 2


def getMNIST_2D():
    scaler = MinMaxScaler()
    mnist_data = np.loadtxt("data/MNIST_2")
    mnist_data = scaler.fit_transform(mnist_data)
    mnist_label = np.loadtxt("data/MNIST_2_Label")

    clunum = len(np.unique(mnist_label))

    return iter(zip(mnist_data, mnist_label)), 2, len(mnist_label), clunum


def getStored(name):
    # name = 'data/artificial/' + name_data
    # print(name)
    try:
        scaler = MinMaxScaler()
        X, Y = read_file(name, 0)
        X = scaler.fit_transform(X)
        idx = np.random.permutation(len(X))
        X, Y = X[idx], Y[idx]
        clunum = len(np.unique(Y))
        return iter(zip(X, Y)), len(X[0]), len(Y), clunum
    except:

        # name = 'data/real-world/' + name_data
        # print(name)
        try:
            scaler = MinMaxScaler()
            X, Y = read_file(name, 1)
            X = scaler.fit_transform(X)
            idx = np.random.permutation(len(X))
            X, Y = X[idx], Y[idx]
            clunum = len(np.unique(Y))
            return iter(zip(X, Y)), len(X[0]), len(Y), clunum
        except:
            raise NotImplementedError


def getKMNIST_128D():
    scaler = MinMaxScaler()
    kmnist_data = np.loadtxt("data/KMNIST_128")
    kmnist_data = scaler.fit_transform(kmnist_data)
    kmnist_label = np.loadtxt("data/KMNIST_128_Label")

    clunum = len(np.unique(kmnist_label))

    return iter(zip(kmnist_data, kmnist_label)), len(kmnist_data[0]), len(kmnist_label), clunum


def getCovertype():
    # fetch dataset
    covertype = fetch_ucirepo(id=31)
    # data (as pandas dataframes)
    X = covertype.data.features.to_numpy()
    y = covertype.data.targets.to_numpy().reshape(-1)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    clunum = len(np.unique(y))

    return iter(zip(X, y)), len(X[0]), len(y), clunum

def getIsolet():
    # fetch dataset
    isolet = fetch_ucirepo(id=54)
    # data (as pandas dataframes)
    X = isolet.data.features.to_numpy()
    y = isolet.data.targets.to_numpy().reshape(-1)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    clunum = len(np.unique(y))

    return iter(zip(X, y)), len(X[0]), len(y), clunum


def getKDD99_data():
    kddcup_all_df = pd.read_csv('data/KDDCup99.csv')
    kddcup_df = kddcup_all_df[kddcup_all_df['service'] == 'http']
    kddcup_df['label'] = pd.to_numeric(kddcup_df['label'].astype('category').cat.codes, errors='coerce')
    kddcup_df['src_bytes'] = np.log2(kddcup_df['src_bytes'] + 0.1)
    kddcup_df['dst_bytes'] = np.log2(kddcup_df['dst_bytes'] + 0.1)
    kddcup_df['duration'] = np.log2(kddcup_df['duration'] + 0.1)
    kddcup_df['src_bytes'] = minmax_scale(kddcup_df['src_bytes'])
    kddcup_df['dst_bytes'] = minmax_scale(kddcup_df['dst_bytes'])
    kddcup_df['duration'] = minmax_scale(kddcup_df['duration'])
    return kddcup_df


def getKDD99():
    kddcup_df = getKDD99_data().select_dtypes(exclude=['object'])
    print(kddcup_df)
    data = kddcup_df.loc[:, kddcup_df.columns != 'label'].to_numpy()
    label = kddcup_df['label'].to_numpy()
    clunum = len(np.unique(label))
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return iter(zip(data, label)), len(data[0]), len(label), clunum


def getKDD99_3D():
    kddcup_df = getKDD99_data()[['src_bytes', 'dst_bytes', 'duration', 'label']]
    data = kddcup_df[['src_bytes', 'dst_bytes', 'duration']].to_numpy()
    label = kddcup_df['label'].to_numpy()
    clunum = len(np.unique(label))
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return iter(zip(data, label)), 3, len(label), clunum


def getKDD99_2D():
    kddcup_df = getKDD99_data()[['src_bytes', 'dst_bytes', 'label']]
    data = kddcup_df[['src_bytes', 'dst_bytes']].to_numpy()
    label = kddcup_df['label'].to_numpy()
    clunum = len(np.unique(label))
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return iter(zip(data, label)), 2, len(label), clunum


def getOptDigits_2D():
    scaler = MinMaxScaler()
    optdigits_data = np.loadtxt("data/optdigitsdata.txt")
    optdigits_data = scaler.fit_transform(optdigits_data)
    optdigits_label = np.loadtxt("data/optdigitslabel.txt")
    clunum = len(np.unique(optdigits_label))
    scaler = MinMaxScaler()
    optdigits_data = scaler.fit_transform(optdigits_data)
    return iter(zip(optdigits_data, optdigits_label)), 2, len(optdigits_label), clunum


def getEMNIST_Digits():
    train_data = torchvision.datasets.EMNIST(root="real_world", split="digits", download=True, train=True)
    X = train_data.train_data
    X = np.reshape(X,[X.shape[0], -1])
    Y = train_data.train_labels

    idx = np.random.permutation(len(X))
    X, Y = X[idx], Y[idx]

    clunum = len(np.unique(Y))
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return iter(zip(X_scaled, Y)), len(X[0]), len(Y), clunum


def getEMNIST_Letters():
    train_data = torchvision.datasets.EMNIST(root="real_world", split="letters", download=True, train=True)
    X = train_data.train_data
    X = np.reshape(X,[X.shape[0], -1])
    Y = train_data.train_labels

    idx = np.random.permutation(len(X))
    X, Y = X[idx], Y[idx]

    clunum = len(np.unique(Y))
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return iter(zip(X_scaled, Y)), len(X[0]), len(Y), clunum


def getMNIST():
    train_data = torchvision.datasets.MNIST(root="real_world", download=True, train=True)
    X = train_data.train_data
    X = np.reshape(X,[X.shape[0], -1])
    Y = train_data.train_labels

    idx = np.random.permutation(len(X))
    X, Y = X[idx], Y[idx]

    clunum = len(np.unique(Y))
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return iter(zip(X_scaled, Y)), len(X[0]), len(Y), clunum


def getKMNIST():
    train_data = torchvision.datasets.KMNIST(root="real_world", download=True, train=True)
    X = train_data.train_data
    X = np.reshape(X,[X.shape[0], -1])
    Y = train_data.train_labels

    idx = np.random.permutation(len(X))
    X, Y = X[idx], Y[idx]

    clunum = len(np.unique(Y))
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return iter(zip(X_scaled, Y)), len(X[0]), len(Y), clunum


def getISOLET():
    isolet = fetch_ucirepo(id=54)

    # data (as pandas dataframes)
    X = isolet.data.features
    y = isolet.data.targets

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    clunum = len(np.unique(y))
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return iter(zip(X_scaled, y)), len(X[0]), len(y), clunum


def getRBF():
    scaler = MinMaxScaler()
    X = np.loadtxt("data/drifting/movingRBF.data")
    y = np.loadtxt("data/drifting/movingRBF.labels")
    X = scaler.fit_transform(X)
    # idx = np.random.permutation(len(X))
    # X, y = X[idx], y[idx]
    clunum = len(np.unique(y))
    return iter(zip(X, y)), len(X[0]), len(y), clunum


def getRotatingHyperplane():
    scaler = MinMaxScaler()
    X = np.loadtxt("data/drifting/rotatingHyperplane.data")
    y = np.loadtxt("data/drifting/rotatingHyperplane.labels")
    X = scaler.fit_transform(X)
    # idx = np.random.permutation(len(X))
    # X, y = X[idx], y[idx]
    clunum = len(np.unique(y))
    return iter(zip(X, y)), len(X[0]), len(y), clunum


def getElectricity():
    scaler = MinMaxScaler()
    X = np.loadtxt("data/drifting/elec2_data.dat")
    y = np.loadtxt("data/drifting/elec2_label.dat")
    print(X.shape)
    X = X[:, [1, 2, 4, 5, 6, 7]]
    X = scaler.fit_transform(X)
    # idx = np.random.permutation(len(X))
    # X, y = X[idx], y[idx]
    clunum = len(np.unique(y))
    return iter(zip(X, y)), len(X[0]), len(y), clunum
