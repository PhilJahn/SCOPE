import sys
from datahandler import load_data
import matplotlib.pyplot as plt
import river.cluster as rcluster
import numpy as np

from dp_clustream import DensityPeak_CluStream


def main():
    X, y = load_data("diamond9", "artificial")
    c = 1000
    X_test = X[c:]
    y_test = y[c:]
    X = X[:c]
    y = y[:c]
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    plt.show()

    method = DensityPeak_CluStream(n_macro_clusters=10, max_micro_clusters=100, sigma=0.5, mu=0.5)
    for x, label in zip(X, y):
        dp = dict(enumerate(x))
        method.learn_one(dp)
    preds_train = []
    for x in X:
        dp = dict(enumerate(x))
        pred = method.predict_one(dp)
        preds_train.append(pred)
    preds_test = []
    for x in X_test:
        dp = dict(enumerate(x))
        pred = method.predict_one(dp)
        preds_test.append(pred)
    centers = []
    for key in method.centers.keys():
        center = method.centers[key]
        centers.append([center[0], center[1]])
    centers = np.array(centers)
    mcs = []
    for mc_key in method.micro_clusters.keys():
        mc = method.micro_clusters[mc_key]
        mccenter = mc.center
        mcs.append([mccenter[0], mccenter[1], mc.radius(1), mc.weight])
    mcs = np.array(mcs)
    print(len(mcs))
    print(preds_train)
    plt.figure(figsize=(10,10))
    plt.scatter(X[:, 0], X[:, 1], c=preds_train)
    plt.scatter(centers[:, 0], centers[:, 1], c="black")
    plt.scatter(mcs[:, 0], mcs[:, 1], c="grey", s=mcs[:, 3])
    plt.ylim(-0.1, 1.1)
    plt.xlim(-0.1, 1.1)
    plt.show()
    plt.figure(figsize=(10,10))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=preds_test)
    plt.scatter(centers[:, 0], centers[:, 1], c="black")
    plt.scatter(mcs[:, 0], mcs[:, 1], c="grey", s=mcs[:, 3])
    plt.ylim(-0.1, 1.1)
    plt.xlim(-0.1, 1.1)
    plt.show()

if __name__ == '__main__':
    main()
