import sys
import time

from datahandler import load_data
import matplotlib.pyplot as plt
from competitors import clustream
import numpy as np

from dp_clustream import DensityPeak_CluStream


def main():
    X, y = load_data("complex9", "artificial", seed=10)
    c = 1000
    X_test = X[c:]
    y_test = y[c:]
    X = X[:c]
    y = y[:c]
    # plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()
    # plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    # plt.show()

    method = clustream.CluStream(n_macro_clusters=9, max_micro_clusters=100, sigma=0.25, mu=0.5, seed=32, time_gap=133)
    preds_train = []
    i = 0
    x_train = []
    for x, label in zip(X, y):
        dp = dict(enumerate(x))

        method.learn_one(dp)
        pred = method.predict_one(dp, True)
        preds_train.append(pred)
        x_train.append(x)
        if i > 0:
            if i% 100 == 99:
                centers = []
                for key in method.centers.keys():
                    center = method.centers[key]
                    centers.append([center[0], center[1], key])
                centers = np.array(centers)
                mcs = []
                for mc_key in method.micro_clusters.keys():
                    mc = method.micro_clusters[mc_key]
                    mccenter = mc.center
                    mcs.append([mccenter[0], mccenter[1], mc.radius(1), mc.weight])
                mcs = np.array(mcs)
                x_train = np.array(x_train)
                plt.figure(figsize=(5, 5))
                plt.scatter(x_train[:, 0], x_train[:, 1], c=preds_train)
                plt.scatter(centers[:, 0], centers[:, 1], c=centers[:, 2], s=200, edgecolors='red')
                plt.scatter(mcs[:, 0], mcs[:, 1], c="grey", s=30, edgecolors='orange')
                plt.ylim(-0.1, 1.1)
                plt.xlim(-0.1, 1.1)
                plt.show()
                preds_train = []
                x_train = []
                time.sleep(2)
        i += 1
    preds_test = []
    for x in X_test:
        dp = dict(enumerate(x))
        pred = method.predict_one(dp)
        preds_test.append(pred)

    print(len(mcs))
    print(preds_test)
    # plt.figure(figsize=(10, 10))
    # plt.scatter(X_test[:, 0], X_test[:, 1], c=preds_test)
    # plt.scatter(centers[:, 0], centers[:, 1], c=centers[:, 2], s=200, edgecolors='red')
    # plt.scatter(mcs[:, 0], mcs[:, 1], c="grey", s=mcs[:, 3] * 10)
    # plt.ylim(-0.1, 1.1)
    # plt.xlim(-0.1, 1.1)
    # plt.show()

if __name__ == '__main__':
    main()
