from __future__ import annotations

import math

from clustpy.partition import XMeans
from numpy.random import PCG64
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, OPTICS, MeanShift, HDBSCAN, AgglomerativeClustering

from competitors.clustream import CluStream
import river.cluster as rcluster
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc

from dbhd_clustering.DBHDALGO import DBHD
from method.CircSCOPE import CircSCOPE
from method.SCOPE import SCOPE
from utils import dps_to_np, dict_to_np


# obtain n uniformly sampled points within a d-sphere with a fixed radius around a given point. Assigns all points to
# cluster code partially based on code provided here:
# http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
def random_ball_num(center, radius, d, n, clunum, generator):
    # print("uniform")
    d = int(d)
    n = int(n)
    dps = []
    for _ in range(n):
        u = generator.normal(0, 1, d + 2)  # an array of (d+2) normally distributed random variables
        norm = np.sum(u ** 2) ** (0.5)
        u = u / norm
        x = u[0:d]  # take the first d coordinates
        x_dict = {}
        i = 0
        for attr in center.keys():
            x_dict[attr] = center[attr] + x[i] * radius
            i += 1
        dps.append(x_dict)
    y = [clunum] * n
    return dps, y


def reconstruct_data(micro_clusters, num, radius_mult, generator):
    new_ds = []
    new_labels = []
    weight_sum = 0
    for _, mc in micro_clusters.items():
        weight_sum += mc.weight
    ratio = weight_sum / num

    for id, mc in micro_clusters.items():
        mc_num = math.ceil(mc.weight / ratio)
        # print(mc.weight, ratio, mc_num)
        new_dps, label = random_ball_num(mc.center, mc.radius(radius_mult), len(mc.center.keys()), mc_num, id, generator)
        for j in range(mc_num):
            new_ds.append(new_dps[j])
        new_labels.extend(label)
    return new_ds, new_labels


def perform_clustering(data, algorithm, args):
    if algorithm == "kMeans":
        kmeans = KMeans(n_clusters=args["n_clusters"], random_state=args["seed"])
        clustering = kmeans.fit_predict(data, None)
        return clustering, kmeans.cluster_centers_
    elif algorithm == "XMeans":
        xmeans = XMeans(random_state=args["seed"], allow_merging=True, check_global_score=False)
        clustering = xmeans.fit_predict(data, None)
        return clustering, xmeans.cluster_centers_
    elif algorithm == "DBHD":
        args = {"min_cluster_size": 10, "rho": 1.2, "beta": 0.1} | args
        print(args)
        dbhd = DBHD(min_cluster_size=args["min_cluster_size"], rho=args["rho"], beta=args["beta"])
        clustering = dbhd.fit_predict(data)
        return clustering, None
    elif algorithm == "Spectral":
        args = {"n_init": 10, "gamma": 1.0, "affinity": "rbf", "n_neighbors": 10, "eigen_tol": "auto",
                "assign_labels": "kmeans", "degree": 3, "coef0": 1} | args

        spectral = SpectralClustering(n_clusters=args["n_clusters"], random_state=args["seed"], n_init=args["n_init"],
                                      gamma=args["gamma"], affinity=args["affinity"], n_neighbors=args["n_neighbors"],
                                      eigen_tol=args["eigen_tol"], assign_labels=args["assign_labels"],
                                      degree=args["degree"], coef0=args["coef0"])
        clustering = spectral.fit_predict(data, None)
        return clustering, None
    elif algorithm == "DBSCAN":
        args = {"eps": 0.5, "min_samples": 5, "algorithm": 'auto', "leaf_size": 30, "p": None, "metric_params": None,
                "metric": "euclidean"} | args
        dbscan = DBSCAN(eps=args["eps"], min_samples=args["min_samples"], metric=args["metric"],
                        metric_params=args["metric_params"], algorithm=args["algorithm"], leaf_size=args["leaf_size"],
                        p=args["p"])
        clustering = dbscan.fit_predict(data, None)
        return clustering, None
    elif algorithm == "HDBSCAN":
        args = {"min_cluster_size":5, "min_samples":None, "cluster_selection_epsilon":0.0, "max_cluster_size":None, "metric":'euclidean', "metric_params":None, "alpha":1.0, "algorithm":'auto', "leaf_size":40, "cluster_selection_method":'eom', "allow_single_cluster":False} | args
        hdbscan = HDBSCAN(min_cluster_size=args["min_cluster_size"], min_samples=args["min_samples"], cluster_selection_epsilon=args["cluster_selection_epsilon"], max_cluster_size=args["max_cluster_size"], metric=args["metric"], metric_params=args["metric_params"], alpha=args["alpha"], algorithm=args["algorithm"], leaf_size=args["leaf_size"], cluster_selection_method=args["cluster_selection_method"], allow_single_cluster=args["allow_single_cluster"], store_centers='centroid')
        clustering = hdbscan.fit_predict(data, None)
        return clustering, hdbscan.centroids_
    elif algorithm == "OPTICS":
        args = {"min_samples": 5, "max_eps": np.inf, "metric": "minkowski", "p": 2, "metric_params": None,
                "cluster_method": "xi", "eps": None, "xi": 0.05, "predecessor_correction": True, "min_cluster_size": None,
                "algorithm": "auto", "leaf_size": 3} | args
        optics = OPTICS(min_samples=args["min_samples"], max_eps=args["max_eps"], metric=args["metric"], p=args["p"],
                        metric_params=args["metric_params"], cluster_method=args["cluster_method"], eps=args["eps"],
                        xi=args["xi"], predecessor_correction=args["predecessor_correction"],
                        min_cluster_size=args["min_cluster_size"], algorithm=args["algorithm"], leaf_size=args["leaf_size"])
        clustering = optics.fit_predict(data, None)
        return clustering, None
    elif algorithm == "MeanShift":
        args = {"bandwidth": None, "seeds": None, "bin_seeding": False, "min_bin_freq": 1, "cluster_all": True,
                "n_jobs": None, "max_iter": 300} | args

        meanshift = MeanShift(bandwidth=args["bandwidth"], seeds=args["seeds"], bin_seeding=args["bin_seeding"],
                              min_bin_freq=args["min_bin_freq"], cluster_all=args["cluster_all"], n_jobs=args["n_jobs"],
                              max_iter=args["max_iter"])
        clustering = meanshift.fit_predict(data, None)
        return clustering, None
    elif algorithm == "Agglomerative":
        args = {"memory": None, "connectivity": None, "compute_full_tree": 'auto', "linkage": 'ward', "distance_threshold": None, "compute_distances": False } | args

        agglomerative = AgglomerativeClustering(n_clusters=args["n_clusters"], memory = args["memory"], connectivity = args["connectivity"], compute_full_tree = args["compute_full_tree"], linkage = args["linkage"], distance_threshold = args["distance_threshold"], compute_distances = args["compute_distances"])

        clustering = agglomerative.fit_predict(data, None)
        return clustering, None
    else:
        raise NotImplementedError


def set_seed(i):
    np.random.seed = i

class GenCluStream(CluStream):

    def __init__(
            self,
            n_macro_clusters: int = 5,
            max_micro_clusters: int = 100,
            micro_cluster_r_factor: int = 2,
            time_window: int = 1000,
            time_gap: int = 100,
            seed: int | None = None,
            offline_algo: str = "kMeans",
            offline_args=None,
            offline_datascale: int = 1000,
            **kwargs,
    ):
        super().__init__(n_macro_clusters, max_micro_clusters, micro_cluster_r_factor, time_window, time_gap, seed,
                         **kwargs)

        self.generator = np.random.Generator(PCG64(seed))
        self.cluster_assignments = {}
        self.offline_algo = offline_algo
        if offline_args is None:
            self.offline_args = {}
        else:
            self.offline_args = offline_args
        if self.offline_algo == "kMeans" or self.offline_algo == "Spectral" or self.offline_algo == "Agglomerative":
            self.offline_args["n_clusters"] = n_macro_clusters
        self.offline_args["seed"] = seed
        self.offline_datascale = offline_datascale

        self.offline_dataset = []
        self.offline_labels = []

    def display_store(self):
        X = dps_to_np(self.datastore)

        mc_assigns = []
        for dp in self.datastore:
            closest_mc_id, _ = self._get_closest_mc(dp)
            mc_assigns.append(closest_mc_id)

        plt.figure(figsize=(10, 10))
        plt.scatter(X[:, 0], X[:, 1], c=mc_assigns)
        for id, mc in self.micro_clusters.items():
            mccenter = dict_to_np(mc.center)
            mc_patch = ptc.Circle((float(mccenter[0]), float(mccenter[1])), mc.radius(self.micro_cluster_r_factor),
                                  alpha=0.2, color="lightgrey")
            plt.scatter(float(mccenter[0]), float(mccenter[1]), c="black", alpha=0.5)
            plt.gca().add_patch(mc_patch)
        plt.ylim(-0.1, 1.1)
        plt.xlim(-0.1, 1.1)
        plt.show()

    def offline_processing(self):
        gen_data, gen_labels = reconstruct_data(self.micro_clusters, self.offline_datascale,
                                                self.micro_cluster_r_factor, self.generator)
        gen_X = dps_to_np(gen_data)
        plt.figure(figsize=(10, 10))
        plt.scatter(gen_X[:, 0], gen_X[:, 1], c=gen_labels)
        for id, mc in self.micro_clusters.items():
            mccenter = dict_to_np(mc.center)
            mc_patch = ptc.Circle((float(mccenter[0]), float(mccenter[1])), mc.radius(self.micro_cluster_r_factor),
                                  alpha=0.2, color="lightgrey")
            plt.scatter(float(mccenter[0]), float(mccenter[1]), c="black", alpha=0.5)
            plt.gca().add_patch(mc_patch)
        plt.ylim(-0.1, 1.1)
        plt.xlim(-0.1, 1.1)
        plt.show()

        clustering, self.centers = perform_clustering(gen_X, self.offline_algo, self.offline_args)
        num_clu = len(np.unique(clustering))
        plt.figure(figsize=(10, 10))
        plt.scatter(gen_X[:, 0], gen_X[:, 1], c=clustering)
        for id, mc in self.micro_clusters.items():
            mccenter = dict_to_np(mc.center)
            mc_patch = ptc.Circle((float(mccenter[0]), float(mccenter[1])), mc.radius(self.micro_cluster_r_factor),
                                  alpha=0.2, color="lightgrey")
            plt.scatter(float(mccenter[0]), float(mccenter[1]), c="black", alpha=0.5)
            plt.gca().add_patch(mc_patch)
        plt.ylim(-0.1, 1.1)
        plt.xlim(-0.1, 1.1)
        plt.show()

        for id, mc in self.micro_clusters.items():
            is_mc = [i for i, x in enumerate(gen_labels) if x == id]
            labels_mc = [0] * num_clu
            for i_mc in is_mc:
                labels_mc[clustering[i_mc]] += 1
            self.cluster_assignments[id] = labels_mc.index(max(labels_mc))

        cluster_labels_gen = []
        for l in gen_labels:
            cluster_labels_gen.append(self.cluster_assignments[l])
        plt.figure(figsize=(10, 10))
        plt.scatter(gen_X[:, 0], gen_X[:, 1], c=cluster_labels_gen)
        for id, mc in self.micro_clusters.items():
            mccenter = dict_to_np(mc.center)
            mc_patch = ptc.Circle((float(mccenter[0]), float(mccenter[1])), mc.radius(self.micro_cluster_r_factor),
                                  alpha=0.2, color="lightgrey")
            plt.scatter(float(mccenter[0]), float(mccenter[1]), c="black", alpha=0.5)
            plt.gca().add_patch(mc_patch)
        plt.ylim(-0.1, 1.1)
        plt.xlim(-0.1, 1.1)
        plt.show()

        self._offline_timestamp = self._timestamp

        self.offline_dataset = gen_data
        self.offline_labels = gen_labels

    def predict_one(self, x, recluster=False, sklearn=None):
        if self._offline_timestamp != self._timestamp:
            self.offline_processing()
        index, _ = self._get_closest_mc(x)

        return self.cluster_assignments[index]

class CircSCOPEOffline(CircSCOPE):

    def __init__(
            self,
            n_macro_clusters: int = 5,
            max_micro_clusters: int = 100,
            singleton_micro_clusters: int = 50,
            micro_cluster_r_factor: int = 2,
            time_window: int = 1000,
            time_gap: int = 100,
            seed: int | None = None,
            offline_algo: str = "kMeans",
            offline_args=None,
            offline_datascale: int = 1000,
            **kwargs,
    ):
        super().__init__(n_macro_clusters=n_macro_clusters, max_micro_clusters=max_micro_clusters, singleton_micro_clusters=singleton_micro_clusters, micro_cluster_r_factor=micro_cluster_r_factor, time_window=time_window, time_gap=time_gap, seed=seed,
                         **kwargs)

        self.generator = np.random.Generator(PCG64(seed))
        self.cluster_assignments = {}
        self.offline_algo = offline_algo
        if offline_args is None:
            self.offline_args = {}
        else:
            self.offline_args = offline_args
        if self.offline_algo == "kMeans" or self.offline_algo == "Spectral" or self.offline_algo == "Agglomerative":
            self.offline_args["n_clusters"] = n_macro_clusters
        self.offline_args["seed"] = seed
        self.offline_datascale = offline_datascale

        self.offline_dataset = []
        self.offline_labels = []

    def display_store(self):
        X = dps_to_np(self.datastore)

        #print(X)

        mc_assigns = []
        i = 0
        for dp in self.datastore:
            closest_mc_id, dist = self._get_best_mc(dp)
            mc_assigns.append(closest_mc_id)
            print(i, dp, dist, closest_mc_id, self.micro_clusters[closest_mc_id])
            i +=1

        plt.figure(figsize=(10, 10))
        plt.scatter(X[:, 0], X[:, 1], c=mc_assigns)
        for id, mc in self.micro_clusters.items():
            mccenter = dict_to_np(mc.center)
            mc_patch = ptc.Circle((float(mccenter[0]), float(mccenter[1])), mc.radius(),
                                  alpha=0.2, color="lightgrey")
            plt.scatter(float(mccenter[0]), float(mccenter[1]), c="black", alpha=0.5)
            plt.gca().add_patch(mc_patch)
        plt.ylim(-0.1, 1.1)
        plt.xlim(-0.1, 1.1)
        plt.show()

        plt.figure(figsize=(30, 30))
        plt.scatter(X[:, 0], X[:, 1], c=mc_assigns)
        for i in range(len(X)):
            plt.text(X[i, 0], X[i, 1], f"{i}: {mc_assigns[i]}")
        for id, mc in self.micro_clusters.items():
            mccenter = dict_to_np(mc.center)
            mc_patch = ptc.Circle((float(mccenter[0]), float(mccenter[1])), mc.radius(),
                                  alpha=0.2, color="lightgrey")
            plt.gca().add_patch(mc_patch)
        for id, mc in self.micro_clusters.items():
            mccenter = dict_to_np(mc.center)
            plt.scatter(float(mccenter[0]), float(mccenter[1]), c="black", alpha=1)
            plt.text(float(mccenter[0]), float(mccenter[1]), str(id), c="pink")
        plt.ylim(-0.1, 1.1)
        plt.xlim(-0.1, 1.1)
        plt.show()

    def offline_processing(self):
        np.random.seed= self.seed
        gen_data, gen_labels = reconstruct_data(self.micro_clusters, self.offline_datascale,
                                                1, self.generator)
        gen_X = dps_to_np(gen_data)
        plt.figure(figsize=(10, 10))
        plt.scatter(gen_X[:, 0], gen_X[:, 1], c=gen_labels)
        for id, mc in self.micro_clusters.items():
            mccenter = dict_to_np(mc.center)
            mc_patch = ptc.Circle((float(mccenter[0]), float(mccenter[1])), mc.radius(),
                                  alpha=0.2, color="lightgrey")
            plt.scatter(float(mccenter[0]), float(mccenter[1]), c="black", alpha=0.5)
            plt.gca().add_patch(mc_patch)
        plt.ylim(-0.1, 1.1)
        plt.xlim(-0.1, 1.1)
        plt.show()

        clustering, self.centers = perform_clustering(gen_X, self.offline_algo, self.offline_args)
        num_clu = len(np.unique(clustering))
        plt.figure(figsize=(10, 10))
        plt.scatter(gen_X[:, 0], gen_X[:, 1], c=clustering)
        for id, mc in self.micro_clusters.items():
            mccenter = dict_to_np(mc.center)
            mc_patch = ptc.Circle((float(mccenter[0]), float(mccenter[1])), mc.radius(),
                                  alpha=0.2, color="lightgrey")
            plt.scatter(float(mccenter[0]), float(mccenter[1]), c="black", alpha=0.5)
            plt.gca().add_patch(mc_patch)
        plt.ylim(-0.1, 1.1)
        plt.xlim(-0.1, 1.1)
        plt.show()

        for id, mc in self.micro_clusters.items():
            is_mc = [i for i, x in enumerate(gen_labels) if x == id]
            labels_mc = [0] * num_clu
            for i_mc in is_mc:
                labels_mc[clustering[i_mc]] += 1
            self.cluster_assignments[id] = labels_mc.index(max(labels_mc))

        cluster_labels_gen = []
        for l in gen_labels:
            cluster_labels_gen.append(self.cluster_assignments[l])
        plt.figure(figsize=(10, 10))
        plt.scatter(gen_X[:, 0], gen_X[:, 1], c=cluster_labels_gen)
        for id, mc in self.micro_clusters.items():
            mccenter = dict_to_np(mc.center)
            mc_patch = ptc.Circle((float(mccenter[0]), float(mccenter[1])), mc.radius(),
                                  alpha=0.2, color="lightgrey")
            plt.scatter(float(mccenter[0]), float(mccenter[1]), c="black", alpha=0.5)
            plt.gca().add_patch(mc_patch)
        plt.ylim(-0.1, 1.1)
        plt.xlim(-0.1, 1.1)
        plt.show()

        self._offline_timestamp = self._timestamp

        self.offline_dataset = gen_data
        self.offline_labels = gen_labels

    def predict_one(self, x, recluster=False, sklearn=None):
        if self._offline_timestamp != self._timestamp:
            self.offline_processing()
        index, _ = self._get_best_mc(x)

        return self.cluster_assignments[index]


class SCOPEOffline(SCOPE):

    def __init__(
            self,
            n_macro_clusters: int = 5,
            max_micro_clusters: int = 100,
            singleton_micro_clusters: int = 50,
            micro_cluster_r_factor: int = 2,
            time_window: int = 1000,
            time_gap: int = 100,
            seed: int | None = None,
            offline_algo: str = "kMeans",
            offline_args=None,
            offline_datascale: int = 1000,
            **kwargs,
    ):
        super().__init__(n_macro_clusters=n_macro_clusters, max_micro_clusters=max_micro_clusters, singleton_micro_clusters=singleton_micro_clusters, micro_cluster_r_factor=micro_cluster_r_factor, time_window=time_window, time_gap=time_gap, seed=seed,
                         **kwargs)

        self.generator = np.random.Generator(PCG64(seed))
        self.cluster_assignments = {}
        self.offline_algo = offline_algo
        if offline_args is None:
            self.offline_args = {}
        else:
            self.offline_args = offline_args
        if self.offline_algo == "kMeans" or self.offline_algo == "Spectral" or self.offline_algo == "Agglomerative":
            self.offline_args["n_clusters"] = n_macro_clusters
        self.offline_args["seed"] = seed
        self.offline_datascale = offline_datascale

        self.offline_dataset = []
        self.offline_labels = []

    def display_store(self):
        X = dps_to_np(self.datastore)

        #print(X)

        mc_assigns = []
        i = 0
        for dp in self.datastore:
            closest_mc_id, dist = self._get_best_mc(dp)
            mc_assigns.append(closest_mc_id)
            print(i, dp, dist, closest_mc_id, self.micro_clusters[closest_mc_id])
            i +=1

        plt.figure(figsize=(10, 10))
        plt.scatter(X[:, 0], X[:, 1], c=mc_assigns)
        for id, mc in self.micro_clusters.items():
            mccenter = dict_to_np(mc.center)
            mcrange = dict_to_np(mc.extent)
            mc_patch = ptc.Rectangle((float(mccenter[0]-mcrange[0]), float(mccenter[1]-mcrange[1])),2*mcrange[0], 2*mcrange[1] ,
                                  alpha=0.2, color="lightgrey")
            plt.scatter(float(mccenter[0]), float(mccenter[1]), c="black", alpha=0.5)
            plt.gca().add_patch(mc_patch)
        plt.ylim(-0.1, 1.1)
        plt.xlim(-0.1, 1.1)
        plt.show()

        plt.figure(figsize=(30, 30))
        plt.scatter(X[:, 0], X[:, 1], c=mc_assigns)
        for i in range(len(X)):
            plt.text(X[i, 0], X[i, 1], f"{i}: {mc_assigns[i]}")
        for id, mc in self.micro_clusters.items():
            mccenter = dict_to_np(mc.center)
            mcrange = dict_to_np(mc.extent)
            mc_patch = ptc.Rectangle((float(mccenter[0]-mcrange[0]), float(mccenter[1]-mcrange[1])),2*mcrange[0], 2*mcrange[1] ,
                                  alpha=0.2, color="lightgrey")
            plt.gca().add_patch(mc_patch)
        for id, mc in self.micro_clusters.items():
            mccenter = dict_to_np(mc.center)
            plt.scatter(float(mccenter[0]), float(mccenter[1]), c="black", alpha=1)
            plt.text(float(mccenter[0]), float(mccenter[1]), str(id), c="pink")
        plt.ylim(-0.1, 1.1)
        plt.xlim(-0.1, 1.1)
        plt.show()

    def offline_processing(self):
        np.random.seed= self.seed
        gen_data, gen_labels = self.reconstruct_data(self.micro_clusters, self.offline_datascale, self.generator)
        gen_X = dps_to_np(gen_data)
        plt.figure(figsize=(10, 10))
        plt.scatter(gen_X[:, 0], gen_X[:, 1], c=gen_labels)
        for id, mc in self.micro_clusters.items():
            mccenter = dict_to_np(mc.center)
            mcrange = dict_to_np(mc.extent)
            mc_patch = ptc.Rectangle((float(mccenter[0]-mcrange[0]), float(mccenter[1]-mcrange[1])),2*mcrange[0], 2*mcrange[1] ,
                                  alpha=0.2, color="lightgrey")
            plt.scatter(float(mccenter[0]), float(mccenter[1]), c="black", alpha=0.5)
            plt.gca().add_patch(mc_patch)
        plt.ylim(-0.1, 1.1)
        plt.xlim(-0.1, 1.1)
        plt.show()

        clustering, self.centers = perform_clustering(gen_X, self.offline_algo, self.offline_args)
        num_clu = len(np.unique(clustering))
        plt.figure(figsize=(10, 10))
        plt.scatter(gen_X[:, 0], gen_X[:, 1], c=clustering)
        for id, mc in self.micro_clusters.items():
            mccenter = dict_to_np(mc.center)
            mcrange = dict_to_np(mc.extent)
            mc_patch = ptc.Rectangle((float(mccenter[0]-mcrange[0]), float(mccenter[1]-mcrange[1])),2*mcrange[0], 2*mcrange[1] ,
                                  alpha=0.2, color="lightgrey")
            plt.scatter(float(mccenter[0]), float(mccenter[1]), c="black", alpha=0.5)
            plt.gca().add_patch(mc_patch)
        plt.ylim(-0.1, 1.1)
        plt.xlim(-0.1, 1.1)
        plt.show()

        for id, mc in self.micro_clusters.items():
            is_mc = [i for i, x in enumerate(gen_labels) if x == id]
            labels_mc = [0] * num_clu
            for i_mc in is_mc:
                labels_mc[clustering[i_mc]] += 1
            self.cluster_assignments[id] = labels_mc.index(max(labels_mc))

        cluster_labels_gen = []
        for l in gen_labels:
            cluster_labels_gen.append(self.cluster_assignments[l])
        plt.figure(figsize=(10, 10))
        plt.scatter(gen_X[:, 0], gen_X[:, 1], c=cluster_labels_gen)
        for id, mc in self.micro_clusters.items():
            mccenter = dict_to_np(mc.center)
            mcrange = dict_to_np(mc.extent)
            mc_patch = ptc.Rectangle((float(mccenter[0]-mcrange[0]), float(mccenter[1]-mcrange[1])),2*mcrange[0], 2*mcrange[1] ,
                                  alpha=0.2, color="lightgrey")
            plt.scatter(float(mccenter[0]), float(mccenter[1]), c="black", alpha=0.5)
            plt.gca().add_patch(mc_patch)
        plt.ylim(-0.1, 1.1)
        plt.xlim(-0.1, 1.1)
        plt.show()

        self._offline_timestamp = self._timestamp

        self.offline_dataset = gen_data
        self.offline_labels = gen_labels

    def predict_one(self, x, recluster=False, sklearn=None):
        if self._offline_timestamp != self._timestamp:
            self.offline_processing()
        index, _ = self._get_best_mc(x)

        return self.cluster_assignments[index]

    def generate_in_box(self, center, ranges, num, clunum):
        uniform_data = self.generator.uniform(low=-1, high=1, size=(num, len(center.keys())))
        dps = []
        for j in range(num):
            x_dict = {}
            i = 0
            for attr in center.keys():
                x_dict[attr] = uniform_data[j,i] * ranges[attr] + center[attr]
                i += 1
            dps.append(x_dict)

        y = [clunum] * num
        return dps, y

    def reconstruct_data(self, micro_clusters, num, radius_mult):
        new_ds = []
        new_labels = []
        weight_sum = 0
        for _, mc in micro_clusters.items():
            weight_sum += mc.weight
        ratio = weight_sum / num

        for id, mc in micro_clusters.items():
            mc_num = math.ceil(mc.weight / ratio)
            # print(mc.weight, ratio, mc_num)


            new_dps, label = self.generate_in_box(mc.center, mc.extent, mc_num, id)
            for j in range(mc_num):
                new_ds.append(new_dps[j])
            new_labels.extend(label)
        return new_ds, new_labels
