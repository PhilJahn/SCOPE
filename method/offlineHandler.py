from __future__ import annotations

import math

from clustpy.partition import XMeans
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN

from competitors.clustream import CluStream
import river.cluster as rcluster
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc

from dbhd_clustering.DBHDALGO import DBHD
from utils import dps_to_np, dict_to_np


# obtain n uniformly sampled points within a d-sphere with a fixed radius around a given point. Assigns all points to
# cluster code partially based on code provided here:
# http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
def random_ball_num(center, radius, d, n, clunum):
	# print("uniform")
	d = int(d)
	n = int(n)
	dps = []
	for _ in range(n):
		u = np.random.normal(0, 1, d + 2)  # an array of (d+2) normally distributed random variables
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


def reconstruct_data(micro_clusters, num, radius_mult):
	new_ds = []
	new_labels = []
	weight_sum = 0
	for _, mc in micro_clusters.items():
		weight_sum += mc.weight
	ratio = weight_sum / num

	for id, mc in micro_clusters.items():
		mc_num = math.ceil(mc.weight / ratio)
		#print(mc.weight, ratio, mc_num)
		new_dps, label = random_ball_num(mc.center, mc.radius(radius_mult), len(mc.center.keys()), mc_num, id)
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
		args = {"min_cluster_size":10, "rho":1.2, "beta": 0.1} | args
		print(args)
		dbhd = DBHD(min_cluster_size=args["min_cluster_size"], rho = args["rho"], beta = args["beta"])
		clustering = dbhd.fit_predict(data)
		return clustering, None
	elif algorithm == "Spectral":
		args = {"n_init": 10, "gamma": 1.0, "affinity": "rbf", "n_neighbors": 10, "eigen_tol": "auto", "assign_labels": "kmeans", "degree": 3, "coef0": 1} | args

		spectral = SpectralClustering(n_clusters=args["n_clusters"], random_state = args["seed"], n_init = args["n_init"],
		                              gamma = args["gamma"], affinity = args["affinity"], n_neighbors = args["n_neighbors"],
									  eigen_tol = args["eigen_tol"], assign_labels = args["assign_labels"],
									  degree= args["degree"], coef0 = args["coef0"])
		clustering = spectral.fit_predict(data, None)
		return clustering, None
	elif algorithm == "DBSCAN":
		args = {"eps": 0.5, "min_samples": 5, "algorithm": 'auto', "leaf_size": 30, "p": None} | args
		dbscan = DBSCAN(eps=args["eps"], min_samples=args["min_samples"],  algorithm=args["algorithm"], leaf_size=args["leaf_size"], p=args["p"])
		clustering = dbscan.fit_predict(data, None)
		return clustering, None
	else:
		raise NotImplementedError


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
		self.cluster_assignments = {}
		self.offline_algo = offline_algo
		if offline_args is None:
			self.offline_args = {}
		else:
			self.offline_args = offline_args
		if self.offline_algo == "kMeans" or self.offline_algo == "Spectral":
			self.offline_args["n_clusters"] = n_macro_clusters
		self.offline_args["seed"] = seed
		self.offline_datascale = offline_datascale

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
			mc_patch = ptc.Circle((float(mccenter[0]), float(mccenter[1])), mc.radius(self.micro_cluster_r_factor), alpha=0.4, color="lightgrey")
			plt.scatter(float(mccenter[0]), float(mccenter[1]), c="black")
			plt.gca().add_patch(mc_patch)
		plt.ylim(-0.1, 1.1)
		plt.xlim(-0.1, 1.1)
		plt.show()

	def offline_processing(self):

		gen_data, gen_labels = reconstruct_data(self.micro_clusters, self.offline_datascale, self.micro_cluster_r_factor)
		gen_X = dps_to_np(gen_data)
		plt.figure(figsize=(10, 10))
		plt.scatter(gen_X[:, 0], gen_X[:, 1], c=gen_labels)
		for id, mc in self.micro_clusters.items():
			mccenter = dict_to_np(mc.center)
			mc_patch = ptc.Circle((float(mccenter[0]), float(mccenter[1])), mc.radius(self.micro_cluster_r_factor), alpha=0.4, color="lightgrey")
			plt.scatter(float(mccenter[0]), float(mccenter[1]), c="black")
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
			mc_patch = ptc.Circle((float(mccenter[0]), float(mccenter[1])), mc.radius(self.micro_cluster_r_factor), alpha=0.4, color="lightgrey")
			plt.scatter(float(mccenter[0]), float(mccenter[1]), c="black")
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
			mc_patch = ptc.Circle((float(mccenter[0]), float(mccenter[1])), mc.radius(self.micro_cluster_r_factor), alpha=0.4, color="lightgrey")
			plt.scatter(float(mccenter[0]), float(mccenter[1]), c="black")
			plt.gca().add_patch(mc_patch)
		plt.ylim(-0.1, 1.1)
		plt.xlim(-0.1, 1.1)
		plt.show()

		self._offline_timestamp = self._timestamp


	def predict_one(self, x, recluster=False, sklearn=None):
		if self._offline_timestamp != self._timestamp:
			self.offline_processing()
		index, _ = self._get_closest_mc(x)

		return self.cluster_assignments[index]