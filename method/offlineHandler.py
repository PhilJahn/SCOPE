from __future__ import annotations

import math

import torch
from clustpy.partition import XMeans, SubKmeans, ProjectedDipMeans
from clustpy.deep import DEC, DipEncoder
from numpy.random import PCG64
from scipy.spatial import distance
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, OPTICS, MeanShift, HDBSCAN, AgglomerativeClustering
from sklearn.neighbors import KDTree
#from offline_methods.SHADE.dcdist import DCTree_Clusterer
#from offline_methods.SHADE.shade.shade import SHADE
from offline_methods.DPC.DPC import DensityPeakCluster
from competitors.clustream import CluStream
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc

#from offline_methods.DCFcluster.src.DCFcluster import DCFcluster
from offline_methods.SNNDPC import SNNDPC
from offline_methods.dbhd_clustering.DBHDALGO import DBHD
from offline_methods.SCAR.SpectralClusteringAcceleratedRobust import SCAR
from offline_methods.mdbscan import MDBSCAN
from offline_methods.rnndbscan import RNNDBSCAN
from offline_methods.spectacl.Spectacl import Spectacl
from utils import dps_to_np, dict_to_np

k_algos = ["kmeans", "spectral", "agglomerative", "scar", "spectacl", "subkmeans"]


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

class DataReconstructor:

	def __init__(self, radii=None, num_tries=1000):
		if radii is None:
			radii = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
		self.weight_scale_factor = {0:0, 1:0}
		self.min1 = np.inf
		self.max0 = 1
		self.mins = [np.inf] * len(radii)
		self.maxs = [1] * len(radii)
		self.radii = radii
		self.num_tries = num_tries

	def calc_weight_scale(self, mc_num, mc_dim):
		if mc_num > self.min1:
			print("n", mc_num, "d", mc_dim, "->", 1, "(skipped due to lower bound)")
			return 1
		if mc_num < self.max0:
			print("n", mc_num, "d", mc_dim, "->", 0, "(skipped due to upper bound)")
			return 0
		for r in range(len(self.radii)):
			if mc_num < self.maxs[r] and mc_num > self.mins[r]:
				print("n", mc_num, "d", mc_dim, "->", self.radii[r], f"(skipped due to {self.mins[r]} and {self.maxs[r]})")
				return self.radii[r]

		min_radius = -1
		min_dist = np.inf
		min_radius_index = -1
		center = {}
		for m in range(mc_dim):
			center[m] = 0
		s0 = 0
		r = 0
		for radius in self.radii:
			radius_dists = []
			for s in range(self.num_tries):
				generator = np.random.Generator(PCG64(s+s0))
				dps_real, _ = random_ball_num(center=center, d=mc_dim, radius=1, n=1, clunum=0,
				                              generator=generator)
				dps_real = dps_to_np(dps_real)
				if radius > 0:
					dps_other, _ = random_ball_num(center=center, d=mc_dim, radius=radius, n=mc_num, clunum=0,
					                               generator=generator)
					dps_other = dps_to_np(dps_other)
					min_dist_seed = np.inf
					for i in range(mc_num):
						min_dist_seed = min(min_dist_seed, distance.euclidean(dps_real[0], dps_other[i]))
				else:
					min_dist_seed = distance.euclidean(dps_real[0], dict_to_np(center)) # no need to do generation for radius 0
				radius_dists.append(min_dist_seed)
			s0 += self.num_tries
			dist = np.mean(radius_dists)
			if dist < min_dist:
				min_dist = dist
				min_radius = radius
				min_radius_index = r
			r+= 1
		if min_radius == 1:
			if mc_num < self.min1:
				self.min1 = mc_num
		elif min_radius == 0:
			if mc_num > self.max0:
				self.max0 = mc_num
		else:
			self.mins[min_radius_index] = min(self.mins[min_radius_index], mc_num)
			self.maxs[min_radius_index] = max(self.maxs[min_radius_index], mc_num)

		print("n", mc_num, "d", mc_dim, "->", min_radius)
		return min_radius

	def reconstruct_data(self,micro_clusters, num, radius_mult, generator, use_centroid=False, mc_import = False, weight_scale = False):
		if mc_import:
			micro_clusters_dict = {}
			i = 0
			for mc in micro_clusters:
				micro_clusters_dict[i] = mc
				i+=1
		else:
			micro_clusters_dict =micro_clusters

		new_ds = []
		new_labels = []
		weight_sum = 0
		for _, mc in micro_clusters_dict.items():
			if not mc_import:
				weight_sum += mc.weight
			else:
				weight_sum += mc[4]
		ratio = weight_sum / num
		for id, mc in micro_clusters_dict.items():
			if not mc_import:
				mc_weight = mc.weight
				mc_center = mc.center
				mc_radius = mc.radius(radius_mult)
				mc_dim = len(mc_center.keys())
			else:
				mc_weight = mc[4]
				mc_center = mc[2]
				mc_radius = mc[3] * radius_mult
				mc_dim = len(mc_center)

			mc_num = math.ceil(mc_weight / ratio)
			if weight_scale:
				if not mc_num-1 in self.weight_scale_factor.keys():
					self.weight_scale_factor[mc_num-1] = self.calc_weight_scale(mc_num-1, mc_dim)
				mc_radius = mc_radius * self.weight_scale_factor[mc_num-1]

			# print(mc.weight, ratio, mc_num)
			if use_centroid:
				mc_num = mc_num - 1
			if mc_num > 0:
				new_dps, label = random_ball_num(mc_center, mc_radius, mc_dim, mc_num, id,
				                                 generator)
				for j in range(mc_num):
					new_ds.append(new_dps[j])
				new_labels.extend(label)
			if use_centroid:
				new_ds.append(mc_center)
				new_labels.append(id)
		return new_ds, new_labels



def perform_clustering(data, algorithm, args):
	if algorithm == "kmeans":
		kmeans = KMeans(n_clusters=args["n_clusters"], random_state=args["alg_seed"])
		clustering = kmeans.fit_predict(data, None)
		return clustering, kmeans.cluster_centers_
	elif algorithm == "xmeans":
		args = {"n_clusters_init": 2,
             "max_n_clusters": np.inf,
             "check_global_score": True,
             "allow_merging": False,
             "n_split_trials": 10} | args
		if args["check_global_score"] == 1:
			args["check_global_score"] = True
		elif args["check_global_score"] == 0:
			args["check_global_score"] = False
		if args["allow_merging"] == 1:
			args["allow_merging"] = True
		elif args["allow_merging"] == 0:
			args["allow_merging"] = False
		xmeans = XMeans(random_state=args["alg_seed"], n_clusters_init=args["n_clusters_init"], allow_merging=args["allow_merging"], check_global_score=args["check_global_score"], n_split_trials=args["n_split_trials"], max_n_clusters=args["max_n_clusters"])
		clustering = xmeans.fit_predict(np.array(data), None)
		return clustering, xmeans.cluster_centers_
	elif algorithm == "dbhd":
		args = {"min_cluster_size": 10, "rho": 1.2, "beta": 0.1} | args
		# print(args)
		dbhd = DBHD(min_cluster_size=args["min_cluster_size"], rho=args["rho"], beta=args["beta"])
		clustering = dbhd.fit_predict(data)
		return clustering, None
	elif algorithm == "spectral":
		args = {"n_init": 10, "gamma": 1.0, "affinity": "rbf", "n_neighbors": 10, "eigen_tol": "auto",
		        "assign_labels": "kmeans", "degree": 3, "coef0": 1} | args

		spectral = SpectralClustering(n_clusters=args["n_clusters"], random_state=args["alg_seed"],
		                              n_init=args["n_init"],
		                              gamma=args["gamma"], affinity=args["affinity"], n_neighbors=args["n_neighbors"],
		                              eigen_tol=args["eigen_tol"], assign_labels=args["assign_labels"],
		                              degree=args["degree"], coef0=args["coef0"])
		clustering = spectral.fit_predict(data, None)
		return clustering, None
	elif algorithm == "dbscan":
		args = {"eps": 0.5, "min_samples": 5, "algorithm": 'auto', "leaf_size": 30, "p": None, "metric_params": None,
		        "metric": "euclidean"} | args
		dbscan = DBSCAN(eps=args["eps"], min_samples=args["min_samples"], metric=args["metric"],
		                metric_params=args["metric_params"], algorithm=args["algorithm"], leaf_size=args["leaf_size"],
		                p=args["p"])
		clustering = dbscan.fit_predict(data, None)
		return clustering, None
	elif algorithm == "hdbscan":
		args = {"min_cluster_size": 5, "min_samples": None, "cluster_selection_epsilon": 0.0, "max_cluster_size": None,
		        "metric": 'euclidean', "metric_params": None, "alpha": 1.0, "algorithm": 'auto', "leaf_size": 40,
		        "cluster_selection_method": 'eom', "allow_single_cluster": False} | args
		if args["allow_single_cluster"] == 1:
			args["allow_single_cluster"] = True
		elif args["allow_single_cluster"] == 0:
			args["allow_single_cluster"] = False
		hdbscan = HDBSCAN(min_cluster_size=args["min_cluster_size"], min_samples=args["min_samples"],
		                  cluster_selection_epsilon=args["cluster_selection_epsilon"],
		                  max_cluster_size=args["max_cluster_size"], metric=args["metric"],
		                  metric_params=args["metric_params"], alpha=args["alpha"], algorithm=args["algorithm"],
		                  leaf_size=args["leaf_size"], cluster_selection_method=args["cluster_selection_method"],
		                  allow_single_cluster=args["allow_single_cluster"], store_centers='centroid')
		clustering = hdbscan.fit_predict(data, None)
		return clustering, hdbscan.centroids_
	elif algorithm == "optics":
		args = {"min_samples": 5, "max_eps": np.inf, "metric": "minkowski", "p": 2, "metric_params": None,
		        "cluster_method": "xi", "eps": None, "xi": 0.05, "predecessor_correction": True,
		        "min_cluster_size": None,
		        "algorithm": "auto", "leaf_size": 3} | args
		if args["predecessor_correction"] == 1:
			args["predecessor_correction"] = True
		elif args["predecessor_correction"] == 0:
			args["predecessor_correction"] = False
		optics = OPTICS(min_samples=args["min_samples"], max_eps=args["max_eps"], metric=args["metric"], p=args["p"],
		                metric_params=args["metric_params"], cluster_method=args["cluster_method"], eps=args["eps"],
		                xi=args["xi"], predecessor_correction=args["predecessor_correction"],
		                min_cluster_size=args["min_cluster_size"], algorithm=args["algorithm"],
		                leaf_size=args["leaf_size"])
		clustering = optics.fit_predict(data, None)
		return clustering, None
	elif algorithm == "meanshift":
		args = {"bandwidth": None, "seeds": None, "bin_seeding": False, "min_bin_freq": 1, "cluster_all": True,
		        "n_jobs": None, "max_iter": 300} | args
		if args["bin_seeding"] == 1:
			args["bin_seeding"] = True
		elif args["bin_seeding"] == 0:
			args["bin_seeding"] = False
		if args["cluster_all"] == 1:
			args["cluster_all"] = True
		elif args["cluster_all"] == 0:
			args["cluster_all"] = False
		meanshift = MeanShift(bandwidth=args["bandwidth"], seeds=args["seeds"], bin_seeding=args["bin_seeding"],
		                      min_bin_freq=args["min_bin_freq"], cluster_all=args["cluster_all"], n_jobs=args["n_jobs"],
		                      max_iter=args["max_iter"])
		clustering = meanshift.fit_predict(data, None)
		return clustering, None
	elif algorithm == "agglomerative":
		args = {"memory": None, "connectivity": None, "compute_full_tree": 'auto', "linkage": 'ward',
		        "distance_threshold": None, "compute_distances": False} | args
		if args["compute_distances"] == 1:
			args["compute_distances"] = True
		elif args["compute_distances"] == 0:
			args["compute_distances"] = False
		agglomerative = AgglomerativeClustering(n_clusters=args["n_clusters"], memory=args["memory"],
		                                        connectivity=args["connectivity"],
		                                        compute_full_tree=args["compute_full_tree"], linkage=args["linkage"],
		                                        distance_threshold=args["distance_threshold"],
		                                        compute_distances=args["compute_distances"])

		clustering = agglomerative.fit_predict(data, None)
		return clustering, None
	elif algorithm == "scar":
		args = {"nn": "size_root", "alpha": 0.5, "theta": 20, "m": 0.5, "laplacian": 0, "n_iter": 50,
		        "normalize": False, "weighted": False} | args
		if args["normalize"] == 1:
			args["normalize"] = True
		elif args["normalize"] == 0:
			args["normalize"] = False
		if args["weighted"] == 1:
			args["weighted"] = True
		elif args["weighted"] == 0:
			args["weighted"] = False
		if args["nn"] == "size_root":
			args["nn"] = round(len(data) ** 0.5)

		scar = SCAR(k=args["n_clusters"], nn=args["nn"], alpha=args["alpha"], theta=args["theta"], m=args["m"],
		            laplacian=args["laplacian"], n_iter=args["n_iter"], normalize=args["normalize"],
		            weighted=args["weighted"], seed=args["alg_seed"])

		clustering = scar.fit_predict(np.array(data))
		return clustering, None
	elif algorithm == "spectacl":
		args = {"affinity": "radius_neighbors", "epsilon": 1.0, "n_jobs": None, "normalize_adjacency": False} | args
		if args["normalize_adjacency"] == 1:
			args["normalize_adjacency"] = True
		elif args["normalize_adjacency"] == 0:
			args["normalize_adjacency"] = False
		spectacl = Spectacl(affinity=args["affinity"], n_clusters=args["n_clusters"], epsilon=args["epsilon"],
		                    n_jobs=args["n_jobs"], normalize_adjacency=args["normalize_adjacency"],
		                    seed=args["alg_seed"])
		clustering = spectacl.fit_predict(data)
		return clustering, None
	#elif algorithm == "dcf":
	#	args = {"k": None, "beta": 0.4} | args
#
	#	dcf_result = DCFcluster.train(X=np.array(data), k=args["k"], beta=args["beta"])
	#	clustering = dcf_result.labels
	#	return clustering, None
	elif algorithm == "mdbscan":
		args = {"eps": 0.5, "min_samples": 5, "n_neighbors": 10, "t": 0.5} | args
		clustering = MDBSCAN(np.array(data), eps=args["eps"], min_samples=args["min_samples"],
		                     n_neighbors=args["n_neighbors"], t=args["t"])
		return clustering, None
	elif algorithm == "subkmeans":
		args = {"V": None, "m": None, "cluster_centers": None, "mdl_for_noisespace": False,
		        "outliers": False, "max_iter": 300, "n_init": 1, "cost_type": "default",
		        "threshold_negative_eigenvalue": -1e-7, "max_distance": None} | args
		if args["mdl_for_noisespace"] == 1:
			args["mdl_for_noisespace"] = True
		elif args["mdl_for_noisespace"] == 0:
			args["mdl_for_noisespace"] = False
		if args["outliers"] == 1:
			args["outliers"] = True
		elif args["outliers"] == 0:
			args["outliers"] = False
		clustering = SubKmeans(n_clusters=args["n_clusters"], random_state=args["alg_seed"], V=args["V"], m=args["m"],
		                       cluster_centers=args["cluster_centers"], mdl_for_noisespace=args["mdl_for_noisespace"],
		                       outliers=args["outliers"], max_iter=args["max_iter"], cost_type=args["cost_type"],
		                       threshold_negative_eigenvalue=args["threshold_negative_eigenvalue"],
		                       max_distance=args["max_distance"]).fit_predict(np.array(data), None)
		return clustering, None
	elif algorithm == "projdipmeans":
		args = {"significance": 0.001,
             "n_random_projections": 0,
             "pval_strategy": "table",
             "n_boots": 1000,
             "n_split_trials":  10,
             "n_clusters_init": 1,
            "max_n_clusters": np.inf} | args
		clustering = ProjectedDipMeans(random_state=args["alg_seed"], significance=args["significance"],
		                               n_random_projections=args["n_random_projections"], pval_strategy=args["pval_strategy"],
		                               n_boots=args["n_boots"], n_split_trials=args["n_split_trials"], n_clusters_init=args["n_clusters_init"],
		                               max_n_clusters=args["max_n_clusters"]).fit_predict(np.array(data), None)
		return clustering, None
	elif algorithm == "snndpc":
		args = {"n_neighbors": 5} | args
		centroid, clustering = SNNDPC(nc=args["n_clusters"], k=args["n_neighbors"], data=np.array(data))
		#print(clustering)
		return clustering, centroid
	elif algorithm == "dec":
		if args["embedding_size"] > len(data[0]):
			return [-1]*len(data), None

		args = {"embedding_size": 10, "alpha": 1.0,
             "batch_size": 256,
            "pretrain_optimizer_params": {"lr": 1e-3},
		"clustering_optimizer_params": {"lr": 1e-4},
             "pretrain_epochs": 100,
             "clustering_epochs": 150,
             "optimizer_class": torch.optim.Adam,
             "loss_fn": torch.nn.MSELoss(),
			"augmentation_invariance": False, "cluster_loss_weight":1} | args
		if args["augmentation_invariance"] == 1:
			args["augmentation_invariance"] = True
		elif args["augmentation_invariance"] == 0:
			args["augmentation_invariance"] = False
		clustering = DEC(n_clusters=args["n_clusters"], random_state=args["alg_seed"], embedding_size=args["embedding_size"],
		                 batch_size=args["batch_size"], pretrain_optimizer_params=args["pretrain_optimizer_params"],
		                 clustering_optimizer_params = args["clustering_optimizer_params"], pretrain_epochs =args["pretrain_epochs"],
		                 clustering_epochs= args["clustering_epochs"], optimizer_class=args["optimizer_class"],
		                 loss_fn=args["loss_fn"], augmentation_invariance=args["augmentation_invariance"], cluster_loss_weight =args["cluster_loss_weight"]
		                 ).fit_predict(np.array(data), None)
		return clustering, None
	elif algorithm == "dipencoder":
		if args["embedding_size"] > len(data[0]):
			return [-1]*len(data), None
		args = {"embedding_size": 10, "pretrain_batch_size":256, "batch_size": None,
             "pretrain_optimizer_params": {"lr": 1e-3},
             "clustering_optimizer_params": {"lr": 1e-4},
             "pretrain_epochs": 100,
             "clustering_epochs": 100,
             "optimizer_class": torch.optim.Adam,
             "loss_fn": torch.nn.MSELoss(),
			"reconstruction_loss_weight": None, "max_cluster_size_diff_factor":3} | args
		clustering = DipEncoder(n_clusters=args["n_clusters"], random_state=args["alg_seed"], embedding_size=args["embedding_size"],
		                        batch_size=args["batch_size"], pretrain_optimizer_params=args["pretrain_optimizer_params"],
		                        clustering_optimizer_params = args["clustering_optimizer_params"],
		                        pretrain_epochs =args["pretrain_epochs"], clustering_epochs= args["clustering_epochs"],
		                        reconstruction_loss_weight =args["reconstruction_loss_weight"],
		                        optimizer_class=args["optimizer_class"], loss_fn=args["loss_fn"], max_cluster_size_diff_factor=args["max_cluster_size_diff_factor"]).fit_predict(np.array(data), None)
		return clustering, None
	elif algorithm == "dpca":
		args = {"dc":None, "distance_metric": 'euclidean',
             "silence": True,
             "gauss_cutoff": True,
             "density_threshold": None,
             "distance_threshold":  None,
             "anormal": True} | args
		if args["silence"] == 1:
			args["silence"] = True
		elif args["silence"] == 0:
			args["silence"] = False
		if args["gauss_cutoff"] == 1:
			args["gauss_cutoff"] = True
		elif args["gauss_cutoff"] == 0:
			args["gauss_cutoff"] = False
		if args["anormal"] == 1:
			args["anormal"] = True
		elif args["anormal"] == 0:
			args["anormal"] = False
		if args["distance_threshold"] < 0:
			args["distance_threshold"] = None
		if args["density_threshold"] < 0:
			args["density_threshold"] = None
		if args["dc"] < 0:
			args["dc"] = None
		dpc = DensityPeakCluster(dc = args["dc"], distance_metric =args["distance_metric"], silence=args["silence"], gauss_cutoff=args["gauss_cutoff"],
		                         density_threshold=args["density_threshold"], distance_threshold=args["distance_threshold"],
		                         anormal=args["anormal"])
		dpc.fit(np.array(data))
		clustering = dpc.labels_
		return clustering, None
	elif algorithm == "rnndbscan":
		args = {"n_neighbors": 5} | args
		clustering = RNNDBSCAN(k=args["n_neighbors"]).fit_predict(data)
		return clustering, None
	# elif algorithm == "shade":
	# 	if args["embedding_size"] > len(data[0]):
	# 		return [-1]*len(data), None
	# 	args = {"batch_size": 500,
    #          "autoencoder":  None,
    #          "min_points": 5,
    #          "use_complete_dc_tree": False,
    #          "use_matrix_dc_distance": True,
    #          "increase_inter_cluster_distance": False,
    #          "pretrain_epochs": 0,
    #          "pretrain_optimizer_params": {"lr": 1e-3},
    #          "clustering_epochs": 100,
    #          "clustering_optimizer_params": {"lr": 1e-3},
    #          "embedding_size": 10,
    #          "optimizer_class": torch.optim.Adam,
    #          "loss_fn": torch.nn.MSELoss(),
    #          "custom_dataloaders": None,
    #          "standardize": True,
    #          "standardize_axis":  0,
    #          "cluster_algorithm": DCTree_Clusterer,
    #          "cluster_algorithm_params": {},
    #          "degree_of_reconstruction":  1.0,
    #          "degree_of_density_preservation": 1.0} | args
	# 	if args["use_complete_dc_tree"] == 1:
	# 		args["use_complete_dc_tree"] = True
	# 	elif args["use_complete_dc_tree"] == 0:
	# 		args["use_complete_dc_tree"] = False
	# 	if args["use_matrix_dc_distance"] == 1:
	# 		args["use_matrix_dc_distance"] = True
	# 	elif args["use_matrix_dc_distance"] == 0:
	# 		args["use_matrix_dc_distance"] = False
	# 	if args["increase_inter_cluster_distance"] == 1:
	# 		args["increase_inter_cluster_distance"] = True
	# 	elif args["increase_inter_cluster_distance"] == 0:
	# 		args["increase_inter_cluster_distance"] = False
	# 	if args["standardize"] == 1:
	# 		args["standardize"] = True
	# 	elif args["standardize"] == 0:
	# 		args["standardize"] = False
	# 	clustering = SHADE(batch_size=args["batch_size"], autoencoder=args["autoencoder"], min_points=args["min_points"],
	# 	                   use_complete_dc_tree=args["use_complete_dc_tree"], use_matrix_dc_distance=args["use_matrix_dc_distance"],
	# 	                   increase_inter_cluster_distance=args["increase_inter_cluster_distance"], pretrain_epochs=args["pretrain_epochs"],
	# 	                   pretrain_optimizer_params=args["pretrain_optimizer_params"], clustering_epochs=args["clustering_epochs"],
	# 	                   clustering_optimizer_params=args["clustering_optimizer_params"], embedding_size=args["embedding_size"],
	# 	                   optimizer_class=args["optimizer_class"], loss_fn=args["loss_fn"], custom_dataloaders=args["custom_dataloaders"],
	# 	                   standardize=args["standardize"], standardize_axis=args["standardize_axis"], n_clusters=None,
	# 	                   cluster_algorithm=args["cluster_algorithm"], cluster_algorithm_params=args["cluster_algorithm_params"],
	# 	                   degree_of_reconstruction=args["degree_of_reconstruction"], degree_of_density_preservation=args["degree_of_density_preservation"],
	# 	                   random_state=args["alg_seed"]
	# 	                   ).fit_predict(np.array(data), None)
	# 	return clustering, None

	else:
		raise NotImplementedError


class WCluStream(CluStream):

	def __init__(
			self,
			n_macro_clusters: int = 5,
			max_micro_clusters: int = 100,
			micro_cluster_r_factor: int = 2,
			time_window: int = 1000,
			time_gap: int = 100,
			seed: int | None = None,
			offline_algo: str = "kmeans",
			offline_args=None,
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
		if self.offline_algo in k_algos:
			self.offline_args["n_clusters"] = n_macro_clusters
		self.offline_args["alg_seed"] = seed

		self.offline_dataset = []
		self.offline_labels = []

	def offline_processing(self):
		gen_data = []
		gen_labels = []
		for id, mc in self.micro_clusters.items():
			mcweight = mc.weight
			for w in range(int(mcweight)):
				gen_data.append(mc.center)
				gen_labels.append(id)

		gen_X = dps_to_np(gen_data)
		clustering, self.centers = perform_clustering(gen_X, self.offline_algo, self.offline_args)
		num_clu = len(np.unique(clustering))

		for id, mc in self.micro_clusters.items():
			is_mc = [i for i, x in enumerate(gen_labels) if x == id]
			labels_mc = [0] * num_clu
			for i_mc in is_mc:
				labels_mc[clustering[i_mc]] += 1
			self.cluster_assignments[id] = labels_mc.index(max(labels_mc))

		cluster_labels_gen = []
		for l in gen_labels:
			cluster_labels_gen.append(self.cluster_assignments[l])

		self._offline_timestamp = self._timestamp

		self.offline_dataset = gen_data
		self.offline_labels = gen_labels

	def predict_one(self, x, recluster=False, sklearn=None, return_mc=False):
		if self._offline_timestamp != self._timestamp:
			self.offline_processing()
		index, _ = self._get_closest_mc(x)

		if return_mc:
			return self.cluster_assignments[index], index
		else:
			return self.cluster_assignments[index]

class OPECluStream(CluStream):

	def __init__(
			self,
			n_macro_clusters: int = 5,
			max_micro_clusters: int = 100,
			micro_cluster_r_factor: int = 2,
			time_window: int = 1000,
			time_gap: int = 100,
			seed: int | None = None,
			offline_algo: str = "kmeans",
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
		if self.offline_algo in k_algos:
			self.offline_args["n_clusters"] = n_macro_clusters
		self.offline_args["alg_seed"] = seed
		self.offline_datascale = offline_datascale

		self.offline_dataset = []
		self.offline_labels = []
		self.data_reconstructor = DataReconstructor()

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
		gen_data, gen_labels = self.data_reconstructor.reconstruct_data(self.micro_clusters, self.offline_datascale,
		                                        self.micro_cluster_r_factor, self.generator)
		gen_X = dps_to_np(gen_data)
		# plt.figure(figsize=(10, 10))
		# plt.scatter(gen_X[:, 0], gen_X[:, 1], c=gen_labels)
		# for id, mc in self.micro_clusters.items():
		# 	mccenter = dict_to_np(mc.center)
		# 	mc_patch = ptc.Circle((float(mccenter[0]), float(mccenter[1])), mc.radius(self.micro_cluster_r_factor),
		# 						  alpha=0.2, color="lightgrey")
		# 	plt.scatter(float(mccenter[0]), float(mccenter[1]), c="black", alpha=0.5)
		# 	plt.gca().add_patch(mc_patch)
		# plt.ylim(-0.1, 1.1)
		# plt.xlim(-0.1, 1.1)
		# plt.show()

		clustering, self.centers = perform_clustering(gen_X, self.offline_algo, self.offline_args)
		num_clu = len(np.unique(clustering))
		# plt.figure(figsize=(10, 10))
		# plt.scatter(gen_X[:, 0], gen_X[:, 1], c=clustering)
		# for id, mc in self.micro_clusters.items():
		# 	mccenter = dict_to_np(mc.center)
		# 	mc_patch = ptc.Circle((float(mccenter[0]), float(mccenter[1])), mc.radius(self.micro_cluster_r_factor),
		# 						  alpha=0.2, color="lightgrey")
		# 	plt.scatter(float(mccenter[0]), float(mccenter[1]), c="black", alpha=0.5)
		# 	plt.gca().add_patch(mc_patch)
		# plt.ylim(-0.1, 1.1)
		# plt.xlim(-0.1, 1.1)
		# plt.show()

		for id, mc in self.micro_clusters.items():
			is_mc = [i for i, x in enumerate(gen_labels) if x == id]
			labels_mc = [0] * num_clu
			for i_mc in is_mc:
				labels_mc[clustering[i_mc]] += 1
			self.cluster_assignments[id] = labels_mc.index(max(labels_mc))

		cluster_labels_gen = []
		for l in gen_labels:
			cluster_labels_gen.append(self.cluster_assignments[l])
		# plt.figure(figsize=(10, 10))
		# plt.scatter(gen_X[:, 0], gen_X[:, 1], c=cluster_labels_gen)
		# for id, mc in self.micro_clusters.items():
		# 	mccenter = dict_to_np(mc.center)
		# 	mc_patch = ptc.Circle((float(mccenter[0]), float(mccenter[1])), mc.radius(self.micro_cluster_r_factor),
		# 						  alpha=0.2, color="lightgrey")
		# 	plt.scatter(float(mccenter[0]), float(mccenter[1]), c="black", alpha=0.5)
		# 	plt.gca().add_patch(mc_patch)
		# plt.ylim(-0.1, 1.1)
		# plt.xlim(-0.1, 1.1)
		# plt.show()

		self._offline_timestamp = self._timestamp

		self.offline_dataset = gen_data
		self.offline_labels = gen_labels

	def predict_one(self, x, recluster=False, sklearn=None, return_mc=False):
		if self._offline_timestamp != self._timestamp:
			self.offline_processing()
		index, _ = self._get_closest_mc(x)

		if return_mc:
			return self.cluster_assignments[index], index
		else:
			return self.cluster_assignments[index]


class SCOPE(CluStream):

	def __init__(
			self,
			n_macro_clusters: int = 5,
			max_micro_clusters: int = 100,
			micro_cluster_r_factor: int = 2,
			time_window: int = 1000,
			time_gap: int = 100,
			seed: int | None = None,
			offline_algo: str = "kmeans",
			offline_args=None,
			offline_datascale: int = 1000,
			weight_scale: bool = True,
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
		if self.offline_algo in k_algos:
			self.offline_args["n_clusters"] = n_macro_clusters
		self.offline_args["alg_seed"] = seed
		self.offline_datascale = offline_datascale

		self.offline_dataset = []
		self.offline_labels = []
		self.kdtree = None
		self.data_reconstructor = DataReconstructor()
		self.weight_scale = weight_scale

	def offline_processing(self):
		gen_data, gen_labels = self.data_reconstructor.reconstruct_data(self.micro_clusters, self.offline_datascale,
		                                        self.micro_cluster_r_factor, self.generator, use_centroid=True, weight_scale=self.weight_scale)
		gen_X = dps_to_np(gen_data)

		clustering, self.centers = perform_clustering(gen_X, self.offline_algo, self.offline_args)


		self.cluster_assignments = clustering

		self._offline_timestamp = self._timestamp

		self.offline_dataset = gen_data
		self.offline_labels = gen_labels
		self.kdtree = KDTree(dps_to_np(self.offline_dataset))

	def predict_one(self, x, recluster=False, sklearn=None, return_mc=False):
		if self._offline_timestamp != self._timestamp:
			self.offline_processing()
		index = self.kdtree.query(dict_to_np(x).reshape(1, -1), 1, return_distance=False)[0][0]
		if return_mc:
			return self.cluster_assignments[index], index
		else:
			return self.cluster_assignments[index]

class ScaledCluStream(CluStream):

	def __init__(
			self,
			n_macro_clusters: int = 5,
			max_micro_clusters: int = 100,
			micro_cluster_r_factor: int = 2,
			time_window: int = 1000,
			time_gap: int = 100,
			seed: int | None = None,
			offline_algo: str = "kmeans",
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
		if self.offline_algo in k_algos:
			self.offline_args["n_clusters"] = n_macro_clusters
		self.offline_args["alg_seed"] = seed
		self.offline_datascale = offline_datascale

		self.offline_dataset = []
		self.offline_labels = []
		self.kdtree = None
		self.data_reconstructor = DataReconstructor()
		self.weight_scale = 0

	def offline_processing(self):
		gen_data, gen_labels = self.data_reconstructor.reconstruct_data(self.micro_clusters, self.offline_datascale,
		                                        0, self.generator, use_centroid=True, weight_scale=False)
		gen_X = dps_to_np(gen_data)

		clustering, self.centers = perform_clustering(gen_X, self.offline_algo, self.offline_args)


		self.cluster_assignments = clustering

		self._offline_timestamp = self._timestamp

		self.offline_dataset = gen_data
		self.offline_labels = gen_labels
		self.kdtree = KDTree(dps_to_np(self.offline_dataset))

	def predict_one(self, x, recluster=False, sklearn=None, return_mc=False):
		if self._offline_timestamp != self._timestamp:
			self.offline_processing()
		index = self.kdtree.query(dict_to_np(x).reshape(1, -1), 1, return_distance=False)[0][0]
		if return_mc:
			return self.cluster_assignments[index], index
		else:
			return self.cluster_assignments[index]