import argparse
import os
import socket
import sys
from copy import copy
from datetime import datetime
from math import floor
from time import sleep

import torch
from sklearn.cluster import KMeans

from competitors.EmCStream import EmcStream
from competitors.MCMSTStream import MCMSTStream
from competitors.MuDi import MuDiDataPoint, MudiHandler
from competitors.dbstream import DBSTREAM
from competitors.denstream import DenStream
from competitors.dstream import DStreamClusterer
from competitors.gbfuzzystream.MBStream import MBStreamHandler

from competitors.full_dataset_learner import full_dataset_leaner
from competitors.streamkmeans import STREAMKMeans
from evaluate import getMetrics
import mlflow_logger
from competitors.clustream import CluStream, CluStreamMicroCluster
from datahandler import load_data
from method.CircSCOPE import CircSCOPE
#from method.SCOPE import SCOPE
from method.offlineHandler import CircSCOPEOffline, OPECluStream, perform_clustering, SCOPE_CluStream #, SCOPE_Offline
from utils import make_param_dicts, dps_to_np, dict_to_np
import numpy as np


def get_offline_dict(args):
	offline_dict = {}
	if args.gpu:
		shade = "shade"
		shade_vals = {"alg_seed": [0, 1, 2, 3, 4], "min_points": [5, 3, 2, 10, 25, 50, 100], "embedding_size": [10, 5, 2], "increase_inter_cluster_distance": [False, True]}
		shade_dicts = make_param_dicts(shade_vals)
		offline_dict[shade] = shade_dicts

		dec = "dec"
		dec_vals = {"alg_seed": [0, 1, 2, 3, 4], "alpha": [1.0, 0.1, 0.25, 0.5, 0.75, 0.9], "embedding_size": [10, 5, 2], "use_reconstruction_loss":[True, False]}
		dec_dicts = make_param_dicts(dec_vals)
		offline_dict[dec] = dec_dicts

		dipenc = "dipencoder"
		dipenc_vals = {"alg_seed": [0, 1, 2, 3, 4], "embedding_size": [10, 5, 2], "max_cluster_size_diff_factor": [3, 2, 5, 10]}
		dipenc_dicts = make_param_dicts(dipenc_vals)
		offline_dict[dipenc] = dipenc_dicts
	else:

		if args.category == "all" or args.category == "kmeans" or args.category == "means":
			kmeans = "kmeans"
			kmeans_vals = {"alg_seed": [0, 1, 2, 3, 4]}
			kmeans_dicts = make_param_dicts(kmeans_vals)
			offline_dict[kmeans] = kmeans_dicts

			subkmeans = "subkmeans"
			subkmeans_vals = {"alg_seed": [0, 1, 2, 3, 4], "outliers": [True, False], "mdl_for_noisespace": [True, False]}
			subkmeans_dicts = make_param_dicts(subkmeans_vals)
			offline_dict[subkmeans] = subkmeans_dicts

		if args.category == "all" or args.category == "xmeans" or args.category == "means":
			xmeans = "xmeans"
			xmeans_vals = {"alg_seed": [0, 1, 2, 3, 4], "n_split_trials": [10, 20], "check_global_score": [True], "allow_merging": [True, False]}
			xmeans_dicts = make_param_dicts(xmeans_vals)
			offline_dict[xmeans] = xmeans_dicts


			projdipmeans = "projdipmeans"
			projdipmeans_vals = {"alg_seed": [0, 1, 2, 3, 4], "significance": [0.001, 0.01, 0.0001], "n_random_projections": [0,1,5], "n_split_trials": [10, 20], "pval_strategy":['table', 'function', 'bootstrap']}
			projdipmeans_dicts = make_param_dicts(projdipmeans_vals)
			offline_dict[projdipmeans] = projdipmeans_dicts

		if args.category == "all" or args.category == "spectral":

			spectral = "spectral"
			spectral_vals_rbf = {"alg_seed": [0, 1, 2, 3, 4], "affinity": ['rbf'], "gamma": [1, 0.5, 1.5, 2]}
			spectral_vals_nn = {"alg_seed": [0, 1, 2, 3, 4], "affinity": ['nearest_neighbors'], "n_neighbors": [10, 5, 2, 20]}
			spectral_dicts = make_param_dicts(spectral_vals_rbf)
			spectral_dicts.extend(make_param_dicts(spectral_vals_nn))
			offline_dict[spectral] = spectral_dicts

			scar = "scar"
			scar_vals = {"alg_seed": [0, 1, 2, 3, 4], "n_neighbors": ["size_root", 10, 5, 2, 20], "theta": [20, 30, 100, 200, 500], "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
			scar_dicts = make_param_dicts(scar_vals)
			offline_dict[scar] = scar_dicts

			spectacl = "spectacl"
			spectacl_vals = {"alg_seed": [0, 1, 2, 3, 4], "epsilon": [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]}
			spectacl_dicts = make_param_dicts(spectacl_vals)
			offline_dict[spectacl] = spectacl_dicts

		if args.category == "all" or args.category == "dbscan" or args.category == "density_all":

			dbscan = "dbscan"
			dbscan_vals = {"eps": [0.5, 0.25, 0.1, 0.05, 0.01], "min_samples": [5, 3, 2, 10, 25, 50, 100]}
			dbscan_dicts = make_param_dicts(dbscan_vals)
			offline_dict[dbscan] = dbscan_dicts

			hdbscan = "hdbscan"
			hdbscan_vals = {"min_cluster_size": [5, 3, 2, 10, 25, 50, 100]}
			hdbscan_dicts = make_param_dicts(hdbscan_vals)
			offline_dict[hdbscan] = hdbscan_dicts

			rnndbscan = "rnndbscan"
			rnndbscan_vals = {"n_neighbors": [10, 8, 5, 3, 2, 15, 20, 25, 50, 75, 100]}
			rnndbscan_dicts = make_param_dicts(rnndbscan_vals)
			offline_dict[rnndbscan] = rnndbscan_dicts

			mdbscan = "mdbscan"
			mdbscan_vals = {"eps": [0.5, 0.25, 0.1, 0.05, 0.01], "min_samples": [5, 3, 2, 10, 25, 50, 100],
			                "n_neighbors": [10, 5, 2, 20], "t": [2, 5, 10, 50, 100, 200]}
			mdbscan_dicts = make_param_dicts(mdbscan_vals)
			offline_dict[mdbscan] = mdbscan_dicts

		if args.category == "all" or args.category == "density" or args.category == "density_all":

			dpca = "dpca"
			dpca_vals = {"distance_threshold": [None, 0.5, 0.25, 0.1, 0.05, 0.01], "density_threshold": [None, 5, 3, 2, 10, 25, 50, 100],
			             "anormal":[True, False], "gauss_cutoff": [True, False]}
			dpca_dicts = make_param_dicts(dpca_vals)
			offline_dict[dpca] = dpca_dicts

			snndpc = "snndpc"
			snndpc_vals = {"n_neighbors": [10, 5, 2, 20]}
			snndpc_dicts = make_param_dicts(snndpc_vals)
			offline_dict[snndpc] = snndpc_dicts

			dbhd = "dbhd"
			dbhd_vals = {"rho": [1.2, 0.5, 0.75, 1, 1.25, 1.5, 2], "beta": [0.1, 0.2, 0.3, 0.4, 0.5, 0.05, 0.01],
			             "min_cluster_size": [5, 3, 2, 10, 25, 50, 100]}
			dbhd_dicts = make_param_dicts(dbhd_vals)
			offline_dict[dbhd] = dbhd_dicts

#		optics = "optics"
#		optics_vals = {"min_samples": [5, 3, 2, 10, 25, 50, 100], "xi": [0.05, 0.03, 0.01, 0.08, 0.1, 0.2]}
#		optics_dicts = make_param_dicts(optics_vals)
#		offline_dict[optics] = optics_dicts

#		meanshift = "meanshift"
#		meanshift_vals = {"default": [1]}
#		meanshift_dicts = make_param_dicts(meanshift_vals)
#		offline_dict[meanshift] = meanshift_dicts

#		agglomerative = "agglomerative"
#		agglomerative_vals = {"linkage": ["ward", "complete", "average", "single"]}
#		agglomerative_dicts = make_param_dicts(agglomerative_vals)
#		offline_dict[agglomerative] = agglomerative_dicts

		#dcf = "dcf"
		#dcf_vals = {"k": [3, 5, 10, 15, 20, 25, 50, 100], "beta": [0.4, 0.3, 0.2, 0.1, 0.5, 0.6, 0.7, 0.8, 0.9]}
		#dcf_dicts = make_param_dicts(dcf_vals)
		#offline_dict[dcf] = dcf_dicts

	return offline_dict


def main(args):
	# print(args, flush=True)

	parser = argparse.ArgumentParser()
	parser.add_argument('--ds', default="densired", type=str, help='Used stream data set')
	parser.add_argument('--offline', default=10000, type=int, help='Timesteps for offline phase')
	parser.add_argument('--method', default="full", type=str, help='Stream Clustering Method')
	parser.add_argument('--sumlimit', default=100, type=int, help='Number of micro-clusters/summarizing structures')
	parser.add_argument('--gennum', default=1000, type=int, help='Scale of generated points')
	parser.add_argument('--gpu', default=False, type=bool, help='GPU usage')
	parser.add_argument('--category', default="all", type=str, help='Offline algorithm category')
	# parser.add_argument('--seed', default=0, type=int, help='Seed')
	parser.add_argument('--startindex', int=0, type=str, help='Start index for parameter configuration')
	args = parser.parse_args()
	method_name = args.method

	scope_list = ["scope", "scope2", "scope3"]

	param_vals = {}
	has_offline = False
	flex_offline = False
	has_mcs = False
	needs_dict = False
	use_one = False
	has_gen = False
	constraint = False
	batch_eval = False

	if args.method == "clustream":
		param_vals["seed"] = [0, 1, 2, 3, 4]  # seed
		param_vals["mmc"] = [args.sumlimit]  # max_micro_clusters
		param_vals["mcrf"] = [2, 1.5, 3]  # micro_cluster_r_factor
		param_vals["tg"] = [args.offline]  # time_gap
		param_vals["tw"] = [1000, 10000, 10000000]  # time window
		param_vals["sigma"] = [0.5]
		param_vals["mu"] = [0.5]
		has_offline = True
		flex_offline = True
		has_mcs = True
		needs_dict = True
		use_one = True
	elif args.method == "opeclustream" or args.method == "scope_clustream":
		param_vals["seed"] = [0, 1, 2, 3, 4]  # seed
		param_vals["mmc"] = [args.sumlimit]  # max_micro_clusters
		param_vals["mcrf"] = [2, 1.5, 3]  # micro_cluster_r_factor
		param_vals["tg"] = [args.offline]  # time_gap
		param_vals["tw"] = [1000, 10000, 10000000]  # time window
		param_vals["sigma"] = [0.5]
		param_vals["mu"] = [0.5]
		param_vals["gen"] = [args.gennum]
		has_offline = True
		flex_offline = True
		has_mcs = True
		needs_dict = True
		use_one = True
		has_gen = True
	elif args.method in scope_list:
		param_vals["seed"] = [0, 1, 2, 3, 4]  # seed
		param_vals["mmc"] = [args.sumlimit]  # max_micro_clusters
		param_vals["msmc"] = [3, 5, floor(args.sumlimit * 0.1),
		                      floor(args.sumlimit * 0.25)]  # max_singleton_micro_clusters
		param_vals["dis"] = [False]
		param_vals["mcrf"] = [2]  # micro_cluster_r_factor
		param_vals["tg"] = [args.offline]  # time_gap
		param_vals["tw"] = [1000]  # time window
		param_vals["sigma"] = [0.5]
		param_vals["mu"] = [0.5]
		param_vals["gen"] = [args.gennum]
		has_offline = True
		flex_offline = True
		has_mcs = True
		needs_dict = True
		use_one = True
		has_gen = True
	elif args.method == "emcstream":
		param_vals["seed"] = [0, 1, 2, 3, 4]  # seed
		param_vals["horizon"] = [20, 50, 80, 100, args.sumlimit, 1000]  # horizon, from paper
		param_vals["ari_threshold"] = [1.0]  # ari_threshold, from code
		param_vals["ari_threshold_step"] = [0.001]  # ari_threshold, from code
	elif args.method == "mcmststream":
		param_vals["W"] = [100, 1000, 2000]  # use round(10^((log_10(upper) - log_10(lower)) / 2) * lower)
		param_vals["N"] = [2, 5, 15]
		param_vals["r"] = [0.001, 0.005, 0.01, 0.015, 0.05, 0.1, 0.25]  # offline optimization
		param_vals["n_micro"] = [2, 7, 25]
	elif args.method == "denstream":
		param_vals["decaying_factor"] = [0.25, 0.125, 0.35, 1]  # decaying_factor
		param_vals["beta"] = [0.75, 0.2, 0.35, 0.6]  # beta
		param_vals["mu"] = [2, 10]  # mu
		param_vals["epsilon"] = [0.02, 0.2, 0.1, 0.05, 0.03, 0.01]  # epsilon
		param_vals["n_samples_init"] = [args.sumlimit]  # n_samples_init
		param_vals["stream_speed"] = [100, 1000]  # stream speed
		needs_dict = True
		use_one = True
		constraint = True
	elif args.method == "dbstream":
		param_vals["clustering_threshold"] = [1.0, 0.2, 0.1, 0.05, 0.03, 0.01]
		param_vals["fading_factor"] = [0.01, 0.001, 0.0001]
		param_vals["cleanup_interval"] = [2, 1000]
		param_vals["intersection_factor"] = [0.3, 0.1, 0.2]
		param_vals["minimum_weight"] = [1.0, 2, 3]
		needs_dict = True
		use_one = True
	elif args.method == "streamkmeans":
		param_vals["seed"] = [0, 1, 2, 3, 4]  # seed
		param_vals["chunk_size"] = [10, 100, 1000]  # chunk_size
		param_vals["sigma"] = [0.5]
		param_vals["mu"] = [0.5]
		needs_dict = True
		use_one = True
	elif args.method == "gbfuzzystream1000":
		param_vals["lam"] = [0.2, 1, 2, 0.5, 1.4]
		param_vals["threshold"] = [0.3, 0.8, 0.5]
		param_vals["eps"] = [10]
		param_vals["m"] = [2]
		param_vals["batchsize"] = [1000]
		param_vals["rand"] = [0, 1, 2, 3, 4]  # seed
		batch_eval = True
	elif args.method == "gbfuzzystream100":
		param_vals["lam"] = [0.2, 1, 2, 0.5, 1.4]
		param_vals["threshold"] = [0.3, 0.8, 0.5]
		param_vals["eps"] = [10]
		param_vals["m"] = [2]
		param_vals["batchsize"] = [100]
		param_vals["rand"] = [0, 1, 2, 3, 4]  # seed
		batch_eval = True
	elif args.method == "dstream":
		param_vals["seed"] = [0, 1, 2, 3, 4]  # seed
		param_vals["dense_threshold_parameter"] = [3, 1, 0.5]
		param_vals["sparse_threshold_parameter"] = [0.8, 0.5, 0.3]
		param_vals["sporadic_threshold_parameter"] = [0.3, 0.1, 0.05]
		param_vals["decay_factor"] = [0.998, 0.999]
		param_vals["partitions_count"] = [5, 2, 10, 30]
		param_vals["gap"] = [100, args.offline, 0]
		use_one = True
		has_offline = True
		constraint = True
	elif args.method == "mudistream":
		X, Y = load_data(args.ds, seed=0)
		num_cls = len(np.unique(Y))
		dim = len(X[0])
		param_vals["seed"] = [0, 1, 2, 3, 4]  # seed
		param_vals["gridgran"] = [30, 20, 10, 2]
		param_vals["alpha"] = [0.03, 0.2, 0.15, 0.16, 0.08]
		param_vals["lamda"] = [0.125, 8, 0.5, 0.25, 2]
		param_vals["minPts"] = [int(2 ** (dim * 3 / 4)), 2, 3, 5, 10, 50, 100]
		constraint = True
	elif args.method == "full":
		has_offline = True
		flex_offline = True
		use_one = True
		has_mcs = True
		param_vals["full"] = [True]

	offline_dict = {}
	if has_offline:
		if flex_offline:
			offline_dict = get_offline_dict(args)

	if constraint:
		param_dicts_temp = make_param_dicts(param_vals)
		param_dicts = []
		for param_dict in param_dicts_temp:
			if args.method == "denstream":
				if param_dict["mu"] > 1 / param_dict["beta"]:
					param_dicts.append(param_dict)
			elif args.method == "dstream":
				if param_dict["dense_threshold_parameter"] > param_dict["sparse_threshold_parameter"] > param_dict[
					"sporadic_threshold_parameter"]:
					param_dicts.append(param_dict)
			elif args.method == "mudistream":
				if param_dict["alpha"] + 2 ** (-param_dict["lamda"]) > 1:
					param_dicts.append(param_dict)
	else:
		param_dicts = make_param_dicts(param_vals)
	print(param_dicts)
	j = 0
	if args.category == "all" and args.startindex == 0:
		f = open(f'run_logs/{args.ds}_{args.method}_{args.offline}_{args.sumlimit}_{args.gennum}_{args.gpu}.txt', 'w', newline='\n',
		         buffering=1000)
	else:
		f = open(f'run_logs/{args.ds}_{args.method}_{args.offline}_{args.sumlimit}_{args.gennum}_{args.gpu}_{args.category}_{args.startindex}.txt', 'w',
		         newline='\n', buffering=1000)

	param_index = -1
	for param_dict in param_dicts:
		param_index += 1
		if param_index < args.start_index:
			continue

		X, Y = load_data(args.ds, seed=0)
		num_cls = len(np.unique(Y))
		dim = len(X[0])

		args.class_num = num_cls
		f.write(f"{method_name} {j} |{vars(args) | param_dict}\n")

		method = None
		if args.method == "clustream":
			method = CluStream(n_macro_clusters=args.class_num, max_micro_clusters=param_dict["mmc"],
			                   micro_cluster_r_factor=param_dict["mcrf"],
			                   time_gap=param_dict["tg"], time_window=param_dict["tw"], sigma=param_dict["sigma"],
			                   mu=param_dict["mu"],
			                   seed=param_dict["seed"])
		elif args.method == "opeclustream":
			method = OPECluStream(n_macro_clusters=args.class_num, max_micro_clusters=param_dict["mmc"],
			                      micro_cluster_r_factor=param_dict["mcrf"],
			                      time_gap=param_dict["tg"], time_window=param_dict["tw"], sigma=param_dict["sigma"],
			                      mu=param_dict["mu"],
			                      seed=param_dict["seed"], offline_datascale=param_dict["gen"])
		elif args.method == "scope_clustream":
			method = SCOPE_CluStream(n_macro_clusters=args.class_num, max_micro_clusters=param_dict["mmc"],
			                      micro_cluster_r_factor=param_dict["mcrf"],
			                      time_gap=param_dict["tg"], time_window=param_dict["tw"], sigma=param_dict["sigma"],
			                      mu=param_dict["mu"],
			                      seed=param_dict["seed"], offline_datascale=param_dict["gen"])
		# elif args.method in scope_list:
		# 	method = SCOPEOffline(n_macro_clusters=args.class_num, max_micro_clusters=param_dict["mmc"],
		# 	                      micro_cluster_r_factor=param_dict["mcrf"],
		# 	                      time_gap=param_dict["tg"], time_window=param_dict["tw"], sigma=param_dict["sigma"],
		# 	                      mu=param_dict["mu"],
		# 	                      seed=param_dict["seed"], dissolve=param_dict["dis"],
		# 	                      max_singletons=param_dict["msmc"], offline_datascale=param_dict["gen"])
		elif args.method == "emcstream":
			method = EmcStream(k=args.class_num, horizon=param_dict["horizon"],
			                   ari_threshold=param_dict["ari_threshold"],
			                   ari_threshold_step=param_dict["ari_threshold_step"], seed=param_dict["seed"])
		elif args.method == "mcmststream":
			method = MCMSTStream(N=param_dict["N"], W=param_dict["W"], r=param_dict["r"], n_micro=param_dict["n_micro"],
			                     d=dim)
		elif args.method == "denstream":
			method = DenStream(decaying_factor=param_dict["decaying_factor"], beta=param_dict["beta"],
			                   mu=param_dict["mu"],
			                   epsilon=param_dict["epsilon"], n_samples_init=param_dict["n_samples_init"],
			                   stream_speed=param_dict["stream_speed"])
		elif args.method == "dbstream":
			method = DBSTREAM(clustering_threshold=param_dict["clustering_threshold"],
			                  fading_factor=param_dict["fading_factor"],
			                  cleanup_interval=param_dict["cleanup_interval"],
			                  intersection_factor=param_dict["intersection_factor"],
			                  minimum_weight=param_dict["minimum_weight"])
		elif args.method == "streamkmeans":
			method = STREAMKMeans(n_clusters=args.class_num, chunk_size=param_dict["chunk_size"],
			                      sigma=param_dict["sigma"], mu=param_dict["mu"], seed=param_dict["seed"])
		elif args.method == "dstream":
			domains_per_dimension = [(0, 1)] * dim
			partitions_per_dimension = [param_dict["partitions_count"]] * dim
			method = DStreamClusterer(initial_cluster_count=args.class_num, seed=param_dict["seed"],
			                          dense_threshold_parameter=param_dict["dense_threshold_parameter"],
			                          sparse_threshold_parameter=param_dict["sparse_threshold_parameter"],
			                          sporadic_threshold_parameter=param_dict["sporadic_threshold_parameter"],
			                          decay_factor=param_dict["decay_factor"],
			                          gap=param_dict["gap"],
			                          domains_per_dimension=domains_per_dimension,
			                          partitions_per_dimension=partitions_per_dimension,
			                          dimensions=dim)
		elif args.method == "mudistream":
			mini = 0
			maxi = 1

			method = MudiHandler(mini=mini, maxi=maxi,
			                     dimension=dim,
			                     lamda=param_dict["lamda"],
			                     gridGranuality=param_dict["gridgran"],
			                     alpha=param_dict["alpha"],
			                     minPts=param_dict["minPts"],
			                     seed=param_dict["seed"])
		elif args.method == "gbfuzzystream100" or args.method == "gbfuzzystream1000":
			method = MBStreamHandler(lam=param_dict["lam"],
			                         batchsize=param_dict["batchsize"],
			                         threshold=param_dict["threshold"],
			                         m=param_dict["m"],
			                         eps=param_dict["eps"])
		elif args.method == "full":
			method = full_dataset_leaner()

		dp_store = []
		pred_store = []
		y_store = []
		assign_store = []
		mc_store = {}
		pred_store_step = {}
		y_store_step = {}
		dp_store_step = {}
		mc_step = {}
		assign_step = {}
		gen_data_step = {}
		gen_label_step = {}

		for i in range(len(X)):
			x = X[i]
			y = Y[i]
			if needs_dict:
				dp = dict(enumerate(x))
			elif args.method == "mudistream":
				dp = [MuDiDataPoint(X[i].tolist(), i)]
			else:
				dp = x
			dp_store.append(dp)
			y_store.append(y)
			is_last = i == len(X) - 1
			if use_one:
				method.learn_one(dp)
				if ((i+1) % args.offline == 0 and i > 0) or is_last:
					if has_offline:
						method.offline_processing()
					if has_gen:
						gen_data_step[i] = dps_to_np(method.offline_dataset)
						gen_label_step[i] = np.array(method.offline_labels).reshape(-1, 1)
					for dp in dp_store:
						if method_name == "scope_clustream":
							pred, gen_id = method.predict_one(dp, return_mc=True)
							assign_store.append(gen_id)
						elif has_mcs:
							pred, mc_id = method.predict_one(dp, return_mc=True)
							assign_store.append(mc_id)
						else:
							pred = method.predict_one(dp)
						pred_store.append(pred)
					pred_store_step[i] = copy(pred_store)
					dp_store_step[i] = copy(dp_store)
					y_store_step[i] = copy(y_store)
					assign_step[i] = copy(assign_store)

					metrics, cm = getMetrics(y_store, pred_store)
					f.write(f"\t{method_name} {j} {i+1} |{metrics}\n")
					if has_mcs:
						mcs = []
						if args.method == "clustream" or args.method == "opeclustream":
							for mcid, mc in method.micro_clusters.items():
								mcs.append([mcid, mc.center, mc.radius(r_factor=1), mc.weight, mc.var_time, mc.var_x])
								mc_store[mcid] = mc
						elif args.method == "circscope":
							for mcid, mc in method.micro_clusters.items():
								mcs.append([mcid, mc.center, mc.radius, mc.weight, mc.var_time, mc.var_x])
								mc_store[mcid] = mc
						elif args.method in scope_list:
							for mcid, mc in method.micro_clusters.items():
								mcs.append([mcid, mc.min, mc.max, mc.center, mc.extent, mc.weight, mc.var_time])
								mc_store[mcid] = mc
						mc_step[i] = copy(mc_store)
					dp_store = []
					y_store = []
					pred_store = []
					assign_store = []
					mc_store = {}
			else:
				if is_last:
					if method_name == "emcstream":
						method.add_label_store(y_store)
					pred_store = method.learn(dp_store)
					if method_name == "emcstream":
						clustered_X, clustered_y = method.get_clustered_data()
						y_store = clustered_y
					elif method_name == "mudistream":
						clusters = pred_store.copy()
						pred_store = [-1] * len(X)
						for clu in clusters:
							cmcs = clu._cmcList
							for cmc in cmcs:
								for point in cmc.storage:
									pred_store[point.t] = clu.name
					if batch_eval:
						batchsize = param_dict["batchsize"]
						for i in range(0, len(pred_store), batchsize):
							end_batch = min(i + batchsize, len(pred_store))
							pred_batch = pred_store[i:end_batch]
							y_batch = y_store[i:end_batch]
							metrics, cm = getMetrics(pred_batch, y_batch)
							f.write(f"\t{method_name} {j} {end_batch} |{metrics}\n")
					else:
						metrics, cm = getMetrics(y_store, pred_store)
						f.write(f"\t{method_name} {j} {i+1} |{metrics}\n")

		if flex_offline:
			for alg in offline_dict.keys():
				k = 0
				for alg_dict in offline_dict[alg]:
					args.class_num = num_cls
					alg_dict["n_clusters"] = args.class_num
					f.write(f"\t{method_name} {j} {alg} {k} |{vars(args) | param_dict | alg_dict}\n")

					steps = sorted(dp_store_step.keys())
					#print(steps)
					off_pred_store_step = {}
					for step in steps:
						# cur_dp = dp_store_step[step]
						cur_y = y_store_step[step]
						assert has_mcs
						cur_mcs = mc_step[step]
						cur_assign = assign_step[step]
						# print(cur_assign)
						cur_pred = []
						cur_mc_clu = {}

						if has_gen:
							cur_gen_data = gen_data_step[step]
							cur_gen_label = gen_label_step[step]
							clustering, _ = perform_clustering(cur_gen_data, alg, alg_dict)
							# print(clustering, flush=True)
							num_clu = len(np.unique(clustering))
							if method_name == "scope_clustream":
								for id in np.unique(cur_assign):
									cur_mc_clu[id] = clustering[id]
							else:
								for id in np.unique(cur_assign):
									# print(id)
									is_mc = [l for l, x in enumerate(cur_gen_label) if x == id]
									labels_mc = [0] * num_clu
									for i_mc in is_mc:
										labels_mc[clustering[i_mc]] += 1
									cur_mc_clu[id] = labels_mc.index(max(labels_mc))
						elif not method_name=="full":
							try:
								cur_mc_centers = [mc.center for l, mc in cur_mcs.items()]
								cur_mc_ids = [l for l, mc in cur_mcs.items()]
								if alg == "kmeans":
									_kmeans = KMeans(n_clusters=alg_dict["n_clusters"],
									                 random_state=alg_dict["alg_seed"])
									cur_mc_weight = [mc.weight for l, mc in cur_mcs.items()]
									clustering = _kmeans.fit_predict(dps_to_np(cur_mc_centers),
									                                 sample_weight=cur_mc_weight)
								else:
									cur_mc_centers_np = dps_to_np(cur_mc_centers)
									clustering, _ = perform_clustering(cur_mc_centers_np, alg, alg_dict)
								for l in range(len(cur_mcs)):
									cur_mc_clu[cur_mc_ids[l]] = clustering[l]
							except:
								cur_mc_ids = [l for l, mc in cur_mcs.items()]
								for l in range(len(cur_mcs)):
									cur_mc_clu[cur_mc_ids[l]] = l  # assume full segementation
								print(f"Clustering for {alg_dict} failed at {step+1}")

						if method_name == "full":
							cur_data = dp_store_step[step]
							#print(len(cur_data))
							cur_pred, _ = perform_clustering(cur_data, alg, alg_dict)
							#print(cur_pred)
						else:
							for mc_id in cur_assign:
								cur_pred.append(cur_mc_clu[mc_id])

						metrics, cm = getMetrics(cur_y, cur_pred)
						f.write(f"\t\t{method_name} {j} {step+1} {alg} {k} |{metrics}\n")

						off_pred_store_step[step] = cur_pred

					k += 1
		# f.flush()

		j += 1
	# f.flush()
	f.close()


if __name__ == '__main__':
	main(sys.argv)
