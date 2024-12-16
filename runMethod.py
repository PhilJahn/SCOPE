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

from evaluate import getMetrics
import mlflow_logger
from competitors.clustream import CluStream, CluStreamMicroCluster
from datahandler import load_data
from method.CircSCOPE import CircSCOPE
from method.SCOPE import SCOPE
from method.offlineHandler import SCOPEOffline, CircSCOPEOffline, OPECluStream, perform_clustering
from utils import make_param_dicts, dps_to_np, dict_to_np
import numpy as np




def get_offline_dict():
	offline_dict = {}
	kmeans = "kmeans"
	kmeans_vals = {"alg_seed": [0, 1, 2, 3, 4]}
	kmeans_dicts = make_param_dicts(kmeans_vals)
	offline_dict[kmeans] = kmeans_dicts

	xmeans = "xmeans"
	xmeans_vals = {"alg_seed": [0, 1, 2, 3, 4]}
	xmeans_dicts = make_param_dicts(xmeans_vals)
	offline_dict[xmeans] = xmeans_dicts

	dbhd = "dbhd"
	dbhd_vals = {"rho": [1.2, 0.5, 0.75, 1, 1.25, 1.5, 2], "beta": [0.1, 0.2, 0.3, 0.4, 0.5, 0.05, 0.01],
	             "min_cluster_size": [5, 3, 2, 10, 25, 50, 100]}
	dbhd_dicts = make_param_dicts(dbhd_vals)
	offline_dict[dbhd] = dbhd_dicts

	spectral = "spectral"
	spectral_vals_rbf = {"alg_seed": [0, 1, 2, 3, 4], "affinity": ['rbf'], "gamma": [1, 0.5, 1.5, 2]}
	spectral_vals_nn = {"alg_seed": [0, 1, 2, 3, 4], "affinity": ['nearest_neighbors'], "n_neighbors": [10, 5, 2, 20]}
	spectral_dicts = make_param_dicts(spectral_vals_rbf)
	spectral_dicts.extend(make_param_dicts(spectral_vals_nn))
	offline_dict[spectral] = spectral_dicts

	dbscan = "dbscan"
	dbscan_vals = {"eps": [0.5, 0.1, 0.05, 0.01], "min_samples": [5, 3, 2, 10, 25, 50, 100]}
	dbscan_dicts = make_param_dicts(dbscan_vals)
	offline_dict[dbscan] = dbscan_dicts

	hdbscan = "hdbscan"
	hdbscan_vals = {"min_cluster_size": [5, 3, 2, 10, 25, 50, 100]}
	hdbscan_dicts = make_param_dicts(hdbscan_vals)
	offline_dict[hdbscan] = hdbscan_dicts

	optics = "optics"
	optics_vals = {"min_samples": [5, 3, 2, 10, 25, 50, 100], "xi": [0.05, 0.03, 0.01, 0.08, 0.1, 0.2]}
	optics_dicts = make_param_dicts(optics_vals)
	offline_dict[optics] = optics_dicts

	meanshift = "meanshift"
	meanshift_vals = {"default": [1]}
	meanshift_dicts = make_param_dicts(meanshift_vals)
	offline_dict[meanshift] = meanshift_dicts

	agglomerative = "agglomerative"
	agglomerative_vals = {"linkage": ["ward",  "complete", "average", "single"]}
	agglomerative_dicts = make_param_dicts(agglomerative_vals)
	offline_dict[agglomerative] = agglomerative_dicts

	return offline_dict


def main(args):
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	# print(args, flush=True)

	parser = argparse.ArgumentParser()
	parser.add_argument('--ds', default="complex9", type=str, help='Used stream data set')
	parser.add_argument('--offline', default=1000, type=int, help='Timesteps for offline phase')
	parser.add_argument('--method', default="opeclustream", type=str, help='Stream Clustering Method')
	parser.add_argument('--sumlimit', default=100, type=int, help='Number of micro-clusters/summarizing structures')
	parser.add_argument('--gennum', default=1000, type=int, help='Scale of generated points')
	# parser.add_argument('--seed', default=0, type=int, help='Seed')
	args = parser.parse_args()
	method_name = args.method

	param_vals = {}
	has_offline = False
	flex_offline = False
	has_mcs = False
	needs_dict = False
	use_one = False
	has_gen = False

	if args.method == "clustream":
		param_vals["seed"] = [0, 1, 2, 3, 4]  # seed
		param_vals["mmc"] = [args.sumlimit]  # max_micro_clusters
		param_vals["mcrf"] = [2]  # micro_cluster_r_factor
		param_vals["tg"] = [args.offline]  # time_gap
		param_vals["tw"] = [1000]  # time window
		param_vals["sigma"] = [0.5]
		param_vals["mu"] = [0.5]
		has_offline = True
		flex_offline = True
		has_mcs = True
		needs_dict = True
		use_one = True
	if args.method == "opeclustream":
		param_vals["seed"] = [0, 1, 2, 3, 4]  # seed
		param_vals["mmc"] = [args.sumlimit]  # max_micro_clusters
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
	elif args.method == "scope":
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

	offline_dict = {}
	if has_offline:
		if flex_offline:
			offline_dict = get_offline_dict()

	param_dicts = make_param_dicts(param_vals)
	print(param_dicts)
	j = 0
	f = open(f'run_logs/{args.ds}_{args.method}_{args.offline}_{args.sumlimit}_{args.gennum}.txt', 'w', newline='\n', buffering=1000)
	for param_dict in param_dicts:

		X, Y = load_data(args.ds, seed=0)
		num_cls = len(np.unique(Y))

		args.class_num = num_cls
		f.write(f"{method_name} {j} {vars(args) | param_dict}\n")

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
		elif args.method == "scope":
			method = SCOPEOffline(n_macro_clusters=args.class_num, max_micro_clusters=param_dict["mmc"],
			                      micro_cluster_r_factor=param_dict["mcrf"],
			                      time_gap=param_dict["tg"], time_window=param_dict["tw"], sigma=param_dict["sigma"],
			                      mu=param_dict["mu"],
			                      seed=param_dict["seed"], dissolve=param_dict["dis"],
			                      max_singletons=param_dict["msmc"], offline_datascale=param_dict["gen"])
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
			else:
				dp = x
			dp_store.append(dp)
			y_store.append(y)
			is_last = i == len(X) - 1
			if use_one:
				method.learn_one(dp)
				if (i % args.offline == 0 and i > 0) or is_last:
					if has_offline:
						method.offline_processing()
					if has_gen:
						gen_data_step[i] = dps_to_np(method.offline_dataset)
						gen_label_step[i] = np.array(method.offline_labels).reshape(-1, 1)
					for dp in dp_store:
						if has_mcs:
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
					f.write(f"\t{method_name} {j} {i} {metrics}\n")
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
						elif args.method == "scope":
							for mcid, mc in method.micro_clusters.items():
								mcs.append([mcid, mc.center, mc.extent, mc.weight, mc.var_time, mc.var_x])
								mc_store[mcid] = mc
						mc_step[i] = copy(mc_store)
					dp_store = []
					y_store = []
					pred_store = []
					assign_store = []
					mc_store = {}
			else:
				if is_last:
					predictions = method.learn(dp_store)
					metrics, cm = getMetrics(y_store, pred_store)
					f.write(f"\t{method_name} {j} {i} {metrics}\n")

		if flex_offline:
			for alg in offline_dict.keys():
				k = 0
				for alg_dict in offline_dict[alg]:
					args.class_num = num_cls
					alg_dict["n_clusters"] = args.class_num
					f.write(f"\t{method_name} {j} {alg} {k} {vars(args) | param_dict | alg_dict}\n")

					steps = sorted(dp_store_step.keys())
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
							for id in np.unique(cur_assign):
								# print(id)
								is_mc = [l for l, x in enumerate(cur_gen_label) if x == id]
								labels_mc = [0] * num_clu
								for i_mc in is_mc:
									labels_mc[clustering[i_mc]] += 1
								cur_mc_clu[id] = labels_mc.index(max(labels_mc))

						else:
							try:
								cur_mc_centers = [mc.center for l, mc in cur_mcs.items()]
								cur_mc_ids = [l for l, mc in cur_mcs.items()]
								if alg == "kmeans":
									_kmeans = KMeans(n_clusters=alg_dict["n_clusters"], random_state=alg_dict["alg_seed"])
									cur_mc_weight = [mc.weight for l, mc in cur_mcs.items()]
									clustering = _kmeans.fit_predict(dps_to_np(cur_mc_centers), sample_weight=cur_mc_weight)
								else:
									cur_mc_centers_np = dps_to_np(cur_mc_centers)
									clustering, _ = perform_clustering(cur_mc_centers_np, alg, alg_dict)
								for l in range(len(cur_mcs)):
									cur_mc_clu[cur_mc_ids[l]] = clustering[l]
							except:
								cur_mc_ids = [l for l, mc in cur_mcs.items()]
								for l in range(len(cur_mcs)):
									cur_mc_clu[cur_mc_ids[l]] = l #assume full segementation
								print(f"Clustering for {alg_dict} failed at {step}")


						for mc_id in cur_assign:
							cur_pred.append(cur_mc_clu[mc_id])

						metrics, cm = getMetrics(cur_y, cur_pred)
						f.write(f"\t\t{method_name} {j} {step} {alg} {k} {metrics}\n")

						off_pred_store_step[step] = cur_pred

					k += 1
					#f.flush()

		j += 1
		#f.flush()
	f.close()


if __name__ == '__main__':
	main(sys.argv)
