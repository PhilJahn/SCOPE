import argparse
import os
import socket
import sys
from copy import copy, deepcopy
from sklearn.neighbors import KDTree

import utils
from evaluate import getMetrics
from datahandler import load_data
from method.offlineHandler import perform_clustering
from utils import make_param_dicts, dps_to_np, dict_to_np
import numpy as np


def main(args):
	# print(args, flush=True)

	parser = argparse.ArgumentParser()
	parser.add_argument('--ds', default="complex9", type=str, help='Used stream data set')
	parser.add_argument('--offline', default=1000, type=int, help='Timesteps for offline phase')
	parser.add_argument('--sumlimit', default=100, type=int, help='Number of micro-clusters/summarizing structures')
	parser.add_argument('--gennum', default=1000, type=int, help='Scale of generated points')
	parser.add_argument('--gpu', default=0, type=int, help='GPU usage')
	parser.add_argument('--category', default="density", type=str, help='Offline algorithm category')
	parser.add_argument('--startindex', default=0, type=int, help='Start index for parameter configuration')
	parser.add_argument('--endindex', default=100000000, type=int, help='End index for parameter configuration')
	parser.add_argument('--used_full', default=1, type=int, help='If AutoML used subsampled data or the full dataset')

	args = parser.parse_args()
	args.method = "scaledclustream"
	method_name = args.method
	args.gpu = args.gpu == 1

	print(args, flush=True)


	param_vals = {}
	has_offline = False
	flex_offline = False
	has_mcs = False
	needs_dict = False
	use_one = False
	has_gen = False
	constraint = False
	batch_eval = False
	dataset_length = 10000000

	param_vals["seed"] = [0, 1, 2, 3, 4]  # seed
	param_vals["mmc"] = [args.sumlimit]  # max_micro_clusters
	param_vals["mc_r_factor"] = [2, 1.5, 3]  # micro_cluster_r_factor
	param_vals["time_gap"] = [1000000000000]  # time_gap
	param_vals["time_window"] = [1000, 10000, 10000000]  # time window
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
	if not os.path.exists("../run_logs"):
		os.mkdir("../run_logs_re")

	f = open(
		f'run_logs_re/{args.ds}_{args.method}_{args.offline}_{args.sumlimit}_{args.gennum}_{args.gpu}_{args.category}_{args.startindex}.txt',
		'w',
		newline='\n', buffering=100)
	offline_dicts = {}
	param_dicts = utils.load_parameters(args.ds, args.method, args.used_full)
	offlinemethods = []
	if args.category == "density":
		offlinemethods.append("dpca")
		offlinemethods.append("snndpc")
		offlinemethods.append("dbhd")
		offlinemethods.append("dbscan")
		offlinemethods.append("hdbscan")
		offlinemethods.append("rnndbscan")
		offlinemethods.append("mdbscan")
	elif args.category == "vardbscan":
		offlinemethods.append("hdbscan")
		offlinemethods.append("rnndbscan")
		offlinemethods.append("mdbscan")
	elif args.category == "nkest":
		offlinemethods.append("kmeans")
		offlinemethods.append("subkmeans")
		offlinemethods.append("spectral")
		offlinemethods.append("spectacl")
		offlinemethods.append("scar")
	elif args.category == "kest":
		offlinemethods.append("xmeans")
	offline_dicts = utils.load_offline_parameters(args.ds, args.method, offlinemethods, args.used_full)
	# pprint(offline_dicts)

	param_index = -1
	j = 0
	for param_dict in param_dicts:
		param_index += 1
		if param_index < args.startindex:
			j += 1
			continue
		if param_index > args.endindex:
			j += 1
			continue

		X, Y = load_data(args.ds, seed=0)
		num_cls = len(np.unique(Y))
		dim = len(X[0])

		args.class_num = num_cls
		f.write(f"{method_name} {j} |{vars(args) | param_dict}\n")

		method = None
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
			if ((i + 1) % args.offline == 0 and i > 0) or is_last:

				gen_data = np.load(f"./gen_data/data_{args.ds}_scaledclustream_{args.offline}_{args.sumlimit}_{args.gennum}_False_{param_index}_{i+1}.npy",
				               allow_pickle=True)

				gen_data_step[i] = gen_data
				unique_offline, unique_indices = np.unique(gen_data, axis=0,
				                                                return_index=True)
				kdtree = KDTree(unique_offline)
				for dp in dp_store:
					index = kdtree.query(dict_to_np(dp).reshape(1, -1), 1, return_distance=False)[0][0]
					assign_store.append(unique_indices[index])

				dp_store_step[i] = copy(dp_store)
				y_store_step[i] = copy(y_store)
				assign_step[i] = copy(assign_store)
				dp_store = []
				y_store = []
				assign_store = []
		print("loaded", flush=True)
		if flex_offline:
			offline_dict = offline_dicts[j]
			for alg in offline_dict.keys():
				k = 0
				for alg_dict in offline_dict[alg]:
					args.class_num = num_cls
					alg_dict["n_clusters"] = args.class_num
					f.write(f"\t{method_name} {j} {alg} {k} |{vars(args) | param_dict | alg_dict}\n")

					steps = sorted(dp_store_step.keys())
					# print(steps)
					off_pred_store_step = {}
					for step in steps:
						# cur_dp = dp_store_step[step]
						cur_y = y_store_step[step]
						cur_assign = assign_step[step]
						cur_pred = []
						cur_mc_clu = {}

						cur_gen_data = gen_data_step[step]
						try:
							clustering, _ = perform_clustering(cur_gen_data, alg, alg_dict)
						except:
							clustering = [-1] * len(cur_gen_data)
							print(f"Clustering for {alg_dict} failed at {step + 1}", flush=True)
							for l in range(len(cur_gen_data)):
								clustering[l] = l
						# print(clustering, flush=True)
						num_clu = len(np.unique(clustering))
						_, clustering = np.unique(clustering, return_inverse=True)
						for id in np.unique(cur_assign):
							cur_mc_clu[id] = clustering[id]
						for mc_id in cur_assign:
							cur_pred.append(cur_mc_clu[mc_id])
						#np.save(
						#	f"preds/preds_{args.ds}_{method_name}_{args.offline}_{args.sumlimit}_{args.gennum}_{args.gpu}_{j}_{step + 1}_{alg}_{k}",
						#	cur_pred)
						metrics, cm = getMetrics(cur_y, cur_pred)
						f.write(f"\t\t{method_name} {j} {step + 1} {alg} {k} |{metrics}\n")

						off_pred_store_step[step] = cur_pred

					k += 1
		# f.flush()

		j += 1
	# f.flush()
	f.close()
	print("Done", flush=True)


if __name__ == '__main__':
	main(sys.argv)
