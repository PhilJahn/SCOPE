import argparse
import os
import socket
import sys
from datetime import datetime
import torch

from evaluate import getMetrics
import mlflow_logger
from competitors.clustream import CluStream, CluStreamMicroCluster
from datahandler import load_data
from method.CircSCOPE import CircSCOPE
from method.SCOPE import SCOPE
from utils import make_param_dicts
import numpy as np


def get_offline_dict():
	offline_dict = {}
	kmeans = "kMeans"
	kmeans_vals = {"":""}
	kmeans_dicts = make_param_dicts(kmeans_vals)
	offline_dict[kmeans] = kmeans_dicts

	return offline_dict

def main(args):
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	# print(args, flush=True)

	parser = argparse.ArgumentParser()
	parser.add_argument('--ds', default="complex9", type=str, help='Used stream data set')
	parser.add_argument('--offline', default=1000000000, type=int, help='Timesteps for offline phase')
	parser.add_argument('--method', default="clustream", type=str, help='Stream Clustering Method')
	parser.add_argument('--sumlimit', default=100, type=str, help='Number of micro-clusters/summarizing structures')
	#parser.add_argument('--seed', default=0, type=int, help='Seed')
	args = parser.parse_args()
	method_name = args.method

	param_vals = {}
	has_offline = False
	flex_offline = False
	has_mcs = False
	needs_dict = False
	use_one = False

	if args.method == "clustream":
		param_vals["seed"] = [0,1,2,3,4] #seed
		param_vals["mmc"] = [args.sumlimit] #max_micro_clusters
		param_vals["mcrf"] = [2] #micro_cluster_r_factor
		param_vals["tg"] = [args.offline] #time_gap
		param_vals["tw"] = [1000] # time window
		param_vals["sigma"] = [0.5]
		param_vals["mu"] = [0.5]
		has_offline = True
		flex_offline = True
		has_mcs = True
		needs_dict = True
		use_one = True



	offline_dict = {}
	if has_offline:
		if flex_offline:
			offline_dict = get_offline_dict()


	param_dicts = make_param_dicts(param_vals)
	print(param_dicts)
	j = 0
	for param_dict in param_dicts:
		if socket.gethostname() == "PhilippPC23":
			flow_logger = mlflow_logger.MLFlowLogger(experiment_name="SCOPE_Local")
		elif args.method == "SCOPE":
			flow_logger = mlflow_logger.MLFlowLogger(experiment_name="SCOPE")
		else:
			flow_logger = mlflow_logger.MLFlowLogger(experiment_name="SCOPE_Competitors")

		X, Y = load_data(args.ds, seed = 0)
		num_cls = len(np.unique(Y))

		args.class_num = num_cls

		mmc = param_dict["mmc"]
		run_name = f"{args.ds}_{method_name}_{mmc}_{j}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
		hp_dict = vars(args) | param_dict
		#print(hp_dict)
		run_id, output_path = flow_logger.init_experiment(run_name, hyper_parameters=hp_dict)

		flow_logger.log_dict(param_dict, "params")


		method = None
		if args.method == "clustream":
			method = CluStream(n_macro_clusters=args.class_num, max_micro_clusters=param_dict["mmc"], micro_cluster_r_factor=param_dict["mcrf"],
			                   time_gap=param_dict["tg"], time_window=param_dict["tw"], sigma=param_dict["sigma"], mu=param_dict["mu"],
			                   seed=param_dict["seed"])
		dp_store = []
		pred_store = []
		pred_store_all = []
		y_store = []
		for i in range(len(X)):
			x = X[i]
			y = Y[i]
			if needs_dict:
				dp = dict(enumerate(x))
			else:
				dp = x
			dp_store.append(dp)
			y_store.append(y)
			is_last = i == len(X)-1
			if use_one:
				method.learn_one(dp)
				if (i % args.offline == 0 and i > 0) or is_last:
					if has_offline:
						method.offline_processing()
					for dp in dp_store:
						pred = method.predict_one(dp)
						pred_store.append(pred)
						pred_store_all.append(pred)
					metrics, cm = getMetrics(y_store, pred_store)
					flow_logger.log_results(metrics, i)
					flow_logger.log_dict(cm, f"Confusion_Matrix_{i}")
					dp_store = []
					y_store = []
					pred_store = []
					if has_mcs:
						mcs = []
						if args.method == "clustream" or args.method == "opeclustream" :
							for mcid, mc in method.micro_clusters.items():
								mcs.append([mcid, mc.center, mc.radius(r_factor=1), mc.weight, mc.var_time, mc.var_x])
						elif args.method == "circscope":
							for mcid, mc in method.micro_clusters.items():
								mcs.append([mcid, mc.center, mc.radius, mc.weight, mc.var_time, mc.var_x])
						elif args.method == "scope":
							for mcid, mc in method.micro_clusters.items():
								mcs.append([mcid, mc.center, mc.extent, mc.weight, mc.var_time, mc.var_x])

						flow_logger.log_numpy(mcs, f"MCs_{i}")

			else:
				if is_last:
					predictions = method.learn(dp_store)
					metrics, cm = getMetrics(y_store, pred_store)
					flow_logger.log_results(metrics, i)
					flow_logger.log_dict(cm, f"Confusion_Matrix_{i}")




		flow_logger.finalise_experiment()

		j+=1

if __name__ == '__main__':
	main(sys.argv)