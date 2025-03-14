import argparse
import copy
import os

from ConfigSpace.hyperparameters import FloatHyperparameter
from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, Categorical, Constant, \
	ForbiddenGreaterThanRelation, ForbiddenLessThanRelation, ForbiddenValueError
import numpy as np
import matplotlib.pyplot as plt

import utils
from competitors.clustream import CluStream

from datahandler import load_data
from evaluate import getMetrics
from utils import dps_to_np

# def train(config: Configuration, seed: int = 0) -> float:
#    classifier = SVC(C=config["C"], random_state=seed)
#    scores = cross_val_score(classifier, iris.data, iris.target, cv=5)
#    return 1 - np.mean(scores)


# configspace = ConfigurationSpace({"C": (0.100, 1000.0)})

# Scenario object specifying the optimization environment
# scenario = Scenario(configspace, deterministic=True, n_trials=200)

# Use SMAC to find the best configuration/hyperparameters
# smac = HyperparameterOptimizationFacade(scenario, train)
# incumbent = smac.optimize()
global data_name
data_name = None
global data_dim
data_dim = 0
global data_length
data_length = 0
global class_num
class_num = 0
global mc_num
mc_num = 100
global offline_timing
offline_timing = 1000
global dps
dps = []
global labels
labels = []
global cur_best_score
cur_best_score = -1
global best_performance
best_performance = -1


def get_clustering_learn_one(clustering_method):
	dp_store = []
	prediction = []
	i = 0
	is_clustream = type(clustering_method) == CluStream
	assignments = []
	mcs = []
	cur_prediction = []
	for dp in dps:
		is_last = i == len(X) - 1
		dp = dict(enumerate(dp))
		dp_store.append(dp)
		clustering_method.learn_one(dp)
		if not is_clustream:
			label = clustering_method.predict_one(dp)
			prediction.append(label)
		else:
			if (i + 1) % offline_timing == 0 or is_last:
				if is_clustream:
					for mcid, mc in clustering_method.micro_clusters.items():
						mcs.append([i, mcid, mc.center, mc.radius(r_factor=1), mc.weight, mc.var_time, mc.var_x])
				cur_prediction = []
				for dp_2 in dp_store:
					if is_clustream:
						label, assign = clustering_method.predict_one(dp_2, recluster=True, sklearn=True,return_mc=True)
						assignments.append(assign)
					else:
						label = clustering_method.predict_one(dp_2)
					cur_prediction.append(label)
				dp_store = []
				prediction.extend(cur_prediction)
		i += 1
	amis = []
	aris = []
	accs = []
	for i in range(0, len(prediction), offline_timing):
		end = min(len(prediction), i + offline_timing)
		length = end - i
		# print(i, end, length)
		metrics, _ = getMetrics(y[i:end], prediction[i:end])
		amis.extend([metrics["AMI"]] * length)
		aris.extend([metrics["ARI"]] * length)
		accs.extend([metrics["accuracy"]] * length)
	# print(len(amis))
	ami = float(np.mean(amis))
	ari = float(np.mean(aris))
	acc = float(np.mean(accs))

	if not os.path.exists("param_data"):
		os.mkdir("param_data")

	np.save(f"param_data/assign_{data_name}", assignments)
	np.save(f"param_data/mcs_{data_name}", mcs)

	return ami + ari


def train_clustream(config: dict, seed: int = 0) -> float:
	# print(data_dim, class_num, data_length, mc_num, offline_timing)
	#global offline_timing
	#offline_timing = config["time_window"]
	clustering_method = CluStream(n_macro_clusters=class_num, seed=seed, max_micro_clusters=mc_num, time_gap=100000000,
	                              micro_cluster_r_factor=config["mc_r_factor"], time_window=config["time_window"])
	score = get_clustering_learn_one(clustering_method)

	print("CluStream", config["mc_r_factor"], config["time_window"], score)
	return 2 - score

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--ds', default="densired10", type=str, help='Used stream data set')
	#parser.add_argument('--method', default="clustream", type=str, help='Stream Clustering Method')
	parser.add_argument('--use_full', default=0, type=int, help='Use full datset')
	args = parser.parse_args()
	args.method = "clustream"
	print(args, flush=True)

	data_name = args.ds
	args.use_full = args.use_full == 1

	param_dicts = utils.load_parameters(args.ds, args.method)

	param_dict = param_dicts[0]
	print(param_dict)
	cur_best_score = -1
	best_performance = -1
	if not args.use_full:
		seed_num = 5
		data_name = args.ds + "_subset_0"
	else:
		seed_num = 1
	X, y = load_data(data_name)
	if not args.use_full:
		data_name = args.ds + "_subset_default"
	else:
		data_name = args.ds + "_default"
	uniques = np.unique(y, return_counts=False)
	data_dim = len(X[0])
	data_length = len(y)
	class_num = len(uniques)
	dps = X

	labels = y
	train_clustream(param_dict, 0)

	for run in range(seed_num):
		run_index = run
		param_dict = param_dicts[5*(run+1)]
		print(param_dict)
		cur_best_score = -1
		best_performance = -1
		if not args.use_full:
			data_name = args.ds + "_subset_" + str(run)
		else:
			data_name = args.ds + "_" + str(run)
		X, y = load_data(data_name)
		uniques = np.unique(y, return_counts=False)
		data_dim = len(X[0])
		data_length = len(y)
		class_num = len(uniques)
		dps = X

		labels = y
		train_clustream(param_dict, 0)
