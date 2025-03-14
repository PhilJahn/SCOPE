import argparse
import copy
import os

from ConfigSpace.hyperparameters import FloatHyperparameter
from numpy.random import PCG64
from sklearn.neighbors import KDTree
from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, Categorical, Constant, \
	ForbiddenGreaterThanRelation, ForbiddenLessThanRelation, ForbiddenValueError, EqualsCondition, \
	ForbiddenEqualsClause, ForbiddenAndConjunction
import numpy as np
import matplotlib.pyplot as plt

import utils
from datahandler import load_data, read_subset
from evaluate import getMetrics
from method.offlineHandler import perform_clustering, DataReconstructor
from utils import dps_to_np, dict_to_np

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
global handler
handler = None
global clusterer
clusterer = None
global mcs
mcs = {}
global assignment
assignment = {}
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
global param_dict
param_dict = {}

def eval_clustering(clustering, assigns):
	amis = []
	aris = []
	accs = []
	for i in assigns.keys():
		clustering_assigned = []
		for assign in assigns[i]:
			#print(clustering[i], assign)
			clustering_assigned.append(clustering[i][assign])
		y_step = labels[i]
		length = len(assigns[i])
		metrics, _ = getMetrics(y_step, clustering_assigned)
		amis.extend([metrics["AMI"]] * length)
		aris.extend([metrics["ARI"]] * length)
		accs.extend([metrics["accuracy"]] * length)
	# print(len(amis))
	ami = float(np.mean(amis))
	ari = float(np.mean(aris))
	acc = float(np.mean(accs))

	global cur_best_score
	global best_performance
	if ami + ari > cur_best_score:
		cur_best_score = ami + ari
		best_performance = {"accuracy": acc, "ARI": ari, "AMI": ami}

	return ami + ari


def get_centroid(mcs):
	data_dict = {}
	for i in mcs.keys():
		mcsi = mcs[i]
		datai = []
		for mc in mcsi:
			datai.append(dict_to_np(mc[2]))
		data_dict[i] = datai
	return data_dict


def get_multi_centroid(mcs, assign):
	data_dict = {}
	new_assign_dict = {}
	for i in mcs.keys():
		mcsi = mcs[i]
		#print(mcsi)
		assigni = assign[i]
		new_assigni = [-1]*len(assigni)
		datai = []
		for mc in mcsi:
			mcid = mc[1]
			cur_pos = len(datai)
			for _ in range(int(mc[4])):
				datai.append(dict_to_np(mc[2]))
			for j in range(len(assigni)):
				if assigni[j] == mcid:
					new_assigni[j] = cur_pos
		data_dict[i] = datai
		new_assign_dict[i] = new_assigni
	return data_dict, new_assign_dict

def get_gendata(mcs, seed, weight_scale):
	data_dict = {}
	new_assign_dict = {}
	generator = np.random.Generator(PCG64(seed))
	data_reconstructor = DataReconstructor()
	for i in mcs.keys():
		dps_i = dps[i]
		mcsi = mcs[i]
		data_new, _ = data_reconstructor.reconstruct_data(micro_clusters=mcsi, num=1000, radius_mult=param_dict['mc_r_factor'],
		                                    generator=generator, use_centroid=True, mc_import=True, weight_scale=weight_scale)
		data_new = dps_to_np(data_new)
		kdtree = KDTree(data_new)
		assign_new = kdtree.query(dps_i, 1, return_distance=False).flatten()
		#print("assign", assign_new.shape)
		#print("dps", dps_i.shape)
		data_dict[i] = data_new
		#print("gen", data_new.shape)
		new_assign_dict[i] = assign_new
	#print(data_dict)
	return data_dict, new_assign_dict

def get_scaleddata(mcs, seed, weight_scale):
	data_dict = {}
	new_assign_dict = {}
	generator = np.random.Generator(PCG64(seed))
	data_reconstructor = DataReconstructor()
	for i in mcs.keys():
		dps_i = dps[i]
		mcsi = mcs[i]
		data_new, _ = data_reconstructor.reconstruct_data(micro_clusters=mcsi, num=1000, radius_mult=0,
		                                    generator=generator, use_centroid=True, mc_import=True, weight_scale=False)
		data_new = dps_to_np(data_new)
		kdtree = KDTree(data_new)
		assign_new = kdtree.query(dps_i, 1, return_distance=False).flatten()
		#print("assign", assign_new.shape)
		#print("dps", dps_i.shape)
		data_dict[i] = data_new
		#print("gen", data_new.shape)
		new_assign_dict[i] = assign_new
	#print(data_dict)
	return data_dict, new_assign_dict


def train_clustream(config: Configuration, seed: int = 0) -> float:
	clustering_dict = {}
	config_dict = config.get_dictionary()
	config_dict["alg_seed"] = seed
	config_dict["n_clusters"] = class_num
	data = get_centroid(mcs)
	for i in assignment.keys():
		clustering, _ = perform_clustering(data[i], clusterer, config_dict)
		#print(clustering)
		clustering_dict[i] = clustering
	score = eval_clustering(clustering_dict, assignment)
	print("CluStream", clusterer, param_dict, config_dict, score)
	return 2 - score

def train_wclustream(config: Configuration, seed: int = 0) -> float:
	clustering_dict = {}
	config_dict = config.get_dictionary()
	config_dict["alg_seed"] = seed
	config_dict["n_clusters"] = class_num
	data, new_assign = get_multi_centroid(mcs, assignment)
	for i in new_assign.keys():
		clustering, _ = perform_clustering(data[i], clusterer, config_dict)
		#print(clustering)
		clustering_dict[i] = clustering
	score = eval_clustering(clustering_dict, new_assign)
	print("Weighted CluStream", clusterer, param_dict, config_dict, score)
	return 2 - score

def train_scaledclustream(config: Configuration, seed: int = 0) -> float:
	clustering_dict = {}
	config_dict = config.get_dictionary()
	config_dict["alg_seed"] = seed
	config_dict["n_clusters"] = class_num
	data, new_assign = get_scaleddata(mcs, seed, False)
	for i in new_assign.keys():
		clustering, _ = perform_clustering(data[i], clusterer, config_dict)
		#print(clustering)
		clustering_dict[i] = clustering
	score = eval_clustering(clustering_dict, new_assign)
	print("Scaled CluStream", clusterer, param_dict, config_dict, score)
	return 2 - score

def train_scope_full(config: Configuration, seed: int = 0) -> float:
	clustering_dict = {}
	config_dict = config.get_dictionary()
	config_dict["alg_seed"] = seed
	config_dict["n_clusters"] = class_num
	data, new_assign = get_gendata(mcs, seed, False)
	for i in new_assign.keys():
		clustering, _ = perform_clustering(data[i], clusterer, config_dict)
		#print(clustering)
		clustering_dict[i] = clustering
	score = eval_clustering(clustering_dict, new_assign)
	print("SCOPE (full)", clusterer, param_dict, config_dict, score)
	return 2 - score

def train_scope(config: Configuration, seed: int = 0) -> float:
	clustering_dict = {}
	config_dict = config.get_dictionary()
	config_dict["alg_seed"] = seed
	config_dict["n_clusters"] = class_num
	data, new_assign = get_gendata(mcs, seed, True)
	for i in new_assign.keys():
		clustering, _ = perform_clustering(data[i], clusterer, config_dict)
		#print(clustering)
		clustering_dict[i] = clustering
	score = eval_clustering(clustering_dict, new_assign)
	print("SCOPE", clusterer, param_dict, config_dict, score)
	return 2 - score


configspaces = {}
kmeans_space = ConfigurationSpace()
kmeans_1=Constant("init", 'k-means++')
kmeans_space.add([kmeans_1])
configspaces["kmeans"] = kmeans_space

subkmeans_space = ConfigurationSpace()
subkmeans_1=Categorical("outliers", [1, 0], default=0)
subkmeans_2=Categorical("mdl_for_noisespace", [1, 0], default=0)
subkmeans_3=Categorical("check_global_score", ["default", "mdl"], default="default")
subkmeans_4=Integer("n_init ", (1, 10), default=1)
subkmeans_space.add([subkmeans_1, subkmeans_2, subkmeans_3, subkmeans_4])
configspaces["subkmeans"] = subkmeans_space

xmeans_space = ConfigurationSpace()
xmeans_1=Integer("n_clusters_init", (2, 20), default=2)
xmeans_2=Categorical("check_global_score", [True, False], default=True)
xmeans_3=Categorical("allow_merging", [True, False], default=False)
xmeans_4=Integer("n_split_trials", (2, 50), default=10)
xmeans_space.add([xmeans_1, xmeans_2, xmeans_3, xmeans_4])
xmeans_forbidden_clause_1 = ForbiddenEqualsClause(xmeans_space["allow_merging"], True)
xmeans_forbidden_clause_2 = ForbiddenEqualsClause(xmeans_space["check_global_score"], True)
xmeans_forbidden_clause = ForbiddenAndConjunction(xmeans_forbidden_clause_1, xmeans_forbidden_clause_2)
xmeans_space.add(xmeans_forbidden_clause)
configspaces["xmeans"] = xmeans_space

projdipmeans_space = ConfigurationSpace()
projdipmeans_1=Float("significance", (0.0001, 0.01), log=True, default=0.001)
projdipmeans_2=Integer("n_random_projections", (0, 5), default=0)
projdipmeans_3=Integer("n_split_trials", (2, 50), default=10)
projdipmeans_4=Categorical("allow_merging", ['table', 'function', 'bootstrap'], default='table')
projdipmeans_space.add([projdipmeans_1,projdipmeans_2,projdipmeans_3,projdipmeans_4])
configspaces["projdipmeans"] = projdipmeans_space

spectral_space = ConfigurationSpace()
spectral_1=Categorical("affinity", ['rbf', 'nearest_neighbors'], default='rbf')
spectral_2=Float("gamma", (0,5), default=1.0)
spectral_3=Integer("n_neighbors", (2,100), default=10)
spectral_space.add([spectral_1,spectral_2,spectral_3])
spectraL_cond_1 = EqualsCondition(spectral_space['gamma'], spectral_space['affinity'], 'rbf')
spectral_space.add(spectraL_cond_1)
spectraL_cond_2 = EqualsCondition(spectral_space['n_neighbors'], spectral_space['affinity'], 'nearest_neighbors')
spectral_space.add(spectraL_cond_2)
configspaces["spectral"] = spectral_space

scar_space = ConfigurationSpace()
scar_1=Categorical("normalize", [0,1], default=0)
scar_2=Categorical("weighted", [0,1], default=0)
scar_3=Float("alpha", (0,1), default=0.5)
scar_4=Integer("nn", (2,100), default=32)
scar_5=Integer("theta", (1,1000), default=50)
scar_6=Float("m", (0,1), default=0.5)
scar_7=Categorical("laplacian", [0,1,2], default=0)
scar_space.add([scar_1,scar_2,scar_3,scar_4,scar_5,scar_6,scar_7])
configspaces["scar"] = scar_space

spectacl_space = ConfigurationSpace()
spectacl_1=Float("epsilon", (0,2), default=1)
spectacl_2=Categorical("normalize_adjacency", [0,1], default=0)
spectacl_space.add([spectacl_1,spectacl_2])
configspaces["spectacl"] = spectacl_space

dbscan_space = ConfigurationSpace()
dbscan_1=Float("eps", (0,0.5), default=0.5)
dbscan_2=Integer("min_samples", (1,100), default=5)
dbscan_space.add([dbscan_1,dbscan_2])
configspaces["dbscan"] = dbscan_space

hdbscan_space = ConfigurationSpace()
hdbscan_1=Float("cluster_selection_epsilon", (0,0.1), default=0.0)
hdbscan_2=Integer("min_cluster_size", (1,100), default=5)
hdbscan_3=Categorical("allow_single_cluster", [0,1], default=0)
hdbscan_4=Categorical("cluster_selection_method", ["eom", "leaf"], default="eom")
hdbscan_5=Float("alpha", (0,1), default=1)
hdbscan_space.add([hdbscan_1,hdbscan_2,hdbscan_3,hdbscan_4,hdbscan_5])
configspaces["hdbscan"] = hdbscan_space

rnndbscan_space = ConfigurationSpace()
rnndbscan_1=Integer("n_neighbors", (2,100), default=5)
rnndbscan_space.add([rnndbscan_1])
configspaces["rnndbscan"] = rnndbscan_space

mdbscan_space = ConfigurationSpace()
mdbscan_1=Integer("n_neighbors", (2,100), default=5)
mdbscan_2=Float("eps", (0,0.5), default=0.5)
mdbscan_3=Integer("min_samples", (1,100), default=5)
mdbscan_space.add([mdbscan_1,mdbscan_2,mdbscan_3])
configspaces["mdbscan"] = mdbscan_space

dpca_space = ConfigurationSpace()
dpca_1=Float("dc", (-0.1,0.5), default=-0.1)
dpca_2=Float("density_threshold", (-0.1,0.5), default=-0.1)
dpca_3=Float("distance_threshold", (-0.1,0.5), default=-0.1)
dpca_4=Categorical("gauss_cutoff", [0,1], default=1)
dpca_5=Categorical("anormal", [0,1], default=1)
dpca_6=Categorical("distance_metric", ['euclidean', "cosine"], default="euclidean")
dpca_space.add([dpca_1,dpca_2,dpca_3,dpca_4,dpca_5,dpca_6])
configspaces["dpca"] = dpca_space

snndpc_space = ConfigurationSpace()
snndpc_1=Integer("n_neighbors", (2,100), default=5)
snndpc_space.add([snndpc_1])
configspaces["snndpc"] = snndpc_space

dbhd_space = ConfigurationSpace()
dbhd_1=Integer("min_cluster_size", (2,100), default=5)
dbhd_2=Float("beta", (0,1), default=0.1)
dbhd_3=Float("rho", (0,5), default=1.2)
dbhd_space.add([dbhd_1,dbhd_2,dbhd_3])
configspaces["dbhd"] = dbhd_space

trainmethods = {}
trainmethods["clustream"] = train_clustream
trainmethods["wclustream"] = train_wclustream
trainmethods["scaledclustream"] = train_scaledclustream
trainmethods["scope_full"] = train_scope_full
trainmethods["scope"] = train_scope


# 86400 * 5
def run_parameter_estimation(method, offline_method, time_budget, seed):
	print("Time Budget:", time_budget)
	scenario = Scenario(configspaces[offline_method], deterministic=False, use_default_config=True, walltime_limit=time_budget,
	                    n_trials=100000, seed=seed, name=f"{data_name}_{method}_{offline_method}_{time_budget}_{seed}")
	smac = HyperparameterOptimizationFacade(scenario, trainmethods[method], overwrite=True)
	incumbent = smac.optimize()
	run_num = len(smac.runhistory.items())
	return incumbent, 2 - smac.runhistory.get_min_cost(incumbent), run_num


def import_mcs(data_name, full_data=None):

	mcs = np.load(f"./param_data/mcs_{data_name}.npy", allow_pickle=True)
	#print(mcs.shape)
	steps = np.unique(mcs[:, 0])
	#print(steps)
	assign = np.load(f"./param_data/assign_{data_name}.npy")
	if full_data is None:
		data_x, data_y = load_data(f"{data_name}", 0)
	else:
		data_x, data_y = load_data(f"{full_data}", 0)
	#print(data_x.shape, data_y.shape)
	#print(assign.shape)
	mindex = 0
	uniques = np.unique(data_y, return_counts=False)

	data_x_dict = {}
	data_y_dict = {}
	mcs_dict = {}
	assign_dict = {}

	i = 0

	for step in sorted(steps):
		data_x_step = data_x[mindex:step + 1]
		data_y_step = data_y[mindex:step + 1]

		assign_step = assign[mindex:step + 1]

		mcs_step = mcs[mcs[:, 0] == step]
		data_x_dict[i] = data_x_step
		data_y_dict[i] = data_y_step
		mcs_dict[i] = mcs_step
		assign_dict[i] = assign_step
		mindex = step + 1
		i += 1

	#print(mcs_dict)
	#print(assign_dict)

	return mcs_dict, assign_dict, data_x_dict, data_y_dict, uniques


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--ds', default="densired10", type=str, help='Used stream data set')
	parser.add_argument('--method', default="clustream", type=str, help='Stream Clustering Method')
	parser.add_argument('--offline', default="kmeans", type=str, help='Offline Clustering Method')
	parser.add_argument('--use_full', default=1, type=int, help='Use full datset')

	args = parser.parse_args()
	print(args, flush=True)

	method = args.method
	offline_method = args.offline
	dataset = args.ds
	args.use_full = args.use_full == 1

	clusterer = offline_method

	#if method not in clustream_methods and method not in offline_methods:
	#	time_budget = 18000 # 5 hours
	#else:
	#	time_budget = 3600

	if not os.path.exists("param_logs"):
		os.mkdir("param_logs")
	if args.use_full:
		seed_num = 2
		time_budget = 86400
		f = open(f'param_logs/params_{dataset}_{method}_{offline_method}_full.txt', 'w', buffering=1)
	else:
		seed_num = 6
		time_budget = 18000
		f = open(f'param_logs/params_{dataset}_{method}_{offline_method}.txt', 'w', buffering=1)
	time_budget = 4*time_budget/5
	f.write(f"{configspaces[offline_method].get_default_configuration().get_dictionary()};-;-;-;-\n")

	data_name = dataset
	param_dicts = utils.load_parameters(args.ds, "clustream", args.use_full)

	for run in range(seed_num):
		param_dict = param_dicts[5*run]
		if run == 0:
			run = "default"
		else:
			run = run-1
		run_index = run

		cur_best_score = -1
		best_performance = -1
		if not args.use_full:
			data_name = dataset + "_subset_" + str(run)
			mcs, assignment, X, y, y_uniques = import_mcs(data_name)
		else:
			data_name = dataset + "_" + str(run)
			mcs, assignment, X, y, y_uniques = import_mcs(data_name, dataset)



		data_dim = len(X[0])
		data_length = len(y)
		class_num = len(y_uniques)
		dps = X
		labels = y

		best_params, score, run_num = run_parameter_estimation(method, offline_method, time_budget, 0)
		f.write(f"{best_params.get_dictionary()};{score};{run_num};{cur_best_score};{best_performance}\n")
	f.close()
