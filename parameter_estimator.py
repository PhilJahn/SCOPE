import argparse
import copy

from ConfigSpace.hyperparameters import FloatHyperparameter
from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, Categorical, Constant, \
	ForbiddenGreaterThanRelation, ForbiddenLessThanRelation, ForbiddenValueError
import numpy as np
import matplotlib.pyplot as plt
from competitors.clustream import CluStream
from competitors.EmCStream import EmcStream
from competitors.MCMSTStream import MCMSTStream
from competitors.MuDi import MuDiDataPoint, MudiHandler
from competitors.dbstream import DBSTREAM
from competitors.denstream import DenStream
from competitors.dstream import DStreamClusterer
from competitors.gbfuzzystream.MBStream import MBStreamHandler

from competitors.full_dataset_learner import full_dataset_leaner
from competitors.streamkmeans import STREAMKMeans
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


def get_clustering_learn_one(clustering_method, nooffline=False):
	dp_store = []
	prediction = []
	i = 0
	is_clustream = type(clustering_method) == CluStream
	is_dstream = type(clustering_method) == DStreamClusterer
	assignments = []
	mcs = []
	cur_prediction = []
	for dp in dps:
		is_last = i == len(X) - 1
		if not is_dstream:
			dp = dict(enumerate(dp))
		dp_store.append(dp)
		clustering_method.learn_one(dp)
		if not is_clustream and not is_dstream:
			label = clustering_method.predict_one(dp)
			prediction.append(label)
		else:
			if (i + 1) % offline_timing == 0 or is_last:
				if is_clustream:
					for mcid, mc in clustering_method.micro_clusters.items():
						mcs.append([i, mcid, mc.center, mc.radius(r_factor=1), mc.weight, mc.var_time, mc.var_x])
				if is_dstream:
					clustering_method.offline_processing()
				cur_prediction = []
				for dp_2 in dp_store:
					if is_clustream:
						label, assign = clustering_method.predict_one(dp_2, recluster=True, sklearn=True,return_mc=True)
						assignments.append(assign)
					else:
						label = clustering_method.predict_one(dp_2)
					if nooffline:
						cur_prediction.append(assign)
					else:
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

	global cur_best_score
	global best_performance
	if ami + ari > cur_best_score:
		cur_best_score = ami + ari
		best_performance = {"accuracy":acc, "ARI": ari, "AMI": ami}
		#if is_clustream:
		#	np.save(f"param_data/assign_{data_name}", assignments)
		#	np.save(f"param_data/mcs_{data_name}", mcs)

	return ami + ari


def get_clustering(method, config=None):
	is_emcstream = type(method) == EmcStream
	is_mudistream = type(method) == MudiHandler
	is_gbstream = type(method) == MBStreamHandler

	dps_np = np.array(dps)

	amis = []
	aris = []
	accs = []
	y_store = copy.copy(y)
	if is_emcstream:
		method.add_label_store(y_store)
	pred_store = method.learn(dps_np)
	if is_emcstream:
		clustered_X, clustered_y = method.get_clustered_data()
		y_store = clustered_y
	elif is_mudistream:
		clusters = pred_store.copy()
		pred_store = [-1] * len(X)
		for clu in clusters:
			cmcs = clu._cmcList
			for cmc in cmcs:
				for point in cmc.storage:
					pred_store[point.t] = clu.name
	if is_gbstream:
		batchsize = config["batchsize"]
		for i in range(0, len(pred_store), batchsize):
			end_batch = min(i + batchsize, len(pred_store))
			pred_batch = pred_store[i:end_batch]
			y_batch = y[i:end_batch]
			length = end_batch - i
			metrics, _ = getMetrics(pred_batch, y_batch)
			amis.extend([metrics["AMI"]] * length)
			aris.extend([metrics["ARI"]] * length)
			accs.extend([metrics["accuracy"]] * length)
	else:
		metrics, _ = getMetrics(y_store, pred_store)
		amis.extend([metrics["AMI"]] * len(y_store))
		aris.extend([metrics["ARI"]] * len(y_store))
		accs.extend([metrics["accuracy"]] * len(y_store))

	ami = float(np.mean(amis))
	ari = float(np.mean(aris))
	acc = float(np.mean(accs))

	global cur_best_score
	global best_performance
	if ami + ari > cur_best_score:
		cur_best_score = ami + ari
		best_performance = {"accuracy":acc, "ARI": ari, "AMI": ami}

	return ami + ari


def eval_clustering(clustering):
	amis = []
	aris = []
	accs = []
	for i in range(0, len(clustering), offline_timing):
		end = min(len(clustering), i + offline_timing)
		length = end - i
		# print(i, end, length)
		metrics, _ = getMetrics(y[i:end], clustering[i:end])
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
		best_performance = {"accuracy":acc, "ARI": ari, "AMI": ami}

	return ami + ari


def train_clustream(config: Configuration, seed: int = 0) -> float:
	# print(data_dim, class_num, data_length, mc_num, offline_timing)
	#global offline_timing
	#offline_timing = config["time_window"]
	clustering_method = CluStream(n_macro_clusters=class_num, seed=seed, max_micro_clusters=mc_num, time_gap=100000000,
	                              micro_cluster_r_factor=config["mc_r_factor"], time_window=config["time_window"])
	score = get_clustering_learn_one(clustering_method)

	print("CluStream", config["mc_r_factor"], config["time_window"], score)
	return 2 - score

def train_clustream_no_offline(config: Configuration, seed: int = 0) -> float:
	# print(data_dim, class_num, data_length, mc_num, offline_timing)
	#global offline_timing
	#offline_timing = config["time_window"]
	clustering_method = CluStream(n_macro_clusters=1, seed=seed, max_micro_clusters=config["mmc"], time_gap=100000000,
	                              micro_cluster_r_factor=config["mc_r_factor"], time_window=config["time_window"])
	score = get_clustering_learn_one(clustering_method, nooffline=True)

	print("CluStream", config["mc_r_factor"], config["time_window"], config["mmc"], score)
	return 2 - score

def train_clustream_no_offline_fixed(config: Configuration, seed: int = 0) -> float:
	# print(data_dim, class_num, data_length, mc_num, offline_timing)
	#global offline_timing
	#offline_timing = config["time_window"]
	config["mmc"] = class_num
	clustering_method = CluStream(n_macro_clusters=1, seed=seed, max_micro_clusters=class_num, time_gap=100000000,
	                              micro_cluster_r_factor=config["mc_r_factor"], time_window=config["time_window"])
	score = get_clustering_learn_one(clustering_method, nooffline=True)

	print("CluStream", config["mc_r_factor"], config["time_window"], class_num, score)
	return 2 - score

def train_dbstream(config: Configuration, seed: int = 0) -> float:
	clustering_method = DBSTREAM(clustering_threshold=config["clustering_threshold"],
	                             fading_factor=config["fading_factor"],
	                             cleanup_interval=config["cleanup_interval"],
	                             intersection_factor=config["intersection_factor"],
	                             minimum_weight=config["minimum_weight"])
	score = get_clustering_learn_one(clustering_method)
	print("DBStream", config["clustering_threshold"], config["fading_factor"], config["cleanup_interval"],
	      config["intersection_factor"], config["minimum_weight"], score)
	return 2 - score


def train_denstream(config: Configuration, seed: int = 0) -> float:
	if config["mu"] > 1 / config["beta"]:
		clustering_method = DenStream(decaying_factor=config["decaying_factor"], beta=config["beta"],
		                              mu=config["mu"], epsilon=config["epsilon"],
		                              n_samples_init=config["n_samples_init"],
		                              stream_speed=config["stream_speed"])
		score = get_clustering_learn_one(clustering_method)
	else:
		score = -np.inf
	print("Denstream", config["decaying_factor"], config["beta"],
	      config["mu"], config["epsilon"],
	      config["n_samples_init"],
	      config["stream_speed"], score)
	return 2 - score


def train_streamkmeans(config: Configuration, seed: int = 0) -> float:
	# print(data_dim, class_num, data_length, mc_num, offline_timing)

	clustering_method = STREAMKMeans(n_clusters=class_num, seed=seed,
	                                 chunk_size=config["chunk_size"], sigma=config["sigma"], mu=config["mu"])
	score = get_clustering_learn_one(clustering_method)

	print("STREAMKMeans", config["chunk_size"], config["sigma"], config["mu"], score)
	return 2 - score


def train_emcstream(config: Configuration, seed: int = 0) -> float:
	# print(data_dim, class_num, data_length, mc_num, offline_timing)
	clustering_method = EmcStream(k=class_num, seed=seed, horizon=config["horizon"],
	                              ari_threshold=config["ari_threshold"],
	                              ari_threshold_step=config["ari_threshold_step"])
	try:
		score = get_clustering(clustering_method)
	except:
		score = -np.inf
	print("EmCStream", config["horizon"], config["ari_threshold"], config["ari_threshold_step"], score)
	return 2 - score


def train_mcmststream(config: Configuration, seed: int = 0) -> float:
	clustering_method = MCMSTStream(N=config["N"], W=config["W"], r=config["r"], n_micro=config["n_micro"],
	                                d=data_dim)
	score = get_clustering(clustering_method)
	print("MCMSTStream", config["N"], config["W"], config["r"], config["n_micro"], score)
	return 2 - score


def train_mudistream(config: Configuration, seed: int = 0) -> float:
	mini = 0
	maxi = 1
	if config["alpha"] + 2 ** (-config["lamda"]) > 1:
		clustering_method = MudiHandler(mini=mini, maxi=maxi,
		                                dimension=data_dim,
		                                lamda=config["lamda"],
		                                gridGranuality=config["gridgran"],
		                                alpha=config["alpha"],
		                                minPts=config["minPts"],
		                                seed=seed)

		score = get_clustering(clustering_method)
	else:
		score = -np.inf
	print("MudiStream", config["lamda"], config["gridgran"], config["alpha"], config["minPts"], score)
	return 2 - score


def train_dstream(config: Configuration, seed: int = 0) -> float:
	domains_per_dimension = [(0, 1)] * data_dim
	partitions_per_dimension = [config["partitions_count"]] * data_dim
	clustering_method = DStreamClusterer(initial_cluster_count=class_num, seed=seed,
	                                     dense_threshold_parameter=config["dense_threshold_parameter"],
	                                     sparse_threshold_parameter=config["sparse_threshold_parameter"],
	                                     sporadic_threshold_parameter=config["sporadic_threshold_parameter"],
	                                     decay_factor=config["decay_factor"],
	                                     gap=config["gap"],
	                                     domains_per_dimension=domains_per_dimension,
	                                     partitions_per_dimension=partitions_per_dimension,
	                                     dimensions=data_dim)
	score = get_clustering_learn_one(clustering_method)
	print("DStream", config["partitions_count"], config["dense_threshold_parameter"],
	      config["sparse_threshold_parameter"],
	      config["sporadic_threshold_parameter"], config["decay_factor"],
	      config["gap"], score)
	return 2 - score

def train_gbfuzzystream(config: Configuration, seed: int = 0) -> float:
	clustering_method = MBStreamHandler(lam=config["lam"],
			                         batchsize=config["batchsize"],
			                         threshold=config["threshold"],
			                         m=config["m"],
			                         eps=config["eps"])
	score = get_clustering(clustering_method, config)
	print("MCMSTStream", config["lam"],	config["batchsize"], config["threshold"], config["m"], config["eps"], score)
	return 2 - score


configspaces = {}
clustream_space = ConfigurationSpace()
clustream_timewindow = Categorical("time_window",
                                   [1000, 1500, 2000, 2500, 5000, 10000],
                                   default=1000)
clustream_mc_r_factor = Float("mc_r_factor", (1.0, 5.0), default=2.0)
clustream_space.add([clustream_timewindow, clustream_mc_r_factor])
configspaces["clustream"] = clustream_space

clustream_no_offline_space = ConfigurationSpace()
clustream_no_offline_space_mmc = Integer("mmc", (1, 100), default=100)
clustream_no_offline_space.add([clustream_timewindow, clustream_mc_r_factor, clustream_no_offline_space_mmc])
configspaces["clustream_no_offline"] = clustream_no_offline_space



denstream_space = ConfigurationSpace()
denstream_decaying_factor = Float("decaying_factor", (0.1, 1), default=0.25)
denstream_beta = Float("beta", (0, 1), default=0.75)
denstream_mu = Float("mu", (1, 100000), log=True, default=2)
denstream_epsilon = Float("epsilon", (0.001, 0.5), log=True, default=0.02)
denstream_nsamples = Categorical("n_samples_init", [5, 10, 25, 50, 75, 100, 250, 500, 750, 1000], default=1000)
denstream_speed = Categorical("stream_speed", [1, 10, 100], default=100)
denstream_space.add(
	[denstream_decaying_factor, denstream_beta, denstream_mu, denstream_epsilon, denstream_nsamples, denstream_speed])
# denstream_mu_beta = ForbiddenLessThanRelation(denstream_space["mu"], FloatHyperparameter(1/denstream_space["beta"]))
# denstream_space.add(denstream_mu_beta)
configspaces["denstream"] = denstream_space

dbstream_space = ConfigurationSpace()
dbstream_clustering_threshold = Float("clustering_threshold", (0.05, 1), default=1.0)
dbstream_fading_factor = Float("fading_factor", (0.005, 0.015), default=0.01, log=True)
dbstream_cleanup_interval = Categorical("cleanup_interval", [2, 5, 10, 100, 1000], default=2)
dbstream_intersection_factor = Float("intersection_factor", (0.1, 0.5), default=0.3)
dbstream_minimum_weight = Float("minimum_weight", (1, 5), default=1)
dbstream_space.add([dbstream_clustering_threshold, dbstream_fading_factor, dbstream_cleanup_interval,
                    dbstream_intersection_factor, dbstream_minimum_weight])
configspaces["dbstream"] = dbstream_space

streamkmeans_space = ConfigurationSpace()
streamkmeans_chunk = Integer("chunk_size", (10, 1000), default=10)
streamkmeans_sigma = Float("sigma", (0, 1), default=0.5)
streamkmeans_mu = Float("mu", (0, 1), default=0.5)
streamkmeans_space.add([streamkmeans_chunk, streamkmeans_sigma, streamkmeans_mu])
configspaces["streamkmeans"] = streamkmeans_space

emcstream_space = ConfigurationSpace()
emcstream_horizon = Integer("horizon", (10, 1000), default=100)
emcstream_ari_threshold = Float("ari_threshold", (0.5, 1), default=1.0)
# emcstream_ari_threshold = Constant("ari_threshold", 1.0)
emcstream_ari_threshold_step = Float("ari_threshold_step", (0.0001, 0.01), default=0.001, log=True)
# emcstream_ari_threshold_step = Constant("ari_threshold_step", 0.001)
emcstream_space.add([emcstream_horizon, emcstream_ari_threshold, emcstream_ari_threshold_step])
configspaces["emcstream"] = emcstream_space

mcmststream_space = ConfigurationSpace()
mcmststream_W = Integer("W", (100, 2000), default=235)
mcmststream_N = Integer("N", (2, 15), default=5)
mcmststream_r = Float("r", (0.001, 0.25), default=0.033, log=True)
mcmststream_n_micro = Integer("n_micro", (2, 25), default=2)
mcmststream_space.add([mcmststream_W, mcmststream_N, mcmststream_r, mcmststream_n_micro])
configspaces["mcmststream"] = mcmststream_space

mudistream_space = ConfigurationSpace()
mudistream_lamda = Float("lamda", (0.03, 32), log=True, default=0.5)
mudistream_gridgran = Integer("gridgran", (2, 40), default=32)
mudistream_alpha = Float("alpha", (0, 1), default=0.5)
mudistream_minpts = Categorical("minPts", [2, 3, 5, 7, 10, 50, 100, 500, 1000], default=3)
mudistream_space.add([mudistream_lamda, mudistream_gridgran, mudistream_alpha, mudistream_minpts])
configspaces["mudistream"] = mudistream_space

dstream_space = ConfigurationSpace()
dstream_dense_threshold = Float("dense_threshold_parameter", (0.1, 5), default=3)
dstream_sparse_threshold = Float("sparse_threshold_parameter", (0.01, 1), default=0.8)
dstream_sporadic_threshold = Float("sporadic_threshold_parameter", (0.001, 0.5), default=0.3)
dstream_decay_factor = Float("decay_factor", (0.5, 1), default=0.998)
dstream_partitions_count = Integer("partitions_count", (2, 40), default=5)
dsteam_gap = Categorical("gap", [0, 10, 100, 1000], default=100)
dstream_space.add([dstream_dense_threshold, dstream_sparse_threshold, dstream_sporadic_threshold, dstream_decay_factor,
                   dstream_partitions_count, dsteam_gap])
denstream_order1 = ForbiddenLessThanRelation(dstream_space["dense_threshold_parameter"],
                                             dstream_space["sparse_threshold_parameter"])
dstream_space.add(denstream_order1)
denstream_order2 = ForbiddenLessThanRelation(dstream_space["sparse_threshold_parameter"],
                                             dstream_space["sporadic_threshold_parameter"])
dstream_space.add(denstream_order2)
configspaces["dstream"] = dstream_space

gbfuzzystream_space = ConfigurationSpace()
gbfuzzystream_lam = Float("lam",(0.1, 5), default=1)
gbfuzzystream_batchsize=Categorical("batchsize",[1000], default=1000)
gbfuzzystream_threshold=Float("threshold",(0.1, 0.8),default=0.3)
gbfuzzystream_m=Constant("m", 2)
gbfuzzystream_eps=Constant("eps", 10)
gbfuzzystream_space.add([gbfuzzystream_lam, gbfuzzystream_batchsize, gbfuzzystream_threshold, gbfuzzystream_m, gbfuzzystream_eps])
configspaces["gbfuzzystream"] = gbfuzzystream_space

trainmethods = {}
trainmethods["clustream"] = train_clustream
trainmethods["clustream_no_offline"] = train_clustream_no_offline
trainmethods["clustream_no_offline_fixed"] = train_clustream_no_offline_fixed
trainmethods["dbstream"] = train_dbstream
trainmethods["denstream"] = train_denstream
trainmethods["streamkmeans"] = train_streamkmeans
trainmethods["emcstream"] = train_emcstream
trainmethods["mcmststream"] = train_mcmststream
trainmethods["mudistream"] = train_mudistream
trainmethods["dstream"] = train_dstream
trainmethods["gbfuzzystream"] = train_gbfuzzystream

# 86400 * 5
def run_parameter_estimation(method, time_budget, seed):
	print("Checksum:", np.sum(labels), " Time Budget:", time_budget)
	det = not method in ["dstream", "clustream", "mudistream", "emcstream", "streamkmeans"]
	scenario = Scenario(configspaces[method], deterministic=det, use_default_config=True, walltime_limit=time_budget,
	                    n_trials=100000, seed=seed, name=f"{data_name}_{method}_{time_budget}_{seed}")
	smac = HyperparameterOptimizationFacade(scenario, trainmethods[method], overwrite=True)
	incumbent = smac.optimize()
	run_num = len(smac.runhistory.items())
	return incumbent, 2 - smac.runhistory.get_min_cost(incumbent), run_num

clustream_methods = {"clustream", "wclustream", "opeclustream", "scope"}
offline_methods = {}
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--ds', default="powersupply", type=str, help='Used stream data set')
	parser.add_argument('--method', default="clustream_no_offline_fixed", type=str, help='Stream Clustering Method')
	parser.add_argument('--use_full', default=0, type=int, help='Use full datset')
	args = parser.parse_args()
	print(args, flush=True)

	method = args.method
	dataset = args.ds
	args.use_full = args.use_full == 1

	#if method not in clustream_methods and method not in offline_methods:
	#	time_budget = 18000 # 5 hours
	#else:
	#	time_budget = 3600


	if args.use_full:
		seed_num = 1
		time_budget = 86400
		f = open(f'param_logs/params_{dataset}_{method}_full.txt', 'w', buffering=1)
	else:
		seed_num = 5
		time_budget = 18000
		f = open(f'param_logs/params_{dataset}_{method}.txt', 'w', buffering=1)

	if args.method in clustream_methods:
		time_budget = time_budget/5
	elif args.method in offline_methods:
		time_budget = 4*time_budget/5
	if not method == "clustream_no_offline_fixed":
		f.write(f"{configspaces[method].get_default_configuration().get_dictionary()};-;-;-;-\n")

	data_name = dataset
	for run in range(seed_num):
		run_index = run

		cur_best_score = -1
		best_performance = -1
		if not args.use_full:
			data_name = dataset + "_subset_" + str(run)
		X, y = load_data(data_name)
		uniques = np.unique(y, return_counts=False)
		data_dim = len(X[0])
		data_length = len(y)
		class_num = len(uniques)
		if method == "clustream_no_offline_fixed" and run == 0:
			clustream_no_offline_fixed_space = ConfigurationSpace()
			clustream_no_offline_fixed_mmc = Constant("mmc", class_num)
			clustream_no_offline_fixed_space.add(
				[clustream_timewindow, clustream_mc_r_factor, clustream_no_offline_fixed_mmc])
			configspaces["clustream_no_offline_fixed"] = clustream_no_offline_fixed_space
			f.write(f"{configspaces[method].get_default_configuration().get_dictionary()};-;-;-;-\n")

		if method == "mudistream":
			dps = []
			for i in range(len(X)):
				dps.append([MuDiDataPoint(X[i], i)])
		else:
			dps = X

		labels = y
		best_params, score, run_num = run_parameter_estimation(method, time_budget, 0)
		f.write(f"{best_params.get_dictionary()};{score};{run_num};{cur_best_score};{best_performance}\n")
	f.close()
