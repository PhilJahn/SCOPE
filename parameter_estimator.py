import copy

from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, Categorical, Constant
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
global batchsize
batchsize = 1000


def get_clustering_learn_one(clustering_method):
    dp_store = []
    prediction = []
    i = 0
    is_clustream = type(clustering_method) == CluStream
    for dp in dps:
        is_last = i == len(X) - 1
        dp = dict(enumerate(dp))
        dp_store.append(dp)
        clustering_method.learn_one(dp)
        if (i + 1) % offline_timing == 0 or is_last:
            cur_prediction = []
            for dp_2 in dp_store:
                if is_clustream:
                    label = clustering_method.predict_one(dp_2, recluster=True, sklearn=True)
                else:
                    label = clustering_method.predict_one(dp_2)
                cur_prediction.append(label)

#            plt.figure(figsize=(10, 10))
#            plt.scatter(dps_to_np(dp_store)[:, 0], dps_to_np(dp_store)[:, 1], c=cur_prediction)
#            plt.ylim(-0.1, 1.1)
#            plt.xlim(-0.1, 1.1)
#            plt.show()
            dp_store = []
            prediction.extend(cur_prediction)
        i += 1
    nmis = []
    aris = []
    accs = []
    for i in range(0, len(prediction), offline_timing):
        end = min(len(prediction), i + offline_timing)
        length = end - i
        # print(i, end, length)
        metrics, _ = getMetrics(y[i:end], prediction[i:end])
        nmis.extend([metrics["NMI"]] * length)
        aris.extend([metrics["ARI"]] * length)
        accs.extend([metrics["accuracy"]] * length)
    # print(len(nmis))
    nmi = np.mean(nmis)
    ari = np.mean(aris)
    acc = np.mean(accs)
    return nmi + ari + acc


def get_clustering(method):
    is_emcstream = type(method) == EmcStream
    is_mudistream = type(method) == MudiHandler
    is_gbstream = type(method) == MBStreamHandler

    dps_np = np.array(dps)

    nmis = []
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
        for i in range(0, len(pred_store), batchsize):
            end_batch = min(i + batchsize, len(pred_store))
            pred_batch = pred_store[i:end_batch]
            y_batch = y[i:end_batch]
            length = end_batch - i
            metrics, _ = getMetrics(pred_batch, y_batch)
            nmis.extend([metrics["NMI"]] * length)
            aris.extend([metrics["ARI"]] * length)
            accs.extend([metrics["accuracy"]] * length)

    else:
        metrics, _ = getMetrics(y_store, pred_store)
        nmis.extend([metrics["NMI"]] * len(y_store))
        aris.extend([metrics["ARI"]] * len(y_store))
        accs.extend([metrics["accuracy"]] * len(y_store))

    nmi = np.mean(nmis)
    ari = np.mean(aris)
    acc = np.mean(accs)
    return nmi + ari + acc

def eval_clustering(clustering):
    nmis = []
    aris = []
    accs = []
    for i in range(0, len(clustering), offline_timing):
        end = min(len(clustering), i+offline_timing)
        length = end - i
        #print(i, end, length)
        metrics, _ = getMetrics(y[i:end], clustering[i:end])
        nmis.extend([metrics["NMI"]]*length)
        aris.extend([metrics["ARI"]]*length)
        accs.extend([metrics["accuracy"]]*length)
    #print(len(nmis))
    nmi = np.mean(nmis)
    ari = np.mean(aris)
    acc = np.mean(accs)
    return nmi + ari + acc


def train_clustream(config: Configuration, seed: int = 0) -> float:
    #print(data_dim, class_num, data_length, mc_num, offline_timing)
    clustering_method = CluStream(n_macro_clusters=class_num, seed=seed, max_micro_clusters=mc_num, time_gap=100000000,
                                  micro_cluster_r_factor=config["mc_r_factor"], time_window=config["time_window"])
    score = get_clustering_learn_one(clustering_method)
    print("CluStream", config["mc_r_factor"], config["time_window"], score)
    return 3 - score

def train_emcstream(config: Configuration, seed: int = 0) -> float:
    #print(data_dim, class_num, data_length, mc_num, offline_timing)
    clustering_method = EmcStream(k=class_num, seed=seed, horizon=config["horizon"],
                                  ari_threshold=config["ari_threshold"], ari_threshold_step=config["ari_threshold_step"])
    try:
        score = get_clustering(clustering_method)
    except:
        score = -np.inf
    print("EmCStream", config["horizon"], config["ari_threshold"], config["ari_threshold_step"], score)
    return 3 - score

def train_mcmststream(config: Configuration, seed: int = 0) -> float:
    clustering_method = MCMSTStream(N=config["N"], W=config["W"], r=config["r"], n_micro=config["n_micro"],
			                     d=data_dim)
    score = get_clustering(clustering_method)
    print("MCMSTStream", config["N"], config["W"], config["r"], config["n_micro"], score)
    return 3 - score

configspaces = {}
clustream_space = ConfigurationSpace()
clustream_timewindow = Categorical("time_window", [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7500, 10000], default=1000)
clustream_mc_r_factor = Float("mc_r_factor", (1.0, 5.0), default=2.0)
clustream_space.add([clustream_timewindow, clustream_mc_r_factor])
configspaces["clustream"] = clustream_space

denstream_space = ConfigurationSpace()
denstream_decaying_factor = Float("decaying_factor", (0.1, 1), default= 0.25)

dbstream_space = ConfigurationSpace()
dbstream_clustering_threshold = Float("clustering_threshold",())
dbstream_fading_factor = Float("fading_factor",())



emcstream_space = ConfigurationSpace()
emcstream_horizon = Integer("horizon", (10, 1000), default=100)
emcstream_ari_threshold = Float("ari_threshold", (0.5, 1), default=1.0)
emcstream_ari_threshold = Constant("ari_threshold", 1.0)
emcstream_ari_threshold_step = Float("ari_threshold_step", (0.0001, 0.01), default=0.001)
emcstream_ari_threshold_step = Constant("ari_threshold_step", 0.001)
emcstream_space.add([emcstream_horizon, emcstream_ari_threshold, emcstream_ari_threshold_step])
configspaces["emcstream"] = emcstream_space

mcmststream_space = ConfigurationSpace()
mcmststream_W = Integer("W", (100, 2000), default=1000)
mcmststream_N = Integer("N", (2,15), default=5)
mcmststream_r = Float("r", (0.001, 0.25), default=0.01)
mcmststream_n_micro = Integer("n_micro", (2, 25), default = 10)
mcmststream_space.add([mcmststream_W, mcmststream_N, mcmststream_r, mcmststream_n_micro])
configspaces["mcmststream"] = mcmststream_space


trainmethods = {}
trainmethods["clustream"] = train_clustream
trainmethods["emcstream"] = train_emcstream
trainmethods["mcmststream"] = train_mcmststream
#86400 * 5
def run_parameter_estimation(method, dataset):
    scenario = Scenario(configspaces[method], deterministic=True, use_default_config=True, cputime_limit=120,
                        n_trials=100000)
    smac = HyperparameterOptimizationFacade(scenario, trainmethods[method], overwrite=True)
    incumbent = smac.optimize()
    #runhistory = smac.runhistory
    #print(runhistory.items())
    #for k, v in runhistory.items():
    #    print(k, v)
    #    config = runhistory.get_config(k.config_id)
    #    print(config)
    print(incumbent.get_dictionary)
    print(incumbent.values())
    return incumbent


if __name__ == '__main__':
    dataset = "complex9"
    method = "mcmststream"
    X, y = load_data(dataset)
    uniques = np.unique(y, return_counts=False)
    data_dim = len(X[0])
    data_length = len(y)
    class_num = len(uniques)
    dps = X
    labels = y

    print(run_parameter_estimation(method, dataset))


