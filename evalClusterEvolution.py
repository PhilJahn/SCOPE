import sys
import numpy as np
from os import listdir
from os.path import isfile, join

from cmm import get_cmm
from datahandler import load_data
from evaluate import getMetrics

import matplotlib.pyplot as plt

from temporalsilhouette.TSindex import tempsil

naming = {}
naming["groundtruth"] = "Ground Truth"
naming["streamkmeans"] = "STREAMKmeans"
naming["denstream"] = "DenStream"
naming["dbstream"] = "DBSTREAM"
naming["emcstream"] = "EMCStream"
naming["mcmststream"] = "MCMSTStream"
naming["gbfuzzystream"] = "GB-FuzzyStream"
naming["clustream_no_offline"] = "CluStream-O - var. $k$"
naming["clustream_no_offline_fixed"] = "CluStream-O - fixed $k$"
naming["clustream"] = "CluStream-C"
naming["clustream2"] = "CluStream"
naming["wclustream"] = "CluStream-W"
naming["scaledclustream"] = "CluStream-S"
naming["scope_full"] = "CluStream-G"
naming["nooffline"] = "-O - $k$=100"
naming["wkmeans"] = " - W$k$-Means"
naming["kmeans"] = " - $k$-Means"
naming["subkmeans"] = " - SubKMeans"
naming["xmeans"] = " - X-Means"
naming["projdipmeans"] = " - P-Dip-M"
naming["spectral"] = " - SC"
naming["scar"] = " - SCAR"
naming["spectacl"] = " - SpectACl"
naming["dbscan"] = " - DBSCAN"
naming["hdbscan"] = " - HDBSCAN"
naming["rnndbscan"] = " - RNN-DBS"
naming["mdbscan"] = " - MDBSCAN"
naming["dpca"] = " - DPC"
naming["snndpc"] = " - SNN-DPC"
naming["dbhd"] = " - DBHD"


def evaluate(online_method, offline_method, y_batches, pred_batches, x_batches):
    #print("Evaluating ", online_method, offline_method, y_batches, pred_batches, x_batches)

    x_all = []
    y_all = []
    pred_all = []
    #pred_all_2 = []

    cmm_score = 0
    min_clu = 2
    for b in range(len(y_batches)):
        y_batch = y_batches[b]
        pred_batch = pred_batches[b]
        x_batch = x_batches[b]
        #print(y_batch, pred_batch)

        mapping = {}
        mapping_overlap = {}
        rev_mapping = {}
        rev_mapping_overlap = {}
        for label in np.unique(y_batch):
            rev_mapping_overlap[label] = 0
        for clu in np.unique(pred_batch):
            clu_ids = np.where(pred_batch == clu)[0]
            #print(clu, clu_ids)
            max_overlap = 0
            for label in np.unique(y_batch):
                label_ids = np.where(y_batch == label)[0]
                overlap = len(np.intersect1d(clu_ids, label_ids))
                #print(label, clu, overlap)
                if overlap > max_overlap:
                    max_overlap = overlap
                    mapping[clu] = label
                    mapping_overlap[clu] = overlap
            label_max_overlap = rev_mapping_overlap[mapping[clu]]
            if label_max_overlap < mapping_overlap[clu]:
                rev_mapping[mapping[clu]] = clu
                rev_mapping_overlap[mapping[clu]] = mapping_overlap[clu]
        assignment = {}
        for clu in np.unique(pred_batch):
            if clu == -1:
                assignment[clu] = -1
            else:
                if clu == rev_mapping[mapping[clu]]:
                    assignment[clu] = mapping[clu]
                else:
                    min_clu +=1
                    assignment[clu] = min_clu

        #print(mapping, mapping_overlap, rev_mapping, rev_mapping_overlap, assignment)

        cmm_score_batch = get_cmm(x_batch, y_batch, pred_batch, [1]*len(x_batch), 5)
        #print("CMM Batch", cmm_score_batch)
        cmm_score += len(x_batch)*cmm_score_batch

        for pred in pred_batch:
            pred_all.append(assignment[pred])

        x_all.extend(x_batch)
        y_all.extend(y_batch)
        #pred_all_2.extend(pred_batch)

    cmm_score /= len(x_all)
    print("CMM", cmm_score)
    x_all = np.array(x_all)
    y_all = np.array(y_all)
    pred_all = np.array(pred_all)
    #pred_all_2 = np.array(pred_all_2)
    timesteps = np.arange(len(x_all))


    _, _, TS = tempsil(timesteps, x_all, pred_all, s=100, kn=1000, c=1)
    print("TS", TS)

    plt.rcParams.update({'font.size': 25})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,5), layout='constrained')
    color = plt.cm.viridis(np.linspace(0.2, 0.8, len(np.unique(pred_all))+1))
    color[0] = plt.cm.viridis(0)
    color[1] = plt.cm.viridis(0.99999999)
    color = np.append(color, [[0, 0, 0, 1]], axis=0)

    if offline_method != "wkmeans":
        title = naming[online_method]
    else:
        title = naming["clustream2"]
    if offline_method is not None:
        title += naming[offline_method]
    title += "\n"
    title += f"TS: {TS:.3f}, CMM: {cmm_score:.3f}"
    ax1.set_title(title)
    ax1.scatter(range(len(x_all)), x_all[:,1], label=pred_all, c=color[pred_all.astype('int32')], s=5)
    ax1.xaxis.set_visible(False)
    ax1.set_ylabel("Income")
    ax2.scatter(range(len(x_all)), x_all[:, 0], label=pred_all, c=color[pred_all.astype('int32')], s=5)
    ax2.set_ylabel("Fertility")
    ax2.set_xlabel("Timestamp")
    figname = f"clusterevolution_figures/evofig_{online_method}"
    if offline_method is not None:
        figname += f"_{offline_method}"
    fig.savefig(figname+".pdf", bbox_inches='tight', transparent=True)
    plt.close()



def main(args):
    setting = "1000_100_1000_False"
    dataset = "fert_vs_gdp"
    method_names = [#"groundtruth", "streamkmeans", "denstream", "dbstream",
                    #"emcstream",
                    #"mcmststream", #"gbfuzzystream",
                    #"clustream_no_offline",
                    #"clustream_no_offline_fixed",
                    "clustream",
                    #"wclustream",
                    #"scaledclustream",
                    #"scope_full"
                    ]
    offline_methods = ["nooffline", "wkmeans",
                    "kmeans",
                    "subkmeans",
                    "xmeans", "projdipmeans",
                    "spectral",# "scar",
	                "spectacl",
                    'dbscan','hdbscan', 'rnndbscan',# 'mdbscan',
	                'dpca', 'snndpc', 'dbhd'
                    ]

    X, y_store = load_data("fert_vs_gdp", seed=0)
    batchsize = 1000
    clu_label_dir = "preds_clueval/"

    for online_method in method_names:
        print("online_method:", online_method)
        if online_method == "groundtruth":
            pred_batches = []
            y_batches = []
            x_batches = []
            for i in range(0, len(y_store), batchsize):
                end_batch = min(i + batchsize, len(y_store))
                pred_batch = y_store[i:end_batch]
                x_batch = X[i:end_batch]
                y_batch = y_store[i:end_batch]
                pred_batches.append(pred_batch)
                y_batches.append(y_batch)
                x_batches.append(x_batch)
            evaluate(online_method, None, y_batches, pred_batches, x_batches)
        elif online_method in ["emcstream", "mcmststream"]:
            best_seed = -1
            best_score = -np.inf
            for seed in range(10):
                filepath = clu_label_dir + "preds_" + dataset + "_" + online_method + "_" + setting + "_" + str(seed) + ".npy"
                print(filepath)
                pred_store = np.load(filepath)
                batchsize = 1000
                score = 0
                for i in range(0, len(pred_store), batchsize):
                    end_batch = min(i + batchsize, len(pred_store))
                    pred_batch = pred_store[i:end_batch]
                    y_batch = y_store[i:end_batch]
                    metrics, cm = getMetrics(y_batch, pred_batch)
                    score += (end_batch-i) * (metrics['ARI'] + metrics['AMI'])
                score = score/len(pred_store)/2
                print(score)
                if score > best_score:
                    best_score = score
                    best_seed = seed
            print(best_seed, best_score)
            filepath = clu_label_dir + "preds_" + dataset + "_" + online_method + "_" + setting + "_" + str(best_seed) + ".npy"
            print(filepath)
            pred_store = np.load(filepath)
            pred_batches = []
            y_batches = []
            x_batches = []
            for i in range(0, len(pred_store), batchsize):
                end_batch = min(i + batchsize, len(pred_store))
                pred_batch = pred_store[i:end_batch]
                x_batch = X[i:end_batch]
                y_batch = y_store[i:end_batch]
                pred_batches.append(pred_batch)
                y_batches.append(y_batch)
                x_batches.append(x_batch)
            evaluate(online_method, None, y_batches, pred_batches, x_batches)
        elif online_method not in ["clustream", "scope_full", "scaledclustream", "wclustream"]:
            best_seed = -1
            best_score = -np.inf
            batches = range(0, len(X), batchsize)
            for seed in range(10):
                score = 0
                for i in batches:
                    end_batch = min(i + batchsize, len(X))
                    filepath = clu_label_dir + "preds_" + dataset + "_" + online_method + "_" + setting + "_" + str(seed) + "_" + str(end_batch) + "_base.npy"
                    print(filepath)
                    pred_batch = np.load(filepath)
                    y_batch = y_store[i:end_batch]
                    metrics, cm = getMetrics(y_batch, pred_batch)
                    score += (end_batch - i) * (metrics['ARI'] + metrics['AMI'])
                score = score * 0.5 / len(X)
                print("Score:", score)
                if score > best_score:
                    best_score = score
                    best_seed = seed
            print(best_seed, best_score)
            pred_batches = []
            y_batches = []
            x_batches = []
            for i in range(0, len(X), 1000):
                end_batch = min(i + batchsize, len(X))
                filepath = clu_label_dir + "preds_" + dataset + "_" + online_method + "_" + setting + "_" + str(
                    best_seed) + "_" + str(end_batch) + "_base.npy"
                pred_batch = np.load(filepath)
                x_batch = X[i:end_batch]
                y_batch = y_store[i:end_batch]
                pred_batches.append(pred_batch)
                y_batches.append(y_batch)
                x_batches.append(x_batch)
            evaluate(online_method, None, y_batches, pred_batches, x_batches)
        else:
            for offline_method in offline_methods:
                best_seed = -1
                best_score = -np.inf
                batches = range(0, len(X), batchsize)
                for seed in range(10):
                    score = 0
                    for i in batches:
                        end_batch = min(i + batchsize, len(X))
                        if offline_method == "nooffline":
                            filepath = clu_label_dir + "preds_" + dataset + "_" + online_method + "_" + setting + "_" + str(
                                seed) + "_" + str(end_batch) + "_" + offline_method + "_0.npy"
                        elif offline_method not in ["dbscan", "dpca", "hdbscan", "mdbscan", "dbhd", "snndpc", "rnndbscan"]:
                            filepath = clu_label_dir + "preds_" + dataset + "_" + online_method + "_" + setting + "_" + str(
                                seed) + "_" + str(end_batch) + "_" + offline_method + "_" + str(
                                seed) + ".npy"
                        elif seed <= 4:
                            filepath = clu_label_dir + "preds_" + dataset + "_" + online_method + "_" + setting + "_" + str(
                                seed) + "_" + str(end_batch) + "_" + offline_method + "_0.npy"
                        else:
                            filepath = clu_label_dir + "preds_" + dataset + "_" + online_method + "_" + setting + "_" + str(
                                seed) + "_" + str(end_batch) + "_" + offline_method + "_1.npy"
                        print(filepath)
                        pred_batch = np.load(filepath)
                        y_batch = y_store[i:end_batch]
                        metrics, cm = getMetrics(y_batch, pred_batch)
                        score += (end_batch - i) * (metrics['ARI'] + metrics['AMI'])
                    score = score * 0.5 / len(X)
                    print("Score:", score)
                    if score > best_score:
                        best_score = score
                        best_seed = seed
                print(best_seed, best_score)
                pred_batches = []
                y_batches = []
                x_batches = []
                for i in range(0, len(X), 1000):
                    end_batch = min(i + batchsize, len(X))
                    if offline_method == "nooffline":
                        filepath = clu_label_dir + "preds_" + dataset + "_" + online_method + "_" + setting + "_" + str(
                            best_seed) + "_" + str(end_batch) + "_" + offline_method + "_0.npy"
                    elif offline_method not in ["dbscan", "dpca", "hdbscan", "mdbscan", "dbhd", "snndpc", "rnndbscan"]:
                        filepath = clu_label_dir + "preds_" + dataset + "_" + online_method + "_" + setting + "_" + str(
                            best_seed) + "_" + str(end_batch) + "_" + offline_method + "_" + str(
                            best_seed) + ".npy"
                    elif best_seed <= 4:
                        filepath = clu_label_dir + "preds_" + dataset + "_" + online_method + "_" + setting + "_" + str(
                            best_seed) + "_" + str(end_batch) + "_" + offline_method + "_0.npy"
                    else:
                        filepath = clu_label_dir + "preds_" + dataset + "_" + online_method + "_" + setting + "_" + str(
                            best_seed) + "_" + str(end_batch) + "_" + offline_method + "_1.npy"
                    pred_batch = np.load(filepath)
                    x_batch = X[i:end_batch]
                    y_batch = y_store[i:end_batch]
                    pred_batches.append(pred_batch)
                    y_batches.append(y_batch)
                    x_batches.append(x_batch)
                evaluate(online_method, offline_method, y_batches, pred_batches, x_batches)


if __name__ == '__main__':
    main(sys.argv)