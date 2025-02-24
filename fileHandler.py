import ast
import copy
import os
import sys
import traceback
from datetime import datetime

import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from pprint import pprint

import dictdiffer

from datahandler import load_data


def process_file(path, metrics):
	reader = open(path)
	param_dict = {}
	result_dict = {}
	seed_mapping = {}
	if "AMI" not in metrics:
		metrics.append("AMI")
	if "ARI" not in metrics:
		metrics = metrics.append("ARI")
	#print(metrics)
	try:
		line = reader.readline()
		while line != '':
			# print(line)
			line = line.replace("\t", "")
			line = line.replace("\n", "")
			#print(line)
			if "|" not in line:
				line = line.replace("{", "|{")
			line_split = line.split("|")
			method = line_split[0].split(" ")
			method_name = method[0]
			#print(line)
			# https://stackoverflow.com/questions/988228/convert-a-string-representation-of-a-dictionary-to-a-dictionary
			dictionary = ast.literal_eval(line_split[1])
			method_index = method[1]
			t = None
			param = True
			#print(method)
			if len(method) == 4:
				t = method[2]
				alg_name = "base"
				offline_index = 0
				param = False
			elif len(method) == 5:
				alg_name = method[2]
				offline_index = method[3]
			elif len(method) == 6:
				t = method[2]
				alg_name = method[3]
				offline_index = method[4]
				param = False
			else:
				alg_name = "base"
				offline_index = 0
				#print("-----")
			#print(method, alg_name, offline_index)
			if param:
				if method_index in param_dict.keys():
					if alg_name in param_dict[method_index].keys():
						if offline_index not in param_dict[method_index][alg_name].keys():
							param_dict[method_index][alg_name][offline_index] = dictionary
					else:
						param_dict[method_index][alg_name] = {offline_index: dictionary}
				else:
					param_dict[method_index] = {alg_name: {offline_index: dictionary}}
			else:
				all_metrics = list(dictionary.keys())
				for metric in all_metrics:
					if metric not in metrics:
						del(dictionary[metric])
				if method_index in result_dict.keys():
					if alg_name in result_dict[method_index].keys():
						if offline_index in result_dict[method_index][alg_name].keys():
							result_dict[method_index][alg_name][offline_index][t] = dictionary
						else:
							result_dict[method_index][alg_name][offline_index] = {t: dictionary}
					else:
						result_dict[method_index][alg_name] = {offline_index: {t: dictionary}}
				else:
					result_dict[method_index] = {alg_name: {offline_index: {t: dictionary}}}

			line = reader.readline()
	except Exception:
		print(traceback.format_exc())
	finally:
		#pprint(result_dict)
		reader.close()
		#pprint(param_dict)
		#pprint(result_dict)

		method_index_mapping = {}
		method_index_params = {}
		min_method_index = -1
		for method_index in param_dict.keys():
			method_index_param = copy.deepcopy(param_dict[method_index]["base"][0])
			if "seed" in method_index_param.keys():
				method_index_param["seed"] = 0
			found = -1
			for j in method_index_params.keys():
				if method_index_params[j] == method_index_param:
					found = j
			if found == -1:
				min_method_index += 1
				method_index_mapping[method_index] = min_method_index
				method_index_params[min_method_index] = method_index_param
			else:
				method_index_mapping[method_index] = found
		#print(f"{method_name}: unique method params: {len(method_index_params.keys())}, mapping: {method_index_mapping}")

		first_method_index = list(param_dict.keys())[0]
		offline_index_mapping = {}
		offline_index_params = {}
		for alg_name in param_dict[first_method_index].keys():
			offline_index_mapping[alg_name] = {}
			offline_index_params[alg_name] = {}
			min_offline_index = -1
			for alg_index in param_dict[first_method_index][alg_name].keys():
				offline_index_param = copy.deepcopy(param_dict[first_method_index][alg_name][alg_index])
				if "alg_seed" in offline_index_param.keys():
					offline_index_param["alg_seed"] = 0
				found = -1
				for j in offline_index_params[alg_name].keys():
					if offline_index_params[alg_name][j] == offline_index_param:
						found = j
				if found == -1:
					min_offline_index += 1
					offline_index_mapping[alg_name][alg_index] = min_offline_index
					offline_index_params[alg_name][min_offline_index] = offline_index_param
				else:
					offline_index_mapping[alg_name][alg_index] = found
			#print(
			#	f"{method_name}: unique {alg_name} params: {len(offline_index_params[alg_name].keys())}, mapping: {offline_index_mapping[alg_name]}")

		#get avg. value across all timesteps
		for method_index in result_dict.keys():
			for alg_name in result_dict[method_index].keys():
				for offline_index in result_dict[method_index][alg_name].keys():
					t_results = result_dict[method_index][alg_name][offline_index]
					t_keys = list(t_results.keys())
					metric_keys = list(t_results[t_keys[0]].keys())
					result_dict[method_index][alg_name][offline_index]["tfull"] = {}

					for metric_key in metric_keys:
						metric_t_list = []
						for t_key in t_keys:
							t_metric = t_results[t_key][metric_key]
							t_num = int(t_key)
							metric_t_list.extend([t_metric]*t_num)
						avg_metric = np.mean(metric_t_list)
						result_dict[method_index][alg_name][offline_index]["tfull"][metric_key] = avg_metric

		#build framework for true indexing
		true_result_dict = {}
		for method_true_index in method_index_params.keys():
			true_result_dict[method_true_index] = {}
			for alg_name in offline_index_params.keys():
				true_result_dict[method_true_index][alg_name] = {}
				for alg_true_index in offline_index_params[alg_name].keys():
					true_result_dict[method_true_index][alg_name][alg_true_index] = {}
		#pprint(true_result_dict)

		#transfer contents into true indexed form
		for method_index in result_dict.keys():
			true_method_index = method_index_mapping[method_index]
			for alg_name in result_dict[method_index].keys():
				for offline_index in result_dict[method_index][alg_name].keys():
					true_offline_index = offline_index_mapping[alg_name][offline_index]
					t_results = result_dict[method_index][alg_name][offline_index]
					seed_key = f"m{method_index}_o{offline_index}"
					#true_result_dict[true_method_index][alg_name][true_offline_index][seed_key] = t_results
					for t_key in t_results.keys():
						if t_key not in true_result_dict[true_method_index][alg_name][true_offline_index].keys():
							true_result_dict[true_method_index][alg_name][true_offline_index][t_key] = {}
						true_result_dict[true_method_index][alg_name][true_offline_index][t_key][seed_key] = t_results[t_key]

		#get mean per timestep
		for method_true_index in true_result_dict.keys():
			for alg_name in true_result_dict[method_true_index].keys():
				for alg_true_index in true_result_dict[method_true_index][alg_name].keys():
					for t_key in true_result_dict[method_true_index][alg_name][alg_true_index].keys():
						tkey_results = true_result_dict[method_true_index][alg_name][alg_true_index][t_key]
						tkey_mean = {}
						tkey_std = {}
						tkey_num = 0
						for metric in metrics:
							tkey_metrics = []
							for seed_key in tkey_results.keys():
								tkey_metrics.append(tkey_results[seed_key][metric])
							tkey_mean[metric] = np.mean(tkey_metrics)
							tkey_std[metric] = np.std(tkey_metrics)
							tkey_num = len(tkey_metrics)
						true_result_dict[method_true_index][alg_name][alg_true_index][t_key]["mean"] = tkey_mean
						true_result_dict[method_true_index][alg_name][alg_true_index][t_key]["std"] = tkey_std
						true_result_dict[method_true_index][alg_name][alg_true_index][t_key]["num"] = tkey_num
		best_dict = {}
		for alg_name in offline_index_params.keys():
			best_method_index = -1
			best_alg_index = -1
			best_score = -np.inf
			for method_true_index in true_result_dict.keys():
				for alg_true_index in true_result_dict[method_true_index][alg_name].keys():
					cur_mean = true_result_dict[method_true_index][alg_name][alg_true_index]["tfull"]["mean"]
					score = cur_mean["ARI"] + cur_mean["AMI"]
					if score > best_score:
						best_score = score
						best_method_index = method_true_index
						best_alg_index = alg_true_index
			best_num = true_result_dict[best_method_index][alg_name][best_alg_index]["tfull"]["num"]
			best_dict[alg_name] = {"method": best_method_index, "offline": best_alg_index, "num": best_num}
			best_mean = true_result_dict[best_method_index][alg_name][best_alg_index]["tfull"]["mean"]
			best_std = true_result_dict[best_method_index][alg_name][best_alg_index]["tfull"]["std"]
			for metric in metrics:
				metric_mean = best_mean[metric]
				metric_std = best_std[metric]
				best_dict[alg_name][metric+ "_mean"] = metric_mean
				best_dict[alg_name][metric + "_std"] = metric_std
		#pprint(best_dict)

		default_best_dict = {}
		for alg_name in offline_index_params.keys():
			method_true_index = 0
			best_alg_index = -1
			best_score = -np.inf
			for alg_true_index in true_result_dict[method_true_index][alg_name].keys():
				cur_mean = true_result_dict[method_true_index][alg_name][alg_true_index]["tfull"]["mean"]

				score = cur_mean["ARI"] + cur_mean["AMI"]
				if score > best_score:
					best_score = score
					best_alg_index = alg_true_index
			best_num = true_result_dict[0][alg_name][best_alg_index]["tfull"]["num"]
			default_best_dict[alg_name] = {"method": 0, "offline": best_alg_index, "num":best_num}
			best_mean = true_result_dict[0][alg_name][best_alg_index]["tfull"]["mean"]
			best_std = true_result_dict[0][alg_name][best_alg_index]["tfull"]["std"]
			for metric in metrics:
				metric_mean = best_mean[metric]
				metric_std = best_std[metric]
				default_best_dict[alg_name][metric + "_mean"] = metric_mean
				default_best_dict[alg_name][metric + "_std"] = metric_std
		#pprint(default_best_dict)

		default_dict = {}
		for alg_name in offline_index_params.keys():
			default_num = true_result_dict[0][alg_name][0]["tfull"]["num"]
			default_dict[alg_name] = {"method": 0, "offline": 0, "num":default_num}
			default_mean = true_result_dict[0][alg_name][0]["tfull"]["mean"]
			default_std = true_result_dict[0][alg_name][0]["tfull"]["std"]
			for metric in metrics:
				metric_mean = default_mean[metric]
				metric_std = default_std[metric]
				default_dict[alg_name][metric + "_mean"] = metric_mean
				default_dict[alg_name][metric + "_std"] = metric_std
		#pprint(default_dict)

		return method_name, param_dict, true_result_dict, best_dict, default_dict, default_best_dict

def main(args):
	result_dir = "run_logs/"
	# https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
	onlyfiles = [f for f in listdir(result_dir) if isfile(join(result_dir, f))]
	setting = "1000_100_1000"
	dataset = "complex9"
	metrics = ["accuracy", "ARI", "AMI", "purity"]
	method_names = ["clustream", "wclustream", "scope_full", "scope", "denstream", "dbstream", "emcstream", "streamkmeans", "mcmststream", "mudistream", "dstream", "gbfuzzystream"]
	best_dicts = {}
	default_dicts = {}
	default_best_dicts = {}
	for f in onlyfiles:
		#https://stackoverflow.com/questions/3682748/converting-unix-timestamp-string-to-readable-date
		last_change = datetime.fromtimestamp(os.path.getmtime(result_dir + f))
		relevant = False
		for method_name in method_names:
			relevant |= method_name in f
		if setting in f and dataset in f and relevant:
			print(f, last_change)
			method_name, param_dict, true_result_dict, best_dict, default_dict, default_best_dict = process_file(result_dir + f, copy.deepcopy(metrics))
			best_dicts[method_name] = best_dict
			default_dicts[method_name] = default_dict
			default_best_dicts[method_name] = default_best_dict


			if method_name in ["clustream", "wclustream", "scope_full", "scope"]:
				try:
					mcs = np.load(f"mc_data/mcs_{f.strip('.txt')}_0_1000.npy", allow_pickle=True)
					print(mcs.shape)
				except:
					print(f"Could not find MCs for {f.strip('.txt')}")
			for alg_name in best_dict.keys():
				line = f"{method_name} {alg_name} ({best_dict[alg_name]['num']}): Best: "
				for metric in metrics:
					line += f"{metric}: {best_dict[alg_name][f'{metric}_mean']:.3f} ±{best_dict[alg_name][f'{metric}_std']:.3f}  "
				print(line)
				line = f"{method_name} {alg_name} ({default_dict[alg_name]['num']}): Default: "
				for metric in metrics:
					line += f"{metric}: {default_dict[alg_name][f'{metric}_mean']:.3f} ±{default_dict[alg_name][f'{metric}_std']:.3f}  "
				#print(line)
				methodindex = best_dict[alg_name]["method"]
				offlineindex = best_dict[alg_name]["offline"]
				try:
					timesteps = list(true_result_dict[methodindex][alg_name][offlineindex].keys())
					#print(timesteps)
					for ti in range(len(timesteps)-1):
						if ti == 0:
							t_start = 0
						else:
							t_start = int(timesteps[ti-1])
						t_stop = int(timesteps[ti])
						#print(t_stop)
						#print(t_start)
						preds = np.load(f"preds/preds_{f.strip('.txt')}_{methodindex}_{t_stop}_{alg_name}_{offlineindex}.npy", allow_pickle=True)
						#print(len(preds))
						X, Y = load_data(dataset, seed=0)
						#print(len(X[t_start:t_stop, 0]))
						#print(X.shape)
						plt.figure(figsize=(8,8))
						plt.scatter(X[t_start:t_stop,0], X[t_start:t_stop,1], c=preds)
						plt.savefig(f"preds_{f.strip('.txt')}_{alg_name}_{t_stop}.pdf")
						plt.close()
						#plt.show()

					#print(preds.shape)
				except:
					print(f"Could not find predictions for {f.strip('.txt')} {alg_name}")


			for metric in metrics:
				plt.figure(figsize=(10, 7))
				alg_names = list(best_dict.keys())
				height = []
				ranges = []
				colors = []
				for alg_name in alg_names:
					if alg_name in ["kmeans", "wkmeans", "subkmeans"]:
						colors.append("blue")
					elif alg_name in ["xmeans", "projdipmeans"]:
						colors.append("lightblue")
					elif alg_name in ["spectral", "scar", "spectacl"]:
						colors.append("pink")
					elif alg_name in ["dec", "idec", "dipencoder", "shade"]:
						colors.append("green")
					elif alg_name in ["dpca", "snndpc", "dbhd"]:
						colors.append("orange")
					elif alg_name in ["dbscan", "hdbscan", "rnndbscan", "mdbscan"]:
						colors.append("red")
					else:
						colors.append("black")
					height.append(best_dict[alg_name][f'{metric}_mean'])
					ranges.append(best_dict[alg_name][f'{metric}_std'])
				plt.bar(alg_names, height, yerr=ranges, color=colors)
				plt.errorbar(alg_names, height, yerr=ranges, fmt="o", color="grey")
				plt.title(metric)
				plt.xticks(rotation=90)
				plt.savefig(f"{f.strip('.txt')}_{metric}_best.pdf")
				plt.close()
				#plt.show()
			print("---")
	for metric in metrics:
		plt.figure(figsize=(10, 7))
		method_names = list(best_dicts.keys())
		alg_names = list(best_dicts[method_names[0]].keys())
		height = []
		ranges = []
		colors = []
		names = []
		hatches = []
		for alg_name in alg_names:
			for method_name in method_names:
				if alg_name in best_dicts[method_name].keys():
					if method_name == "clustream":
						hatches.append("///")
					elif method_name == "wclustream":
						hatches.append("\\\\\\")
					elif method_name == "scope":
						hatches.append("+++")
					elif method_name == "scope_full":
						hatches.append("xxx")
					else:
						hatches.append("")
					if alg_name in ["kmeans", "wkmeans", "subkmeans"]:
						colors.append("blue")
					elif alg_name in ["xmeans", "projdipmeans"]:
						colors.append("lightblue")
					elif alg_name in ["spectral", "scar", "spectacl"]:
						colors.append("pink")
					elif alg_name in ["dec", "idec", "dipencoder", "shade"]:
						colors.append("green")
					elif alg_name in ["dpca", "snndpc", "dbhd"]:
						colors.append("orange")
					elif alg_name in ["dbscan", "hdbscan", "rnndbscan", "mdbscan"]:
						colors.append("red")
					else:
						colors.append("lightgrey")
					names.append(f"{method_name} {alg_name}")
					height.append(best_dicts[method_name][alg_name][f'{metric}_mean'])
					ranges.append(best_dicts[method_name][alg_name][f'{metric}_std'])
		plt.bar(names, height, yerr=ranges, color=colors, hatch=hatches, edgecolor="black")
		plt.errorbar(names, height, yerr=ranges, fmt="o", color="grey")
		plt.title(metric)
		plt.xticks(rotation=90)
		plt.subplots_adjust(bottom=0.30)
		plt.savefig(f"{dataset}_all_{setting}_{metric}_best.pdf")
		plt.close()

	for metric in metrics:
		plt.figure(figsize=(10, 7))
		method_names = list(best_dicts.keys())
		alg_names = list(best_dicts[method_names[0]].keys())
		height = []
		ranges = []
		colors = []
		names = []
		hatches = []
		for alg_name in alg_names:
			for method_name in method_names:
				if alg_name in best_dicts[method_name].keys():
					if method_name == "clustream":
						hatches.append("///")
					elif method_name == "wclustream":
						hatches.append("\\\\\\")
					elif method_name == "scope":
						hatches.append("+++")
					elif method_name == "scope_full":
						hatches.append("xxx")
					else:
						hatches.append("")
					if alg_name in ["kmeans", "wkmeans", "subkmeans"]:
						colors.append("blue")
					elif alg_name in ["xmeans", "projdipmeans"]:
						colors.append("lightblue")
					elif alg_name in ["spectral", "scar", "spectacl"]:
						colors.append("pink")
					elif alg_name in ["dec", "idec", "dipencoder", "shade"]:
						colors.append("green")
					elif alg_name in ["dpca", "snndpc", "dbhd"]:
						colors.append("orange")
					elif alg_name in ["dbscan", "hdbscan", "rnndbscan", "mdbscan"]:
						colors.append("red")
					else:
						colors.append("lightgrey")
					names.append(f"{method_name} {alg_name}")
					height.append(default_dicts[method_name][alg_name][f'{metric}_mean'])
					ranges.append(default_dicts[method_name][alg_name][f'{metric}_std'])
		plt.bar(names, height, yerr=ranges, color=colors, hatch=hatches, edgecolor="black")
		plt.errorbar(names, height, yerr=ranges, fmt="o", color="grey")
		plt.title(metric)
		plt.xticks(rotation=90)
		plt.subplots_adjust(bottom=0.30)
		plt.savefig(f"{dataset}_all_{setting}_{metric}_default.pdf")
		plt.close()
		#plt.show()

	for metric in metrics:
		plt.figure(figsize=(10, 7))
		method_names = list(best_dicts.keys())
		alg_names = list(best_dicts[method_names[0]].keys())
		height = []
		ranges = []
		colors = []
		names = []
		hatches = []
		for alg_name in alg_names:
			for method_name in method_names:
				if alg_name in best_dicts[method_name].keys():
					if method_name == "clustream":
						hatches.append("///")
					elif method_name == "wclustream":
						hatches.append("\\\\\\")
					elif method_name == "scope":
						hatches.append("+++")
					elif method_name == "scope_full":
						hatches.append("xxx")
					else:
						hatches.append("")
					if alg_name in ["kmeans", "wkmeans", "subkmeans"]:
						colors.append("blue")
					elif alg_name in ["xmeans", "projdipmeans"]:
						colors.append("lightblue")
					elif alg_name in ["spectral", "scar", "spectacl"]:
						colors.append("pink")
					elif alg_name in ["dec", "idec", "dipencoder", "shade"]:
						colors.append("green")
					elif alg_name in ["dpca", "snndpc", "dbhd"]:
						colors.append("orange")
					elif alg_name in ["dbscan", "hdbscan", "rnndbscan", "mdbscan"]:
						colors.append("red")
					else:
						colors.append("lightgrey")
					names.append(f"{method_name} {alg_name}")
					height.append(default_best_dicts[method_name][alg_name][f'{metric}_mean'])
					ranges.append(default_best_dicts[method_name][alg_name][f'{metric}_std'])
		plt.bar(names, height, yerr=ranges, color=colors, hatch=hatches, edgecolor="black")
		plt.errorbar(names, height, yerr=ranges, fmt="o", color="grey")
		plt.title(metric)
		plt.xticks(rotation=90)
		plt.subplots_adjust(bottom=0.30)
		plt.savefig(f"{dataset}_all_{setting}_{metric}_default_best.pdf")
		plt.close()


if __name__ == '__main__':
	main(sys.argv)
