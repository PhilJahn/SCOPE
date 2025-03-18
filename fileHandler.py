import ast
import copy
import os
import pickle
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


def process_file(path):
	reader = open(path)
	param_dict = {}
	result_dict = {}
	seed_mapping = {}
	#if "AMI" not in metrics:
	#	metrics.append("AMI")
	#if "ARI" not in metrics:
	#	metrics = metrics.append("ARI")

	metrics = ['accuracy', 'ARI', 'AMI', 'NMI', 'completeness', 'fowl', 'homogeneity', 'purity', 'F1', 'precision', 'recall', 'cluster_num']
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
		#print("File Loaded", flush=True)
		reader.close()
		#pprint(param_dict)
		#pprint(result_dict)

		method_index_mapping = {}
		method_index_params = {}
		min_method_index = -1
		for method_index in param_dict.keys():
			method_index_param = copy.deepcopy(param_dict[method_index]["base"][0])
			if "seed" in method_index_param.keys():
				method_index_param["seed"] = int(method_index) // 5
				#print(method_index_param["seed"])
			if "startindex" in method_index_param.keys():
				method_index_param['startindex'] = 0
			if "category" in method_index_param.keys():
				method_index_param['category'] = 0
			if "endindex" in method_index_param.keys():
				method_index_param['endindex'] = 0
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

		offline_index_mapping = {}
		offline_index_params = {}



		for method_index in param_dict.keys():
			mapped_method_index = method_index_mapping[method_index]
			if mapped_method_index not in offline_index_mapping.keys():
				offline_index_mapping[mapped_method_index] = {}
				offline_index_params[mapped_method_index] = {}
				for alg_name in param_dict[method_index].keys():
					offline_index_mapping[mapped_method_index][alg_name] = {}
					offline_index_params[mapped_method_index][alg_name] = {}
					min_offline_index = -1
					for alg_index in param_dict[method_index][alg_name].keys():
						offline_index_param = copy.deepcopy(param_dict[method_index][alg_name][alg_index])
						if "alg_seed" in offline_index_param.keys():
							offline_index_param["alg_seed"] = offline_index_param["alg_seed"] // 5
						if "seed" in offline_index_param.keys():
							offline_index_param["seed"] = offline_index_param["seed"] // 5
						found = -1
						for j in offline_index_params[mapped_method_index][alg_name].keys():
							if offline_index_params[mapped_method_index][alg_name][j] == offline_index_param:
								found = j
						if found == -1:
							min_offline_index += 1
							offline_index_mapping[mapped_method_index][alg_name][alg_index] = min_offline_index
							offline_index_params[mapped_method_index][alg_name][min_offline_index] = offline_index_param
						else:
							offline_index_mapping[mapped_method_index][alg_name][alg_index] = found
					#print(f"{method_name} {mapped_method_index}: unique {alg_name} params: {len(offline_index_params[mapped_method_index][alg_name].keys())}, mapping: {offline_index_mapping[mapped_method_index][alg_name]}")

		#get avg. value across all timesteps
		for method_index in result_dict.keys():
			for alg_name in result_dict[method_index].keys():
				for offline_index in result_dict[method_index][alg_name].keys():
					t_results = result_dict[method_index][alg_name][offline_index]
					t_keys = list(t_results.keys())
					metric_keys = list(t_results[t_keys[0]].keys())
					result_dict[method_index][alg_name][offline_index]["tfull"] = {}

					for metric_key in metric_keys:
						metric_t_sum = 0
						t_count = 0
						old_t_key = 0
						for t_key in t_keys:
							t_metric = t_results[t_key][metric_key]
							t_num = int(t_key) - old_t_key
							old_t_key = int(t_key)
							metric_t_sum += t_metric*t_num
							t_count += t_num
						#print(t_count)
						#https://stackoverflow.com/questions/57343516/easier-way-to-find-the-average-of-a-set-of-numbers-in-python
						avg_metric = metric_t_sum/t_count
						result_dict[method_index][alg_name][offline_index]["tfull"][metric_key] = avg_metric
		#print("Got avg. value across timesteps", flush=True)
		#build framework for true indexing
		true_result_dict = {}
		for method_true_index in method_index_params.keys():
			true_result_dict[method_true_index] = {}
			for alg_name in offline_index_params[method_true_index].keys():
				true_result_dict[method_true_index][alg_name] = {}
				for alg_true_index in offline_index_params[method_true_index][alg_name].keys():
					true_result_dict[method_true_index][alg_name][alg_true_index] = {}

		#pprint(true_result_dict)
		#transfer contents into true indexed form

		for method_index in result_dict.keys():
			true_method_index = method_index_mapping[method_index]
			for alg_name in result_dict[method_index].keys():
				for offline_index in result_dict[method_index][alg_name].keys():
					#print(true_method_index, alg_name, offline_index)
					true_offline_index = offline_index_mapping[true_method_index][alg_name][offline_index]
					t_results = result_dict[method_index][alg_name][offline_index]
					seed_key = f"m{method_index}_o{offline_index}"
					#true_result_dict[true_method_index][alg_name][true_offline_index][seed_key] = t_results
					for t_key in t_results.keys():
						if t_key not in true_result_dict[true_method_index][alg_name][true_offline_index].keys():
							true_result_dict[true_method_index][alg_name][true_offline_index][t_key] = {}
						true_result_dict[true_method_index][alg_name][true_offline_index][t_key][seed_key] = t_results[t_key]
		#print("Content transferred", flush=True)
		#get mean per timestep

		#for key2 in true_result_dict.keys():
		#	for key1 in true_result_dict[key2].keys():
		#		print(key2, key1, true_result_dict[key2][key1].keys(), true_result_dict[key2][key1][list(true_result_dict[key2][key1].keys())[0]].keys())

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
		#print("Got mean per timestep", flush=True)
		best_dict = {}
		for alg_name in offline_index_params[0].keys():
			best_method_index = -1
			best_alg_index = -1
			best_score = -np.inf
			for method_true_index in true_result_dict.keys():
				#print(method_true_index, alg_name)
				if alg_name in true_result_dict[method_true_index].keys():
					alg_index_num = len(true_result_dict[method_true_index][alg_name].keys())
					for alg_true_index in true_result_dict[method_true_index][alg_name].keys():
						if not ((method_true_index == 0 and alg_true_index != 0) or (alg_index_num > 1 and method_true_index != 0 and alg_true_index == 0)): # skip optimized for default and default for optimized
							#print(true_result_dict[method_true_index][alg_name][alg_true_index])
							if len(true_result_dict[method_true_index][alg_name][alg_true_index]) > 0:
							#print(method_true_index, alg_name, alg_true_index)
								cur_mean = true_result_dict[method_true_index][alg_name][alg_true_index]["tfull"]["mean"]
								score = cur_mean["ARI"] + cur_mean["AMI"]
								if score > best_score:
									best_score = score
									best_method_index = method_true_index
									best_alg_index = alg_true_index
			if best_alg_index != -1:
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
		#print("Got best", flush=True)
		default_best_dict = {}
		for alg_name in offline_index_params[0].keys():

			best_alg_index = -1
			best_score = -np.inf
			#print(alg_name)
			for alg_true_index in true_result_dict[0][alg_name].keys():
				if len(true_result_dict[0][alg_name][alg_true_index]) > 0:
					cur_mean = true_result_dict[0][alg_name][alg_true_index]["tfull"]["mean"]

					score = cur_mean["ARI"] + cur_mean["AMI"]
					print(alg_true_index, score, cur_mean["ARI"],  cur_mean["AMI"])
					if score > best_score:
						best_score = score
						best_alg_index = alg_true_index
			#print(best_alg_index,best_score)
			if best_alg_index != -1:
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
		#print("Got default-best", flush=True)
		default_dict = {}
		for alg_name in offline_index_params[0].keys():
			if len(true_result_dict[0][alg_name][0]) > 0:
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
		#print("Got default", flush=True)

		return method_name, offline_index_params, true_result_dict, best_dict, default_dict, default_best_dict

#from https://stackoverflow.com/questions/13613336/how-do-i-concatenate-text-files-in-python
def combine_files(prefix, files, result_dir):
	filenames = []
	suffixes = []
	outputpath = prefix + "_all_combined.txt"
	if outputpath in files:
		files.remove(outputpath)
	for file in files:
		if prefix in file:
			filenames.append(file)
			suffix = file.split("_")[-1].strip(".txt")
			if suffix == "False":
				suffix = 0
			suffixes.append(int(suffix))
	#https://stackoverflow.com/questions/6618515/sorting-list-according-to-corresponding-values-from-a-parallel-list
	filenames = [x for _, x in sorted(zip(suffixes, filenames))]
	print(filenames)
	base_ctr = -1
	last_change_timestamp = 0
	incomplete = False
	suffix_counter = {}
	first = True
	with open(result_dir + outputpath, 'w') as outfile:
		for fname in filenames:
			ctr = 0
			suffix_subsets = fname.split("_")[6:-1]
			suffix = ""
			for suffix_subset in suffix_subsets:
				suffix += suffix_subset
			last_change_timestamp += os.path.getmtime(result_dir + fname)
			with open(result_dir + fname) as infile:
				for line in infile:
					outfile.write(line)
					ctr += 1
			if suffix in suffix_counter.keys():
				if ctr != suffix_counter[suffix] and not first:
					print(f"{fname} is incomplete", flush=True)
					incomplete = True
			else:
				suffix_counter[suffix] = ctr
			first = False
	outfile.close()
	if incomplete and not ("kddcup" in prefix and "scaledclustream"):
		raise Exception
	return outputpath, last_change_timestamp

def main(args):
	result_dir = "run_logs/"
	# https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
	onlyfiles = [f for f in listdir(result_dir) if isfile(join(result_dir, f))]
	setting = "1000_100_1000"
	dataset = "densired10"
	metrics = ["accuracy", "ARI", "AMI", "purity", "cluster_num"]
	method_names = ["streamkmeans", "denstream", "dbstream",
	                #"emcstream",
					#"mcmststream", #"gbfuzzystream",
					"clustream_no_offline", "clustream_no_offline_fixed",
					"clustream",
					"wclustream",
					"scaledclustream",
					"scope_full"
	                ]
	best_dicts = {}
	default_dicts = {}
	default_best_dicts = {}
	for f in onlyfiles:
		#https://stackoverflow.com/questions/3682748/converting-unix-timestamp-string-to-readable-date
		last_change_timestamp = os.path.getmtime(result_dir + f)
		last_change = datetime.fromtimestamp(last_change_timestamp)
		relevant = False
		for method_name in method_names:
			relevant |= "_" + method_name + "_" in f

		if "opeclustream" in f:
			continue
		if "True" in f:
			continue

		if setting in f and dataset in f and relevant:
			if f.removesuffix('.txt') + "_all_1.txt" in onlyfiles or f.removesuffix('.txt') + "_all_12.txt" in onlyfiles:
				f, last_change_timestamp = combine_files(f.removesuffix('.txt'),onlyfiles,result_dir)
			elif f.removesuffix('_not_projdipmeans_0.txt') + "_not_projdipmeans_1.txt" in onlyfiles:
				f, last_change_timestamp = combine_files(f.removesuffix('_not_projdipmeans_0.txt'),onlyfiles,result_dir)
			elif f.removesuffix('_density_0.txt') + "_density_1.txt" in onlyfiles:
				f, last_change_timestamp = combine_files(f.removesuffix('_density_0.txt'),onlyfiles,result_dir)
			elif "combined" in f:
				continue
			elif "_all_" in f or "_not_projdipmeans_" in f or "_projdipmeans_" in f or "_density_" in f or "vardbscan_" in f \
					or "_kest_" in f or "nkest" in f:
				already_seen = False
				for j in range(0,29):
					if f"_all_{j}" in f or f"_not_projdipmeans_{j}" in f or f"_projdipmeans_{j}" in f or f"_density_{j}" in f\
							or f"_vardbscan_{j}" in f or f"_kest_{j}" in f or f"_nkest_{j}" in f:
						already_seen = True
				if already_seen:
					continue

			print(f, last_change)
			# https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict-or-any-other-python-object
			method_name, param_dict, true_result_dict, best_dict, default_dict, default_best_dict = process_file(result_dir + f)
			if not os.path.exists("dicts"):
				os.mkdir("dicts")
			with open(f"dicts/{f.strip('.txt')}_result.pkl", 'wb') as out:
				pickle.dump(true_result_dict, out)
			with open(f"dicts/{f.strip('.txt')}_best.pkl", 'wb') as out:
				pickle.dump(best_dict, out)
			with open(f"dicts/{f.strip('.txt')}_default.pkl", 'wb') as out:
				pickle.dump(default_dict, out)
			with open(f"dicts/{f.strip('.txt')}_default_best.pkl", 'wb') as out:
				pickle.dump(default_best_dict, out)
			with open(f"dicts/{f.strip('.txt')}_param.pkl", 'wb') as out:
				pickle.dump(param_dict, out)

			best_dicts[method_name] = best_dict
			default_dicts[method_name] = default_dict
			default_best_dicts[method_name] = default_best_dict

			#pprint(param_dict)

			if method_name in ["clustream", "wclustream", "scope_full", "scope"] and dataset=="complex9":
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
				#methodindex = best_dict[alg_name]["method"]
				#offlineindex = best_dict[alg_name]["offline"]
				# if method_name in ["clustream", "wclustream", "scope_full"] and dataset == "complex9":
				# 	try:
				# 		timesteps = list(true_result_dict[methodindex][alg_name][offlineindex].keys())
				# 		#print(timesteps)
				# 		for ti in range(len(timesteps)-1):
				# 			if ti == 0:
				# 				t_start = 0
				# 			else:
				# 				t_start = int(timesteps[ti-1])
				# 			t_stop = int(timesteps[ti])
				# 			#print(t_stop)
				# 			#print(t_start)
				# 			preds = np.load(f"preds/preds_{f.strip('.txt')}_{methodindex}_{t_stop}_{alg_name}_{offlineindex}.npy", allow_pickle=True)
				# 			#print(len(preds))
				# 			X, Y = load_data(dataset, seed=0)
				# 			#print(len(X[t_start:t_stop, 0]))
				# 			#print(X.shape)
				# 			plt.figure(figsize=(8,8))
				# 			plt.scatter(X[t_start:t_stop,0], X[t_start:t_stop,1], c=preds)
				# 			plt.savefig(f"figures/preds_{f.strip('.txt')}_{alg_name}_{t_stop}.pdf", bbox_inches='tight')
				# 			plt.close()
				# 			#plt.show()
				#
				# 		#print(preds.shape)
				# 	except:
				# 		print(f"Could not find predictions for {f.strip('.txt')} {alg_name}")


			# for metric in metrics:
			# 	plt.figure(figsize=(10, 7))
			# 	alg_names = list(best_dict.keys())
			# 	height = []
			# 	ranges = []
			# 	colors = []
			# 	for alg_name in alg_names:
			# 		if alg_name in ["kmeans", "wkmeans", "subkmeans"]:
			# 			colors.append("blue")
			# 		elif alg_name in ["xmeans", "projdipmeans"]:
			# 			colors.append("lightblue")
			# 		elif alg_name in ["spectral", "scar", "spectacl"]:
			# 			colors.append("pink")
			# 		elif alg_name in ["dec", "idec", "dipencoder", "shade"]:
			# 			colors.append("green")
			# 		elif alg_name in ["dpca", "snndpc", "dbhd"]:
			# 			colors.append("orange")
			# 		elif alg_name in ["dbscan", "hdbscan", "rnndbscan", "mdbscan"]:
			# 			colors.append("red")
			# 		else:
			# 			colors.append("black")
			# 		height.append(best_dict[alg_name][f'{metric}_mean'])
			# 		ranges.append(best_dict[alg_name][f'{metric}_std'])
			# 	plt.bar(alg_names, height, yerr=ranges, color=colors)
			# 	plt.errorbar(alg_names, height, yerr=ranges, fmt="o", color="grey")
			# 	plt.title(metric)
			# 	plt.xticks(rotation=90)
			# 	plt.savefig(f"figures/{f.strip('.txt')}_{metric}_best.pdf", bbox_inches='tight')
			# 	plt.close()
				#plt.show()
			print("---")

if __name__ == '__main__':
	main(sys.argv)
