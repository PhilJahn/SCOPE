import ast
import copy
import sys
import numpy as np
from os import listdir
from os.path import isfile, join
from pprint import pprint

import dictdiffer


def process_file(path, metrics):
	reader = open(path)

	param_dict = {}
	result_dict = {}
	seed_mapping = {}
	try:
		line = reader.readline()
		while line != '':
			# print(line)
			line = line.replace("\t", "")
			line = line.replace("\n", "")
			if "|" not in line:
				line = line.replace("{", "|{")
			line_split = line.split("|")
			method = line_split[0].split(" ")
			method_name = method[0]
			# https://stackoverflow.com/questions/988228/convert-a-string-representation-of-a-dictionary-to-a-dictionary
			dictionary = ast.literal_eval(line_split[1])
			seed = method[1]
			t = None
			param = True
			#print(method)
			if len(method) == 4:
				t = method[2]
				alg = "base"
				alg_index = 0
				param = False
			elif len(method) == 5:
				alg = method[2]
				alg_index = method[3]
			elif len(method) == 6:
				t = method[2]
				alg = method[3]
				alg_index = method[4]
				param = False
			else:
				alg = "base"
				alg_index = 0
			if param:
				if method_name in seed_mapping.keys():
					if alg in param_dict[method_name].keys():
						seed_mapping[method_name][alg][alg_index] = alg_index
					else:
						seed_mapping[method_name][alg] = {alg_index: alg_index}
				else:
					seed_mapping[method_name] = {alg: {alg_index: alg_index}}

				if "alg_seed" in dictionary.keys():
					if dictionary["alg_seed"] != 0:
						dictionary["alg_seed"] = 0
						for alg_index_2 in param_dict[method_name][alg].keys():
							dictionary_2 = param_dict[method_name][alg][alg_index_2]
							if len(list(dictdiffer.diff(dictionary, dictionary_2))) == 0:
								seed_mapping[method_name][alg][alg_index] = alg_index_2
				alg_index = seed_mapping[method_name][alg][alg_index]
				if method_name in param_dict.keys():
					if alg in param_dict[method_name].keys():
						if alg_index not in param_dict[method_name][alg].keys():
							param_dict[method_name][alg][alg_index] = dictionary
					else:
						param_dict[method_name][alg] = {alg_index: dictionary}
				else:
					param_dict[method_name] = {alg: {alg_index: dictionary}}
			else:
				sm = seed_mapping[method_name][alg][alg_index]
				if sm != alg_index:
					seed = f"{seed}_{alg_index}"
					alg_index = sm
				if method_name in result_dict.keys():
					if alg in result_dict[method_name].keys():
						if alg_index in result_dict[method_name][alg].keys():
							if t in result_dict[method_name][alg][alg_index].keys():
								result_dict[method_name][alg][alg_index][t][seed] = dictionary
							else:
								result_dict[method_name][alg][alg_index][t] = {seed: dictionary}
						else:
							result_dict[method_name][alg][alg_index] = {t: {seed: dictionary}}
					else:
						result_dict[method_name][alg] = {alg_index: {t: {seed: dictionary}}}
				else:
					result_dict[method_name] = {alg: {alg_index: {t: {seed: dictionary}}}}

			line = reader.readline()

	finally:
		#pprint(param_dict)
		#pprint(result_dict)
		reader.close()

		for method_name in result_dict.keys():
			for alg_name in result_dict[method_name].keys():
				for alg_index in result_dict[method_name][alg_name].keys():
					t_results = result_dict[method_name][alg_name][alg_index]
					t_keys = list(t_results.keys())
					i_keys = list(t_results[t_keys[0]].keys())
					result_dict[method_name][alg_name][alg_index]["tmean"] = {}
					result_dict[method_name][alg_name][alg_index]["tstd"] = {}
					for i in i_keys:
						#print(i_keys, t_keys, t_results)
						for metric in metrics:
							metric_vals = {t: t_results[t][i][metric] for t in t_keys}
							metric_vals_w = []
							for t_key in t_keys:
								t_int = int(t_key)
								num = t_int - len(metric_vals_w)
								metric_vals_w.extend([metric_vals[t_key]]*num)
							mean = np.mean(metric_vals_w)
							std = np.std(metric_vals_w)
							result_dict[method_name][alg_name][alg_index]["tmean"][metric] = mean
							result_dict[method_name][alg_name][alg_index]["tstd"][metric] = std
					for t in t_keys:
						cur_results = result_dict[method_name][alg_name][alg_index][t]
						result_dict[method_name][alg_name][alg_index][t]["mean"] = {}
						result_dict[method_name][alg_name][alg_index][t]["std"] = {}
						for metric in metrics:
							#print(i_keys)
							metric_vals = [cur_results[i][metric] for i in i_keys]
							mean = np.mean(metric_vals)
							std = np.std(metric_vals)
							result_dict[method_name][alg_name][alg_index][t]["mean"][metric] = mean
							result_dict[method_name][alg_name][alg_index][t]["std"][metric] = std
					result_dict[method_name][alg_name][alg_index]["all_mean"] = {}
					result_dict[method_name][alg_name][alg_index]["all_std"] = {}
					for metric in metrics:
						metric_vals_w = []
						for i in i_keys:
							metric_vals = {t: t_results[t][i][metric] for t in t_keys}
							for t_key in t_keys:
								t_int = int(t_key)
								num = t_int - len(metric_vals_w)
								metric_vals_w.extend([metric_vals[t_key]] * num)
						mean = np.mean(metric_vals_w)
						std = np.std(metric_vals_w)
						result_dict[method_name][alg_name][alg_index]["all_mean"][metric] = mean
						result_dict[method_name][alg_name][alg_index]["all_std"][metric] = std
				best = {}
				base = {}
				for metric in metrics:
					best[metric] = -np.inf
				#for metric in metrics:
				#	base[metric] = result_dict[method_name][alg_name]["base"]["all_mean"][metric]
				for alg_index in result_dict[method_name][alg_name].keys():
					for metric in metrics:
						metric_val = result_dict[method_name][alg_name][alg_index]["all_mean"][metric]
						if metric_val > best[metric]:
							best[metric] = metric_val
				#print("base", method_name, alg_name, base)
				print("best", method_name, alg_name, best)
		return param_dict, result_dict

def main(args):
	result_dir = "run_logs/"
	# https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
	onlyfiles = [f for f in listdir(result_dir) if isfile(join(result_dir, f))]
	setting = "1000000_100_1000"
	dataset = "pendigits"
	metrics = ["accuracy", "ARI", "NMI"]
	for f in onlyfiles:
		if setting in f and dataset in f:
			f_params, f_results = process_file(result_dir + f, metrics)



if __name__ == '__main__':
	main(sys.argv)
