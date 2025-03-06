from pprint import pprint

import numpy as np
import itertools
import ast
import copy

def dict_to_np(dp):
	value_list = [value for key, value in sorted(dp.items())]
	return np.array(value_list)


def dps_to_np(dps):
	dps_list = []
	for dp in dps:
		dp_list = dict_to_np(dp)
		dps_list.append(dp_list)
	return np.array(dps_list)

def flatten_dict(d):
	"""
    Function to transform a nested dictionary to a flattened dot notation dictionary.

    :param d: Dict
        The dictionary to flatten.

    :return: Dict
        The flattened dictionary.
    """

	def expand(key, value):
		if isinstance(value, dict):
			return [(key + '.' + k, v) for k, v in flatten_dict(value).items()]
		else:
			return [(key, value)]

	items = [item for k, v in d.items() for item in expand(k, v)]

	return dict(items)

def load_parameters(dataset, method, use_full=False):
	org_method = method
	if org_method == "wclustream" or org_method == "scaledclustream"  or org_method == "scope" or org_method == "scope_full":
		method = "clustream"
	if not use_full:
		path = f"./param_logs/params_{dataset}_{method}.txt"
	else:
		path = f"./param_logs/params_{dataset}_{method}_full.txt"
	reader = open(path)
	param_list = []
	try:
		line = reader.readline()
		while line != '':
			line_split = line.split(";")
			#https://stackoverflow.com/questions/988228/convert-a-string-representation-of-a-dictionary-to-a-dictionary
			params = ast.literal_eval(line_split[0])
			for i in range(5):
				params_copy = copy.deepcopy(params)
				params_copy["seed"] = i
				param_list.append(params_copy)
			line = reader.readline()
	finally:

		reader.close()
		if "subset" in dataset and len(param_list) != 30:
			raise Exception(f"Parameter Estimator error for {org_method} in {dataset}")
		return param_list

seed_algs = ["kmeans", "wkmeans", "subkmeans", "xmeans", "projdipmeans", "spectral", "scar", "spectacl", "shade", "dec", "dipencoder"]

def load_offline_parameters(dataset, method, offlinemethods, use_full=False):
	param_lists = {}
	if use_full:
		size = 2
	else:
		size = 6
	for i in range(size):
		for j in range(5):
			param_lists[5*i+j] = {}
	for offlinemethod in offlinemethods:
		if offlinemethod == "nooffline":
			for i in range(size):
				for j in range(5):
					param_lists[5*i+j][offlinemethod] = [{"offline": False}]
			continue
		needs_seeds = offlinemethod in seed_algs
		if not use_full:
			if offlinemethod == "wkmeans":
				path = f"./param_logs/params_{dataset}_{method}_kmeans.txt"
			else:
				path = f"./param_logs/params_{dataset}_{method}_{offlinemethod}.txt"
		else:
			if offlinemethod == "wkmeans":
				path = f"./param_logs/params_{dataset}_{method}_kmeans_full.txt"
			else:
				path = f"./param_logs/params_{dataset}_{method}_{offlinemethod}_full.txt"
		reader = open(path)
		param_list = []
		try:
			line = reader.readline()
			while line != '':
				line_split = line.split(";")
				#https://stackoverflow.com/questions/988228/convert-a-string-representation-of-a-dictionary-to-a-dictionary
				params = ast.literal_eval(line_split[0])
				if needs_seeds:
					for i in range(5):
						params_copy = copy.deepcopy(params)
						params_copy["alg_seed"] = i
						param_list.append(params_copy)
				else:
					param_list.append(params)
				line = reader.readline()
		finally:
			reader.close()
			if not use_full:
				if (needs_seeds	and len(param_list) != 35) or (not needs_seeds and len(param_list) != 7):
					raise Exception(f"Parameter Estimator error for {offlinemethod} for {method} in {dataset}")
			if needs_seeds:
				defaults = param_list[0:5]
			else:
				defaults = [param_list[0]]

			for i in range(size):
				if needs_seeds:
					non_defaults = param_list[5*(i+1):5*(i+2)]
				else:
					non_defaults = [param_list[i+1]]
				for j in range(5):
					param_lists[5*i+j][offlinemethod] = copy.deepcopy(defaults)
					param_lists[5*i+j][offlinemethod].extend(non_defaults)
	return param_lists

# made with ChatGPT
def make_param_dicts(param_dict):
	keys = param_dict.keys()
	values = param_dict.values()
	combinations = list(itertools.product(*values))
	list_of_dicts = [dict(zip(keys, combo)) for combo in combinations]
	return list_of_dicts

