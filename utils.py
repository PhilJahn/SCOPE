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
			raise Exception(f"Parameter Estimator error for {method} in {dataset}")
		return param_list

# made with ChatGPT
def make_param_dicts(param_dict):
	keys = param_dict.keys()
	values = param_dict.values()
	combinations = list(itertools.product(*values))
	list_of_dicts = [dict(zip(keys, combo)) for combo in combinations]
	return list_of_dicts

