import numpy as np


def dict_to_np(dp):
	value_list = [value for key, value in sorted(dp.items())]
	return np.array(value_list)


def dps_to_np(dps):
	dps_list = []
	for dp in dps:
		dp_list = dict_to_np(dp)
		dps_list.append(dp_list)
	return np.array(dps_list)
