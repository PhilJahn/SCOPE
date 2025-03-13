from pprint import pprint

from numpy.random import PCG64
import matplotlib.pyplot as plt
from datahandler import load_data
import numpy as np
import os

from sklearn.model_selection import StratifiedKFold
def select_subset(dataset, ratio):
	print(dataset)
	X, y = load_data(dataset)
	uniques, uniquenum = np.unique(y, return_counts=True)
	unique_dict_full = {}
	for i in range(len(uniques)):
		unique_dict_full[uniques[i]] = uniquenum[i]
	print(len(X), len(X[0]))
	print(unique_dict_full)


	num = round(len(X)*ratio)

	for seed in range(5):
		generator = np.random.Generator(PCG64(seed))
		rand_subset_indices = np.sort(generator.choice(len(X), size=num, replace=False))
		X_rand_subset, y_rand_subset = X[rand_subset_indices], y[rand_subset_indices]
		print("----")
		uniques_rand_subset, uniquenum_rand_subset = np.unique(y_rand_subset, return_counts=True)
		unique_dict_rand_subset = {}
		for i in range(len(uniques_rand_subset)):
			unique_dict_rand_subset[uniques_rand_subset[i]] = uniquenum_rand_subset[i]
		for i in range(len(uniques)):
			if uniques[i] not in uniques_rand_subset:
				unique_dict_rand_subset[uniques[i]] = 0
		#https://stackoverflow.com/questions/9001509/how-do-i-sort-a-dictionary-by-key/47017849#47017849
		unique_dict_rand_subset = dict(sorted(unique_dict_rand_subset.items()))
		print(len(X_rand_subset), len(X_rand_subset[0]))
		print(unique_dict_rand_subset)

		scale = len(X)/num

		plt.figure(figsize=(20, 10))
		plt.bar(list(unique_dict_full.keys()), height = list(unique_dict_full.values()), width=0.8)
		plt.bar(np.array(list(unique_dict_rand_subset.keys())), height = np.array(list(unique_dict_rand_subset.values()))*scale, width=0.6)
		plt.bar(np.array(list(unique_dict_rand_subset.keys())), height = list(unique_dict_rand_subset.values()), width=0.4)
		plt.show()

		if not os.path.exists("./data/rand_subset"):
			os.mkdir("./data/rand_subset")
		np.save(f"./data/rand_subset/X_{dataset}_subset_{num}_{seed}.npy", X_rand_subset)
		np.save(f"./data/rand_subset/y_{dataset}_subset_{num}_{seed}.npy", y_rand_subset)


if __name__ == '__main__':
	select_subset("kddcup", 0.01)
	#select_subset("covertype", 0.02)
	#select_subset("powersupply", 0.2)
	select_subset("gassensor", 0.5)
	#select_subset("rotatinghyperplane", 0.05)
	#select_subset("movingrbf", 0.05)
	select_subset("rbf3", 0.2)
	#select_subset("starlight", 0.5)
	#select_subset("letter", 0.25)
	#select_subset("electricity", 0.15)
	select_subset("densired10", 0.5)

