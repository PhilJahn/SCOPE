import argparse
import os.path
from pprint import pprint
# --{bop} --
#from similarity.BoP import BoP
# --{bop} --
import numpy as np
import matplotlib.pyplot as plt
from datahandler import load_data
# --{ks} --
#from similarity.dataset_similarity_metrics import ks_test
# --{ks} --
from utils import dict_to_np, dps_to_np
from sklearn.neighbors import KDTree
# -- {mmd} --
#from similarity.mmd_pytorch import MMD_loss
# -- {mmd} --

def get_mc_centers(dataname, parameters, timestep, mc_folder):
	mcs = np.load(f"./{mc_folder}/mcs_{dataname}_clustream_1000_100_1000_False_{parameters}_{timestep}.npy",
	              allow_pickle=True)
	mc_centers = []
	for mc in mcs:
		mc_center = dict_to_np(mc[1])
		mc_centers.append(mc_center)
	return mc_centers


def get_gen_data(dataname, parameters, timestep, gen_folder, gen_type):
	data = np.load(f"./{gen_folder}/data_{dataname}_{gen_type}_1000_100_1000_False_{parameters}_{timestep}.npy",
	               allow_pickle=True)
	return data


def main(args):
	dataname = args.ds
	method = args.method
	parameters = args.index
	gen_folder = args.gen_folder
	mc_folder = args.mc_folder
	bop_folder = args.bop_folder
	bop_centroids = args.bop_centroids
	max_length = args.max_length
	batch_size = args.batch_size
	sc = args.value_scale

	true_X, true_y = load_data(dataname, seed=0)
	datalength = min(len(true_y), max_length)

	if not os.path.exists("rec_logs"):
		os.mkdir("rec_logs")
	f = open(f'rec_logs/{dataname}_{method}_{parameters}.txt', 'w',
	         newline='\n',
	         buffering=100)
	majority_sum = 0
	bop_jsd_sum = 0
	c2st_real_sum = 0
	c2st_offline_sum = 0
	nndists_sum = 0
	inndists_sum = 0
	mmd_rbf_sum = 0
	ks_sum = 0
	for i in range(batch_size, datalength + batch_size, batch_size):
		cur_ts = min(i, datalength)
		real_subset = true_X[i - batch_size:cur_ts]
		labels = true_y[i - batch_size:cur_ts]

		real_length = len(real_subset)
		if method == "clustream":
			offline_data = get_mc_centers(dataname, parameters, cur_ts, mc_folder)
			offline_data = np.array(offline_data)
		else:
			offline_data = get_gen_data(dataname, parameters, cur_ts, gen_folder, method)
			offline_data = np.array(offline_data)

		# --{bop} --
		#bop_subset = BoP(real_subset, min(bop_centroids, real_length),
		#                 f"{bop_folder}2/{dataname}/{method}/{i}")
		#bop_scores = bop_subset.evaluate(offline_data)
		#bop_jsd_sum += bop_scores['JS'] * real_length
		# --{bop} -- instead:
		bop_scores= {'JS':0}
		# --{bop} --

		kdtree_offline = KDTree(np.unique(offline_data, axis=0))
		nndists_offline, assignment = kdtree_offline.query(real_subset)
		kdtree_offline_nonuni = KDTree(offline_data)
		inndists_offline, _ = kdtree_offline_nonuni.query(offline_data, k=2)
		kdtree_real = KDTree(real_subset)
		nndists_online, _ = kdtree_real.query(real_subset, k=2)
		inndists_online, _ = kdtree_real.query(offline_data)
		true_online = 0
		false_online = 0
		for j in range(len(nndists_offline)):
			# print(nndists_offline[j][0], nndists_online[j][1])
			if nndists_offline[j][0] <= nndists_online[j][1]:
				false_online += 1
			else:
				true_online += 1
		c2st_real = true_online / len(nndists_offline)
		c2st_real_sum += c2st_real * real_length

		false_offline = 0
		true_offline = 0
		for j in range(len(inndists_offline)):
			if inndists_offline[j][1] <= inndists_online[j][0]:
				true_offline += 1
			else:
				false_offline += 1
		c2st_offline = true_offline / len(inndists_offline)
		c2st_offline_sum += c2st_offline * real_length

		cm = {}
		for j in range(len(assignment)):
			index = assignment[j][0]
			if not index in cm.keys():
				cm[index] = [0] * (int(max(labels)) + 1)
			cm[index][int(labels[j])] += 1

		majority = 0
		for j in cm.keys():
			majority += max(cm[j])
		# print(j, cm[j], offline_data[j])

		purity = majority / len(real_subset)
		majority_sum += purity * len(real_subset)

		nndist_avg = np.sum(nndists_offline) / len(nndists_offline)
		nndists_sum += np.sum(nndists_offline)

		inndist_avg = np.sum(inndists_online) / len(inndists_online)
		inndists_sum += inndist_avg * real_length

		# --{ks} --
		#p, ks_stat, _, _ , _ = ks_test(real_subset, offline_data)
		#ks_sum += ks_stat * len(real_subset)
		# --{ks} -- instead:
		ks_stat = 0
		p = 0
		# --{ks} --

		# -- {mmd} --
		#mmd_loss = MMD_loss()
		#mmd_rbf_val = mmd_loss.forward(real_subset, offline_data)
		#mmd_rbf_sum += mmd_rbf_val * real_length
		# -- {mmd} -- instead:
		mmd_rbf_val = 0
		# -- {mmd} --

		ts_output = f"\t{method} {cur_ts} BoP: {bop_scores['JS'] * sc:.3f} Impurity: {(1 - purity) * sc:.3f} "
		ts_output += f"C2ST-R: {c2st_real * sc:.3f} C2ST-O: {c2st_offline * sc:.3f} "
		ts_output += f"NN-dist {nndist_avg * sc:.3f} iNN-dist {inndist_avg * sc:.3f} "
		ts_output += f"KS {ks_stat * sc:.3f} ({p:.2f}) MMD-rbf {mmd_rbf_val * sc:.3f}"
		print(ts_output, flush=True)
		f.write(ts_output + "\n")

	purity = majority_sum / datalength
	bop_jsd_avg = bop_jsd_sum / datalength
	c2st_real_avg = c2st_real_sum / datalength
	c2st_offline_avg = c2st_offline_sum / datalength
	nndists_avg = nndists_sum / datalength
	inndists_avg = inndists_sum / datalength
	ks_avg = ks_sum /datalength
	mmd_rbf_avg = mmd_rbf_sum / datalength
	output = f"{method} BoP: {bop_jsd_avg * sc:.3f} Impurity: {(1 - purity) * sc:.3f} "
	output += f"C2ST-R: {c2st_real_avg * sc:.3f} C2ST-O: {c2st_offline_avg * sc:.3f} "
	output += f"NN-dist {nndists_avg * sc:.3f} iNN-dist {inndists_avg * sc:.3f} "
	output += f"KS {ks_avg * sc:.3f} MMD-rbf {mmd_rbf_avg * sc:.3f}"
	print(output, flush=True)
	f.write(output)
	f.close()


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--ds', default="complex9", type=str, help='Used stream data set')
	parser.add_argument('--method', default="clustream", type=str, help='Stream Clustering Method')
	parser.add_argument('--index', default=0, type=int, help='Index Stream Clustering Method')
	parser.add_argument('--gen_folder', default="./gen_data", type=str, help='Path to generated data')
	parser.add_argument('--mc_folder', default="./mc_data", type=str, help='Path to microclusters')
	parser.add_argument('--bop_folder', default="./bop", type=str, help='Path to BoP storage')
	parser.add_argument('--bop_centroids', default=100, type=int, help='Number of BoP centroids')
	parser.add_argument('--max_length', default=np.inf, type=int, help='Maximum length of examined dataset')
	parser.add_argument('--batch_size', default=1000, type=int,
	                    help='Batch size for examination (must match evaluation length of stream clustering)')
	parser.add_argument('--value_scale', default=100, type=int, help='Scaling of metrics')

	args = parser.parse_args()
	#for ds in ["complex9", "rbf3", "densired2", "densired5", "densired10", "densired50", "densired100", "powersupply", "electricity", "letter", "segment", "gassensor"]:
	#for ds in ["kddcup"]:
	main(args)
