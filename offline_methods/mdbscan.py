from sklearn.neighbors import KDTree
import numpy as np
from sklearn.cluster import DBSCAN

def relative_density(data, n_neighbors):
	kdtree = KDTree(data)
	#print("k", n_neighbors)
	dists, _ = kdtree.query(data, k=n_neighbors + 1, return_distance=True)
	rds = []
	for i in range(len(data)):
		#print(dists[i,1:])
		rd = n_neighbors/np.sum(dists[i,1:])
		rds.append(rd)
	return rds

def SNNC(ld_data, n_neighbors):
	kdtree = KDTree(ld_data)
	_, nearest_neighbors = kdtree.query(ld_data, k=n_neighbors+1)
	kc = [[i] for i in range(len(ld_data))]

	for i in range(len(ld_data)):
		for j in range(i+1, len(ld_data)):
			if set(nearest_neighbors[i]).intersection(nearest_neighbors[j]):
				kc[i].append(j)
	for _ in range(100):
		merged = False
		for i in range(len(ld_data)):
			for j in range(i+1, len(ld_data)):
				if set(kc[i]).intersection(kc[j]):
					for h in kc[j]:
						if h not in kc[i]:
							kc[i].append(h)
					kc[j] = []
					merged = True
		if not merged:
			break
	kcnum = [len(kc[i]) for i in range(len(kc)) if len(kc[i]) > 0]
	kcmean = np.mean(kcnum)
	#print(kcmean)
	labels = [-1] * len(ld_data)
	cur_label = 0
	for i in range(len(ld_data)):
		if len(kc[i]) >= kcmean:
			for j in kc[i]:
				labels[j] = cur_label
			cur_label += 1
	return labels


def MDBSCAN(data, eps=0.5, min_samples=5, n_neighbors=10, t=0.1):
	data = np.array(data)
	rds = relative_density(data, n_neighbors)
	#print(rds)
	filter_density = np.array([rds[i] > t for i in range(len(data))])
	indices_ld = np.arange(len(data))[~filter_density]
	#print(indices_ld)
	data_ld = data[indices_ld]

	#print(data_ld)

	labels = [-1]*len(data)

	if len(data_ld) > n_neighbors:
		labels_snnc = SNNC(data_ld, n_neighbors)
		for i in range(len(indices_ld)):
			labels[indices_ld[i]] = labels_snnc[i]
	max_f_label = np.max(labels)
	#print(labels)
	filter_rest = np.array([labels[i] == -1 for i in range(len(data))])
	indices_rest = np.arange(len(data))[filter_rest]

	#print(filter_rest)
	data_rest = data[indices_rest]
	if len(data_rest) > 0:
		dbscan = DBSCAN(eps=eps, min_samples=min_samples)
		labels_dbscan = dbscan.fit_predict(data_rest)
		for i in range(len(indices_rest)):
			if labels_dbscan[i] >= 0:
				labels[indices_rest[i]] = labels_dbscan[i] + 1 + max_f_label
		if -1 in labels_dbscan:
			filter_cluster = np.array([labels[i] != -1 for i in range(len(data))])
			indices_cluster = np.arange(len(data))[filter_cluster]
			data_cluster = data[indices_cluster]
			labels_cluster = np.array(labels)[indices_cluster]
			if len(data_cluster) > 0:
				kdtree = KDTree(data_cluster)

				for i in range(len(indices_rest)):
					if labels_dbscan[i] == -1:
						dist, closest_clustered = kdtree.query(data[indices_rest[i]].reshape(1, -1), k=1, return_distance=True)
						#print(data[indices_rest[i]], data_cluster[closest_clustered], dist)
						labels[indices_rest[i]] = labels_cluster[closest_clustered][0][0]

			# find closest labeled point

	#print(len(labels))
	return labels


# Example usage
if __name__ == "__main__":
	# Example dataset
	dataset = np.array([
		[1, 2], [2, 3], [3, 4], [5,7], [10, 10], [10, 11], [12,8], [18,7], [22,15], [25, 30],
		[30, 30], [30, 31], [26,25], [ 23,25], [100, 100], [101, 101]
	])

	# Parameters
	Eps = 5
	MinPts = 3
	density_threshold = 0.3
	k = 2
	#print(SNNC(dataset, n_neighbors= k))
	print(MDBSCAN(dataset, eps=Eps, min_samples=MinPts, t=density_threshold, n_neighbors =k))