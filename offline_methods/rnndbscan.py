from sklearn.neighbors import KDTree
from scipy.spatial import distance
import numpy as np


class RNNDBSCAN:
	def __init__(self, k):
		self.k = k

	def fit_predict(self, X):

		if len(X) < self.k:
			return [-1]*len(X)

		assign = [-2] * len(X)  # -2 -> UNCLASSIFIED
		cluster = 0
		kdtree = KDTree(X)

		knn_dist, _ = kdtree.query(X, k=self.k + 1)
		r = np.max(knn_dist, axis=1)
		#print(r)
		knns, knn_dist = kdtree.query_radius(X, r, return_distance=True, sort_results=True)

		# print("knns", knns)
		# print("knn_dist", knn_dist)
		knns_dict = {}
		knn_dist_dict = {}
		for i in range(len(X)):
			knns_dict[i] = knns[i].tolist()
			knns_dict[i].remove(i)
			knn_dist_dict[i] = knn_dist[i][1:].tolist()

		rnns_dict = {}
		for i in range(len(X)):
			rnn = []
			for j in range(len(X)):
				if i in knns_dict[j]:
					rnn.append(j)
			rnns_dict[i] = rnn

		cores = [False] * len(X)
		for i, rnn in rnns_dict.items():
			if len(rnn) >= self.k:
				cores[i] = True
		# print(knns_dict)
		# print(rnns_dict)
		# print(cores)

		for i in range(len(X)):
			x = X[i]
			if assign[i] == -2:
				finished, assign = self.expandCluster(i, X, cluster, assign, knns_dict, rnns_dict, cores)
				if not finished:
					cluster += 1
		#print(assign)
		assign = self.expandClusterFinal(X, assign, knns_dict, cores)
		#print(assign)


		return assign

	def expandCluster(self, i, X, cluster, assign, knns, rnns, cores):
		if not cores[i]:
			finished = True
			assign[i] = -1
			return finished, assign
		else:
			queue = []
			for j in knns[i]:
				assign[j] = cluster
				queue.append(j)
			for j in rnns[i]:
				if assign[j] != cluster and cores[j]:
					assign[j] = cluster
					queue.append(j)
			assign[i] = cluster
			while len(queue) > 0:
				y = queue.pop()
				for j in knns[y]:
					if assign[j] == -2:
						assign[j] = cluster
						queue.append(j)
					elif assign[j] == -1:
						assign[j] = cluster
				for j in rnns[y]:
					if assign[j] == -2 and cores[j]:
						assign[j] = cluster
						queue.append(j)
					elif assign[j] == -1 and cores[j]:
						assign[j] = cluster
		return False, assign

	def density(self, X, cluid, assign, cores, knns_dict):
		max_dist = 0
		for i in range(len(X)):
			if assign[i] == cluid and cores[i]:
				neighbors = knns_dict[i]
				for j in neighbors:
					if assign[j] == cluid and cores[j]:
						dist = distance.euclidean(X[i],X[j])
						if dist > max_dist:
							max_dist = dist
		return max_dist



	def expandClusterFinal(self, X, assign, knns_dict, cores):
		density_dict = {}
		for i in range(len(X)):
			if assign[i] == -1:
				neighbors = knns_dict[i]
				mincluster = -1
				mindist = np.inf
				for j in neighbors:
					if cores[j]:
						cluid = assign[j]
						dist = distance.euclidean(X[i],X[j])
						#print(dist)
						if dist < mindist:
							if cluid not in density_dict.keys():
								density_dict[cluid] = self.density(X, cluid, assign, cores, knns_dict)
							density_dist = density_dict[cluid]
							if dist < density_dist:
								mincluster = cluid
								mindist = dist
								#print("i was here")
				assign[i] = mincluster



		return assign


#data = [[0, 1], [0, 2], [-1, 1], [0, 0], [1, 2], [10, 400], [3, 400], [4, 300], [4, 4], [40, 0], [40, 0], [200, 20]]
#rnndbscan = RNNDBSCAN(2)
#labels = rnndbscan.fit_predict(data)
#print(labels)
