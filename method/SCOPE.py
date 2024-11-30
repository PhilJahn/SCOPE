from __future__ import annotations

import copy
import math
from collections import defaultdict

from river import base, cluster, stats, utils

# altered from River repository (https://github.com/online-ml/river), BSD 3-Clause License
# Copyright (c) 2020, the river developers
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Applies to orginal River code

class SCOPE(base.Clusterer):

	def __init__(
			self,
			n_macro_clusters: int = 5,
			max_micro_clusters: int = 100,
			micro_cluster_r_factor: int = 2,
			time_window: int = 1000,
			time_gap: int = 100,
			seed: int | None = None,
			**kwargs,
	):
		super().__init__()
		self.n_macro_clusters = n_macro_clusters
		self.max_micro_clusters = max_micro_clusters
		self.micro_cluster_r_factor = micro_cluster_r_factor
		self.time_window = time_window
		self.time_gap = time_gap
		self.seed = seed

		self.kwargs = kwargs

		self.centers: dict[int, defaultdict] = {}
		self.micro_clusters: dict[int, SCOPEMicroCluster] = {}

		self._timestamp = -1
		self._offline_timestamp = -1
		self._initialized = False

		self._mc_centers: dict[int, defaultdict] = {}
		self._kmeans_mc = None

		self.datastore = []

		self.max_key = -1

	def _maintain_micro_clusters(self, x, w):
		# Calculate the threshold to delete old micro-clusters
		threshold = self._timestamp - self.time_window

		# Delete old micro-cluster if its relevance stamp is smaller than the threshold
		del_id = None
		for i, mc in self.micro_clusters.items():
			if mc.relevance_stamp(self.max_micro_clusters) < threshold:
				del_id = i
				break

		if del_id is not None:
			self.micro_clusters[del_id] = SCOPEMicroCluster(
				x=x,
				w=w,
				timestamp=self._timestamp,
			)
			return

		# Merge the two closest micro-clusters
		closest_a = 0
		closest_b = 0
		min_distance = math.inf
		for i, mc_a in self.micro_clusters.items():
			for j, mc_b in self.micro_clusters.items():
				if i <= j:
					continue
				dist = self._distance(mc_a.center, mc_b.center)
				if dist < min_distance:
					min_distance = dist
					closest_a = i
					closest_b = j

		self.micro_clusters[closest_a] += self.micro_clusters[closest_b]
		self.micro_clusters[closest_b] = SCOPEMicroCluster(
			x=x,
			w=w,
			timestamp=self._timestamp,
		)

	def _get_best_mc(self, x):
		best_dist = math.inf
		closest_radius = math.inf
		best_idx = -1

		for mc_idx, mc in self.micro_clusters.items():
			distance = self._distance(mc.center, x)
			radius = mc.radius

			if distance <= closest_radius:
				best_dist = distance
				best_idx = mc_idx
				closest_radius = radius if radius <= distance else distance

		return best_idx, best_dist

	@staticmethod
	def _distance(point_a, point_b):
		return utils.math.minkowski_distance(point_a, point_b, 2)

	def learn_one(self, x, w=1.0):
		self._timestamp += 1

		self.datastore.append(x)


		if not self._initialized:
			self.max_key += 1
			self.micro_clusters[self.max_key] = SCOPEMicroCluster(
				x=x,
				w=w,
				# When initialized, all micro clusters generated previously will have the timestamp reset to the current
				# time stamp at the time of initialization (i.e. self.max_micro_cluster - 1). Thus, the timestamp is set
				# as follows.
				timestamp=self.max_micro_clusters - 1,
			)

			if len(self.micro_clusters) == self.max_micro_clusters:
				self._initialized = True

			return
		else:

			# TODO update empty_pos, for now just new key at end
			self.max_key += 1
			self.micro_clusters[self.max_key] = SCOPEMicroCluster(
				x=x,
				w=w,
				timestamp=self._timestamp,
			)

		# Determine the closest micro-cluster with respect to the new point instance
		best_id, best_dist = self._get_best_mc(x)
		best_mc = self.micro_clusters[best_id]

		radius = best_mc.radius

		if best_dist < radius:
			best_mc.insert(x, w, self._timestamp)

		else:

			# If the new point does not fit in the micro-cluster, micro-clusters
			# whose relevance stamps are less than the threshold are deleted.
			# Otherwise, closest micro-clusters are merged with each other.
			self._maintain_micro_clusters(x=x, w=w)

		# Apply incremental K-Means on micro-clusters after each time_gap
		if self._timestamp % self.time_gap == self.time_gap - 1:
			self.offline_processing()

	def offline_processing(self):
		# Micro-cluster centers will only be saved when the calculation of macro-cluster centers
		# is required, in order not to take up memory and time unnecessarily
		self._mc_centers = {i: mc.center for i, mc in self.micro_clusters.items()}
		self._kmeans_mc = cluster.KMeans(
			n_clusters=self.n_macro_clusters, seed=self.seed, **self.kwargs
		)
		for center in self._mc_centers.values():
			self._kmeans_mc.learn_one(center)

		self.centers = self._kmeans_mc.centers
		self._offline_timestamp = self._timestamp

	def predict_one(self, x, recluster=False, sklearn=False):
		raise NotImplementedError

	def set_mcs(self, mcs):
		self.micro_clusters = mcs
		self.offline_processing()

	def calc_gain(self, child, parent_weight, parent_radius):
		d = child.d
		child_weight = child.weight
		child_density = child.density
		return (child_density - get_density(parent_weight+child_weight, parent_radius, d)) * child_weight





def get_volume(r, d):
	return (math.pi**(d/2)/math.gamma(d/2 + 1)) * math.pow(r,d)

def get_density(weight, r, d):
	return weight/get_volume(r,d)

class SCOPEMicroCluster(base.Base):
	"""Micro-cluster class."""

	def __init__(
			self,
			x: dict = defaultdict(float),
			w: float | None = None,
			radius: float = 0,
			timestamp: int | None = None,
			var_x = None,
			var_time = None,
	):
		# Initialize with sample x
		self.x = x
		self.w = w
		self.radius = radius
		self.timestamp = timestamp
		self.d = len(x.keys())
		if var_x is None:
			self.var_x = {}
			for k in x:
				v = stats.Var()
				v.update(x[k], w)
				self.var_x[k] = v
			self.var_time = stats.Var()
			self.var_time.update(timestamp, w)
		else:
			self.var_x = var_x
			self.var_time = var_time

	@property
	def center(self):
		return {k: var.mean.get() for k, var in self.var_x.items()}

	def _deviation(self):
		dev_sum = 0
		for var in self.var_x.values():
			dev_sum += math.sqrt(var.get())
		return dev_sum / len(self.var_x) if len(self.var_x) > 0 else math.inf

	@property
	def weight(self):
		return self.var_time.n

	@property
	def volume(self):
		return get_volume(self.radius, self.d)

	@property
	def density(self):
		return get_density(self.weight, self.radius, self.d)

	def insert(self, x, w, timestamp):
		self.var_time.update(timestamp, w)
		for x_idx, x_val in x.items():
			self.var_x[x_idx].update(x_val, w)

	def relevance_stamp(self, max_mc):
		mu_time = self.var_time.mean.get()
		if self.weight < 2 * max_mc:
			return mu_time

		sigma_time = math.sqrt(self.var_time.get())
		return mu_time + sigma_time * self._quantile(max_mc / (2 * self.weight))

	def _quantile(self, z):
		return math.sqrt(2) * self.inverse_error(2 * z - 1)

	@staticmethod
	def inverse_error(x):
		z = math.sqrt(math.pi) * x
		res = x / 2
		z2 = z * z

		zprod = z2 * z
		res += (1.0 / 24) * zprod

		zprod *= z2  # z5
		res += (7.0 / 960) * zprod

		zprod *= z2  # z ^ 7
		res += (127 * zprod) / 80640

		zprod *= z2  # z ^ 9
		res += (4369 * zprod) / 11612160

		zprod *= z2  # z ^ 11
		res += (34807 * zprod) / 364953600

		zprod *= z2  # z ^ 13
		res += (20036983 * zprod) / 797058662400

		return res

	def __iadd__(self, other: SCOPEMicroCluster):
		self.var_time += other.var_time
		self.var_x = {k: self.var_x[k] + other.var_x.get(k, stats.Var()) for k in self.var_x}
		return self

class SingletonQueue:
	def __init__(self, max_length):
		self.max_length = max_length
		self.keys = set()
		self.length = 0
		self.head = None
		self.tail = None
		self.mid = None

	def insert(self, index):
		sp = SingeltonPointer(index, self.tail, None)
		self.keys.add(index)
		self.length += 1
		removed_index = None
		needs_handling = False
		if self.head is not None: #not first element
			if self.length < self.max_length: #not initilaization
				if self.mid.inMC:
					self.mid, removed_index, _ = self.remove(self.mid)
				else:
					self.head, removed_index, inMC = self.remove(self.head)
					needs_handling = not inMC
					if not needs_handling:
						print("Unexpected behaviour: inserted MC after half of queue", flush=True)
			else:
				mid_pos = self.max_length/2
				if self.length == mid_pos:
					self.mid = sp
			self.tail.update_after(sp)
		else:
			self.head = sp
		self.tail = sp

		return needs_handling, removed_index

	def latestInMC(self):
		self.tail.inMC = True

	def remove(self, sp: SingeltonPointer):
		before = sp.before
		after = sp.after
		index = sp.index
		inMC = sp.inMC

		if before is not None:
			before.update_after(after)
		after.update_before(before)
		del sp
		self.keys.discard(index)
		self.length -= 1

		return after, index, inMC


class SingeltonPointer:
	def __init__(self, index, before, after):
		self.index = index
		self.before = before
		self.after = after
		self.inMC = False

	def update_after(self, after):
		self.after = after

	def update_before(self, before):
		self.before = before

	def intoMC(self):
		self.inMC = True