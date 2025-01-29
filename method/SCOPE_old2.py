from __future__ import annotations

import copy
import math
from collections import defaultdict

import numpy as np
from river import base, stats, utils

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
			max_singletons: int = 25,
			micro_cluster_r_factor: int = 2,
			time_window: int = 1000,
			time_gap: int = 100,
			seed: int | None = None,
			dissolve = False,
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

		self.max_key = -1 #increment before insertion
		self.max_singletons = max_singletons
		self.singleton_queue = SingletonQueue(max_singletons)

		self.dissolve = dissolve

	def clean_old(self):
		# Calculate the threshold to delete old micro-clusters
		threshold = self._timestamp - self.time_window

		# Delete old micro-cluster if its relevance stamp is smaller than the threshold
		del_id = -1
		for i, mc in self.micro_clusters.items():
			if mc.relevance_stamp(self.max_micro_clusters) < threshold:
				del_id = i
				break
		if del_id != -1:
			print(del_id, "aged out")
			self.micro_clusters.__delitem__(del_id)

	def _get_best_mc(self, x, self_id=-1, verbose=False):
		in_volume = math.inf
		cur_dist = math.inf
		was_within = False
		best_idx = -1

		for mc_idx, mc in self.micro_clusters.items():
			if mc_idx != self_id:
				distance = self._distance(mc.get_closest_point(x), x)

				if mc.is_within(x):
					was_within = True
					if mc.volume <= in_volume:
						best_idx = mc_idx
						in_volume = mc.volume
						cur_dist = distance
				else:
					if distance <= cur_dist and not was_within:
						best_idx = mc_idx
						cur_dist = distance
		return best_idx, cur_dist

	@staticmethod
	def _distance(point_a, point_b):
		return utils.math.minkowski_distance(point_a, point_b, 1)

	def learn_one(self, x, w=1.0):
		self._timestamp += 1

		self.datastore.append(x)

		self.max_key += 1
		insert_key = self.max_key

		if not self._initialized:

			self.micro_clusters[insert_key] = SCOPEMicroCluster(
				min=x,
				max=x,
				w=w,
				# When initialized, all micro clusters generated previously will have the timestamp reset to the current
				# time stamp at the time of initialization (i.e. self.max_micro_cluster - 1). Thus, the timestamp is set
				# as follows.
				timestamp=self.max_micro_clusters - 1,
			)

			if len(self.micro_clusters) == self.max_micro_clusters:
				self._initialized = True
		else:

			# TODO update empty_pos, for now just new key at end
			self.max_key += 1
			self.micro_clusters[insert_key] = SCOPEMicroCluster(
				min=x,
				max=x,
				w=w,
				timestamp=self._timestamp,
			)
			print(self._timestamp, f"New DP: {insert_key}, {self.micro_clusters[insert_key]}", flush=True)

		need_handling, emit_mc_id = self.singleton_queue.insert(insert_key)

		if emit_mc_id is not None:
			if not self._initialized:
				emit_mc = self.micro_clusters[emit_mc_id]

				return
			# Determine the closest micro-cluster with respect to the new point instance
			best_id, best_dist = self._get_best_mc(x, insert_key)
			best_mc = self.micro_clusters[best_id]

			if best_dist == 0:
				self.singleton_queue.latestInMC()
				best_mc.insert(x, w, self._timestamp)
				print(insert_key, f"directly inserted DP: {best_id} {best_mc}, {self.singleton_queue.tail.index}, {self.singleton_queue.tail.inMC}", flush=True)

			self.clean_old()

			if need_handling:
				emit_mc = self.micro_clusters[emit_mc_id]
				best_id, best_dist = self._get_best_mc(emit_mc.center, emit_mc_id)
				best_mc = self.micro_clusters[best_id]
				if best_dist == 0:
					best_mc.insert(emit_mc.min, w, emit_mc.timestamp)
					print(emit_mc_id, f"MC insert: {emit_mc_id} -> {best_id} {best_mc}", flush=True)
					self.micro_clusters.__delitem__(emit_mc_id)
				else: # far away TODO either copy radius of closest neighbor or treat as large irrelevant space
					print(emit_mc_id, f"New MC {emit_mc_id}", flush=True)
			else:
				self.micro_clusters.__delitem__(emit_mc_id)

			while (len(self.micro_clusters) - len(self.singleton_queue.keys)) > (self.max_micro_clusters - self.max_singletons): # more MCs than desired, dissolve until fulfilled
				self.dissolve_one()


	def offline_processing(self):
		raise NotImplementedError

	def predict_one(self, x, recluster=False, sklearn=False):
		raise NotImplementedError

	def set_mcs(self, mcs):
		self.micro_clusters = mcs
		self.offline_processing()

	def dissolve_one(self):
		min_i, min_j = self.get_lowest_gain_pair()
		print(min_i, min_j)
		if min_i == min_j:
			print(min_i, "Dissolution")
			self.micro_clusters.__delitem__(min_i)
			return
		mc_a = self.micro_clusters[min_i]
		mc_b = self.micro_clusters[min_j]
		if self.is_child(mc_a, mc_b):
			self.micro_clusters[min_j] += self.micro_clusters[min_i]
			print(min_j, f"Parent merge: {min_i} -> {min_j} {self.micro_clusters[min_j]}")
			self.micro_clusters.__delitem__(min_i)
		elif self.is_child(mc_b, mc_a):
			self.micro_clusters[min_i] += self.micro_clusters[min_j]
			print(min_i, f"Parent merge: {min_j} -> {min_i} {self.micro_clusters[min_i]}")
			self.micro_clusters.__delitem__(min_j)
		else:
			parent = self.get_parent(mc_a, mc_b)
			self.max_key += 1
			self.micro_clusters[self.max_key] = parent
			print(self.max_key, f"New MC merge: {min_i} + {min_j} -> {self.max_key} {self.micro_clusters[self.max_key]}")
			self.micro_clusters.__delitem__(min_i)
			self.micro_clusters.__delitem__(min_j)

	def get_lowest_gain_pair(self):
		min_i = 0
		min_j = 0
		min_gain = np.inf
		for i, mc_a in self.micro_clusters.items():
			for j, mc_b in self.micro_clusters.items():
				if i not in self.singleton_queue.keys and j not in self.singleton_queue.keys:
					if i < j:
						#try:
						gain = self.get_gain(mc_a, mc_b)
						if gain < min_gain:
							min_gain = gain
							min_i = i
							min_j = j
						#except:
						#	print(i, mc_a, j , mc_b)
						#	raise Exception("")
					if i == j and self.dissolve:
						gain = mc_a.weight
						if gain < min_gain:
							min_gain = gain
							min_i = i
							min_j = j

		return min_i, min_j


	def get_gain(self, mc_a:SCOPEMicroCluster, mc_b:SCOPEMicroCluster):
		if self.is_child(mc_a, mc_b):
			return self.calc_gain(mc_a, mc_b)
		elif self.is_child(mc_b, mc_a):
			return self.calc_gain(mc_b, mc_a)
		else:
			parent = self.get_parent(mc_a, mc_b)
			return self.calc_gain(mc_a, parent)

	def is_child(self, mc_a:SCOPEMicroCluster, mc_b:SCOPEMicroCluster):
		return mc_a.is_within(mc_b.max) and mc_a.is_within(mc_b.min)

	def get_half_parent(self, mc_a:SCOPEMicroCluster, mc_b:SCOPEMicroCluster): # parent with only weights from mc_a
		actual_parent = self.get_parent(mc_a, mc_b)
		half_parent = SCOPEMicroCluster(min=actual_parent.min, max=actual_parent.max, w=mc_a.weight, timestamp=mc_a.timestamp, var_time=mc_a.var_time)
		return half_parent

	def get_parent(self, mc_a:SCOPEMicroCluster, mc_b:SCOPEMicroCluster):
		parent = SCOPEMicroCluster(min=mc_a.min, max=mc_a.max, w=mc_a.w, timestamp=mc_a.timestamp, var_time=copy.deepcopy(mc_a.var_time))
		parent += mc_b
		return parent

	def calc_gain(self, child:SCOPEMicroCluster, parent:SCOPEMicroCluster):
		expansion = self._distance(child.min, parent.min)
		expansion += self._distance(child.max, parent.max)

		childsize = self._distance(child.max, child.min)

		expansion = self._distance(parent.max, parent.min)
		return expansion#/parent.weight - childsize/child.weight




class SCOPEMicroCluster(base.Base):
	"""Micro-cluster class."""

	def __init__(
			self,
			min: dict = defaultdict(float),
			max: dict = defaultdict(float),
			w: float | None = None,
			timestamp: int | None = None,
			var_time = None,
	):
		# Initialize with sample x
		self.min = copy.copy(min)
		self.max = copy.copy(max)
		self.w = w
		self.timestamp = timestamp
		self.d = len(min.keys())

		if var_time is None:
			self.var_time = stats.Var()
			self.var_time.update(timestamp, w)
		else:
			self.var_time = var_time

	@property
	def center(self):
		return {i: (self.max[i] + self.min[i])/2 for i in self.min.keys()}

	@property
	def extent(self):
		return {i: (self.max[i] - self.min[i])/2 for i in self.min.keys()}

	@property
	def weight(self):
		return self.var_time.n

	# check if x is within MC
	def is_within(self, x):
		within = True
		for i in self.min.keys():
			if within:
				within &= (x[i] <= self.max[i] and x[i] >= self.min[i])
		return within

	@property
	def volume(self):
		v = 1
		for i in self.min.keys():
			v *= self.max[i] - self.min[i]
		return v

	@property
	def density(self):
		return self.weight / self.volume

	def insert(self, x, w, timestamp):
		assert self.is_within(x)
		self.var_time.update(timestamp, w)

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
		shared_keys = set(self.min.keys()).union(other.min.keys())
		#print(self.min.keys(), other.min.keys(), shared_keys)
		for i in shared_keys:
			#print(self.min[i], other.min[i], min(self.min[i], other.min[i]))
			self.min[i] = min(self.min[i], other.min[i])
			#print(self.min[i])
			#print(self.max[i], other.max[i], max(self.max[i], other.max[i]))
			self.max[i] = max(self.max[i], other.max[i])
			#print(self.max[i])
		for i in set(other.min.keys()).difference(self.min.keys()):
			self.min[i] = other.min[i]
			self.max[i] = other.max[i]
		return self

	def __str__(self):
		return f"[{self.center}, extents=+/- {self.extent}, weight={self.weight}, time={self.var_time.mean}]"

	def get_closest_point(self, dp):
		closest = {}
		for d in dp.keys():
			low = self.min[d]
			high = self.max[d]
			if dp[d] < low:
				closest[d] = low
			elif dp[d] > high:
				closest[d] = high
			else:
				closest[d] = dp[d]

		return closest


class SingletonQueue:
	def __init__(self, max_length):
		self.max_length = max_length
		self.keys = set()
		self.length = 0
		self.head = None
		self.tail = None
		self.mid = None

	def insert(self, index):
		#print(index, flush=True)
		sp = SingeltonPointer(index, self.tail, None)
		self.keys.add(index)
		#print(self.keys, flush=True)
		self.length += 1
		removed_index = None
		needs_handling = False
		if self.head is not None: #not first element
			if self.length > self.max_length:
				if self.mid.inMC:
					removed_index, _ = self.remove(self.mid)
				else:
					removed_index, inMC = self.remove(self.head)
					needs_handling = not inMC
					if not needs_handling:
						print("Unexpected behaviour: inserted MC after half of queue", flush=True)
			else:
				mid_pos = math.ceil(self.max_length/2)
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
		else: # move up next element to tracked positions
			self.head = after
		if self.mid.index == index:
			self.mid = after
		if self.tail.index == index:
			self.tail = after

		after.update_before(before)
		del sp
		self.keys.discard(index)
		self.length -= 1

		return index, inMC


class SingeltonPointer:
	def __init__(self, index, before, after):
		self.index = index # index of singleton MC in scope.micro_clusters
		self.before = before # pointer added before this in queue
		self.after = after # pointer added after this in queue
		self.inMC = False # specifically: inserted at initial arrival, not for singletons that got merged with other singleton at end of queue

	def update_after(self, after):
		self.after = after

	def update_before(self, before):
		self.before = before

	def intoMC(self):
		self.inMC = True