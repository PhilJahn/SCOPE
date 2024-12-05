from __future__ import annotations

import copy
import math
from collections import defaultdict

import numpy as np
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
		for i, mc in self.micro_clusters.items():
			if mc.relevance_stamp(self.max_micro_clusters) < threshold:
				print(i, "aged out")
				self.micro_clusters.__delitem__(i)

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
		return utils.math.minkowski_distance(point_a, point_b, 2)

	def learn_one(self, x, w=1.0):
		self._timestamp += 1

		self.datastore.append(x)

		self.max_key += 1
		insert_key = self.max_key

		if not self._initialized:

			self.micro_clusters[insert_key] = SCOPEMicroCluster(
				x=x,
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
				x=x,
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
					best_mc.insert(emit_mc.x, w, emit_mc.timestamp)
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
						gain = self.d*mc_a.weight
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
		return mc_a.is_within(mc_b.get_max()) and mc_a.is_within(mc_b.get_min())

	def get_half_parent(self, mc_a:SCOPEMicroCluster, mc_b:SCOPEMicroCluster): # parent with only weights from mc_a
		actual_parent = self.get_parent(mc_a, mc_b)
		half_parent = SCOPEMicroCluster(x=actual_parent.center, w=mc_a.weight, timestamp=mc_a.timestamp, var_time=mc_a.var_time, r=actual_parent.r)
		return half_parent

	def get_parent(self, mc_a:SCOPEMicroCluster, mc_b:SCOPEMicroCluster):
		parent = SCOPEMicroCluster(x=mc_a.x, w=mc_a.w, timestamp=mc_a.timestamp, var_time=copy.deepcopy(mc_a.var_time),
		                           var_x=copy.deepcopy(mc_a.var_x))
		#print(mc_a, parent)
		parent += mc_b
		expansion = 0
		for i in parent.center.keys():
			upper_a = mc_a.center[i] + mc_a.extent[i]
			lower_a = mc_a.center[i] - mc_a.extent[i]
			upper_b = mc_b.center[i] + mc_b.extent[i]
			lower_b = mc_b.center[i] - mc_b.extent[i]
			mid = parent.center[i]
			dist = max(abs(upper_a-mid), abs(lower_a-mid), abs(upper_b-mid), abs(lower_b-mid))

			parent.extent[i] = dist


		return parent

	def calc_gain(self, child:SCOPEMicroCluster, parent:SCOPEMicroCluster):
		expansion = self._distance(child.get_min(), parent.get_min())
		expansion += self._distance(child.get_max(), parent.get_max())

		childsize = self._distance(child.get_max(), child.get_min())
		return expansion#/parent.weight - childsize/child.weight




class SCOPEMicroCluster(base.Base):
	"""Micro-cluster class."""

	def __init__(
			self,
			x: dict = defaultdict(float),
			w: float | None = None,
			timestamp: int | None = None,
			var_time = None,
			var_x = None
	):
		# Initialize with sample x
		self.x = x
		self.w = w
		self.extent = {}
		for k in x.keys():
			self.extent[k] = 0
		self.timestamp = timestamp
		self.d = len(x.keys())
		if var_x is None:
			self.var_x = {}
			for k in x:
				v = stats.Var()
				v.update(x[k], w)
				self.var_x[k] = v
		else:
			self.var_x = var_x

		if var_time is None:
			self.var_time = stats.Var()
			self.var_time.update(timestamp, w)
		else:
			self.var_time = var_time

	@property
	def center(self):
		return {k: var.mean.get() for k, var in self.var_x.items()}


	@property
	def weight(self):
		return self.var_time.n

	# check if x is within MC
	def is_within(self, x):
		within = True
		for i in self.extent.keys():
			if within:
				within &= (x[i] < self.center[i] + self.extent[i] and x[i] > self.center[i] - self.extent[i])
		return within

	@property
	def volume(self):
		v = 1
		for r in self.extent.values():
			v *= 2*r
		return v

	@property
	def density(self):
		return self.weight / self.volume

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

	def __str__(self):
		return f"[{self.center}, extents=+/- {self.extent}, weight={self.weight}, time={self.var_time.mean}]"

	def get_closest_point(self, dp):
		closest = {}
		for d in dp.keys():
			mid = self.center[d]
			low = mid - self.extent[d]
			high = mid + self.extent[d]
			if dp[d] < low:
				closest[d] = low
			elif dp[d] > high:
				closest[d] = high
			else:
				closest[d] = dp[d]

		return closest

	def get_max(self):
		max_dp = {}
		for d in self.center.keys():
			max_dp[d] = self.center[d] + self.extent[d]
		return max_dp

	def get_min(self):
		min_dp = {}
		for d in self.center.keys():
			min_dp[d] = self.center[d] - self.extent[d]
		return min_dp


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