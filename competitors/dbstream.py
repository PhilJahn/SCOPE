from __future__ import annotations

import collections
import copy
import math
from abc import ABCMeta

from river import base, utils

# from River repository (https://github.com/online-ml/river), BSD 3-Clause License
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


class DBSTREAM(base.Clusterer):
    r"""DBSTREAM

    DBSTREAM [^1] is a clustering algorithm for evolving data streams.
    It is the first micro-cluster-based online clustering component that
    explicitely captures the density between micro-clusters via a shared
    density graph. The density information in the graph is then exploited
    for reclustering based on actual density between adjacent micro clusters.

    The algorithm is divided into two parts:

    **Online micro-cluster maintenance (learning)**

    For a new point `p`:

    * Find all micro clusters for which `p` falls within the fixed radius
    (clustering threshold). If no neighbor is found, a new micro cluster
    with a weight of 1 is created for `p`.

    * If no neighbor is found, a new micro cluster with a weight of 1 is
    created for `p`. If one or more neighbors of `p` are found, we update
    the micro clusters by applying the appropriate fading, increasing
    their weight and then we try to move them closer to `p` using the
    Gaussian neighborhood function.

    * Next, the shared density graph is updated. To prevent collapsing
    micro clusters, we will restrict the movement for micro clusters in case
    they come closer than $r$ (clustering threshold) to each other. Finishing
    this process, the time stamp is also increased by 1.

    * Finally, the cleanup will be processed. It is executed every `t_gap`
    time steps, removing weak micro clusters and weak entries in the
    shared density graph to recover memory and improve the clustering algorithm's
    processing speed.

    **Offline generation of macro clusters (clustering)**

    The offline generation of macro clusters is generated through the two following steps:

    * The connectivity graph `C` is constructed using shared density entries
    between strong micro clusters. The edges in this connectivity graph with
    a connectivity value greater than the intersection threshold ($\alpha$)
    are used to find connected components representing the final cluster.

    * After the connectivity graph is generated, a variant of the DBSCAN algorithm
    proposed by Ester et al. is applied to form all macro clusters
    from $\alpha$-connected micro clusters.

    Parameters
    ----------
    clustering_threshold
        DBStream represents each micro cluster by a leader (a data point defining the
        micro cluster's center) and the density in an area of a user-specified radius
        $r$ (`clustering_threshold`) around the center.
    fading_factor
        Parameter that controls the importance of historical data to current cluster.
        Note that `fading_factor` has to be different from `0`.
    cleanup_interval
        The time interval between two consecutive time points when the cleanup process is
         conducted.
    minimum_weight
        The minimum weight for a cluster to be not "noisy".
    intersection_factor
        The intersection factor related to the area of the overlap of the micro clusters
        relative to the area cover by micro clusters. This parameter is used to determine
        whether a micro cluster or a shared density is weak.

    Attributes
    ----------
    n_clusters
        Number of clusters generated by the algorithm.
    clusters
        A set of final clusters of type `DBStreamMicroCluster`. However, these are either
        micro clusters, or macro clusters that are generated by merging all $\alpha$-connected
        micro clusters. This set is generated through the offline phase of the algorithm.
    centers
        Final clusters' centers.
    micro_clusters
        Micro clusters generated by the algorithm. Instead of updating directly the new instance points
        into a nearest micro cluster, through each iteration, the weight and center will be modified
        so that the clusters are closer to the new points, using the Gaussian neighborhood function.

    References
    ----------
    [^1]: Michael Hahsler and Matthew Bolanos (2016, pp 1449-1461). Clustering Data Streams Based on
          Shared Density between Micro-Clusters, IEEE Transactions on Knowledge and Data Engineering 28(6) .
          In Proceedings of the Sixth SIAM International Conference on Data Mining,
          April 20–22, 2006, Bethesda, MD, USA.
    [^2]: Ester et al (1996). A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases
          with Noise. In KDD-96 Proceedings, AAAI.
    """

    def __init__(
        self,
        clustering_threshold: float = 1.0,
        fading_factor: float = 0.01,
        cleanup_interval: float = 2,
        intersection_factor: float = 0.3,
        minimum_weight: float = 1.0,
    ):
        super().__init__()
        self._time_stamp = 0

        self.clustering_threshold = clustering_threshold
        self.fading_factor = fading_factor
        self.cleanup_interval = cleanup_interval
        self.intersection_factor = intersection_factor
        self.minimum_weight = minimum_weight

        self._n_clusters: int = 0
        self._clusters: dict[int, DBSTREAMMicroCluster] = {}
        self._centers: dict = {}
        self._micro_clusters: dict[int, DBSTREAMMicroCluster] = {}

        self.s: dict[int, dict[int, float]] = {}
        self.s_t: dict[int, dict[int, float]] = {}

        self.clustering_is_up_to_date = False

    @staticmethod
    def _distance(point_a, point_b):
        return utils.math.minkowski_distance(point_a, point_b, 2)

    def _find_fixed_radius_nn(self, x):
        fixed_radius_nn = {}
        for i in self._micro_clusters.keys():
            if self._distance(self._micro_clusters[i].center, x) < self.clustering_threshold:
                fixed_radius_nn[i] = self._micro_clusters[i]
        return fixed_radius_nn

    def _gaussian_neighborhood(self, point_a, point_b):
        distance = self._distance(point_a, point_b)
        sigma = self.clustering_threshold / 3
        gaussian_neighborhood = math.exp(-(distance * distance) / (2 * (sigma * sigma)))
        return gaussian_neighborhood

    def _update(self, x):
        # Algorithm 1 of Michael Hahsler and Matthew Bolanos
        neighbor_clusters = self._find_fixed_radius_nn(x)

        if len(neighbor_clusters) < 1:
            # create new micro cluster
            if len(self._micro_clusters) > 0:
                self._micro_clusters[max(self._micro_clusters.keys()) + 1] = DBSTREAMMicroCluster(
                    x=x, last_update=self._time_stamp, weight=1
                )
            else:
                self._micro_clusters[0] = DBSTREAMMicroCluster(
                    x=x, last_update=self._time_stamp, weight=1
                )
        else:
            # update existing micro clusters
            current_centers = {}
            for i in neighbor_clusters.keys():
                current_centers[i] = self._micro_clusters[i].center
                self._micro_clusters[i].weight = (
                    self._micro_clusters[i].weight
                    * 2
                    ** (
                        -self.fading_factor
                        * (self._time_stamp - self._micro_clusters[i].last_update)
                    )
                    + 1
                )

                # Update the center (i) with overlapping keys (j)
                self._micro_clusters[i].center = {
                    j: self._micro_clusters[i].center[j]
                    + self._gaussian_neighborhood(x, self._micro_clusters[i].center)
                    * (x[j] - self._micro_clusters[i].center[j])
                    for j in self._micro_clusters[i].center.keys()
                    if j in x
                }
                self._micro_clusters[i].last_update = self._time_stamp

                # update shared density
                for j in neighbor_clusters.keys():
                    if j > i:
                        try:
                            self.s[i][j] = (
                                self.s[i][j]
                                * 2 ** (-self.fading_factor * (self._time_stamp - self.s_t[i][j]))
                                + 1
                            )
                            self.s_t[i][j] = self._time_stamp
                        except KeyError:
                            try:
                                self.s[i][j] = 1
                                self.s_t[i][j] = self._time_stamp
                            except KeyError:
                                self.s[i] = {j: 1}
                                self.s_t[i] = {j: self._time_stamp}

            # prevent collapsing clusters
            for i in neighbor_clusters.keys():
                for j in neighbor_clusters.keys():
                    if j > i:
                        if (
                            self._distance(
                                self._micro_clusters[i].center,
                                self._micro_clusters[j].center,
                            )
                            < self.clustering_threshold
                        ):
                            # revert centers of mc_i and mc_j to previous positions
                            self._micro_clusters[i].center = current_centers[i]
                            self._micro_clusters[j].center = current_centers[j]

        self._time_stamp += 1

    def _cleanup(self):
        # Algorithm 2 of Michael Hahsler and Matthew Bolanos: Cleanup process to remove
        # inactive clusters and shared density entries from memory
        weight_weak = 2 ** (-self.fading_factor * self.cleanup_interval)

        micro_clusters = copy.deepcopy(self._micro_clusters)
        for i, micro_cluster_i in self._micro_clusters.items():
            try:
                value = 2 ** (
                    -self.fading_factor * (self._time_stamp - micro_cluster_i.last_update)
                )
            except OverflowError:
                continue

            if micro_cluster_i.weight * value < weight_weak:
                micro_clusters.pop(i)
                self.s.pop(i, None)
                self.s_t.pop(i, None)
                # Since self.s and self.s_t always have the same keys and are arranged in ascending orders
                for j in self.s:
                    if j < i:
                        self.s[j].pop(i, None)
                        self.s_t[j].pop(i, None)
                    else:
                        break

        # Update microclusters
        self._micro_clusters = micro_clusters

        for i in self.s.keys():
            for j in self.s[i].keys():
                try:
                    value = 2 ** (-self.fading_factor * (self._time_stamp - self.s_t[i][j]))
                except OverflowError:
                    continue

                if self.s[i][j] * value < self.intersection_factor * weight_weak:
                    self.s[i][j] = 0
                    self.s_t[i][j] = 0

    def _generate_weighted_adjacency_matrix(self):
        # Algorithm 3 of Michael Hahsler and Matthew Bolanos: Reclustering using
        # shared density graph
        weighted_adjacency_matrix = {}
        for i in list(self.s.keys()):
            for j in list(self.s[i].keys()):
                try:
                    if (
                        self._micro_clusters[i].weight <= self.minimum_weight
                        or self._micro_clusters[j].weight <= self.minimum_weight
                    ):
                        continue
                except KeyError:
                    continue

                value = self.s[i][j] / (
                    (self._micro_clusters[i].weight + self._micro_clusters[j].weight) / 2
                )
                if value > self.intersection_factor:
                    try:
                        weighted_adjacency_matrix[i][j] = value
                    except KeyError:
                        weighted_adjacency_matrix[i] = {j: value}

        return weighted_adjacency_matrix

    def _generate_labels(self, weighted_adjacency_list):
        # This function handles the weighted adjacency list created above and
        # generate a cluster label for all micro clusters, using a variant of
        # the DBSCAN algorithm proposed by Ester et al. for alpha-connected micro clusters

        # initiate labels of micro clusters to None
        labels = {i: None for i in self._micro_clusters.keys()}

        # cluster counter; in this algorithm, cluster labels starts with 0
        count = -1

        for index in labels.keys():
            if labels[index] is not None:
                continue
            count += 1
            labels[index] = count
            # if it is not in list of alpha-connected micro-clusters, label and continue
            if index not in weighted_adjacency_list.keys():
                continue
            seed_set = collections.deque(weighted_adjacency_list[index].keys())
            while seed_set:
                # check previously processed points
                if labels[seed_set[0]] is not None:
                    seed_set.popleft()
                    continue
                # proceed DBSCAN when seed set is not blank
                if seed_set:
                    labels[seed_set[0]] = count
                    # find neighbors
                    if seed_set[0] in weighted_adjacency_list.keys():
                        neighbor_neighbors = collections.deque(
                            weighted_adjacency_list[seed_set[0]].keys()
                        )
                        # add new neighbors to seed set
                        for neighbor_neighbor in neighbor_neighbors:
                            if labels[neighbor_neighbor] is None:
                                seed_set.append(neighbor_neighbor)

        return labels

    def _generate_clusters_from_labels(self, cluster_labels):
        # initiate the set for final clusters
        clusters = {}

        # generate set of clusters with the same label with the structure {j: micro_cluster_index}
        for i in range(max(cluster_labels.values()) + 1):
            j = 0
            mcs_with_label_i = {}
            for index, label in cluster_labels.items():
                if label == i:
                    mcs_with_label_i[j] = self._micro_clusters[index]
                    j += 1

            # generate a final macro-cluster from clusters with the same label using the
            # merge function of DBStreamMicroCluster
            macro_cluster = copy.deepcopy(mcs_with_label_i[0])
            for m in range(1, len(mcs_with_label_i)):
                macro_cluster.merge(mcs_with_label_i[m])

            clusters[i] = macro_cluster

        n_clusters = len(clusters)

        return n_clusters, clusters

    def _recluster(self):
        # Algorithm 3 of Michael Hahsler and Matthew Bolanos: Reclustering
        # using shared density graph
        if self.clustering_is_up_to_date:
            return

        weighted_adjacency_list = self._generate_weighted_adjacency_matrix()

        labels = self._generate_labels(weighted_adjacency_list)

        # We can only update given we have labels (possibly not on first pass)
        if labels:
            self._n_clusters, self._clusters = self._generate_clusters_from_labels(labels)
            self._centers = {i: self._clusters[i].center for i in self._clusters.keys()}

        self.clustering_is_up_to_date = True

    def learn_one(self, x, w=None):
        self._update(x)

        if self._time_stamp % self.cleanup_interval == 0:
            self._cleanup()

        self.clustering_is_up_to_date = False

    def predict_one(self, x, w=None):
        self._recluster()

        min_distance = math.inf

        # default result of all clustering results, regardless of whether there already
        # exists macro-clusters
        closest_cluster_index = 0

        for i, center_i in self._centers.items():
            distance = self._distance(center_i, x)
            if distance < min_distance:
                min_distance = distance
                closest_cluster_index = i
        return closest_cluster_index

    @property
    def n_clusters(self) -> int:
        self._recluster()
        return self._n_clusters

    @property
    def clusters(self) -> dict[int, DBSTREAMMicroCluster]:
        self._recluster()
        return self._clusters

    @property
    def centers(self) -> dict:
        self._recluster()
        return self._centers

    @property
    def micro_clusters(self) -> dict[int, DBSTREAMMicroCluster]:
        return self._micro_clusters


class DBSTREAMMicroCluster(metaclass=ABCMeta):
    """DBStream Micro-cluster class"""

    def __init__(self, x=None, last_update=None, weight=None):
        self.center = x
        self.last_update = last_update
        self.weight = weight

    def merge(self, cluster):
        # Using cluster.center.get allows updating clusters with different features
        self.center = {
            i: (self.center[i] * self.weight + cluster.center.get(i, 0.0) * cluster.weight)
            / (self.weight + cluster.weight)
            for i in self.center.keys()
        }
        self.weight += cluster.weight
