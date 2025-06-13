import copy

import numpy as np
import sklearn.neighbors as skn

# Assumes that each item only has one label -> only penalty gets max penalty out of all gt classes

knh_store = {}

def get_gt_clusters(gt, x):
    gt_distributions = {}
    for gtl in np.unique(gt):
        gtlindices = np.where(gt == gtl)[0]
        xgtl = x[gtlindices]
        centergtl = np.mean(xgtl, axis=0)
        diffgtl = xgtl - centergtl
        distgtl = np.sum(diffgtl**2, axis = 1)**0.5
        radiusgtl = max(distgtl)
        diffall = x - centergtl
        distall = np.sum(diffall**2, axis = 1)**0.5
        clugtlindices = np.where(distall <= radiusgtl)[0]
        gt_distributions[gtl] = {}
        for clugtli in clugtlindices:
            gtgtl = gt[clugtli]
            if gtgtl in gt_distributions[gtl].keys():
                gt_distributions[gtl][gtgtl] += 1
            else:
                gt_distributions[gtl][gtgtl] = 1
    return gt_distributions

def get_clu_clusters(clu, gt):
    clu_distributions = {}
    for clul in np.unique(clu):
        clulindices = np.where(clu == clul)[0]
        clu_distributions[clul] = {}
        for gtcluli in clulindices:
            gtgtl = gt[gtcluli]
            if gtgtl in clu_distributions[clul].keys():
                clu_distributions[clul][gtgtl] += 1
            else:
                clu_distributions[clul][gtgtl] = 1
    return clu_distributions


def cluster_mapping(gt_distributions, clu_distibutions, clu, gt):
    assignments = {}
    for clui in clu_distibutions.keys():
        mindiff = np.inf
        maxoverlap = 0
        for clu_gt in gt_distributions.keys():
            diff = 0
            for gti in clu_distibutions[clui].keys():
                if gti in gt_distributions[clu_gt].keys():
                    diff += max(0, clu_distibutions[clui][gti] - gt_distributions[clu_gt][gti])
                else:
                    diff += clu_distibutions[clui][gti]
            if diff < mindiff:
                mindiff = diff
                assignments[clui] = clu_gt
            if diff == 0:
                gtindices = np.where(gt == clu_gt)[0]
                cluindices = np.where(clu == clui)[0]
                overlap = len(np.intersect1d(gtindices, cluindices))
                if overlap > maxoverlap:
                    maxoverlap = overlap
                    assignments[clui] = clu_gt
    if '-1' in assignments:
        assignments['-1'] = -2
    if -1 in assignments:
        assignments[-1] = -2
    return assignments

def get_knhdist(x, index, neighbors, k):
    neighbors = neighbors.tolist()
    if index in neighbors:
        neighbors.remove(index)
    #print(neighbors)
    ns = min(k, len(neighbors))
    if ns > 0:
        ndists, nns = skn.NearestNeighbors(n_neighbors=ns).fit(x[neighbors]).kneighbors(x[index].reshape(1,-1), return_distance=True)
        #print(index, ndists)
        knhdist = sum(ndists[0])/k
    else:
        knhdist = 0
    return knhdist

def get_knhdist_c(x, clui, k):
    if str(clui) in knh_store.keys():
        return knh_store[str(clui)]
    else:
        sum_knh = 0
        for i in clui:
            knhi = get_knhdist(x, i, clui, k)
            #print(knhi)
            sum_knh += knhi
        knhdist = sum_knh/len(clui)
        knh_store[str(clui)] = knhdist
        return knhdist


def get_conn(x, index, clui, k):
    if len(clui) == 0:
        return 0
    knhc = get_knhdist_c(x, clui, k)
    knh = get_knhdist(x, index, clui, k)
    #print("conn", index, clui, k, knhc, "/", knh)
    if knh == 0:
        return 1
    if knh < knhc:
        return 1
    else:
        return knhc/knh


def get_fault_set(gt, clu, cluid, mapping):
    clulindices = np.where(clu == cluid)[0]
    if cluid == -1 or cluid == '-1':
        return clulindices
    else:
        label = mapping[cluid]
        fault_set = []
        for clui in clulindices:
            if gt[clui] != label:
                fault_set.append(clui)
        return fault_set

def get_penalty(x, gt, index, true_clu, fault_clu, k):
    trueindices = np.where(gt == true_clu)[0]
    assignindices = np.where(gt == fault_clu)[0]
    first = get_conn(x, index, trueindices, k)
    second = get_conn(x, index, assignindices, k)
    penalty = first * (1-second)
    #print("pen", first, second, penalty)
    return penalty

def get_cmm(x, gt, clu, w, k):
    gt = np.array(gt)
    clu = np.array(clu)
    x = np.array(x)
    gt_dist = get_gt_clusters(gt, x)
    clu_dist = get_clu_clusters(clu, gt)
    mapping = cluster_mapping(gt_dist, clu_dist, clu, gt)
    top_sum = 0
    bottom_sum = 0
    for clui in mapping.keys():
        fault_set_clui = get_fault_set(gt, clu, clui, mapping)
        for i in range(len(fault_set_clui)):
            findex = fault_set_clui[i]
            gtclu_indices = np.where(gt == gt[findex])[0]
            #print("fault", clui, findex, gt[findex], mapping[clui], gtclu_indices, flush=True)
            bottom = w[findex] * get_conn(x, findex, gtclu_indices, k)
            top = w[findex] * get_penalty(x, gt, findex, gt[findex], mapping[clui], k)
            #print("F", bottom, top)
            top_sum += top
            bottom_sum += bottom
    #print("final", top_sum, bottom_sum)
    if bottom_sum == 0:
        return 1
    return 1 - top_sum/bottom_sum



if __name__ == '__main__':

    gt = ['A','A','B','A','B','B','B','B']
    clu = ['a', 'a', 'b', 'a', 'c', 'c', 'b', 'a']

    x = [[1,1],[2,1],[3,3],[4,3],[5,6],[6,6],[7,6],[8,1]]
    w = [1]*len(gt)
    k = 2
    #print(len(gt), len(clu), len(x))

    print(get_cmm(x,gt,clu,w,k))

