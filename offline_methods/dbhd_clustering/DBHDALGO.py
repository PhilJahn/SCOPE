import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree

#from DPoint import DPoint
#from ExpandCluster import start_sigma


class DPoint:


    def __init__(self, coordinates, name):
        self.coordinates = coordinates
        self.neighbours = None
        self.avg_k_distance = None
        self.name = name



ind_to_name = {}
name_to_label = {}
name_to_dPoint = {}
label_counter = 1
data_matrix = None




delete_set = set([])


def start_sigma(sigma, ind_to_name, name_to_dPoint, rho, beta):
    global delete_set
    delete_set = set([])
    new_neighbours = set([])
    sigma.neighbours = set(sigma.neighbours)
    for neighbour in sigma.neighbours:
        dPoint = name_to_dPoint[ind_to_name[neighbour]]
        new_neighbours.update(disjunkt(sigma, dPoint, rho, beta, neighbour))

    chain_cluster(sigma, ind_to_name, name_to_dPoint, rho, beta, new_neighbours)
    return sigma


def chain_cluster(sigma, ind_to_name, name_to_dPoint, rho, beta, new_neighbours):
    global delete_set
    sigma.neighbours.update(new_neighbours)
    sigma.neighbours = sigma.neighbours.difference(delete_set)
    delete_set = set([])
    tmp_neighbours = set([])
    if len(new_neighbours) > 0:
        for neighbour in new_neighbours:
            dPoint = name_to_dPoint[ind_to_name[neighbour]]
            tmp_neighbours.update(disjunkt(sigma, dPoint, rho, beta, neighbour))
        chain_cluster(sigma, ind_to_name, name_to_dPoint, rho, beta, tmp_neighbours)


def disjunkt(sigma, neighour, rho, beta, ind):
    disjunctSet = [_ for _ in neighour.neighbours if _ not in sigma.neighbours]
    rho_con = len(neighour.neighbours) - len(disjunctSet) < rho * len(disjunctSet)
    avg_con = neighour.avg_k_distance * beta > sigma.avg_k_distance
    if (rho_con or avg_con):
        delete_set.add(ind)
        return set([])
    else:
        return set(disjunctSet)





def start_clustering(min_cluster_size, rho, beta):
    global label_counter
    X = index_to_X()
    # label_counter += 1
    epsilon = search_epsilon(min_cluster_size, X)
    if (epsilon == None):
        for key, item in name_to_dPoint.items():
            name = item.name
            name_to_label[name] = label_counter
        return collectLabels()
    kd_query(X, epsilon)
    sigma = sigma_suche()
    sigma_final_result = start_sigma(sigma, ind_to_name, name_to_dPoint, rho, beta)
    for ind in sigma_final_result.neighbours:
        tmp_name = ind_to_name[ind]
        name_to_label[tmp_name] = label_counter
        del name_to_dPoint[tmp_name]
    label_counter += 1
    size_item = len(name_to_dPoint.items())
    if size_item < min_cluster_size:
        for key, item in name_to_dPoint.items():
            name = item.name
            name_to_label[name] = -1
        return collectLabels()
    else:
        return start_clustering(min_cluster_size, rho, beta)


def str_to_name(dPoint):
    name = ""
    for item in dPoint:
        name += str(item) + "x"
    return name


def init(X):
    global data_matrix
    data_matrix = X
    count = 0
    for item in X:
        tmp_name = str_to_name(item)
        if tmp_name not in name_to_dPoint:
            count += 1
            d2Punkt = DPoint(item, tmp_name)
            name_to_dPoint[tmp_name] = d2Punkt
    # print(count)


def index_to_X():
    list = []
    for key, item in name_to_dPoint.items():
        list.append(item.coordinates)
    return np.array(list)


def search_epsilon(min_cluster_size, X):
    if min_cluster_size < X.shape[0]:
        factor = 1
        metricNN = NearestNeighbors(n_neighbors=min_cluster_size * factor + 1, leaf_size=min_cluster_size * factor + 1,
                                    n_jobs=-1).fit(X)
        distances, indices = metricNN.kneighbors(X)

        # epsilon = 0
        # for i in distances[0]:
        #     epsilon += i
        # epsilon /= epsilon / min_cluster_size
        epsilon = distances[0][min_cluster_size]
        for i, distance in enumerate(distances):
            tmp_epsilon = distance[min_cluster_size]
            tmp_distance = 0
            for dis in distance:
                tmp_distance += dis
            name = str_to_name(X[i])
            name_to_dPoint[name].avg_k_distance = tmp_distance / min_cluster_size
            if epsilon > tmp_epsilon:
                epsilon = tmp_epsilon
        return epsilon
    else:
        return None


def kd_query(X, epsilon):
    kd_tree = KDTree(X)
    ind = kd_tree.query_radius(X, r=epsilon, )
    for i, item in enumerate(X):
        tmp_name = str_to_name(item)
        name_to_dPoint[tmp_name].neighbours = ind[i]
        ind_to_name[i] = tmp_name


def sigma_suche():
    sigma = None
    for key, d2Punkt in name_to_dPoint.items():
        if sigma == None:
            sigma = d2Punkt
        tmp_avg = d2Punkt.avg_k_distance
        tmp_neighours_len = len(d2Punkt.neighbours)
        # if tmp_avg < sigma.avg_k_distance and tmp_nachbarn > len(sigma.neighbours):
        #     sigma = d2Punkt
        if tmp_avg < sigma.avg_k_distance:
            sigma = d2Punkt
        if tmp_avg == sigma.avg_k_distance:
            sigma = compare(sigma, d2Punkt)
    return sigma


def compare(d1, d2):
    d2Coor = d2.coordinates
    for i, value in enumerate(d1.coordinates):
        if value == d2Coor[i]:
            continue
        elif value < d2Coor[i]:
            return d1
        else:
            return d2
    return d1


def collectLabels():
    y_label = []
    for datapoint in data_matrix:
        name = str_to_name(datapoint)
        if name in name_to_label:
            y_label.append(name_to_label[name])
        else:
            y_label.append(0)
    global label_counter
    return np.array(y_label).reshape(-1)


def run(X, min_cluster_size, rho=1, beta=0.1):
    global ind_to_name, name_to_dPoint, name_to_label, label_counter, data_matrix
    ind_to_name = {}
    name_to_label = {}
    name_to_dPoint = {}
    label_counter = 0
    data_matrix = None
    init(X)
    return start_clustering(min_cluster_size, rho, beta)
class DBHD:

    def __init__(self, min_cluster_size, rho=1.2, beta=0.1):
        self.min_cluster_size = min_cluster_size
        self.rho = rho
        self.beta = beta

    def fit_predict(self, X):
       return  run(X, self.min_cluster_size, self.rho, self.beta)



