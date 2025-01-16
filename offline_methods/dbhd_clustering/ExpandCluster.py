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
