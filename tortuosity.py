import random
from typing import List, Tuple, Set, Dict, Callable
from math import sqrt, isclose
from sys import float_info
from functools import partial

from deap import creator, base, tools
import networkx

from cube import calc_ijk, calc_m


def calc_tortuosity(
    m_remain: List,
    num: int,
    shape: Tuple,
    seed: int,
    target_lengths: Tuple,
    remove_axix: Tuple = None,
):
    # TODO: docstring
    wx, wy, wz = 1.0, 1.0, 1.0
    if remove_axix is None:
        wx, wy, wz = 1.0, 1.0, 1.0
    elif "x" in remove_axix:
        wx = 0.0
    elif "y" in remove_axix:
        wy = 0.0
    elif "z" in remove_axix:
        wz = 0.0

    __callback: Callable = partial(
        calc_objective,
        shape=shape,
        weights=(wx, wy, wz),
        target_lengths=target_lengths,
    )
    best_ind = fit(m_remain, num, shape, __callback, seed)

    return best_ind

def calc_objective(
    m_selected: List,
    shape: Tuple,
    weights: Tuple,
    target_lengths: Tuple,
) -> float:
    # TODO: docstring
    wx, wy, wz = weights
    _max: float = float_info.max
    max_val = (_max,)
    if len(m_selected) > len(list(set(m_selected))):
        return max_val

    # convert to int
    m_selected: List = [int(m) for m in m_selected]

    G, boundary, nout = __calc_graph_props(m_selected, shape)

    nz, ny, nx = shape
    nxyz = nz * ny * nx

    # x
    not_has_x_lim: bool = not ("x0" in boundary and "x1" in boundary)
    if not_has_x_lim:
        lx = nxyz
    else:
        dist = __calc_bounds_dist(G, boundary["x0"], boundary["x1"])
        if dist is None:
            lx = nxyz
        else:
            lx = dist

    # y
    not_has_y_lim: bool = not ("y0" in boundary and "y1" in boundary)
    if not_has_y_lim:
        ly = nxyz
    else:
        dist = __calc_bounds_dist(
            G,
            boundary["y0"],
            boundary["y1"],
        )
        if dist is None:
            ly = nxyz
        else:
            ly = dist

    # z
    not_has_z_lim: bool = not ("z0" in boundary and "z1" in boundary)
    if not_has_z_lim:
        lz = nxyz
    else:
        dist = __calc_bounds_dist(
            G,
            boundary["z0"],
            boundary["z1"],
        )
        if dist is None:
            lz = nxyz
        else:
            lz = dist
    tx, ty, tz = target_lengths
    return tuple(
        [
            abs(wx * (lx - tx))
            + abs(wy * (ly - ty))
            + abs(wz * (lz - tz))
            + nout * sum(shape) / len(shape)
        ]
    )


def __calc_bounds_dist(
    G: networkx.DiGraph,
    set0: Set,
    set1: Set,
) -> None or float:
    # TODO: docstring
    dist_ls: List = []
    for m1 in list(set1):
        # for m1 in list(set1):
        lengths: float = networkx.multi_source_dijkstra_path_length(
            G, set0, m1, weight="dist"
        )
        dist_ls.extend(
            [lengths[s1] for s1 in list(set(lengths.keys()).intersection(set1))]
        )
    if len(dist_ls) == 0:
        return None
    else:
        return min(dist_ls)


def fit(
    m_remain: List,
    num: int,
    shape: Tuple,
    evaluate_callback: Callable,
    seed: int = 42,
    n_gen: int = 300,
    pop_size: int = 20,
    cx_pb: float = 0.8,
    mut_pb: float = 0.05,
):
    # TODO: docstring
    random.seed(seed)
    # set properties
    _methods: Set[str] = set(dir(creator))
    # set creator
    if "FitnessMin" not in _methods:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in _methods:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    # set toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_gene", lambda: random.choice(m_remain))
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_gene,
        num,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_callback)
    toolbox.register("mate", tools.cxBlend, alpha=0.2)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.01)
    toolbox.register("select", tools.selTournament, tournsize=10)
    cou: int = 0
    f_ls: List[float] = []
    pop = toolbox.population(n=pop_size)
    # Set initial values for no dupulicates
    shortest_set = __search_init(m_remain, shape)
    n_diff: int = num - len(list(shortest_set))
    _diff: List = list(set(m_remain).difference(shortest_set))
    for ind in pop:
        ind.clear()
        _shortest = list(shortest_set)
        _shortest.extend(list(random.sample(_diff, n_diff)))
        ind.extend(_shortest)
    fitnesses = list(map(toolbox.evaluate, pop))
    while cou < n_gen:
        cou += 1
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_pb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mut_pb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = offspring
        fits = [ind.fitness.values[0] for ind in pop]
        _f = min(fits)
        print(_f)
        f_ls.append(min(fits))
        if _f == 0.:
            break

    best_ind = tools.selBest(pop, 1)[0]
    return [int(m) for m in best_ind]

def __search_init(m_remain: List, shape: List):
    m_remain = m_remain.copy()
    G, boundary, _ = __calc_graph_props(m_remain, shape)
    shortest_x: List = []
    if "x0" in boundary and "x1" in boundary:
        _ls: List or None = __get_concat_shortest_path(G, boundary["x0"], boundary["x1"])
        if shortest_x is not None:
            shortest_x.extend(_ls)
    shortest_y: List = []
    if "y0" in boundary and "y1" in boundary:
        _ls: List or None = __get_concat_shortest_path(G, boundary["y0"], boundary["y1"])
        if _ls is not None:
            shortest_y.extend(_ls)
    shortest_z: List = []
    if "z0" in boundary and "z1" in boundary:
        _ls: List or None = __get_concat_shortest_path(G, boundary["z0"], boundary["z1"])
        if _ls is not None:
            shortest_z.extend(_ls)
    shortest_ls: List = []
    shortest_ls.extend(shortest_x)
    shortest_ls.extend(shortest_y)
    shortest_ls.extend(shortest_z)
    shortest_set = set(shortest_ls)

    return shortest_set


def __get_concat_shortest_path(G, set0, set1):
    for m0 in list(set0):
        for m1 in list(set1):
            if networkx.has_path(G, m0, m1):
                return networkx.dijkstra_path(G, m0, m1)
    return None


def __calc_graph_props(m_selected, shape):
    # convert to int
    m_selected: List = [int(m) for m in m_selected]

    # set graph attributes
    addnodes: List[Tuple[int, Dict]] = []
    addedges: List[Tuple[Tuple[int], Dict]] = []
    boundary: Dict[str, Set[int]] = {}
    nz, ny, nx = shape
    nxyz = nx * ny * nz
    nout: int = 0
    for m in m_selected:
        if not 0 <= m <= nxyz - 1:
            nout += 1
            continue
        i, j, k = calc_ijk(m, nx, ny)
        if i == 0:
            boundary.setdefault("x0", set()).add(m)
        elif i == nx - 1:
            boundary.setdefault("x1", set()).add(m)
        if j == 0:
            boundary.setdefault("y0", set()).add(m)
        elif j == ny - 1:
            boundary.setdefault("y1", set()).add(m)
        if k == 0:
            boundary.setdefault("z0", set()).add(m)
        elif k == nz - 1:
            boundary.setdefault("z1", set()).add(m)

        for i_tmp in (i - 1, i, i + 1):
            for j_tmp in (j - 1, j, j + 1):
                for k_tmp in (k - 1, k, k + 1):
                    if i_tmp < 0 or i_tmp >= nx:
                        continue
                    if j_tmp < 0 or j_tmp >= ny:
                        continue
                    if k_tmp < 0 or k_tmp >= nz:
                        continue
                    m_tmp = calc_m(i_tmp, j_tmp, k_tmp, nx, ny)
                    addnodes.append((m_tmp, dict()))
                    if i_tmp == i and j_tmp == j and k_tmp == k:
                        continue
                    dist = sqrt((i - i_tmp) ** 2 + (j - j_tmp) ** 2 + (k - k_tmp) ** 2)
                    if m_tmp in m_selected and isclose(dist, 1.0):
                        addedges.append((m, m_tmp, {"dist": dist}))

    # create graph
    G = networkx.DiGraph()
    G.add_nodes_from(addnodes)
    G.add_edges_from(addedges)

    return G, boundary, nout


from output import plt_any_val

if __name__ == "__main__":
    val = 15
    m_remain = [i for i in range(1000)]
    _ind = calc_tortuosity(m_remain, 200, (10, 10, 10), 42, (val, val, val))
    val_ls: List = []
    print(_ind)  #!
    for m in range(1000):
        if m in _ind:
            val_ls.append(1)
        else:
            val_ls.append(0)
    plt_any_val(val_ls, (10, 10, 10), f"./tortuosity/{val}", 1.0e-6)
