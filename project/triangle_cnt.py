from typing import List

from pygraphblas import types, Matrix


def triangles_count(adjacency_matrix: Matrix) -> List[int]:
    """
    Takes a graph and for each vertex calculates
    the number of triangles that contain that vertex

    Parameters
    ----------
    adjacency_matrix: Matrix
        Adjacency matrix of graph
    Returns
    -------
    List[int]:
        List that specifies for each vertex
        how many triangles it participates in.
    """
    if not adjacency_matrix.square:
        raise ValueError("Must be a square matrix: adjacency matrix")

    if adjacency_matrix.type != types.BOOL:
        raise ValueError(
            f"Wrong type of matrix: Actual: {adjacency_matrix.type}, but Expected: BOOL"
        )

    prepared_graph = _prepare_graph(adjacency_matrix)

    # Triangle {i,j,k} means we have a two-edged path ik, kj (mxm) and edge ij (mask) itself
    tr_matrix = prepared_graph.mxm(prepared_graph, cast=types.INT64, mask=prepared_graph)
    res_tr_counts = []

    for v in range(adjacency_matrix.nrows):
        tr_cnt_v = tr_matrix[v].reduce() // 2
        res_tr_counts.append(tr_cnt_v)

    return res_tr_counts


def _prepare_graph(adjacency_matrix: Matrix) -> Matrix:
    """
    Prepare graph: make it undirected, get rid of self-loops

    Parameters
    ----------
    adjacency_matrix: Matrix
        Adjacency matrix of graph
    Returns
    -------
        Adjacency matrix of prepared graph:
        undirected, without self-loops
    """
    transposed_am = adjacency_matrix.transpose()
    undirected = adjacency_matrix.union(transposed_am)

    for i in range(adjacency_matrix.nrows):
        undirected[i, i] = False

    return undirected
