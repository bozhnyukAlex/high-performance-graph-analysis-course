from typing import List, Tuple

import pygraphblas as gb

__all__ = ["single_ssp", "multiple_ssp"]


def single_ssp(adjacency_matrix: gb.Matrix, start_vertex: int) -> List[int]:
    """
    Solve SSSP problem for given graph and start vertex
    Parameters
    ----------
    adjacency_matrix: Matrix
        Adjacency matrix of given graph
    start_vertex: int
        Vertex for which the shortest paths are counted

    Returns
    -------
    List[int]:
        List with resulted shortest paths from start_vertex to others
    """

    _check_conditions(adjacency_matrix, start_vertex)

    calculated_ssp = _calculate_ssp(adjacency_matrix, start_vertex)
    vertices, s_paths = calculated_ssp.to_lists()

    result = [-1] * adjacency_matrix.nrows
    for i, vertex in enumerate(vertices):
        result[vertex] = s_paths[i]

    return result


def multiple_ssp(
    adjacency_matrix: gb.Matrix, start_vertices: List[int]
) -> List[Tuple[int, List[int]]]:
    """
    Solve Multiple SSP problem for given graph and list of start vertices
    Parameters
    ----------
    adjacency_matrix: Matrix
        Adjacency matrix of given graph
    start_vertices: List[int]
        List of vertices for which the shortest paths are counted

    Returns
    -------
    List[Tuple[int, List[int]]]:
        List of tuples:
            (start_vertex, list of resulted shortest paths from start_vertex to others)

    """

    return [
        (start_vertex, single_ssp(adjacency_matrix, start_vertex))
        for start_vertex in start_vertices
    ]


def _calculate_ssp(adjacency_matrix: gb.Matrix, start_vertex: int) -> gb.Vector:
    """
    Calculate the shortest paths from start_vertex to others

    Parameters
    ----------
    adjacency_matrix: Matrix
        Adjacency matrix of given graph
    start_vertex: int
        Vertex for which the shortest paths are counted

    Returns
    -------
    Vector:
        vector with resulted shortest paths
    """
    current_front = gb.Vector.sparse(adjacency_matrix.type, size=adjacency_matrix.nrows)
    current_front[start_vertex] = 0

    rounds = 0
    while rounds < adjacency_matrix.nrows:
        current_front.vxm(
            adjacency_matrix,
            semiring=adjacency_matrix.type.min_plus,
            out=current_front,
            accum=adjacency_matrix.type.min,
        )
        rounds = rounds + 1

    return current_front


def _check_conditions(adjacency_matrix: gb.Matrix, start_vertex: int) -> None:
    """
    Check that graph matrix is square and
    start vertices are not negative and not more the number of vertices

    Parameters
    ----------
    adjacency_matrix: Matrix
        Adjacency matrix of given graph
    start_vertex: int
        Vertex for which the shortest paths are counted
    """
    if not adjacency_matrix.square:
        raise ValueError("Error: adjacency matrix must be a square matrix")

    if start_vertex < 0 or start_vertex >= adjacency_matrix.nrows:
        raise ValueError(
            f"Error: count of start vertices must be in range of 0 to {adjacency_matrix.nrows - 1}"
        )
