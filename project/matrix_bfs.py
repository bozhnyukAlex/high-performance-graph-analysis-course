from pygraphblas import Matrix, types, descriptor
from itertools import count
from typing import List, Tuple

__all__ = ["bfs", "multiple_source_bfs"]


def bfs(adjacency_matrix: Matrix, start_vertex: int) -> List[int]:
    """
    Implementation of LA BFS algorithm for given graph and start vertex
    Counting a number of steps from start vertex to others.

    Parameters
    ----------
    adjacency_matrix: Matrix
        Adjacency matrix of given graph
    start_vertex: int
        Vertex from BFS if started

    Returns
    -------
    List[int]:
        List with counts of steps from start_vertex to others.
        If vertex is not reachable, value in list equals -1.
    """
    return multiple_source_bfs(adjacency_matrix, [start_vertex])[0][1]


def multiple_source_bfs(
    adjacency_matrix: Matrix, start_vertices: List[int]
) -> List[Tuple[int, List[int]]]:
    """
    Implementation of LA Multiple Source BFS algorithm for given graph and list of start vertices
    Counting a number of steps from start vertices to others.

    Parameters
    ----------
    adjacency_matrix: Matrix
        Adjacency matrix of given graph
    start_vertices: List[int]
        List of start vertices

    Returns
    -------
    List[Tuple[int, List[int]]]:
         List of tuples:
            (start_vertex, list of steps counts from start_vertex to others)
    """
    return _bfs_helper(adjacency_matrix, start_vertices)


def _bfs_helper(
    adjacency_matrix: Matrix, start_vertices: List[int]
) -> List[Tuple[int, List[int]]]:
    """
    Helper function for LA BFS and LA Multiple Source BFS for
    given graph and list of start vertices
    For LA Single Source BFS just put list of vertices with one vertex
    Parameters
    ----------
    adjacency_matrix: Matrix
        Adjacency matrix of given graph
    start_vertices: List[int]
        List of start vertices

    Returns
    -------
    List[Tuple[int, List[int]]]:
         List of tuples:
            (start_vertex, list of steps counts from start_vertex to others)
    """
    _check_conditions(adjacency_matrix, start_vertices)

    res_matrix = Matrix.dense(
        types.INT64, nrows=len(start_vertices), ncols=adjacency_matrix.ncols, fill=-1
    )
    curr_front = Matrix.sparse(
        types.BOOL, nrows=len(start_vertices), ncols=adjacency_matrix.ncols
    )
    was_mask = Matrix.sparse(
        types.BOOL, nrows=len(start_vertices), ncols=adjacency_matrix.ncols
    )

    for i, vertex in enumerate(start_vertices):
        res_matrix[i, vertex] = 0
        curr_front[i, vertex] = True
        was_mask[i, vertex] = True

    step_number = 1
    prev_vals_number = -1
    while prev_vals_number != was_mask.nvals:
        prev_vals_number = was_mask.nvals
        curr_front.mxm(
            adjacency_matrix, mask=was_mask, out=curr_front, desc=descriptor.RC
        )
        was_mask.eadd(
            curr_front,
            curr_front.type.lxor_monoid,
            out=was_mask,
            desc=descriptor.R,
        )
        res_matrix.assign_scalar(step_number, mask=curr_front)
        step_number = step_number + 1

    return [
        (vertex, list(res_matrix[i].vals)) for i, vertex in enumerate(start_vertices)
    ]


def _check_conditions(adjacency_matrix: Matrix, start_vertices: List[int]):
    """
    Check that graph matrix is square,
    matrix has boolean type,
    start vertices are not negative and not more the number of vertices

    Parameters
    ----------
    adjacency_matrix: Matrix
       Adjacency matrix of given graph
    start_vertices: List[int]
       List of start vertices
    """
    if not adjacency_matrix.square:
        raise ValueError("Adjacency matrix must be a square matrix")

    if adjacency_matrix.type != types.BOOL:
        raise ValueError(
            f"Wrong matrix type: Actual: {adjacency_matrix.type}, but Expected: BOOL"
        )

    if any(start < 0 or start >= adjacency_matrix.nrows for start in start_vertices):
        raise ValueError(
            f"The number of the starting vertex or vertices must be between 0 and {adjacency_matrix.nrows - 1}"
        )
