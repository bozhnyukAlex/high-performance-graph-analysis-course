from typing import List, Tuple

import pygraphblas as gb

__all__ = ["sssp", "mssp"]


def sssp(adj_matrix: gb.Matrix, start_vertex: int) -> List[int]:
    return tuple(next(iter(mssp(adj_matrix, [start_vertex]))))[1]


def mssp(
    adj_matrix: gb.Matrix, start_vertices: List[int]
) -> List[Tuple[int, List[int]]]:
    if not adj_matrix.square:
        raise ValueError("Error: adjacency matrix must be a square matrix")

    if any(
        start_vertex < 0 or start_vertex >= adj_matrix.nrows
        for start_vertex in start_vertices
    ):
        raise ValueError(
            f"Error: count of start vertices must be in range of 0 to {adj_matrix.nrows - 1}"
        )

    curr_front = gb.Matrix.sparse(
        adj_matrix.type, nrows=len(start_vertices), ncols=adj_matrix.ncols
    )

    for i, j in enumerate(start_vertices):
        curr_front.assign_scalar(0, i, j)

    for _ in range(adj_matrix.nrows):
        curr_front.mxm(
            adj_matrix,
            semiring=adj_matrix.type.min_plus,
            out=curr_front,
            accum=adj_matrix.type.min,
        )

    def __get_sp(vertices, distances):
        result = [-1] * adj_matrix.nrows
        for i, vertex in enumerate(vertices):
            result[vertex] = distances[i]
        return result

    return [
        (vertex, __get_sp(*curr_front[i].to_lists()))
        for i, vertex in enumerate(start_vertices)
    ]
