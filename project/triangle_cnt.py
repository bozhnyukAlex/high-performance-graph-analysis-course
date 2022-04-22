from typing import List

from pygraphblas import types, Matrix


def triangles_count(adjacency_matrix: Matrix) -> List[int]:
    if not adjacency_matrix.square:
        raise ValueError("Must be a square matrix: adjacency matrix")

    if adjacency_matrix.type != types.BOOL:
        raise ValueError(
            f"Wrong type of matrix: Actual: {adjacency_matrix.type}, but Expected: BOOL"
        )

    res = adjacency_matrix

    for _ in range(2):
        res = adjacency_matrix.mxm(res, cast=types.INT64, accum=types.INT64.PLUS)

    res = res.diag().reduce_vector()
    res /= 2

    vertices, triangles = res.to_lists()
    tr_count = [0] * adjacency_matrix.nrows
    for i, vertex in enumerate(vertices):
        tr_count[vertex] = triangles[i]
    return tr_count
