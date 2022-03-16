from pygraphblas import Matrix, types, descriptor
from itertools import count
from typing import List, Tuple

__all__ = ["bfs", "multiple_source_bfs"]


def _bfs_helper(
    adjacency_matrix: Matrix, start_vertices: List[int]
) -> List[Tuple[int, List[int]]]:
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

    res_matrix = Matrix.dense(
        types.INT64, nrows=len(start_vertices), ncols=adjacency_matrix.ncols, fill=-1
    )
    curr_front = Matrix.sparse(
        types.BOOL, nrows=len(start_vertices), ncols=adjacency_matrix.ncols
    )
    was_mask = Matrix.sparse(
        types.BOOL, nrows=len(start_vertices), ncols=adjacency_matrix.ncols
    )

    for i, j in enumerate(start_vertices):
        res_matrix.assign_scalar(0, i, j)
        curr_front.assign_scalar(True, i, j)
        was_mask.assign_scalar(True, i, j)

    for step_number in count(1):
        prev_nvals = was_mask.nvals
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
        if was_mask.nvals == prev_nvals:
            break

    return [
        (vertex, list(res_matrix[i].vals)) for i, vertex in enumerate(start_vertices)
    ]


def bfs(adjacency_matrix: Matrix, start_vertex: int) -> List[int]:
    return tuple(next(iter(_bfs_helper(adjacency_matrix, [start_vertex]))))[1]


def multiple_source_bfs(
    adjacency_matrix: Matrix, start_vertices: List[int]
) -> List[Tuple[int, List[int]]]:
    return _bfs_helper(adjacency_matrix, start_vertices)
