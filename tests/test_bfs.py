import pygraphblas as pgb
import pytest

from project import bfs, multiple_source_bfs


@pytest.fixture(params=[pgb.INT64, pgb.INT32, pgb.FC64, pgb.UINT8])
def pgb_types(request):
    return request.param


def test_wrong_matrix_type_bfs(pgb_types):
    adjacency_matrix = pgb.Matrix.dense(pgb_types, nrows=3, ncols=3)
    with pytest.raises(ValueError):
        bfs(adjacency_matrix, 0)


def test_wrong_matrix_type_multiple_bfs(pgb_types):
    adjacency_matrix = pgb.Matrix.dense(pgb_types, nrows=3, ncols=3)
    with pytest.raises(ValueError):
        multiple_source_bfs(adjacency_matrix, [0])


def test_non_square_bfs():
    adj_matrix = pgb.Matrix.dense(pgb.BOOL, nrows=2, ncols=5)
    with pytest.raises(ValueError):
        bfs(adj_matrix, 0)


def test_non_square_multiple_bfs():
    adjacency_matrix = pgb.Matrix.dense(pgb.BOOL, nrows=6, ncols=7)
    with pytest.raises(ValueError):
        multiple_source_bfs(adjacency_matrix, [0])


def test_wrong_start_vertex_bfs():
    adjacency_matrix = pgb.Matrix.dense(pgb.BOOL, nrows=3, ncols=3)
    with pytest.raises(ValueError):
        bfs(adjacency_matrix, -1)


def test_wrong_start_vertex_multiple_bfs():
    adjacency_matrix = pgb.Matrix.dense(pgb.BOOL, nrows=3, ncols=3)
    with pytest.raises(ValueError):
        multiple_source_bfs(adjacency_matrix, [10])


@pytest.mark.parametrize(
    "I, J, V, size, start_vertex, expected_ans",
    [
        (
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [True, True, True, True],
            5,
            0,
            [0, 1, 2, 3, 4],
        ),
        ([0], [1], [False], 5, 0, [0, -1, -1, -1, -1]),
        (
            [0, 0, 2, 3, 4],
            [1, 2, 1, 4, 3],
            [True, True, True, True, True],
            5,
            0,
            [0, 1, 1, -1, -1],
        ),
        (
            [0, 0, 2, 3, 4],
            [1, 2, 1, 4, 3],
            [False, True, True, True, True],
            5,
            0,
            [0, 2, 1, -1, -1],
        ),
        (
            [0, 0, 2, 3, 4],
            [1, 2, 1, 4, 3],
            [True, True, True, True, True],
            5,
            3,
            [-1, -1, -1, 0, 1],
        ),
    ],
)
def test_bfs(I, J, V, size, start_vertex, expected_ans):
    adjacency_matrix = pgb.Matrix.from_lists(I, J, V, nrows=size, ncols=size)
    assert bfs(adjacency_matrix, start_vertex) == expected_ans


@pytest.mark.parametrize(
    "I, J, V, size, start_vertices, expected_ans",
    [
        (
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [True, True, True, True],
            5,
            [0, 1, 2, 3, 4],
            [
                (0, [0, 1, 2, 3, 4]),
                (1, [-1, 0, 1, 2, 3]),
                (2, [-1, -1, 0, 1, 2]),
                (3, [-1, -1, -1, 0, 1]),
                (4, [-1, -1, -1, -1, 0]),
            ],
        ),
        (
            [0, 0, 2, 3, 4],
            [1, 2, 1, 4, 3],
            [True, True, True, True, True],
            5,
            [0, 0],
            [(0, [0, 1, 1, -1, -1]), (0, [0, 1, 1, -1, -1])],
        ),
        (
            [0, 0, 2, 3, 4],
            [1, 2, 1, 4, 3],
            [True, True, True, True, True],
            5,
            [0, 3],
            [(0, [0, 1, 1, -1, -1]), (3, [-1, -1, -1, 0, 1])],
        ),
        (
            [0],
            [1],
            [False],
            5,
            [0, 2],
            [(0, [0, -1, -1, -1, -1]), (2, [-1, -1, 0, -1, -1])],
        ),
    ],
)
def test_multiple_bfs(I, J, V, size, start_vertices, expected_ans):
    adjacency_matrix = pgb.Matrix.from_lists(I, J, V, nrows=size, ncols=size)
    assert multiple_source_bfs(adjacency_matrix, start_vertices) == expected_ans
