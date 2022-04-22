import pygraphblas as pgb
import pytest

from project import sssp, mssp


def test_bad_start_vertex_sssp():
    adjacency_matrix = pgb.Matrix.dense(pgb.BOOL, nrows=4, ncols=4)
    with pytest.raises(ValueError):
        sssp(adjacency_matrix, -1)


def test_bad_start_vertex_mssp():
    adjacency_matrix = pgb.Matrix.dense(pgb.BOOL, nrows=2, ncols=2)
    with pytest.raises(ValueError):
        mssp(adjacency_matrix, [10])


def test_non_square_single():
    adj_matrix = pgb.Matrix.dense(pgb.BOOL, nrows=4, ncols=5)
    with pytest.raises(ValueError):
        sssp(adj_matrix, 0)


def test_non_square_multi():
    adjacency_matrix = pgb.Matrix.dense(pgb.BOOL, nrows=7, ncols=8)
    with pytest.raises(ValueError):
        mssp(adjacency_matrix, [0])


@pytest.mark.parametrize(
    "size, I, J, V, start_vertex, expected_ans",
    [
        (2, [0], [1], [2.0], 0, [0.0, 2.0]),
        (
            3,
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0.4, 19.0, 1.0, 3761.9, 3110.0, 5000.7, 1122.0, 1.1, 9999.5],
            2,
            [1122.0, 1.1, 0.0],
        ),
        (
            4,
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 2, 10, 100, 7, 9, 21, 12, 23],
            0,
            [0, 2, 10, -1],
        ),
        (5, [0], [1], [7], 0, [0, 7, -1, -1, -1]),
    ],
)
def test_single_ssp(I, J, V, size, start_vertex, expected_ans):
    adj_matrix = pgb.Matrix.from_lists(I, J, V, nrows=size, ncols=size)
    assert sssp(adj_matrix, start_vertex) == expected_ans


@pytest.mark.parametrize(
    "size, I, J, V, start_vertices, expected",
    [
        (2, [0], [1], [2.0], [0, 1], [(0, [0.0, 2.0]), (1, [-1, 0.0])]),
        (
            3,
            [0, 0, 1, 1, 2, 2],
            [0, 2, 0, 2, 1, 2],
            [0.0, 5.0, 1.0, 3000.0, 0.0, 3871.0],
            {0, 2},
            [(0, [0.0, 5.0, 5.0]), (2, [1.0, 0.0, 0.0])],
        ),
        (
            4,
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 3, 1, 5000, 5000, 5000, 5000, 1, 5000],
            [0, 1, 2],
            [
                (0, [0.0, 2.0, 1.0, -1]),
                (1, [5000.0, 0.0, 5000.0, -1]),
                (2, [5000.0, 1.0, 0.0, -1]),
            ],
        ),
        (
            5,
            [0],
            [1],
            [7],
            [3, 4],
            [(3, [-1, -1, -1, 0, -1]), (4, [-1, -1, -1, -1, 0])],
        ),
    ],
)
def test_multi_ssp(size, I, J, V, start_vertices, expected):
    adj_matrix = pgb.Matrix.from_lists(I, J, V, nrows=size, ncols=size)
    assert mssp(adj_matrix, start_vertices) == expected
