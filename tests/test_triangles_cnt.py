import pygraphblas as pgb
import pytest

from project import triangles_count


@pytest.fixture(params=[pgb.INT64, pgb.INT32, pgb.FC64, pgb.UINT8])
def pgb_types(request):
    return request.param


@pytest.mark.parametrize(
    "size, I, J, V, expected_ans",
    [
        (3, [0, 1, 0, 1, 1, 2], [1, 2, 2, 1, 1, 2], [True] * 6, [1, 1, 1]),
        (
            4,
            [0, 1, 2, 0, 3, 1, 3, 2],
            [1, 2, 1, 3, 0, 3, 2, 0],
            [True] * 8,
            [3, 3, 3, 3],
        ),
        (
            5,
            [0, 1, 2, 2, 1, 3, 0, 3, 3, 4, 0, 2, 3],
            [1, 2, 1, 2, 3, 1, 3, 0, 3, 4, 4, 3, 2],
            [True] * 13,
            [1, 2, 1, 2, 0],
        ),
        (
            6,
            [0, 1, 0, 5, 0, 4, 4, 5, 4, 3, 3, 5, 3, 2, 5, 2, 1, 5, 1, 2],
            [1, 0, 5, 0, 4, 0, 5, 4, 3, 4, 5, 3, 2, 3, 2, 5, 5, 1, 2, 1],
            [True] * 20,
            [2, 2, 2, 2, 2, 5],
        ),
        (
            12,
            [0, 1, 0, 5, 0, 4, 4, 5, 4, 3, 3, 5, 3, 2, 5, 2, 1, 5, 1, 2],
            [1, 0, 5, 0, 4, 0, 5, 4, 3, 4, 5, 3, 2, 3, 2, 5, 5, 1, 2, 1],
            [True] * 20,
            [2, 2, 2, 2, 2, 5, 0, 0, 0, 0, 0, 0],
        ),
    ],
)
def test_count_triangles(size, I, J, V, expected_ans):
    adj_matrix = pgb.Matrix.from_lists(I, J, V, nrows=size, ncols=size)
    assert triangles_count(adj_matrix) == expected_ans


def test_wrong_matrix_type_triangle_cnt(pgb_types):
    adjacency_matrix = pgb.Matrix.dense(pgb_types, nrows=3, ncols=3)
    with pytest.raises(ValueError):
        triangles_count(adjacency_matrix)


def test_non_square_triangle_cnt():
    adjacency_matrix = pgb.Matrix.dense(pgb.BOOL, nrows=2, ncols=5)
    with pytest.raises(ValueError):
        triangles_count(adjacency_matrix)
