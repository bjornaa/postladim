from postladim import cellcount

# Miss some failing tests
#   len(X) != len(Y)
#   len(W) is wrong
#   i0 > i1 or j0 > j1


def test_default():
    """Default arguments"""
    X = [11.2, 11.8, 12.2, 12.3]
    Y = [0.8, 1.2, 1.4, 3.1]
    C = cellcount(X, Y)
    # Test subgrid
    assert C.shape == (3, 2)
    assert all(C.X == [11, 12])
    assert all(C.Y == [1, 2, 3])
    assert (C == [[1, 2], [0, 0], [0, 1]]).all()
    # Test values
    assert C.sum() == len(X)
    assert C.sel(X=11, Y=1) == 1
    assert C.sel(X=12, Y=1) == 2
    assert C.sel(X=12, Y=3) == 1
    assert C.sel(X=11, Y=2) == 0


def test_grid_limits():
    """Test grid limit specificaton"""
    X = [11.2, 11.8, 12.2, 12.3]
    Y = [0.8, 1.2, 1.4, 3.1]
    i0, i1, j0, j1 = 10, 14, 0, 2
    C = cellcount(X, Y, grid_limits=(i0, i1, j0, j1))
    assert C.shape == (j1 - j0, i1 - i0)  # (2, 4)
    assert C.sum() == len(X) - 1  # 3, one point outside
    assert (C == [[0, 0, 0, 0], [0, 1, 2, 0]]).all()
    assert C.sel(X=11, Y=1) == 1
    assert C.sel(X=12, Y=1) == 2
    assert C.sel(X=11, Y=0) == 0


def test_weight():
    """Test weighted counting"""
    X = [11.2, 11.8, 12.2, 12.3]
    Y = [0.8, 1.2, 1.4, 3.1]
    W = [1, 2, 3, 4]
    C = cellcount(X, Y, W)
    assert C.shape == (3, 2)
    assert C.sum() == sum(W)
    assert (C == [[W[0], W[1] + W[2]], [0, 0], [0, W[3]]]).all()
    assert C.sel(X=11, Y=1) == W[0]
    assert C.sel(X=12, Y=1) == W[1] + W[2]
    assert C.sel(X=12, Y=3) == W[3]
