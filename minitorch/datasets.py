import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """
    Generates a simple dataset of N points in two dimensions.

    The classification function is y = 1 if x_1 < 0.5 else 0.

    Parameters
    N : int
        The number of points to generate.

    Returns
    Graph
        A Graph dataclass containing the X and y values of the dataset.
    """

    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """
    Generates a diagonal dataset of N points in two dimensions.

    The classification function is y = 1 if x_1 + x_2 < 0.5 else 0.

    Parameters
    N : int
        The number of points to generate.

    Returns
    Graph
        A Graph dataclass containing the X and y values of the dataset.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """
    Generates a split dataset of N points in two dimensions.

    The classification function is y = 1 if x_1 < 0.2 or x_1 > 0.8 else 0.

    Parameters
    N : int
        The number of points to generate.

    Returns
    Graph
        A Graph dataclass containing the X and y values of the dataset.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """
    Generates an xor dataset of N points in two dimensions.

    The classification function is y = 1 if (x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5) else 0.

    Parameters
    N : int
        The number of points to generate.

    Returns
    Graph
        A Graph dataclass containing the X and y values of the dataset.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """
    Generates a circle dataset of N points in two dimensions.

    The classification function is y = 1 if x_1^2 + x_2^2 > 0.1 else 0.

    Parameters
    N : int
        The number of points to generate.

    Returns
    Graph
        A Graph dataclass containing the X and y values of the dataset.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """
    Generates a spiral dataset of N points in two dimensions.

    The classification function is y = 0 if the point is on the first spiral arm and y = 1
    if the point is on the second spiral arm.

    Parameters
    N : int
        The number of points to generate.

    Returns
    Graph
        A Graph dataclass containing the X and y values of the dataset.
    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
