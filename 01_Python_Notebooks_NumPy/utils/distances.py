"""Метрики за разстояние и сходство.

Разстоянието между два обекта е основата на всеки метод, който групира или
класифицира по сходство.
"""

import numpy as np


def euclidean(a, b):
    """Евклидово разстояние между два вектора.

    Формулата е ||a - b||, тоест коренът от сумата на квадратите на разликите.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.sum((a - b) ** 2)))


def pairwise_euclidean(X, Y=None):
    """Разстояния между всяка двойка редове, без нито един цикъл.

    Parameters
    ----------
    X : ndarray с форма (n, d)
    Y : ndarray с форма (m, d); ако е None, се използва X

    Returns
    -------
    ndarray с форма (n, m), където елемент [i, j] е разстоянието между X[i] и Y[j]

    Notes
    -----
    Трикът е broadcasting: X[:, None, :] има форма (n, 1, d), Y[None, :, :] има
    форма (1, m, d). При изваждане двете се разширяват до (n, m, d) и сумирането
    по последната ос дава търсената матрица (n, m).

    За големи набори това пази (n, m, d) стойности в паметта наведнъж. Тогава се
    използва разлагането ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y, което иска само
    (n, m).
    """
    X = np.asarray(X, dtype=float)
    Y = X if Y is None else np.asarray(Y, dtype=float)
    diff = X[:, None, :] - Y[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def cosine_similarity(X, Y=None):
    """Косинусово сходство между всяка двойка редове.

    Връща стойности в [-1, 1]: 1 означава еднопосочни вектори, 0 означава ортогонални.
    За разлика от евклидовото разстояние мери само посока, не големина, затова
    е предпочитано при текст, където дължината на документа не бива да доминира.
    """
    X = np.asarray(X, dtype=float)
    Y = X if Y is None else np.asarray(Y, dtype=float)
    Xn = X / np.where((n := np.linalg.norm(X, axis=1, keepdims=True)) == 0, 1.0, n)
    Yn = Y / np.where((n := np.linalg.norm(Y, axis=1, keepdims=True)) == 0, 1.0, n)
    return Xn @ Yn.T
