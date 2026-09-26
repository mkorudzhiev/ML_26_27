"""Мащабиране на признаци.

Навсякъде долу X е матрица с форма (n_samples, n_features): един ред = един
обект, една колона = един признак. Това е конвенцията в целия курс и в scikit-learn.
"""

import numpy as np


def standardize(X):
    """Стандартизира всеки признак до средно 0 и стандартно отклонение 1 (z-score).

    Parameters
    ----------
    X : ndarray с форма (n_samples, n_features)

    Returns
    -------
    ndarray със същата форма

    Notes
    -----
    Делението е по колони, затова `axis=0`. Признаци с нулева дисперсия (константни
    колони) биха дали деление на нула, затова заместваме 0 с 1, при което колоната
    просто остава центрирана в нулата.
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std


def minmax_scale(X, feature_range=(0.0, 1.0)):
    """Линейно преобразува всеки признак в зададения интервал.

    Parameters
    ----------
    X : ndarray с форма (n_samples, n_features)
    feature_range : двойка (lo, hi)

    Returns
    -------
    ndarray със същата форма
    """
    X = np.asarray(X, dtype=float)
    lo, hi = feature_range
    xmin = X.min(axis=0)
    xspan = X.max(axis=0) - xmin
    xspan = np.where(xspan == 0, 1.0, xspan)
    return lo + (X - xmin) / xspan * (hi - lo)
