# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:45:49 2017

@author: Subhy

Compute linear fits for simulations of maximum distortion of Gaussian random
manifolds under random projections
"""

from typing import List, Optional
import numpy as np
from numpy import ndarray as array

# =============================================================================
# %%* fitting
# =============================================================================


def linear_fit(Xs: array, MeK: array) -> (array, array, array):
    """
    linear least-squares regression to find dependence on X

    Parameters
    ----------
    Xs
        ndarray of values of ln V / K or ln N (#NV,)
    MeK
        ndarray of values of M * epsilon**2 / K (#NV,)

    Returns
    -------
    m
        ndarray of coefficients (2,)
    y
        predicted y's (#NV,)
    err
        estimator covariance (diagonal is squared std error of m) (2,2)
    """
    x = np.stack((np.ones_like(Xs), Xs), axis=-1)
    m, sigma = np.linalg.lstsq(x, MeK)[:2]
    y = x @ m
    sigma /= MeK.shape[0]
    err = sigma * np.linalg.inv(x.T @ x)
    return m, y, err


def linear_fit_all(Xs: array, MeKs: array) -> (array, array, array):
    """
    linear least-squares regression to find dependence on X

    Parameters
    ----------
    Xs
        ndarray of values of ln V / K or ln N (#NV,)
    MeKs
        ndarray of values of M * epsilon**2 / K (#K,#epsilon,#NV)

    Returns
    -------
    m
        ndarray of coefficients (2,#K,#epsilon)
    y
        predicted y's (#K,#epsilon,#NV)
    err
        estimator covariance (diagonal is squared std error of m) (2,2,#K,#eps)
    """
    siz = MeKs.shape
    nsiz = (np.prod(siz[:-1]), siz[-1])
    x = np.stack((np.ones_like(Xs), Xs), axis=-1)
    y0 = MeKs.reshape(nsiz).T
    m, sigma = np.linalg.lstsq(x, y0)[:2]
    y = x @ m
    sigma /= y0.shape[0]
    err = sigma * np.linalg.inv(x.T @ x)[..., None]
    return (m.reshape((2,) + siz[:-1]), y.T.reshape(siz),
            err.reshape((2, 2) + siz[:-1]))


def calc_for_disp(m: array, err: array, ind: int) -> (float, float):
    """
    calculate one coefficient & standard erroe

    Parameters
    ----------
    m
        ndarray of coefficients (2,#K,#epsilon)
    err
        estimator covariance (diagonal is squared std error of m) (2,2,#K,#eps)
    ind
        which coeff?

    Returns
    -------
    coeff
        weigthed mean coefficient value
    std_err
        standard error of `coeff`
    """
    weights = 1. / err[ind, ind, ...]
    coeff = np.average(m[ind, ...], weights=weights)
    std_err = np.sqrt(1. / np.sum(weights))
    return coeff, std_err


def multi_lin_fit(Ks: array,
                  epsilons: array,
                  Ns: array,
                  Vs: array,
                  MeK: array,
                  ix: Optional[array] = None) -> (array, array, array,
                                                  List[str]):
    """
    Ks
        ndarray of K, dimensionality of manifold (#K,)
    epsilons
        ndarray of allowed distortions (#epsilon,)
    Ns
        ndarray of N's, dimensionality of ambient space when, (#N,)
    Vs
        ndarray of V^1/K, (#V,)
    MeK
        values of M \\epsilon^2 / K, ndarray (#K,#epsilon,#V,#N)
    ix
        ndarray of indices of variables to include in linear regression.
        chosen from: const, ln K, -ln e, ln N, ln V

    Returns
    -------
    m
        ndarray of coefficients (2,#K,#epsilon)
    y
        predicted y's (#K,#epsilon,#NV)
    err
        estimator covariance (diagonal is squared std error of m) (2,2,#K,#eps)
    names
        ndarray of strings of variables in linear regression
    """
    names = np.array(['const', ' ln K', '-ln e', ' ln N', ' ln V'])

    K = np.log(Ks)
    epsilon = - np.log(epsilons)
    N = np.log(Ns)
    V = np.log(Vs)
    c = np.array([1])

    indepvars = (c, K, epsilon, N, V)
    x = np.stack(np.broadcast_arrays(*np.ix_(*indepvars)),
                 axis=-1).reshape((-1, len(indepvars)))
    y0 = MeK.ravel()

    if ix is not None:
        x = x[:, ix]
        names = names[ix]

    m, sigma = np.linalg.lstsq(x, y0)[:2]
    y = x @ m
    sigma /= y0.shape[0]
    err = sigma * np.linalg.inv(x.T @ x)

    return m, y, err, names


def multi_lin_fit2(Ks: array,
                   epsilons: array,
                   Ns_N: array,
                   Ns_V: array,
                   Vs_N: array,
                   Vs_V: array,
                   MeK_N: array,
                   MeK_V: array,
                   ix: Optional[array] = None) -> (array, array, array,
                                                   List[str]):
    """
    Ks
        ndarray of K, dimensionality of manifold (#K)
    epsilons
        ndarray of allowed distortions (#epsilon)
    Ns_N
        ndarray of N's, dimensionality of ambient space when varying N, (#N,)
    Ns_V
        ndarray of N's, dimensionality of ambient space when varying V, ()
    Vs_N
         ndarray of V^1/K, ()
    Vs_V
        ndarray V^1/K,  (#V)
    MeK_N
        values of M \\epsilon^2 / K when varying N, ndarray (#K,#epsilon,#N)
    MeK_V
        values of M \\epsilon^2 / K when varying V, ndarray (#K,#epsilon,#N)
    ix
        ndarray of indices of variables to include in linear regression.
        chosen from: const, ln K, -ln e, ln N, ln V

    Returns
    -------
    m
        ndarray of coefficients (2,#K,#epsilon)
    y
        predicted y's (#K,#epsilon,#NV)
    err
        estimator covariance (diagonal is squared std error of m) (2,2,#K,#eps)
    names
        ndarray of strings of variables in linear regression
    """
    names = np.array(['const', ' ln K', '-ln e', ' ln N', ' ln V'])

    MeK_N = MeK_N[..., None]
    MeK_V = MeK_V[..., None, :]
    K = np.log(Ks)
    epsilon = - np.log(epsilons)
    N_N = np.log(Ns_N)
    N_V = np.log(Ns_V)
    V_N = np.log(Vs_N)
    V_V = np.log(Vs_V)
    c = np.array([1])

    indepvars = (c, K, epsilon, N_N, V_N)
    x_N = np.stack(np.broadcast_arrays(*np.ix_(*indepvars)),
                   axis=-1).reshape((-1, len(indepvars)))
    indepvars = (c, K, epsilon, N_V, V_V)
    x_V = np.stack(np.broadcast_arrays(*np.ix_(*indepvars)),
                   axis=-1).reshape((-1, len(indepvars)))

    x = np.concatenate((np.array(x_N), np.array(x_V)))
    y0 = np.concatenate((MeK_N.ravel(), MeK_V.ravel()))

    if ix is not None:
        x = x[:, ix]
        names = names[ix]

    m, sigma = np.linalg.lstsq(x, y0)[:2]
    y = x @ m
    sigma /= y0.shape[0]
    err = sigma * np.linalg.inv(x.T @ x)

    return m, y, err, names


# =============================================================================
# %%* reading
# =============================================================================


def get_data(fileobj: np.lib.npyio.NpzFile) -> (array, array, array, array,
                                                array, array, array, array):
    """
    Parameters
    ----------
    fileobj
        instance of NpzFile class from .npz file with data, with fields:

        num_N
            values of M when varying N, ndarray (#K,#epsilon,#N)
        num_V
            values of M when varying V, ndarray (#K,#epsilon,#N)
        prob
            allowed failure probability
        ambient_dims
            list of N's, dimensionality of ambient space, ((#N,), 1)
            tuple for varying N and V, second one a scalar
        vols_N
             tuple of V^1/K, for each K,
        vols_V
            tuple of V^1/K, for each K, each member is an ndarray (#V)
        epsilons
            list of allowed distortions (#epsilon)
    Returns
    -------
    Ks
        ndarray of K, dimensionality of manifold (#K)
    epsilons
        ndarray of allowed distortions (#epsilon)
    Ns_N
        ndarray of N's, dimensionality of ambient space when varying N, (#N,)
    Ns_V
        ndarray of N's, dimensionality of ambient space when varying V, ()
    Vs_N
         ndarray of V^1/K, ()
    Vs_V
        ndarray V^1/K,  (#V)
    MeK_N
        values of M \\epsilon^2 / K when varying N, ndarray (#K,#epsilon,#N)
    MeK_V
        values of M \\epsilon^2 / K when varying V, ndarray (#K,#epsilon,#N)
    """
    nums = fileobj['ambient_dims']
    vols = fileobj['vols'][0]

    epsilons = fileobj['epsilons']
    Mes_num = fileobj['M_num']
    Ks = np.arange(1, 1 + len(Mes_num))
    Mes_num *= epsilons[..., None, None]**2 / Ks[..., None, None, None]
#   Mes_num_N = fileobj['num_N'] * epsilons[..., None]**2 / Ks[..., None, None]
#   Mes_num_V = fileobj['num_V'] * epsilons[..., None]**2 / Ks[..., None, None]
#
#    return Ks, epsilons, nums, num, vol, vols, Mes_num_N, Mes_num_V

    return Ks, epsilons, nums, vols, Mes_num


# =============================================================================
# %%* displaying
# =============================================================================


def disp_coeffs(Xs: array, MeKs: array, prefix: str = ''):
    """
    Calculate and display fit coefficients with error bars

    Parameters
    ----------
    Xs
        ndarray of values of ln V / K or ln N (#NV,)
    MeKs
        ndarray of values of M * epsilon**2 / K (#K,#epsilon,#NV)
    prefix
        disply prefix, usually name of `Xs`
    """
    m, y, err = linear_fit_all(Xs, MeKs)
    const, const_err = calc_for_disp(m, err, 0)
    slope, slope_err = calc_for_disp(m, err, 1)

    print(prefix, 'intercept:', const, '+/-', const_err)
    print(prefix, 'slope:', slope, '+/-', slope_err)


def disp_multi(fileobj: np.lib.npyio.NpzFile, ix: Optional[array] = None):
    """
    Parameters
    ----------
    fileobj
        instance of NpzFile class from .npz file with data, with fields:

        num_N
            values of M when varying N, ndarray (#K,#epsilon,#N)
        num_V
            values of M when varying V, ndarray (#K,#epsilon,#N)
        prob
            allowed failure probability
        ambient_dims
            list of N's, dimensionality of ambient space, ((#N,), 1)
            tuple for varying N and V, second one a scalar
        vols_N
             tuple of V^1/K, for each K,
        vols_V
            tuple of V^1/K, for each K, each member is an ndarray (#V)
        epsilons
            list of allowed distortions (#epsilon)
    ix
        ndarray of indices of variables to include in linear regression.
        chosen from: const, ln K, -ln e, ln N, ln V
    """
    Ks, eps, nums, vols, Mes_num = get_data(fileobj)

    disp_coeffs(np.log(nums), Mes_num[..., -1, :], 'N:')
    disp_coeffs(np.log(vols), Mes_num[..., -1], 'V:')

    m, y, err, names = multi_lin_fit(Ks, eps, nums, vols, Mes_num, ix)
    print(list(names))
    print(["{: .2f}".format(x) for x in m], "+/-")
    print(["{: .2f}".format(x) for x in np.sqrt(np.diag(err))])


def disp_multi2(fileobj: np.lib.npyio.NpzFile, ix: Optional[array] = None):
    """
    Parameters
    ----------
    fileobj
        instance of NpzFile class from .npz file with data, with fields:

        num_N
            values of M when varying N, ndarray (#K,#epsilon,#N)
        num_V
            values of M when varying V, ndarray (#K,#epsilon,#N)
        prob
            allowed failure probability
        ambient_dims
            list of N's, dimensionality of ambient space, ((#N,), 1)
            tuple for varying N and V, second one a scalar
        vols_N
             tuple of V^1/K, for each K,
        vols_V
            tuple of V^1/K, for each K, each member is an ndarray (#V)
        epsilons
            list of allowed distortions (#epsilon)
    ix
        ndarray of indices of variables to include in linear regression.
        chosen from: const, ln K, -ln e, ln N, ln V
    """
    Ks, eps, nums, num, vol, vols, Mes_num_N, Mes_num_V = get_data(fileobj)

    disp_coeffs(np.log(nums), Mes_num_N, 'N:')
    disp_coeffs(np.log(vols), Mes_num_V, 'V:')

    m, y, err, names = multi_lin_fit2(Ks, eps, nums, num, vol, vols,
                                      Mes_num_N, Mes_num_V, ix)
    print(names)
    print(["{: .2f}".format(x) for x in m], "+/-")
    print(["{: .2f}".format(x) for x in np.sqrt(np.diag(err))])
