# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 17:09:03 2017

@author: Subhy
Compute theoretical distribution of maximum distortion of Gaussian random
manifolds under random projections
"""
import numpy as np
from numpy import ndarray as array

# =============================================================================
# calculate theory
# =============================================================================


def numerator_LGG(mfld_dim: int,
                  ambient_dim: array,
                  vol: array,
                  epsilon: array,
                  prob: float) -> array:  # our theory
    """
    Theoretical M * epsilon^2 / K, our formula

    Parameters
    ----------
    mfld_dim
        K, dimensionality of manifold
    ambient_dim
        N, dimensionality of ambient space
    vol
        V, volume of manifold
    epsilon
        allowed distortion
    prob
        allowed failure probability
    """
    onev = np.ones_like(epsilon)
    Me_K = (np.log(vol / prob) / mfld_dim + 0.5 * np.log(27. / mfld_dim) +
            np.log(ambient_dim / 4.) + 1.5 * onev)
    return 16 * Me_K


def numerator_BW(mfld_dim: int,
                 ambient_dim: array,
                 vol: array,
                 epsilon: array,
                 prob: float) -> array:  # BW theory
    """
    Theoretical M * epsilon^2 / K, Baraniuk & Wakin's formula

    Parameters
    ----------
    mfld_dim
        K, dimensionality of manifold
    ambient_dim
        N, dimensionality of ambient space
    vol
        V, volume of manifold
    epsilon
        allowed distortion
    prob
        allowed failure probability
    """
    R = 1. / np.sqrt(2. * np.pi * np.e)
    tau = 1.1 * np.sqrt(2.)

    Me_K = (np.log(vol**2 / prob) / mfld_dim +
            np.log(3100.**4 * mfld_dim * (1. * ambient_dim)**3 * R**2 /
                   (epsilon**6 * tau**2)))

    return 676 * Me_K


def numerator_Verma(mfld_dim: int,
                    ambient_dim: array,
                    vol: array,
                    epsilon: array,
                    prob: float) -> array:  # Verma theory
    """
    Theoretical M * epsilon^2 / K, Verma's formula

    Parameters
    ----------
    mfld_dim
        K, dimensionality of manifold
    ambient_dim
        N, dimensionality of ambient space
    vol
        V, volume of manifold
    epsilon     =
        allowed distortion
    prob
        allowed failure probability
    """
    onev = np.ones_like(ambient_dim)
    Me_K = (np.log(vol / prob) / mfld_dim +
            onev * np.log(2**35 * 3**5 * 13**2 * mfld_dim /
                          (epsilon**6 * np.pi * np.e)) / 2.)

    return 64 * Me_K


def numerator_EW(mfld_dim: int,
                 ambient_dim: array,
                 vol: array,
                 epsilon: array,
                 prob: float) -> array:  # Verma theory
    """
    Theoretical M * epsilon^2 / K, Eftekhari & Wakin's formula

    Parameters
    ----------
    mfld_dim
        K, dimensionality of manifold
    ambient_dim
        N, dimensionality of ambient space
    vol
        V, volume of manifold
    epsilon     =
        allowed distortion
    prob
        allowed failure probability
    """
    onev = np.ones_like(ambient_dim)
    tau = 1.1 * np.sqrt(2.)
    Me_K = (np.log(2 * vol**2) / mfld_dim +
            onev * (np.log(mfld_dim / (epsilon**4 * tau**2)) + 24))
    Me_K = np.maximum(Me_K, np.log(8 / prob))

    return 18 * Me_K


# =============================================================================
# analytic data
# =============================================================================


def get_all_analytic(epsilons: array,
                     ambient_dims: array,
                     vols: array,
                     prob: float) -> (array, array, array, array, array, array,
                                      array, array, array, array):
    """
    Calculate all theory

    Parameters
    ----------
    epsilons
        list of allowed distortions
    proj_dims
        list of M's, dimensionalities of projected space
    ambient_dims
        ndarray of N's, dimensionality of ambient space
    vols
        tuple of tuples of ndarrays of (V^1/K)'s, volume of manifold,
        one for eack K, one for varying N/V
    prob
        allowed failure probability

    Returns
    -------
    Ns, Vs, LGG_num, LGG_vol, BW_num, BW_vol, Vr_num, Vr_vol:
        values of (M * epsilon^2 / K)
        from: our theory, Baraniuk & Wakin's, Verma's formula for differnt N,V
    """

    eps = np.array(epsilons)[..., np.newaxis]

    Vs = np.logspace(np.log10(vols[:, 0].min()),
                     np.log10(vols[:, -1].max()), 10)

    Ns = np.logspace(np.log10(ambient_dims[0]),
                     np.log10(ambient_dims[-1]), 10)

    N = ambient_dims[-1]
    V = vols[:, -1]

    LGG_num = np.stack((numerator_LGG(1, Ns, V[0], eps, prob),
                        numerator_LGG(2, Ns, V[1]**2, eps, prob)), axis=0)

    LGG_vol = np.stack((numerator_LGG(1, N, Vs, eps, prob),
                        numerator_LGG(2, N, Vs**2, eps, prob)), axis=0)

    BW_num = np.stack((numerator_BW(1, Ns, V[0], eps, prob),
                       numerator_BW(2, Ns, V[1]**2, eps, prob)), axis=0)

    BW_vol = np.stack((numerator_BW(1, N, Vs, eps, prob),
                       numerator_BW(2, N, Vs**2, eps, prob)), axis=0)

    Verma_num = np.stack((numerator_Verma(1, Ns, V[0], eps, prob),
                          numerator_Verma(2, Ns, V[1]**2, eps, prob)), axis=0)

    Verma_vol = np.stack((numerator_Verma(1, N, Vs, eps, prob),
                          numerator_Verma(2, N, Vs**2, eps, prob)), axis=0)

    EW_num = np.stack((numerator_EW(1, Ns, V[0], eps, prob),
                       numerator_EW(2, Ns, V[1]**2, eps, prob)), axis=0)

    EW_vol = np.stack((numerator_EW(1, N, Vs, eps, prob),
                       numerator_EW(2, N, Vs**2, eps, prob)), axis=0)

    return (Ns, Vs, LGG_num, LGG_vol, BW_num, BW_vol, Verma_num, Verma_vol,
            EW_num, EW_vol)


# =============================================================================
# test code
# =============================================================================


if __name__ == "__main__":
    print('Run from outside package.')
