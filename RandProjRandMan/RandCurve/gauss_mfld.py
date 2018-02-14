# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 13:46:16 2016

@author: Subhy

Numerically compute distance, principal angles between tangent spaces and
curvature as a function of position on a Gaussian random surface in a high
dimensional space

Functions
=========
numeric_distance
    numeric distance between points on surface
numeric_sines
    numeric angles between tangent planes to surface
numeric_proj
    numeric angles between chords and tangent planes to surface
numeric_curv
    numeric curvature of surface
get_all_numeric
    calculate all numeric quantities
default_options
    default options for long numerics for paper
quick_options
    default options for quick numerics for demo
make_and_save
    generate data and save npz file
"""
from typing import Sequence, Tuple
import numpy as np
import scipy.linalg as sla
from . import gauss_curve as gc
from . import gauss_surf as gs
from . import gauss_surf_theory as gst
from ..iter_tricks import denumerate, dcontext

# =============================================================================
# generate surface
# =============================================================================


def spatial_freq(intrinsic_range: Sequence[float],
                 intrinsic_num: Sequence[int],
                 expand: int=2) -> Tuple[np.ndarray, ...]:
    """
    Vectors of spatial frequencies

    Returns
    -------
    kvecs : (K,)(L1,L2,...,LK/2+1)
        Tuple of vectors of spatial frequencies used in FFT, with singletons
        added to broadcast with `embed_ft`.

    Parameters
    ----------
    intrinsic_range
        tuple of ranges of intrinsic coords [-intrinsic_range, intrinsic_range]
    intrinsic_num
        tuple of numbers of sampling points on surface
    expand
        factor to increase size by, to subsample later
    """
    kvecs = ()

    for intr_ran, intr_num in zip(intrinsic_range[:-1], intrinsic_num):
        intr_res = 2. * intr_ran / intr_num
        kvecs += (2*np.pi * np.fft.fftfreq(expand * intr_num, intr_res),)

    intr_res = 2 * intrinsic_range[-1] / intrinsic_num[-1]
    kvecs += (2*np.pi * np.fft.rfftfreq(expand * intrinsic_num[-1], intr_res),)

    return np.ix_(kvecs, np.array([1]))[:-1]


def random_embed_ft(num_dim: int,
                    kvecs: Sequence[np.ndarray],
                    width: Sequence[float] = (1.0, 1.0)) -> np.ndarray:
    """
    Generate Fourier transform of ramndom Gaussian curve with a covariance
    matrix that is a Gaussian function of difference in position

    Returns
    -------
    embed_ft
        Fourier transform of embedding functions,
        embed_ft[s,t,...,i] = phi^i(k1[s], k2[t], ...)

    Parameters
    ----------
    num_dim
        dimensionality ofambient space
    kvecs : (K,)(L1,L2,...,LK/2+1)
        Tuple of vectors of spatial frequencies used in FFT, with singletons
        added to broadcast with `embed_ft`.
    width
        tuple of std devs of gaussian cov along each intrinsic axis
    """
    sqrt_cov = 1.
    for k, w in zip(kvecs, width):
        sqrt_cov = sqrt_cov * gc.gauss_sqrt_cov_ft(k, w)
    siz = tuple(k.size for k in kvecs) + (num_dim,)
    emb_ft_r = np.random.standard_normal(siz) * sqrt_cov[..., None]
    emb_ft_i = np.random.standard_normal(siz) * sqrt_cov[..., None]
    emb_ft_i[..., 0, :] = 0.
    return (emb_ft_r + 1j * emb_ft_i) / np.sqrt(2 * num_dim)


# =============================================================================
# calculate intermediaries
# =============================================================================


def stack_vec(*cmpts: Sequence[np.ndarray]) -> np.ndarray:
    """
    Stack components in vector

    Returns
    -------
    vec
        vector, with extra index last

    Parameters
    ----------
    cmpts
        tuple of components of vector
    """
    vec = np.stack(cmpts, axis=-1)
    return vec


def embed(embed_ft: np.ndarray) -> np.ndarray:
    """
    Calculate embedding functions

    Returns
    -------
    emb
        emb[s,t,...,i] = phi^i(x1[s], x2[t], ...)

    Parameters
    ----------
    embed_ft
        Fourier transform of embedding functions,
        embed_ft[s,t,...,i] = phi^i(k1[s], k2[t], ...)
    """
    axs = tuple(range(embed_ft.ndim - 1))
    return np.fft.irfftn(embed_ft, axes=axs)


def embed_grad(embed_ft: np.ndarray,
               kvecs: Sequence[np.ndarray]) -> np.ndarray:
    """
    Calculate gradient of embedding functions

    Returns
    -------
    grad
        grad[s,t,...,i,a] = phi_a^i(x1[s], x2[t], ...)

    Parameters
    ----------
    embed_ft
        Fourier transform of embedding functions,
        embed_ft[s,t,...,i] = phi^i(k1[s], k2[t], ...)
    kvecs : (K,)(L1,L2,...,LK/2+1)
        Tuple of vectors of spatial frequencies used in FFT, with singletons
        added to broadcast with `embed_ft`.
    """
    K = len(kvecs)
    axs = tuple(range(K))
    siz = (2*(embed_ft.shape[-2] - 1), embed_ft.shape[-1], K)
    grad = np.empty(embed_ft.shape[:-2] + siz)
    for i, k in enumerate(kvecs):
        grad[i] = (np.fft.irfftn(1j * embed_ft * k, axes=axs),)
    return grad


def embed_hess(embed_ft: np.ndarray,
               kvecs: Sequence[np.ndarray]) -> np.ndarray:
    """
    Calculate hessian of embedding functions

    Returns
    -------
    hess
        hess[s,t,...,i,a,b] = phi_ab^i(x1[s], x2[t], ...)

    Parameters
    ----------
    embed_ft
        Fourier transform of embedding functions,
        embed_ft[s,t,...,i] = phi^i(k1[s], k2[t], ...)
    kvecs : (K,)(L1,L2,...,LK/2+1)
        Tuple of vectors of spatial frequencies used in FFT, with singletons
        added to broadcast with `embed_ft`.
    """
    K = len(kvecs)
    axs = tuple(range(K))
    siz = (2*(embed_ft.shape[-2] - 1), embed_ft.shape[-1], K, K)
    hess = np.empty(embed_ft.shape[:-2] + siz)
    for i, ka in enumerate(kvecs):
        for j, kb in enumerate(kvecs[i:], i):
            hess[..., i, j] = np.fft.irfftn(-embed_ft * ka * kb, axes=axs)
            hess[..., j, i] = hess[..., i, j]
    return hess


def vielbein(grad: np.ndarray) -> np.ndarray:
    """
    Orthonormal basis for tangent space, push-forward of vielbein.

    Returns
    -------
    vbein
        orthonormal basis for tangent space,
        vbein[s,t,...,i,A] = e_A^i(x1[s], x2[t], ...).

        vbein[...,  0] parallel to dx^0.
        vbein[...,  1] perpendicular to dx^0, in (dx^0,dx^1) plane.
        vbein[...,  1] perpendicular to (dx^0,dx^1), in (dx^0,dx^1,dx^3) plane.
        etc.

    Parameters
    ----------
    grad
        grad[s,t,...,i,a] = phi_a^i(x1[s], x2[t], ...)
    """
    if grad.shape[-1] == 1:
        return gc.vielbein(grad.squeeze(-1))[..., None]
    if grad.shape[-1] == 2:
        return gs.vielbein(grad)

    return sla.qr(grad)[0]


def induced_metric(grad: np.ndarray) -> np.ndarray:
    """
    Induced metric on embedded surface

    Returns
    -------
    h
        induced metric
        h[s,t,...,a,b] = h_ab(x1[s], x2[t], ...)

    Parameters
    ----------
    grad
        grad[s,t,...,i,a] = phi_a^i(x1[s], x2[t], ...)
    """
    return grad.swapaxes(-2, -1) @ grad


def raise_hess(embed_ft: np.ndarray,
               kvecs: Sequence[np.ndarray],
               grad: np.ndarray) -> np.ndarray:
    """
    Hessian with second index raised

    Returns
    -------
    hess
        hess[s,t,i,a,b] = phi_a^bi(x1[s], x2[t], ...)

    Parameters
    ----------
    embed_ft
        Fourier transform of embedding functions,
        embed_ft[s,t,...,i] = phi^i(k1[s], k2[t], ...)
    kvecs : (K,)(L1,L2,...,LK/2+1)
        Tuple of vectors of spatial frequencies used in FFT, with singletons
        added to broadcast with `embed_ft`.
    grad
        grad[s,t,...,i,a] = phi_a^i(x1[s], x2[t], ...)
    """
    met = induced_metric(grad)
    hess = embed_hess(embed_ft, kvecs)
    return np.linalg.solve(met[..., None, :, :], hess).swapaxes(-1, -2)


def mat_field_evals(mat_field: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Eigenvalues of 2nd rank tensor field, `mat_field`

    Returns
    -------
    (eval1, eval2, ...)
        eigenvalues, `eval1` > `eval2`
    """
    if mat_field.shape[-1] == 1:
        return mat_field.squeeze(-1)
    if mat_field.shape[-1] == 2:
        return np.stack(gs.mat_field_evals(mat_field), axis=-1)

    return np.linalg.eigvals(mat_field).real


def mat_field_svals(mat_field: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Squared singular values of 2nd rank tensor field, `mat_field`

    Returns
    -------
    (sval1^2, sval2^2, ...)
        squared singular values, `sval1` > `sval2`
    """
    if mat_field.shape[-1] == 1:
        return mat_field.squeeze(-1)**2
    if mat_field.shape[-1] == 2:
        return np.stack(gs.mat_field_svals(mat_field), axis=-1)

    return np.linalg.svd(mat_field, compute_uv=False)**2


# =============================================================================
# calculate distances, angles and curvature
# =============================================================================


def numeric_distance(embed_ft: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate Euclidean distance from central point on curve as a fuction of
    position on curve.

    Returns
    -------
    d
        chord length.
        d[s,t,...] = ||phi(x[s,t,...]) - phi(x[mid])||
    ndx
        chord direction.
        ndx[s,t,...,i] = (phi^i(x[s,t,...]) - phi_i(x[mid])) / d[s,t,...]

    Parameters
    ----------
    embed_ft
        Fourier transform of embedding functions,
        embed_ft[s,t,...,i] = phi^i(k1[s], k2[t], ...)
    """
    pos = embed(embed_ft)
    # chords
    mid = tuple(L // 2 for L in pos.shape[:-1]) + (slice(None),)
    dx = pos - pos[mid]
    # chord length
    d = np.linalg.norm(dx, axis=-1)
    # unit vectors along dx
    zero = d < 1e-7
    d[zero] = 1.
    ndx = np.where(zero[..., None], 0., dx / d[..., None])
    d[zero] = 0.
    return d, ndx


def numeric_sines(kbein: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Sine of angle between tangent vectors

    Returns
    -------
    sin(theta_max), sin(theta_min)
        S[a][s,t,...] = tuple of sin theta_a[s,t,...]
    theta_a
        principal angles between tangent space at (x1[s], x2[t], ...) and
        tangent space at center

    Parameters
    ----------
    kbein
        orthonormal basis for tangent space,
        kbein[s,t,...,i,A] = e_A^i(x[s,t]),
    """
    mid = tuple(L // 2 for L in kbein.shape[:-2]) + (slice(None),)*2
    base_bein = kbein[mid]
    bein_prod = base_bein.T @ kbein
    cosangs = mat_field_svals(bein_prod)
    cosangs[cosangs > 1.] = 1.
    return np.flip(np.sqrt(1. - cosangs), axis=-1)


def numeric_proj(ndx: np.ndarray,
                 kbein: np.ndarray,
                 inds: slice) -> (np.ndarray, np.ndarray):
    """
    Cosine of angle between chord and tangent vectors

    Returns
    -------
    costh
        costh[s,t,...] = max_u,v,... (cos angle between tangent vector at
        x[u,v,...] and chord between x[mid] and x[s,t,...]).
    costh
        costh[s,t,...] = cos angle between tangent vector at
        x[(mid+s)/2,(mid+t)/2,...] and chord between x[mid] and x[s,t,...].

    Parameters
    ----------
    ndx
        chord direction.
        ndx[s,t,...,i] = (phi^i(x[s,t,...]) - phi_i(x[mid])) / d[s,t,...]
    kbein
        orthonormal basis for tangent space,
        kbein[s,t,...,i,A] = e_A^i(x1[s], x2[t], ...),
    """
    # find max cos for each chord
    costh = np.empty(ndx.shape[:2])
    inds += (slice(None), slice(None))
    for i, row in denumerate('i', ndx):
        for j, chord in denumerate('j', row):
            costh[i, j] = np.linalg.norm(chord @ kbein[inds], axis=-1).max()
    # find middle range in each dim
    x = ndx.shape[0] // 4
    y = ndx.shape[1] // 4
    # project chord direction on to tangent space at midpoint
    with dcontext('matmult'):
        ndx_pr = ndx[::2, ::2, None, ...] @ kbein[x:-x, y:-y, ...]
    with dcontext('norm'):
        costh_mid = np.linalg.norm(ndx_pr.squeeze(), axis=-1)
    costh[ndx.shape[0] // 2, ndx.shape[1] // 2] = 1.
    costh_mid[ndx.shape[0] // 4, ndx.shape[1] // 4] = 1.
    return costh, costh_mid


def numeric_curv(hessr: np.ndarray,
                 kbein: np.ndarray) -> np.ndarray:
    """
    Extrinsic curvature

    Returns
    -------
    kappa
        Third fundamental form.
        kappa[s,t,...,a,b] = kappa^a_b(x1[s], x2[t], ...)

    Parameters
    ----------
    hessr
        hessian with one index raised
        hessr[s,t,...,i,a,b] = phi_a^bi(x1[s], x2[t], ...)
    kbein
        orthonormal basis for tangent space,
        kbein[s,t,...,i,a] = e_a^i(x1[s], x2[t], ...),
    """
    # hessian projected onto tangent space (L1,L2,...,K,K,K): H^A_a^b
    hessrt = (hessr.swapaxes(-1, -3) @
              kbein[..., None, :, :]).swapaxes(-1, -3)
#    hessrt = hessr.swapaxes(-3, -2).swapaxes(-2, -1) @ kbein
    return np.sum(hessr @ hessr, axis=-3) - np.sum(hessrt @ hessrt, axis=-3)


# =============================================================================
# the whole thing
# =============================================================================


def get_all_numeric(ambient_dim: int,
                    intrinsic_range: Sequence[float],
                    intrinsic_num: Sequence[int],
                    width: Sequence[float]=(1.0, 1.0),
                    expand: int=2) -> (np.ndarray,
                                       Sequence[np.ndarray],
                                       Sequence[np.ndarray],
                                       Sequence[np.ndarray]):
    """
    Calculate everything

    Returns
    -------
    nud
        numeric distances
    nus
        numeric sines, tuple,
        sine 1 > sine 2
    nup
        numeric projection of chord onto tangent space, tuple (best, mid)
    nuc
        numeric curvatures, tuple,
        curvature 1 > curvature 2

    Parameters
    ----------
    ambient_dim
        N, dimensionality of ambient space
    intrinsic_range
        tuple of ranges of intrinsic coords [-intrinsic_range, intrinsic_range]
    intrinsic_num
        tuple of numbers of sampling points on surface
    width
        tuple of std devs of gaussian covariance along each intrinsic axis
    expand
        factor to increase size by, to subsample later
    """

    with dcontext('k'):
        kvecs = spatial_freq(intrinsic_range, intrinsic_num, expand)
    with dcontext('mfld'):
        embed_ft = random_embed_ft(ambient_dim, kvecs, width)
    with dcontext('grad'):
        grad = embed_grad(embed_ft, kvecs)
    with dcontext('hess'):
        hessr = raise_hess(embed_ft, kvecs, grad)

    with dcontext('e'):
        kbein = vielbein(grad)
#    print('U')
#    tang_proj = tangent_proj(kbein)
    with dcontext('K'):
        curvature = numeric_curv(hessr, kbein)

    int_begin = [(expand - 1) * inum // 2 for inum in intrinsic_num]
    int_end = [inum + ibeg for inum, ibeg in zip(intrinsic_num, int_begin)]

    region = tuple(slice(ibeg, iend) for ibeg, iend in zip(int_begin, int_end))
    regionm = tuple(slice(ibeg // 2, iend // 2) for
                    ibeg, iend in zip(int_begin, int_end))

    with dcontext('d'):
        num_dist, ndx = numeric_distance(embed_ft)
    with dcontext('a'):
        num_sin_max, num_sin_min = numeric_sines(kbein)
    with dcontext('p'):
        num_pr, num_pm = numeric_proj(ndx, kbein, region)
    with dcontext('c'):
        num_curv1, num_curv2 = mat_field_evals(curvature)

    nud = num_dist[region]
    nua = (num_sin_max[region], num_sin_min[region])
    nup = (num_pr[region], num_pm[regionm])
    nuc = (num_curv1[region], num_curv2[region])

    return nud, nua, nup, nuc


# =============================================================================
# options
# =============================================================================


def default_options():
    """
    Default options for generating data

    Returns
    -------
    ambient_dim
        N, dimensionality of ambient space
    intrinsic_range
        tuple of ranges of intrinsic coords [-intrinsic_range, intrinsic_range]
    intrinsic_num
        tuple of numbers of sampling points on surface
    width
        tuple of std devs of gaussian covariance along each intrinsic axis
    """
    # choose parameters
    np.random.seed(0)
    ambient_dim = 1000    # dimensionality of ambient space
    intrinsic_range = (6.0, 10.0)  # x-coordinate lies between +/- this
    intrinsic_num = (128, 256)  # number of points to sample
    width = (1.0, 1.8)

    return ambient_dim, intrinsic_range, intrinsic_num, width


def quick_options():
    """
    Default options for generating test data

    Returns
    -------
    ambient_dim
        N, dimensionality of ambient space
    intrinsic_range
        tuple of ranges of intrinsic coords [-intrinsic_range, intrinsic_range]
    intrinsic_num
        tuple of numbers of sampling points on surface
    width
        tuple of std devs of gaussian covariance along each intrinsic axis
    """
    # choose parameters
    np.random.seed(0)
    ambient_dim = 100    # dimensionality of ambient space
    intrinsic_range = (6.0, 10.0)  # x-coordinate lies between +/- this
    intrinsic_num = (64, 128)  # number of points to sample
    width = (1.0, 1.8)

    return ambient_dim, intrinsic_range, intrinsic_num, width


# =============================================================================
# running code
# =============================================================================


def make_and_save(filename: str,
                  ambient_dim: int,
                  intrinsic_range: Sequence[float],
                  intrinsic_num: Sequence[int],
                  width: Sequence[float]):  # generate data and save
    """
    Generate data and save in ``.npz`` file

    Parameters
    ----------
    filenamee
        name of ``.npz`` file, w/o extension, for data
    ambient_dim
        N, dimensionality of ambient space
    intrinsic_range
        tuple of ranges of intrinsic coords [-intrinsic_range, intrinsic_range]
    intrinsic_num
        tuple of numbers of sampling points on surface
    width
        tuple of std devs of gaussian covariance along each intrinsic axis
    """
    with dcontext('analytic 1'):
        theory = gst.get_all_analytic(ambient_dim, intrinsic_range,
                                      intrinsic_num, width)
    x, y, rho, thr_dis, thr_sin, thr_pro, thr_cur = theory

    with dcontext('analytic 2'):
        theoryl = gst.get_all_analytic_line(rho, np.maximum(*intrinsic_num))
    rhol, thr_dsl, thr_snl, thr_prl, thr_crl = theoryl

    with dcontext('numeric'):
        num_dis, num_sin, num_pro, num_cur = get_all_numeric(ambient_dim,
                                                             intrinsic_range,
                                                             intrinsic_num,
                                                             width)

    np.savez_compressed(filename + '.npz', x=x, y=y, rho=rho, rhol=rhol,
                        thr_dis=thr_dis, thr_sin=thr_sin, thr_pro=thr_pro,
                        thr_cur=thr_cur, thr_disl=thr_dsl, thr_sinl=thr_snl,
                        thr_prol=thr_prl, thr_curl=thr_crl, num_dis=num_dis,
                        num_sin=num_sin, num_pro=num_pro, num_cur=num_cur)


# =============================================================================
# test code
# =============================================================================


if __name__ == "__main__":
    print('Run from outside package.')
