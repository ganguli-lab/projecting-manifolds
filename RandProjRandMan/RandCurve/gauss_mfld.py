# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 13:46:16 2016

@author: Subhy

Numerically compute distance, principal angles between tangent spaces and
curvature as a function of position on a Gaussian random manifold in a high
dimensional space

Functions
=========
numeric_distance
    numeric distance between points on manifold
numeric_sines
    numeric angles between tangent planes to manifold
numeric_proj
    numeric angles between chords and tangent planes to manifold
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
from . import gauss_mfld_theory as gmt
from ..iter_tricks import dcontext

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

    return np.ix_(*kvecs, np.array([1]))[:-1]


def gauss_sqrt_cov_ft(k: np.ndarray, width: float=1.0) -> np.ndarray:
    """sqrt of FFT of 1D Gaussian covariance matrix

    Square root of Fourier transform of a covariance matrix that is a Gaussian
    function of difference in position

    Returns
    -------
    cov(k)
        sqrt(sqrt(2pi) width * exp(-1/2 width**2 k**2))

    Parameters
    ----------
    k
        vector of spatial frequencies
    width
        std dev of gaussian covariance. Default=1.0
    """
    # length of grid
    num_pt = k.size
    # check if k came from np.fft.rfftfreq instead of np.fft.fftfreq
    if k.ravel()[-1] > 0:
        num_pt = 2. * (k.size - 1.)
    dk = k.ravel()[1]
    cov_ft = (dk / np.sqrt(2 * np.pi)) * width * np.exp(-0.5 * width**2 * k**2)
    return num_pt * np.sqrt(cov_ft)


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
        sqrt_cov = sqrt_cov * gauss_sqrt_cov_ft(k, w)
    siz = tuple(k.size for k in kvecs) + (num_dim,)
    emb_ft_r = np.random.standard_normal(siz) * sqrt_cov
    emb_ft_i = np.random.standard_normal(siz) * sqrt_cov
    emb_ft_i[..., 0, :] = 0.
    return (emb_ft_r + 1j * emb_ft_i) / np.sqrt(2 * num_dim)


# =============================================================================
# calculate intermediaries
# =============================================================================


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
        grad[..., i] = np.fft.irfftn(1j * embed_ft * k, axes=axs)
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
    vbein = np.empty_like(grad)
    for k in range(grad.shape[-1]):
        vbein[..., k] = grad[..., k]
        vbein[..., k] -= (vbein[..., :k] @ vbein[..., :k].swapaxes(-2, -1) @
                          grad[..., k:k+1]).squeeze(-1)
        vbein[..., k] /= np.linalg.norm(vbein[..., k], axis=-1, keepdims=True)
    return vbein  # sla.qr(grad)[0]


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
        eigenvalues, `eval1` > `eval2`, (L1,L2,...,K)
    """
    if mat_field.shape[-1] == 1:
        return mat_field.squeeze(-1)
    if mat_field.shape[-1] > 2:
        return np.linalg.eigvals(mat_field).real

    tr_field = (mat_field[..., 0, 0] + mat_field[..., 1, 1]) / 2.0
    det_field = (mat_field[..., 0, 0] * mat_field[..., 1, 1] -
                 mat_field[..., 0, 1] * mat_field[..., 1, 0])
    disc_sq = tr_field**2 - det_field
    disc_sq[np.logical_and(disc_sq < 0., disc_sq > -1e-3)] = 0.0
    disc_field = np.sqrt(disc_sq)
    return np.stack((tr_field + disc_field, tr_field - disc_field), axis=-1)


def mat_field_svals(mat_field: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Squared singular values of 2nd rank tensor field, `mat_field`

    Returns
    -------
    (sval1^2, sval2^2, ...)
        squared singular values, `sval1` > `sval2`, (L1,L2,...,K)
    """
    if mat_field.shape[-1] == 1:
        return np.linalg.norm(mat_field, axis=-2)**2
    if mat_field.shape[-1] > 2:
        return np.linalg.svd(mat_field, compute_uv=False)**2

    frob_field = (mat_field**2 / 2.0).sum(axis=(-2, -1))
    det_field = ((mat_field**2).sum(axis=-2).prod(axis=-1)
                 - mat_field.prod(axis=-1).sum(axis=-1)**2)
    disc_sq = frob_field**2 - det_field
    disc_sq[np.logical_and(disc_sq < 0., disc_sq > -1e-3)] = 0.0
    dsc_field = np.sqrt(disc_sq)
    return np.stack((frob_field + dsc_field, frob_field - dsc_field), axis=-1)


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
                 inds: Tuple[slice, ...]) -> (np.ndarray, np.ndarray):
    """
    Cosine of angle between chord and tangent vectors

    Returns
    -------
    costh
        costh[s,t,...] = max_u,v,... (cos angle between tangent vector at
        x[u,v,...] and chord between x[mid] and x[s,t,...]).
    costh_mid
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
    def calc_costh(chord):
        """Calculate max cos(angle) between chord and any tangent vector"""
        return np.linalg.norm(chord @ kbein[inds], axis=-1).max()

    with dcontext('max matmult'):
        costh = np.apply_along_axis(calc_costh, -1, ndx)

#    costh = np.empty(ndx.shape[:-1])
#    for i, row in denumerate('i', ndx):
#        for j, chord in denumerate('j', row):
#            costh[i, j] = np.linalg.norm(chord @ kbein[inds], axis=-1).max()

    # find middle range in each dim
    mid_edges = [siz // 4 for siz in ndx.shape[:-1]]
    mid = tuple(slice(x, -x) for x in mid_edges)
    alternate = (slice(None, None, 2),) * (ndx.ndim - 1) + (None,)
    # project chord direction on to tangent space at midpoint
    with dcontext('mid matmult'):
        ndx_pr = ndx[alternate] @ kbein[mid]
    with dcontext('norm'):
        costh_mid = np.linalg.norm(ndx_pr.squeeze(-2), axis=-1)

    costh[tuple(siz // 2 for siz in ndx.shape[:-1])] = 1.
    costh_mid[tuple(mid_edges)] = 1.
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
        num_sin = numeric_sines(kbein)
    with dcontext('p'):
        num_pr, num_pm = numeric_proj(ndx, kbein, region)
    with dcontext('c'):
        num_curv = mat_field_evals(curvature)

    nud = num_dist[region]
    nua = num_sin[region]
    nup = (num_pr[region], num_pm[regionm])
    nuc = num_curv[region]

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
    intrinsic_range = (6., 8., 10.)  # x-coordinate lies between +/- this
    intrinsic_num = (8, 16, 32)  # number of points to sample
    width = (1., 1.8, 1.3)

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
        theory = gmt.get_all_analytic(ambient_dim, intrinsic_range,
                                      intrinsic_num, width)
    x, rho, thr_dis, thr_sin, thr_pro, thr_cur = theory

    with dcontext('analytic 2'):
        theoryl = gmt.get_all_analytic_line(rho, max(intrinsic_num))
    rhol, thr_dsl, thr_snl, thr_prl, thr_crl = theoryl

    with dcontext('numeric'):
        num_dis, num_sin, num_pro, num_cur = get_all_numeric(ambient_dim,
                                                             intrinsic_range,
                                                             intrinsic_num,
                                                             width)

    np.savez_compressed(filename + '.npz', x=x, rho=rho, rhol=rhol,
                        thr_dis=thr_dis, thr_sin=thr_sin, thr_pro=thr_pro,
                        thr_cur=thr_cur, thr_disl=thr_dsl, thr_sinl=thr_snl,
                        thr_prol=thr_prl, thr_curl=thr_crl, num_dis=num_dis,
                        num_sin=num_sin, num_pro=num_pro, num_cur=num_cur)


# =============================================================================
# test code
# =============================================================================


if __name__ == "__main__":
    print('Run from outside package.')
