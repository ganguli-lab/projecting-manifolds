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
from ..iter_tricks import dcontext, dndindex
from ..larray import larray, wrap_one, solve

# =============================================================================
# generate surface
# =============================================================================


@wrap_one
def spatial_freq(intrinsic_range: Sequence[float],
                 intrinsic_num: Sequence[int],
                 expand: int = 2) -> larray:
    """
    Vectors of spatial frequencies

    Returns
    -------
    karr : (L1,L2,...,LK/2+1,1,K)
        Array of spatial frequencies used in FFT, stacked to broadcast with
        `embed_ft`.

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
    kvecs = np.ix_(*kvecs, np.array([1]))[:-1]
    return np.stack(np.broadcast_arrays(*kvecs), axis=-1)


def gauss_sqrt_cov_ft(karr: larray, width: float = 1.0) -> larray:
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
    K = karr.shape[-1]
    LK_real = karr.shape[-3]  # = L_K/2 + 1, due to rfft
    # length of grid
    num_pt = karr.size * 2*(LK_real - 1) // (K*LK_real)  # size of irfft
    dk = np.prod([np.diff(karr, axis=i).max() for i in range(K)])  # k-cell vol
    ksq = np.sum((width * karr)**2, axis=-1)
    # scaled to convert continuum FT to DFT
    cov_ft = (dk * np.prod(width) / np.sqrt(2 * np.pi)) * np.exp(-0.5 * ksq)
    return num_pt * np.sqrt(cov_ft)


@wrap_one
def random_embed_ft(num_dim: int,
                    karr: larray,
                    width: Sequence[float] = (1.0, 1.0)) -> larray:
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
        dimensionality of ambient space
    karr : (L1,L2,...,LK/2+1,1,K)
        Array of vectors of spatial frequencies used in FFT, with singletons
        added to broadcast with `embed_ft`.
    width
        tuple of std devs of gaussian cov along each intrinsic axis
    """
    sqrt_cov = gauss_sqrt_cov_ft(karr, np.array(width))
    siz = karr.shape[:-2] + (num_dim,)
    emb_ft_r = np.random.randn(*siz)
    emb_ft_i = np.random.randn(*siz)

    flipinds = tuple(-np.arange(k) for k in siz[:-2]) + (np.array([0]),)
    repinds = tuple(np.array([0, k//2]) for k in siz[:-2]) + (np.array([0]),)

    emb_ft_r[..., :1, :] += emb_ft_r[np.ix_(*flipinds)]
    emb_ft_r[..., :1, :] /= np.sqrt(2)
    emb_ft_r[np.ix_(*repinds)] /= np.sqrt(2)
    emb_ft_i[..., :1, :] -= emb_ft_i[np.ix_(*flipinds)]
    emb_ft_i[..., :1, :] /= np.sqrt(2)

    return (emb_ft_r + 1j * emb_ft_i) * sqrt_cov / np.sqrt(2 * num_dim)


# =============================================================================
# calculate intermediaries
# =============================================================================


@wrap_one
def embed(embed_ft: larray) -> larray:
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


@wrap_one
def embed_grad(embed_ft: larray,
               karr: larray) -> larray:
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
    karr : (L1,L2,...,LK/2+1,1,K)
        Array of vectors of spatial frequencies used in FFT, with singletons
        added to broadcast with `embed_ft`.
    """
    axs = tuple(range(karr.shape[-1]))
    return np.fft.irfftn(1j * embed_ft.c * karr, axes=axs)


@wrap_one
def embed_hess(embed_ft: larray,
               karr: larray) -> larray:
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
    karr : (L1,L2,...,LK/2+1,1,K)
        Array of vectors of spatial frequencies used in FFT, with singletons
        added to broadcast with `embed_ft`.
    """
    axs = tuple(range(karr.shape[-1]))
    ksq = -karr.c * karr.r
    return np.fft.irfftn(embed_ft.s * ksq, axes=axs)


def vielbein(grad: larray) -> larray:
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
        return grad / np.linalg.norm(grad, axis=-2, keepdims=True)
    vbein = np.empty_like(grad)
    N = grad.shape[-2]
    proj = np.zeros(grad.shape[:-2] + (N, N)) + np.eye(N)
    for k in range(grad.shape[-1]):
        vbein[..., k] = (proj @ grad[..., k].c).uc
        vbein[..., k] /= np.linalg.norm(vbein[..., k], axis=-1, keepdims=True)
        proj -= vbein[..., k].c * vbein[..., k].r
    return vbein  # sla.qr(grad)[0]


def induced_metric(grad: larray) -> larray:
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
    return grad.t @ grad


def raise_hess(embed_ft: larray,
               karr: larray,
               grad: larray) -> larray:
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
    karr : (L1,L2,...,LK/2+1,1,K)
        Array of vectors of spatial frequencies used in FFT, with singletons
        added to broadcast with `embed_ft`.
    grad
        grad[s,t,...,i,a] = phi_a^i(x1[s], x2[t], ...)
    """
    met = induced_metric(grad)[..., None, :, :]
    hess = embed_hess(embed_ft, karr)
    if karr.shape[-1] == 1:
        return hess / met
    if karr.shape[-1] > 2:
        return solve(met, hess).t

    hessr = np.empty_like(hess)
    hessr[..., 0, 0] = (hess[..., 0, 0] * met[..., 1, 1] -
                        hess[..., 0, 1] * met[..., 1, 0])
    hessr[..., 0, 1] = (hess[..., 0, 1] * met[..., 0, 0] -
                        hess[..., 0, 0] * met[..., 0, 1])
    hessr[..., 1, 0] = (hess[..., 1, 0] * met[..., 1, 1] -
                        hess[..., 1, 1] * met[..., 1, 0])
    hessr[..., 1, 1] = (hess[..., 1, 1] * met[..., 0, 0] -
                        hess[..., 1, 0] * met[..., 0, 1])
    # divide by determinant
    hessr /= (met[..., 0, 0] * met[..., 1, 1] - met[..., 0, 1]**2).s
    return hessr


@wrap_one
def mat_field_evals(mat_field: larray) -> larray:
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


@wrap_one
def mat_field_svals(mat_field: larray) -> larray:
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


def numeric_distance(embed_ft: larray) -> (larray, larray):
    """
    Calculate Euclidean distance from central point on curve as a fuction of
    position on curve.

    Returns
    -------
    d
        chord length.
        d[s,t,...] = ||phi(x[s,t,...]) - phi(x[mid])||
    ndx
        unit vector in chord direction. Central vector is undefined.
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
    d = np.linalg.norm(dx, axis=-1, keepdims=True)
    # unit vectors along dx
    zero = d < 1e-7
    d[zero] = 1.
    ndx = np.where(zero, 0., dx / d)
    d[zero] = 0.
    return d.squeeze(), ndx


def numeric_sines(kbein: larray) -> (larray, larray):
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


@wrap_one
def numeric_proj(ndx: larray,
                 kbein: larray,
                 inds: Tuple[slice, ...]) -> larray:
    """
    Cosine of angle between chord and tangent vectors

    Returns
    -------
    costh
        costh[s,t,...] = max_u,v,... (cos angle between tangent vector at
        x[u,v,...] and chord between x[mid] and x[s,t,...]).

    Parameters
    ----------
    ndx : larray (L1,...,LK,N)
        unit vector in chord direction. Central vector is undefined.
        ndx[s,t,...,i] = (phi^i(x[s,t,...]) - phi_i(x[mid])) / d[s,t,...]
    kbein : larray (L1,...,LK,N,K)
        orthonormal basis for tangent space,
        kbein[s,t,...,i,A] = e_A^i(x1[s], x2[t], ...),
    inds
        K-tuple of slices for region to search over for lowest angle
    """
    flat_bein = kbein[inds].flatter(0, -2)  # (L,N,K)
    # The limit here corresponds to 2GB memory per K (memory ~ size^2)
    if np.prod(ndx.shape[:-1]) <= 2**14:
        with dcontext('matmult'):
            # (L1,...,LK,1,N) @ (L,N,K) -> (L1,...,LK,L,K) -> (L1,...,LK,L)
            costh = np.linalg.norm(ndx.r.r @ flat_bein, axis=-1).max(axis=-2)
        # deal with central vector
        costh[tuple(siz // 2 for siz in ndx.shape[:-1])] = 1.
        return costh.squeeze()

    def calc_costh(chord):
        """Calculate max cos(angle) between chord and any tangent vector"""
        return np.linalg.norm(chord @ flat_bein, axis=-1).max()

#    with dcontext('max matmult'):
#        costh = np.apply_along_axis(calc_costh, -1, ndx)

    costh = np.empty(ndx.shape[:-1])
    for ii in dndindex(*ndx.shape[:-1]):
        costh[ii] = np.apply_along_axis(calc_costh, -1, ndx[ii])
    costh[tuple(siz // 2 for siz in ndx.shape[:-1])] = 1.

    return costh


def numeric_curv(hessr: larray,
                 kbein: larray) -> larray:
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
    hessr = hessr.swapaxes(-1, -3)
    # hessian projected onto tangent space (L1,L2,...,K,K,K): H^A_a^b
    hesst = (hessr @ kbein[..., None, :, :]).swapaxes(-1, -3)
#    hessrt = hessr.swapaxes(-3, -2).swapaxes(-2, -1) @ kbein
    return np.sum(hessr @ np.moveaxis(hessr, -3, -1) - hesst @ hesst, axis=-3)


# =============================================================================
# the whole thing
# =============================================================================


def get_all_numeric(ambient_dim: int,
                    intrinsic_range: Sequence[float],
                    intrinsic_num: Sequence[int],
                    width: Sequence[float] = (1.0, 1.0),
                    expand: int = 2) -> (larray, larray, larray, larray):
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
        numeric projection of chord onto tangent space
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
        karr = spatial_freq(intrinsic_range, intrinsic_num, expand)
    with dcontext('mfld'):
        embed_ft = random_embed_ft(ambient_dim, karr, width)
    with dcontext('grad'):
        grad = embed_grad(embed_ft, karr)
    with dcontext('hess'):
        hessr = raise_hess(embed_ft, karr, grad)
    with dcontext('e'):
        kbein = vielbein(grad)
#    print('U')
#    tang_proj = tangent_proj(kbein)
    with dcontext('K'):
        curvature = numeric_curv(hessr, kbein)

    int_begin = [(expand - 1) * inum // 2 for inum in intrinsic_num]
    int_end = [inum + ibeg for inum, ibeg in zip(intrinsic_num, int_begin)]

    region = tuple(slice(ibeg, iend) for ibeg, iend in zip(int_begin, int_end))

    with dcontext('d'):
        num_dist, ndx = numeric_distance(embed_ft)
    with dcontext('a'):
        num_sin = numeric_sines(kbein)
    with dcontext('p'):
        num_pr = numeric_proj(ndx.view(larray), kbein, region)
    with dcontext('c'):
        num_curv = np.sqrt(mat_field_evals(curvature))

    nud = num_dist[region]
    nua = num_sin[region]
    nup = num_pr[region]
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
    intrinsic_num = (128, 256)  # number of points to sample (will be expanded)
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
    intrinsic_num = (32, 64)  # number of points to sample (will be expanded)
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
