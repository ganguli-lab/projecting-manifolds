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
from typing import Sequence, Tuple, Optional
import numpy as np
from . import gauss_mfld_theory as gmt
from ..iter_tricks import dcontext, dndindex
from ..myarray import array, wrap_one, solve, norm, qr, eigvalsh, singvals

# =============================================================================
# generate surface
# =============================================================================


@wrap_one
def spatial_freq(intrinsic_range: Sequence[float],
                 intrinsic_num: Sequence[int],
                 expand: int = 2) -> array:
    """
    Vectors of spatial frequencies

    Returns
    -------
    karr : (L1,L2,...,LK/2+1,1,K)
        Array of vectors of spatial frequencies used in FFT, with singletons
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
    kvecs = np.ix_(*kvecs, np.array([1]))[:-1]
    return np.stack(np.broadcast_arrays(*kvecs), axis=-1)


def gauss_sqrt_cov_ft(karr: array, width: float = 1.0) -> array:
    """sqrt of FFT of KD Gaussian covariance matrix

    Square root of Fourier transform of a covariance matrix that is a Gaussian
    function of difference in position

    Returns
    -------
    karr : (L1,L2,...,LK/2+1,1,K)
        Array of vectors of spatial frequencies used in FFT, with singletons
        added to broadcast with `embed_ft`.

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
                    karr: array,
                    width: Sequence[float] = (1.0, 1.0)) -> array:
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
# Manifold class
# =============================================================================


class SubmanifoldFTbundle():
    """Class describing a submanifold of R^N and its tangent bundle

    Constructed via its Fourier tranform wrt intrinsic coordinates.

    mfld
        Embedding functions of random surface
        mfld[s,t,...,i] = phi_i(x[s],y[t],...), (Lx,Ly,...,N)
    grad
        Gradient of embedding
        grad[s,t,...,i,a] = phi_a^i(x1[s], x2[t], ...)
    hess
        Hessian of embedding
        hess[s,t,i,a,b] = phi_ab^i(x1[s], x2[t], ...)
    gmap
        orthonormal basis for tangent space, (Lx,Ly,N,K)
        gmap[s,t,i,A] = e_A^i(x[s], y[t]).
        e_(A=0)^i must be parallel to d(phi^i)/dx^(a=0)
    """
    ft: Optional[array]  # Fourier transform of embedding, (L1,...,N)
    k: Optional[array]  # Spatial frequencies, (L1,...,K)
    mfld: Optional[array]  # Embedding funrction, (L1,...,N)
    grad: Optional[array]  # Gradient of embedding, (L1,...,N,K)
    hess: Optional[array]  # Hessian of embedding, (L1,...,N,K,K)
    gmap: Optional[array]  # Gauss map of embedding, (L1,...,N,K)
    shape: Tuple[int]
    ambient: int
    intrinsic: int
    flat: bool

    def __init__(self,
                 embed_ft: Optional[array] = None,
                 karr: Optional[array] = None):
        self.ft = embed_ft
        self.k = karr
        if embed_ft is not None:
            self.ambient = embed_ft.shape[-1]
            kshape = embed_ft.shape[:-1]
            self.shape = kshape[:-1] + (2 * (kshape[-1] - 1),)
            self.intrinsic = embed_ft.ndim - 1
        self.flat = False
        self.mfld = None
        self.grad = None
        self.hess = None
        self.gmap = None

    def calc_embed(self):
        """
        Calculate embedding functions

        Computes
        --------
        self.mfld
            emb[s,t,...,i] = phi^i(x1[s], x2[t], ...)

        Requires
        --------
        self.ft
            Fourier transform of embedding functions,
            embed_ft[s,t,...,i] = phi^i(k1[s], k2[t], ...)
        """
        axs = tuple(range(self.ft.ndim - 1))
        self.mfld = np.fft.irfftn(self.ft, axes=axs).view(array)

    def calc_grad(self):
        """
        Calculate gradient of embedding functions

        Returns
        -------
        grad
            grad[s,t,...,i,a] = phi_a^i(x1[s], x2[t], ...)

        Requires
        --------
        embed_ft
            Fourier transform of embedding functions,
            embed_ft[s,t,...,i] = phi^i(k1[s], k2[t], ...)
        karr : (L1,L2,...,LK/2+1,1,K)
            Array of vectors of spatial frequencies used in FFT, with
            singletons added to broadcast with `embed_ft`.
        """
        axs = tuple(range(self.k.shape[-1]))
        self.grad = np.fft.irfftn(1j * self.k * self.ft[..., None],
                                  axes=axs).view(array)

    def calc_hess(self):
        """
        Calculate hessian of embedding functions

        Computes
        --------
        self.hess
            hess[s,t,...,i,a,b] = phi_ab^i(x1[s], x2[t], ...)

        Requires
        --------
        embed_ft
            Fourier transform of embedding functions,
            embed_ft[s,t,...,i] = phi^i(k1[s], k2[t], ...)
        karr : (L1,L2,...,LK/2+1,1,K)
            Array of vectors of spatial frequencies used in FFT, with
            singletons added to broadcast with `embed_ft`.
        """
        axs = tuple(range(self.k.shape[-1]))
        ksq = self.k[..., None] * self.k[..., None, :]
        self.hess = np.fft.irfftn(-ksq * self.ft[..., None, None],
                                  axes=axs).view(array)

    def calc_gmap(self):
        """
        Orthonormal basis for tangent space, push-forward of vielbein.

        Computes
        --------
        self.gmap
            orthonormal basis for tangent space,
            vbein[s,t,...,i,A] = e_A^i(x1[s], x2[t], ...).

            vbein[...,  0] parallel to dx^0.
            vbein[...,  1] perpendicular to dx^0, in (dx^0,dx^1) plane.
            vbein[...,  2] perp to (dx^0,dx^1), in (dx^0,dx^1,dx^2) plane.
            etc.

        Requires
        --------
        grad
            grad[s,t,...,i,a] = phi_a^i(x1[s], x2[t], ...)
        """
        norm_opts = {'axis': -2 + self.flat, 'keepdims': True}
        if self.intrinsic == 1:
            self.gmap = self.grad / norm(self.grad, **norm_opts)
        else:
            self.gmap = qr(self.grad)

    def dump_ft(self):
        """Delete stored Fourier transform information
        """
        self.ft = None
        self.k = None

    def dump_grad(self):
        """Delete stored gradient
        """
        self.grad = None

    def copy_basic(self):
        """Copy scalar attributes
        """
        other = SubmanifoldFTbundle()
        other.shape = self.shape
        other.ambient = self.ambient
        other.intrinsic = self.intrinsic
        other.flat = self.flat
        return other

    def copy(self):
        """Deep copy
        """
        other = self.copy_basic()
        if self.k is not None:
            other.k = self.k.copy()
        if self.ft is not None:
            other.ft = self.ft.copy()
        if self.mfld is not None:
            other.mfld = self.mfld.copy()
        if self.grad is not None:
            other.grad = self.grad.copy()
        if self.ft is not None:
            other.hess = self.hess.copy()
        if self.gmap is not None:
            other.gmap = self.gmap.copy()
        return other

    def sel_ambient(self, N: int):
        """Restrict to the first N ambient dimensions in a shallow copy
        """
        other = SubmanifoldFTbundle()
        other.shape = self.shape
        other.ambient = N
        other.intrinsic = self.intrinsic
        other.flat = self.flat
        if self.ft is not None:
            other.ft = self.ft[..., :N]
        if self.mfld is not None:
            other.mfld = self.mfld[..., :N]
        if self.grad is not None:
            other.grad = self.grad[..., :N, :]
        if self.ft is not None:
            other.hess = self.hess[..., :N, :, :]
        if self.gmap is not None:
            other.gmap = self.gmap[..., :N, :]
        return other

    def sel_intrinsic(self, K: int):
        """Restrict tangent space to the first K dimensions in a shallow copy
        """
        other = SubmanifoldFTbundle()
        other.shape = self.shape
        other.ambient = self.ambient
        other.intrinsic = K
        other.flat = self.flat
        if self.k is not None:
            other.k = self.k[..., :K]
        if self.grad is not None:
            other.grad = self.grad[..., :K]
        if self.ft is not None:
            other.hess = self.hess[..., :K, :K]
        if self.gmap is not None:
            other.gmap = self.gmap[..., :K]

    def flattish(self):
        """Flatten intrinsic location indeces
        """
        self.shape = (np.prod(self.shape),)
        K = self.intrinsic
        N = self.ambient
        if self.k is not None:
            self.k = self.k.reshape((-1, 1, K))
        if self.ft is not None:
            self.ft = self.ft.reshape((-1, N))
        if self.mfld is not None:
            self.mfld = self.mfld.reshape((-1, N))
        if self.grad is not None:
            self.grad = self.grad.reshape((-1, N, K))
        if self.ft is not None:
            self.hess = self.hess.reshape((-1, N, K, K))
        if self.gmap is not None:
            self.gmap = self.gmap.reshape((-1, N, K))
        self.flat = True


# =============================================================================
# calculate intermediaries
# =============================================================================


def induced_metric(mfld: SubmanifoldFTbundle) -> array:
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
    return mfld.grad.t @ mfld.grad


def raise_hess(mfld: SubmanifoldFTbundle) -> array:
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
    met = induced_metric(mfld)[..., None, :, :]
    hess = mfld.hess
    if mfld.intrinsic == 1:
        return hess / met
    if mfld.intrinsic > 2:
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
def mat_field_evals(mat_field: array) -> array:
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
        return eigvalsh(mat_field)

    tr_field = (mat_field[..., 0, 0] + mat_field[..., 1, 1]) / 2.0
    det_field = (mat_field[..., 0, 0] * mat_field[..., 1, 1] -
                 mat_field[..., 0, 1] * mat_field[..., 1, 0])
    disc_sq = tr_field**2 - det_field
    disc_sq[np.logical_and(disc_sq < 0., disc_sq > -1e-3)] = 0.0
    disc_field = np.sqrt(disc_sq)
    return np.stack((tr_field + disc_field, tr_field - disc_field), axis=-1)


@wrap_one
def mat_field_svals(mat_field: array) -> array:
    """
    Squared singular values of 2nd rank tensor field, `mat_field`

    Returns
    -------
    (sval1^2, sval2^2, ...)
        squared singular values, `sval1` > `sval2`, (L1,L2,...,K)
    """
    if mat_field.shape[-1] == 1:
        return norm(mat_field, axis=-2)**2
    if mat_field.shape[-1] > 2:
        return singvals(mat_field)**2

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


def numeric_distance(mfld: SubmanifoldFTbundle) -> (array, array):
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
    pos = mfld.mfld
    # chords
    mid = tuple(L // 2 for L in pos.shape[:-1]) + (slice(None),)
    dx = pos - pos[mid]
    # chord length
    d = norm(dx, axis=-1, keepdims=True)
    # unit vectors along dx
    zero = d < 1e-7
    d[zero] = 1.
    ndx = np.where(zero, 0., dx / d)
    d[zero] = 0.
    return d.uc, ndx.view(array)


def numeric_sines(mfld: SubmanifoldFTbundle) -> (array, array):
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
    kbein = mfld.gmap
    mid = tuple(L // 2 for L in kbein.shape[:-2]) + (slice(None),)*2
    base_bein = kbein[mid]
    bein_prod = base_bein.T @ kbein
    cosangs = mat_field_svals(bein_prod)
    cosangs[cosangs > 1.] = 1.
    return np.flip(np.sqrt(1. - cosangs), axis=-1)


@wrap_one
def numeric_proj(ndx: array,
                 mfld: SubmanifoldFTbundle,
                 inds: Tuple[slice, ...]) -> array:
    """
    Cosine of angle between chord and tangent vectors

    Returns
    -------
    costh
        costh[s,t,...] = max_u,v,... (cos angle between tangent vector at
        x[u,v,...] and chord between x[mid] and x[s,t,...]).

    Parameters
    ----------
    ndx : array (L1,...,LK,N)
        unit vector in chord direction. Central vector is undefined.
        ndx[s,t,...,i] = (phi^i(x[s,t,...]) - phi_i(x[mid])) / d[s,t,...]
    kbein : array (L1,...,LK,N,K)
        orthonormal basis for tangent space,
        kbein[s,t,...,i,A] = e_A^i(x1[s], x2[t], ...),
    inds
        K-tuple of slices for region to search over for lowest angle
    """
    flat_bein = mfld.gmap[inds].flatter(0, -2)  # (L,N,K)
    # The limit here corresponds to 2GB memory per K (memory ~ size^2)
    if np.prod(ndx.shape[:-1]) <= 2**14:
        with dcontext('matmult'):
            # (L1,...,LK,1,1,N) @ (L,N,K) -> (L1,...,LK,L,1,K)
            # -> (L1,...,LK,L,1) -> (L1,...,LK)
            costh = norm(ndx.r.r @ flat_bein, axis=-1).max(axis=(-2, -1))
        # deal with central vector
        costh[tuple(siz // 2 for siz in costh.shape)] = 1.
        return costh

    def calc_costh(chord):
        """Calculate max cos(angle) between chord and any tangent vector"""
        # (1,N) @ (L,N,K) -> (L,1,K) -> (L,1) -> ()
        return norm(chord.r @ flat_bein, axis=-1).max()

#    with dcontext('max matmult'):
#        costh = np.apply_along_axis(calc_costh, -1, ndx)

    costh = np.empty(ndx.shape[:-1])
    for ii in dndindex(*ndx.shape[:-1]):
        costh[ii] = np.apply_along_axis(calc_costh, -1, ndx[ii])
    costh[tuple(siz // 2 for siz in ndx.shape[:-1])] = 1.

    return costh


def numeric_curv(mfld: SubmanifoldFTbundle) -> array:
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
    hessr = raise_hess(mfld).swapaxes(-1, -3)
    # hessian projected onto tangent space (L1,L2,...,K,K,K): H^A_a^b
    hesst = (hessr @ mfld.gmap[..., None, :, :]).swapaxes(-1, -3)
#    hessrt = hessr.swapaxes(-3, -2).swapaxes(-2, -1) @ kbein
    return np.sum(hessr @ np.moveaxis(hessr, -3, -1) - hesst @ hesst, axis=-3)


# =============================================================================
# the whole thing
# =============================================================================


def get_all_numeric(ambient_dim: int,
                    intrinsic_range: Sequence[float],
                    intrinsic_num: Sequence[int],
                    width: Sequence[float] = (1.0, 1.0),
                    expand: int = 2) -> (array, array, array, array):
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
        mfld = SubmanifoldFTbundle(embed_ft, karr)
        mfld.calc_embed()
    with dcontext('grad'):
        mfld.calc_grad()
    with dcontext('hess'):
        mfld.calc_hess()
    with dcontext('e'):
        mfld.calc_gmap()
    with dcontext('K'):
        curvature = numeric_curv(mfld)
    mfld.dump_ft()

    int_begin = [(expand - 1) * inum // 2 for inum in intrinsic_num]
    int_end = [inum + ibeg for inum, ibeg in zip(intrinsic_num, int_begin)]

    region = tuple(slice(ibeg, iend) for ibeg, iend in zip(int_begin, int_end))

    with dcontext('d'):
        num_dist, ndx = numeric_distance(mfld)
    with dcontext('a'):
        num_sin = numeric_sines(mfld)
    with dcontext('p'):
        num_pr = numeric_proj(ndx, mfld, region)
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
