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
import numpy as np
from . import gauss_curve as gc
from . import gauss_curve_theory as gct
from . import gauss_surf_theory as gst

# =============================================================================
# generate surface
# =============================================================================


def gauss_cov(dx, dy, width=(1.0, 1.0)):  # Gaussian covariance matrix
    """
    Covariance matrix that is a Gaussian function of difference in position

    Returns
    -------
    covmat
        matrix exp(-1/2 * sum_a (dx_a^2 / width_a^2)

    Parameters
    ----------
    dx,dy
        vectors of position differences
    width
        tuple of std devs of gaussian covariance along each intrinsic axis
    """
    return np.outer(gct.gauss_cov(dx, width[0]), gct.gauss_cov(dy, width[1]))


def random_embed_ft(num_dim, kx, ky,
                    width=(1.0, 1.0)):  # generate FFT of rand Gaussn curve
    """
    Generate Fourier transform of ramndom Gaussian curve with a covariance
    matrix that is a Gaussian function of difference in position

    Returns
    -------
    embed_ft
        Fourier transform of embedding functions,
        embed_ft[s,t,i] = phi^i(kx[s], ky[t])

    Parameters
    ----------
    num_dim
        dimensionality ofambient space
    kx,ky
        vectors of spatial frequencies
    width
        tuple of std devs of gaussian cov along each intrinsic axis
    """
    sqrt_cov = (gc.gauss_sqrt_cov_ft(kx, width[0]) *
                gc.gauss_sqrt_cov_ft(ky, width[1]))
    emb_ft_r = np.random.randn(kx.size, ky.size, num_dim) * sqrt_cov
    emb_ft_i = np.random.randn(kx.size, ky.size, num_dim) * sqrt_cov
    return (emb_ft_r + 1j * emb_ft_i) / np.sqrt(2 * num_dim)


# =============================================================================
# calculate intermediaries
# =============================================================================


def spatial_freq(intrinsic_range, intrinsic_num,
                 expand=2):  # vector of spatial frequencies
    """
    Vectors of spatial frequencies

    Returns
    -------
    kx[s,,], ky[,t,]
        Vectors of spatial frequencies used in FFT. Appropriate singleton
        dimensions added to broadcast with embed_ft

    Parameters
    ----------
    intrinsic_range
        tuple of ranges of intrinsic coords [-intrinsic_range, intrinsic_range]
    intrinsic_num
        tuple of numbers of sampling points on surface
    expand
        factor to increase size by, to subsample later
    """
    intrinsic_res = (2. * intrinsic_range[0] / intrinsic_num[0],
                     2. * intrinsic_range[1] / intrinsic_num[1])
    kx = 2 * np.pi * np.fft.fftfreq(expand * intrinsic_num[0],
                                    intrinsic_res[0])
    ky = 2 * np.pi * np.fft.rfftfreq(expand * intrinsic_num[1],
                                     intrinsic_res[1])
    return kx[:, None, None], ky[None, :, None]


def stack_vec(*cmpts):  # stack components in a vector
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
#    vec = cmpts[0][np.newaxis, ...]
#    for i in range(1, len(cmpts)):
#        vec = np.concatenate((vec, cmpts[i][np.newaxis, ...]), axis=0)
    return vec


def embed(embed_ft):  # calculate embedding functions
    """
    Calculate embedding functions

    Returns
    -------
    emb
        emb[s,t,i] = phi^i(x[s], y[t])

    Parameters
    ----------
    embed_ft
        Fourier transform of embedding functions,
        embed_ft[s,t,i] = phi^i(kx[s], ky[t])
    kx,ky
        vectors of spatial frequencies
    """
    return np.fft.irfft2(embed_ft, axes=(0, 1))


def embed_grad(embed_ft, kx, ky):  # calculate gradient of embedding functions
    """
    Calculate gradient of embedding functions

    Returns
    -------
    grad
        grad[s,t,i,a] = phi_a^i(x[s], y[t])

    Parameters
    ----------
    embed_ft
        Fourier transform of embedding functions,
        embed_ft[i,s,t] = phi^i(kx[s], ky[t])
    kx,ky
        vectors of spatial frequencies
    """
    gradx = np.fft.irfft2(1j * embed_ft * kx, axes=(0, 1))
    grady = np.fft.irfft2(1j * embed_ft * ky, axes=(0, 1))
    return stack_vec(gradx, grady)


def embed_hess(embed_ft, kx, ky):  # calculate hessian of embedding functions
    """
    Calculate hessian of embedding functions

    Returns
    -------
    hess
        hess[s,t,i,a,b] = phi_ab^i(x[s], y[t])

    Parameters
    ----------
    embed_ft
        Fourier transform of embedding functions,
        embed_ft[s,t,i] = phi^i(kx[s], ky[t])
    kx,ky
        vectors of spatial frequencies
    """
    gradxx = np.fft.irfft2(-embed_ft * kx**2, axes=(0, 1))
    gradxy = np.fft.irfft2(-embed_ft * kx * ky, axes=(0, 1))
    gradyy = np.fft.irfft2(-embed_ft * ky**2, axes=(0, 1))
    gradx = stack_vec(gradxx, gradxy)
    grady = stack_vec(gradxy, gradyy)
    return stack_vec(gradx, grady)


def vielbein(grad):  # orthonormal basis for tangent space
    """
    Orthonormal basis for tangent space, push-forward of vielbein.

    Returns
    -------
    vbein
        orthonormal basis for tangent space,
        vbein[s,t,i,A] = e_A^i(x[s], y[t]).

        vbein[...,  0] parallel to dx^0.
        vbein[...,  1] perpendicular to dx^0, not necessarily parallel to dx^1.

    Parameters
    ----------
    grad
        grad[s,t,i,a] = phi_a^i(x[s], y[t])
    """
    u1 = grad[..., 0]
    u2 = grad[..., 1]
    u1 /= np.linalg.norm(u1, axis=-1, keepdims=True)
    np.nan_to_num(u1, copy=False)
    u2 -= u1 * np.sum(u1 * u2, axis=-1, keepdims=True)
    u2 /= np.linalg.norm(u2, axis=-1, keepdims=True)
    np.nan_to_num(u2, copy=False)
    return stack_vec(u1, u2)


def induced_metric(grad):  # induced metric
    """
    Induced metric on embedded surface

    Returns
    -------
    h
        induced metric
        h[s,t,a,b] = h_ab(x[s], y[t])

    Parameters
    ----------
    grad
        grad[s,t,i,a] = phi_a^i(x[s], y[t])
    """
#    hxx = np.sum(grad[0,...]**2, axis=0)
#    hxy = np.sum(grad[0,...] * grad[1,...], axis=0)
#    hyy = np.sum(grad[1,...]**2, axis=0)
#    hx = np.concatenate((hxx[np.newaxis,...], hxy[np.newaxis,...]), axis=0)
#    hy = np.concatenate((hxy[np.newaxis,...], hyy[np.newaxis,...]), axis=0)
#    return np.concatenate((hx[np.newaxis,...], hy[np.newaxis,...]), axis=0)
#    return np.einsum('stia,stib->stab', grad, grad)
    return grad.swapaxes(-2, -1) @ grad


def inverse_metric(grad):  # inverse induced metric
    """
    Inverse of induced metric on embedded surface

    Returns
    -------
    h
        inverse induced metric,
        h[s,t,a,b] = h^ab(x[s], y[t])

    Parameters
    ----------
    grad
        grad[s,t,i,a] = phi_a^i(x[s], y[t])
    """
    ind_met = induced_metric(grad)
    hxx = ind_met[..., 1, 1]
    hxy = -ind_met[..., 0, 1]
    hyy = ind_met[..., 0, 0]
    hx = stack_vec(hxx, hxy)
    hy = stack_vec(hxy, hyy)
    h = stack_vec(hx, hy)
#    hx = np.concatenate((hxx[np.newaxis,...], hxy[np.newaxis,...]), axis=0)
#    hy = np.concatenate((hxy[np.newaxis,...], hyy[np.newaxis,...]), axis=0)
#    h = np.concatenate((hx[np.newaxis,...], hy[np.newaxis,...]), axis=0)
    hdet = h[..., 0, 0] * h[..., 1, 1] - h[..., 0, 1] * h[..., 1, 0]
    return h / hdet[..., None, None]


def raise_hess(embed_ft, kx, ky, grad):
    """
    Hessian with second index raised

    Returns
    -------
    hess
        hess[s,t,i,a,b] = phi_a^bi(x[s], y[t])

    Parameters
    ----------
    embed_ft
        Fourier transform of embedding functions,
        embed_ft[s,t,i] = phi^i(kx[s], ky[t])
    kx,ky
        vectors of spatial frequencies
    grad
        grad[s,t,i,a] = phi_a^i(x[s], y[t])
    """
    inv_met = inverse_metric(grad)
    hess = embed_hess(embed_ft, kx, ky)
    return hess @ inv_met[..., None, :, :]
#    return np.einsum('abist,bcst->acist', hess, inv_met)


def tangent_proj(zweibein):  # projection operator for tangent space
    """
    Projection operator for tangent space of surface

    Returns
    -------
    h
        push forward of inverse induced metric,
        h[s,t,i,j] = h^ij(x[s],y[t])

    Parameters
    ----------
    zweibein
        orthonormal basis for tangent space,
        zweibein[s,t,i,A] = e_A^i(x[s], x[t]),
    """
#    hx = inv_met[0,0,...] * grad[0,...] + inv_met[0,1,...] * grad[1,...]
#    hy = inv_met[1,0,...] * grad[0,...] + inv_met[1,1,...] * grad[1,...]
#    return hx * grad[0,:,np.newaxis,...] + hy * grad[1,:,np.newaxis,...]
    return zweibein @ zweibein.swapaxes(-2, -1)
#    return np.einsum('aist,ajst->ijst', zweibein, zweibein)


def mat_field_evals(mat_field):  # eigenvalues of 2nd rank tensor field
    """
    Eigenvalues of 2nd rank tensor field, mat_field

    Returns
    -------
    eval1, eval2
        eval1 > eval2
    """
    tr_field = (mat_field[..., 0, 0] + mat_field[..., 1, 1]) / 2.0
    det_field = (mat_field[..., 0, 0] * mat_field[..., 1, 1] -
                 mat_field[..., 0, 1] * mat_field[..., 1, 0])
    disc_sq = tr_field**2 - det_field
    disc_sq[np.logical_and(disc_sq < 0., disc_sq > -1e-3)] = 0.0
    disc_field = np.sqrt(disc_sq)
    return tr_field + disc_field, tr_field - disc_field


def mat_field_svals(mat_field):  # sqrd sing vals of 2nd rank tensor field
    """
    Squared singular values of 2nd rank tensor field, mat_field

    Returns
    -------
    sval1^2, sval2^2
        sval1 > sval2
    """
    frob_field = (mat_field[..., 0, 0]**2 + mat_field[..., 0, 1]**2 +
                  mat_field[..., 1, 0]**2 + mat_field[..., 1, 1]**2) / 2.0
    det_field = (mat_field[..., 0, 0] * mat_field[..., 1, 1] -
                 mat_field[..., 0, 1] * mat_field[..., 1, 0])
    disc_sq = frob_field**2 - det_field**2
    disc_sq[np.logical_and(disc_sq < 0., disc_sq > -1e-3)] = 0.0
    disc_field = np.sqrt(disc_sq)
    return frob_field + disc_field, frob_field - disc_field


# =============================================================================
# calculate distances, angles and curvature
# =============================================================================


def numeric_distance(embed_ft):  # Euclidean distance from centre
    """
    Calculate Euclidean distance from central point on curve as a fuction of
    position on curve.

    Returns
    -------
    d
        d[s,t] = ||phi(x[s,t]) - phi(x[mid])||

    Parameters
    ----------
    embed_ft
        Fourier transform of embedding functions,
        embed_ft[s,t,i] = phi^i(kx[s], ky[t])
    kx,ky
        vectors of spatial frequencies
    """
    pos = embed(embed_ft)
    dpos = pos - pos[pos.shape[0] // 2, pos.shape[1] // 2, :]
    return np.linalg.norm(dpos, axis=-1)


# sine of angle between tangent vectors
# def numeric_sines(zweibein, tang_proj):
def numeric_sines(zweibein):  # sine of angle between tangent vectors
    """
    Sine of angle between tangent vectors

    Returns
    -------
    sin(theta_max), sin(theta_min)
        S[a][s,t] = tuple of sin theta_a[s,t]
    theta_a
        principal angle between tangent spaces at x[s], x[t] and at center

    Parameters
    ----------
    zweibein
        orthonormal basis for tangent space,
        zweibein[s,t,i,A] = e_A^i(x[s], x[t]),
    """
    base_bein = zweibein[zweibein.shape[0] // 2, zweibein.shape[1] // 2, ...]
#    bein_prod = np.einsum('aist,bi->abst', zweibein, base_bein)
    bein_prod = base_bein.T @ zweibein
#    Returns sum_a sin^2 theta_a[s,t], sin^2 theta_max[s,t]
#     # projection operator for tangent space
#    tang_proj[i,j,s,t] = h^ij(x[s], y[t]),
#    base_proj = tang_proj[..., tang_proj.shape[-2] // 2,
#                           tang_proj.shape[-1] // 2]
#    proj_prod = np.einsum('ikst,jk->ijst', tang_proj, base_proj)
    cosang0, cosang1 = mat_field_svals(bein_prod)
    cosang0[cosang0 > 1.] = 1.
    cosang1[cosang1 > 1.] = 1.
    return np.sqrt(1. - cosang1), np.sqrt(1. - cosang0)
#    return (2 - np.einsum('ijst,ij->st', tang_proj, base_proj),
#            1 - mat_field_svals(bein_prod)[1])


def numeric_curv(hessr, zweibein):  # curvature of curve
    """
    Extrinsic curvature

    Returns
    -------
    kappa
        kappa[a,b,s,t] = kappa^a_b(x[s],y[t])

    Parameters
    ----------
    hessr
        hessian with one index raised
        hessr[s,t,i,a,b] = phi_a^bi(x[s], y[t])
    zweibein
        orthonormal basis for tangent space,
        zweibein[s,t,i,a] = e_a^i(x[s], x[t]),
    """
#    hessrt = np.einsum('abist,cist->abcst', hessr, zweibein)
#    return (np.einsum('acist,cbist->abst',  hessr, hessr) -
#            np.einsum('acdst,cbdst->abst',  hessrt, hessrt))
    # hessian projected onto tangent space
    hessrt = (hessr.transpose(0, 1, 3, 4, 2) @
              zweibein[..., None, :, :]).transpose(0, 1, 4, 2, 3)
#    hessrt = hessr.swapaxes(-3, -2).swapaxes(-2, -1) @ zweibein
    return np.sum(hessr @ hessr, axis=-3) - np.sum(hessrt @ hessrt, axis=-3)


# =============================================================================
# the whole thing
# =============================================================================


def get_all_numeric(ambient_dim, intrinsic_range, intrinsic_num,
                    width=(1.0, 1.0), expand=2):  # calculate everything
    """
    Calculate everything

    Returns
    -------
    nud
        numeric distances
    nus
        numeric sines, tuple,
        sine 1 > sine 2
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
#    intrinsic_res = (4 * intrinsic_range[0] / intrinsic_num[0],
#                     4 * intrinsic_range[1] / intrinsic_num[1])
#    kx = 2 * np.pi * np.fft.fftfreq(intrinsic_num[0], intrinsic_res[0])
#    ky = 2 * np.pi * np.fft.rfftfreq(intrinsic_num[1], intrinsic_res[1])

    print('k')
    kx, ky = spatial_freq(intrinsic_range, intrinsic_num, expand)
    print('mfld')
    embed_ft = random_embed_ft(ambient_dim, kx, ky, width)
    print('grad')
    grad = embed_grad(embed_ft, kx, ky)
    print('hess')
    hessr = raise_hess(embed_ft, kx, ky, grad)

    print('e')
    zweibein = vielbein(grad)
#    print('U')
#    tang_proj = tangent_proj(zweibein)
    print('K')
    curvature = numeric_curv(hessr, zweibein)

    print('d')
    num_dist = numeric_distance(embed_ft)
    print('a')
    num_sin_max, num_sin_min = numeric_sines(zweibein)
    print('c')
    num_curv1, num_curv2 = mat_field_evals(curvature)

    int_begin = ((expand - 1) * intrinsic_num[0] // 2,
                 (expand - 1) * intrinsic_num[1] // 2)
    int_end = (intrinsic_num[0] + int_begin[0],
               intrinsic_num[1] + int_begin[1])
    region = (slice(int_begin[0], int_end[0]), slice(int_begin[1], int_end[1]))

    nud = num_dist[region]
    nua = (num_sin_max[region], num_sin_min[region])
    nuc = (num_curv1[region], num_curv2[region])

    return nud, nua, nuc


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
    ambient_dim = 200    # dimensionality of ambient space
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


def make_and_save(filename, ambient_dim, intrinsic_range, intrinsic_num,
                  width):  # generate data and save
    """
    Generate data and save in .npz file

    Parameters
    ----------
    filenamee
        name of .npz file, w/o extension, for data
    ambient_dim
        N, dimensionality of ambient space
    intrinsic_range
        tuple of ranges of intrinsic coords [-intrinsic_range, intrinsic_range]
    intrinsic_num
        tuple of numbers of sampling points on surface
    width
        tuple of std devs of gaussian covariance along each intrinsic axis
    """
    print('analytic 1')
    theory = gst.get_all_analytic(ambient_dim, intrinsic_range, intrinsic_num,
                                  width)
    x, y, rho, thr_dis, thr_sin, thr_cur = theory

    print('analytic 2')
    theoryl = gst.get_all_analytic_line(rho, np.maximum(*intrinsic_num))
    rhol, thr_dsl, thr_snl, thr_crl = theoryl

    print('numeric')
    num_dis, num_sin, num_cur = get_all_numeric(ambient_dim,
                                                intrinsic_range,
                                                intrinsic_num,
                                                width)

    np.savez_compressed(filename + '.npz', x=x, y=y, rho=rho, rhol=rhol,
                        thr_dis=thr_dis, thr_sin=thr_sin, thr_cur=thr_cur,
                        thr_disl=thr_dsl, thr_sinl=thr_snl, thr_curl=thr_crl,
                        num_dis=num_dis, num_sin=num_sin, num_cur=num_cur)


# =============================================================================
# test code
# =============================================================================


if __name__ == "__main__":
    print('Run from outside package.')
