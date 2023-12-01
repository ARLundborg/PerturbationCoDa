import numpy as np
from scipy.interpolate import BSpline
from scipy import sparse
import warnings


def iqr(x):
    return np.subtract(*np.percentile(x, [75, 25]))


def design_matrix_2_deriv_cubic(x, all_knots):

    knots_diff_offset3 = all_knots[3:] - all_knots[:-3]
    knots_diff_offset2 = all_knots[2:] - all_knots[:-2]

    spline_mat = BSpline.design_matrix(x, all_knots, 1)

    # The following is a hack to fix a boundary issue with the splines
    # as defined in scipy. On the right-most inner knot, the splines are 0
    # but we need them to be 1.

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sparse.SparseEfficiencyWarning)
        for i, is_boundary in enumerate(x == np.max(all_knots)):
            if is_boundary:
                spline_mat[i, -3] = 1.0

    # Save intermediate computations
    tmp1 = knots_diff_offset3[:-1]*knots_diff_offset2[:-2]
    tmp2 = knots_diff_offset3[:-1]*knots_diff_offset2[1:-1]
    tmp3 = knots_diff_offset3[1:]*knots_diff_offset2[1:-1]
    tmp4 = knots_diff_offset3[1:]*knots_diff_offset2[2:]

    res = 6*(np.reciprocal(tmp1, where=tmp1 != 0, out=np.zeros_like(tmp1)) * spline_mat[:, :-2] - (np.reciprocal(tmp2, where=tmp2 != 0, out=np.zeros_like(tmp2)) + np.reciprocal(
        tmp3, where=tmp3 != 0, out=np.zeros_like(tmp3))) * spline_mat[:, 1:-1] + np.reciprocal(tmp4, where=tmp4 != 0, out=np.zeros_like(tmp4))*spline_mat[:, 2:])

    return res


def pen_mat(knots, all_knots):
    # Based on example R code from the book Computational Statistics with R by Niels Richard Hansen
    diff_knots = np.diff(knots)
    g_ab = design_matrix_2_deriv_cubic(knots, all_knots)
    knots_mid = knots[:(-1)] + diff_knots/2
    g_ab_mid = design_matrix_2_deriv_cubic(knots_mid, all_knots)

    g_a = g_ab[:-(1), :]
    g_b = g_ab[1:, :]

    pen_mat = (diff_knots*g_a.T).dot(g_a) + 4*(diff_knots *
                                               g_ab_mid.T).dot(g_ab_mid) + (diff_knots*g_b.T).dot(g_b)

    return pen_mat/6


def make_posdef(A, mineig=10**(-15)):
    aa = np.linalg.eigvals(A)
    if (np.min(aa) < mineig):
        if (min(aa) < 0):
            A = A + np.eye(A.shape[0])*(-np.min(aa) + mineig)
        else:
            A = A + np.eye(A.shape[0])*mineig
    return (A)


def smoothing_spline_setup(x, nknots=None, tol=None):

    degree = 3

    # The defaults below are taken from the R smooth.spline function
    if tol is None:
        tol = 1e-6*iqr(x)

    n = len(x) - np.sum(np.unique(np.round((x - np.mean(x)) / tol),
                                  return_counts=True)[1] > 1)

    if nknots is None:
        a1 = np.log2(50)
        a2 = np.log2(100)
        a3 = np.log2(140)
        a4 = np.log2(200)
        if n < 50:
            nknots = n
        elif n < 200:
            nknots = np.trunc(2**(a1 + (a2 - a1) * (n - 50)/150))
        elif n < 800:
            nknots = np.trunc(2**(a2 + (a3 - a2) * (n - 200)/600))
        elif n < 3200:
            nknots = np.trunc(2**(a3 + (a4 - a3) * (n - 800)/2400))
        else:
            nknots = np.trunc(200 + (n - 3200)**0.2)

    nknots = int(nknots)
    xmin = np.min(x)
    xmax = np.max(x)
    knots = np.linspace(xmin, xmax, nknots)
    all_knots = np.r_[(xmin,)*(degree), knots, (xmax,)*(degree)]

    Phi = BSpline.design_matrix(x, all_knots, degree)
    Omega = pen_mat(knots, all_knots)

    return Phi, Omega, all_knots


def smoothing_spline_fit(Phi, y, w, Omega, all_knots, penalties):
    p = len(penalties)
    k = Phi.shape[1]

    if w is None:
        w = np.ones(len(y))

    w = w / np.sum(w) * len(y)
    A = (w*(Phi.T)).dot(Phi)
    b = Phi.T.dot(w*y)

    splines = tuple(BSpline(all_knots, np.zeros(k), 3) for _ in range(p))

    for i, penalty in enumerate(penalties):
        M = A + penalty*Omega

        coef = sparse.linalg.spsolve(M, b)
        splines[i].c = coef

    return splines


def smoothing_spline(x, y, w=None, penalty=1, nknots=None, tol=None):

    Phi, Omega, all_knots = smoothing_spline_setup(x, nknots, tol)
    spline = smoothing_spline_fit(Phi, y, w, Omega, all_knots, [penalty])[0]

    return spline


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    x = np.linspace(0, 1, 10)
    y = x**2 + np.random.randn(len(x))*0.1 + 1
    w = np.array([1, 1, 1, 1, 1, 2, 2, 2, 1, 1])

    plt.scatter(x, y)
    plt.scatter(x, x**2+1, color="red")
    spline = smoothing_spline(x, y, w=w, penalty=0.0000001)
    plt.plot(x, spline(x), color="yellow")

    spline2 = smoothing_spline(x, y, w=w, penalty=0.001)
    plt.plot(x, spline2(x), color="green")
    for xc in spline2.t[spline2.k:(-spline2.k)]:
        plt.axvline(x=xc, alpha=0.1)
