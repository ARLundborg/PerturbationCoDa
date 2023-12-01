import numpy as np
import scipy.sparse as sparse
from scipy.linalg import solveh_banded
from scipy.stats import binned_statistic
import main.smoothing_spline as smoothing_spline
from sklearn.model_selection import KFold


def spline_score_cv(x, penalties=None, folds=5, tol=1e-5, rule="min", random_state=None):

    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)

    x_sort = np.sort(x)
    x_sort, w = bin_sorted_x(x_sort, tol=tol)
    pseudo_y = ng_pseudo_response(sorted_x=x_sort, w=w)

    Phi, Omega, all_knots = smoothing_spline.smoothing_spline_setup(
        x_sort, nknots=None, tol=tol)

    if penalties is None:
        w = w/np.sum(w)*len(x_sort)
        A = (w*(Phi.T)).dot(Phi)
        penalties = 10.0**np.arange(start=-5, stop=10, step=0.5) * \
            sparse.linalg.norm(A)/sparse.linalg.norm(Omega)

    loss = np.zeros((folds, len(penalties)))
    for i, (train_ind, test_ind) in enumerate(kf.split(x)):
        splines = spline_score(x[train_ind], penalties, tol)
        loss[i, :] = [np.mean(spline(
            x[test_ind])**2 + 2*spline.derivative()(x[test_ind])) for spline in splines]

    cv_mean = np.mean(loss, axis=0)
    cv_se = np.std(loss, axis=0)/np.sqrt(folds)

    i_min = np.argmin(cv_mean)
    if rule == "min":
        return smoothing_spline.smoothing_spline_fit(Phi, pseudo_y, w, Omega, all_knots, [penalties[i_min]])[0]
    elif rule == "1se":
        i_1se = len(penalties) - \
            np.argmax(np.flip(cv_mean[i_min] + cv_se[i_min] > cv_mean)) - 1
        return smoothing_spline.smoothing_spline_fit(Phi, pseudo_y, w, Omega, all_knots, [penalties[i_1se]])[0]
    elif rule == "2se":
        i_1se = len(
            penalties) - np.argmax(np.flip(cv_mean[i_min] + 2*cv_se[i_min] > cv_mean)) - 1
        return smoothing_spline.smoothing_spline_fit(Phi, pseudo_y, w, Omega, all_knots, [penalties[i_1se]])[0]
    else:
        return smoothing_spline.smoothing_spline_fit(Phi, pseudo_y, w, Omega, all_knots, penalties)


def spline_score(x, penalties=0, tol=1e-5):
    '''
    Univariate score estimation via the smoothing spline method of Cox [1985] and Ng [1994].
    '''

    x_sort = np.sort(x)
    x_sort, w = bin_sorted_x(x_sort, tol=tol)
    pseudo_y = ng_pseudo_response(sorted_x=x_sort, w=w)

    Phi, Omega, all_knots = smoothing_spline.smoothing_spline_setup(
        x_sort, nknots=None, tol=tol)
    splines = smoothing_spline.smoothing_spline_fit(
        Phi, pseudo_y, w, Omega, all_knots, penalties)
    if len(penalties) == 1:
        return splines[0]
    else:
        return splines


def bin_sorted_x(sorted_x, tol=1e-5):
    '''
    Binning function to avoid singularities.
    '''
    if tol is None:
        return sorted_x, np.ones(len(sorted_x))

    binned_x, _, bin_number = binned_statistic(
        sorted_x, sorted_x, bins=np.ceil((sorted_x[-1]-sorted_x[0])/tol))

    binned_x = binned_x[~np.isnan(binned_x)]
    w = np.unique(bin_number, return_counts=True)[1]

    return binned_x, w


def ng_pseudo_response(sorted_x, w):
    '''
    Generate pseudo responses as in Ng [1994] to enable univariate score estimation by standard smoothing spline regression.
    Pseudo responses should be regarded as a computational tool, not as an estimate of the score itself.
    '''

    n = len(sorted_x)
    h = np.diff(sorted_x)

    wih = np.append(w[0:(n-2)]/h[0:(n-2)], (w[n-2]+w[n-1])/h[n-2])
    wh = np.append(w[0:(n-2)]*h[0:(n-2)], (w[n-2]-w[n-1]/2)*h[n-2])

    # Notation as in Ng [1994] and Ng [2003].

    a_vec = np.append(wih, 0) - np.insert(wih, 0, 0)
    c_vec = (wh[0:(n-2)] + 2*wh[1:(n-1)])/3
    ih = 1/h

    # Specifying the R matrix as simply the diagonal and off-diagonal to be solved later
    R_diag = np.array(h[0:(n-2)] + h[1:(n-1)])*2/3
    R_offdiag = h[1:(n-2)]/3

    Q = np.diag(ih[0:(n-2)])
    Q = np.concatenate((Q, np.zeros((1, n-2)), np.zeros((1, n-2))), axis=0)
    Q[np.arange(n-2)+1, np.arange(n-2)] = -(ih[0:(n-2)] + ih[1:(n-1)])
    Q[np.arange(n-2)+2, np.arange(n-2)] = ih[1:(n-1)]

    Q = sparse.dia_matrix(Q)

    # (Not entirely sure this works -- unclear if R is always pos-def)
    z = solveh_banded(
        np.stack((R_diag, np.append(R_offdiag, 0)), axis=0), c_vec, lower=True)
    y = 1/w * (a_vec + Q.dot(z))

    return y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x = np.random.normal(0, 1, 100)
    res = spline_score_cv(x, folds=10, rule="1se")

    plt.scatter(x, res(x))
    plt.plot(np.sort(x), -np.sort(x), c="red")
    plt.xlim(-3, 2)
    plt.ylim(-3, 2)
