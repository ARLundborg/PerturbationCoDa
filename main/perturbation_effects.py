import main.semiparametric_estimators as semi_est
import numpy as np
from scipy.spatial import distance_matrix


def correct_nonzero(result, indices, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()

    p_hat = np.mean(indices)
    sigma_2 = result["variance"]
    theta = result["estimate"]
    new_theta = theta*p_hat
    new_sigma_2 = sigma_2*p_hat + theta**2*p_hat*(1-p_hat)
    result["estimate"] = new_theta
    result["variance"] = new_sigma_2
    result["standard_error"] = np.sqrt(new_sigma_2/indices.shape[0])
    return result


def cont_pert_effect(Y, L, W, Y_regression, L_regression, method, L_cond_var_regression=None, folds=2, random_state=None):
    if method == "partially_linear":
        res = semi_est.partially_linear_model(Y, L, W,
                                              Y_regression, L_regression, folds=folds, random_state=random_state)
    elif method == "nonparametric":
        if L_cond_var_regression is None:
            raise TypeError(
                "If method is nonparametric, then L_cond_var_regression needs to be given.")
        res = semi_est.average_partial_effect(
            Y, L, W, Y_regression, L_regression, L_cond_var_regression, folds=folds, random_state=random_state)
    else:
        raise ValueError(
            "method arguments needs to be either `nonparametric` or `partially_linear`.")
    return res


def cdi_gini(Y, Z, Y_regression, L_regression, X=None, method="partially_linear", L_cond_var_regression=None, folds=2, random_state=None):
    if not np.allclose(Z.sum(axis=1), 1):
        raise ValueError("Z is not in the simplex.")

    d = Z.shape[1]
    norms = np.linalg.norm(Z-1/d, axis=1, ord=1)
    ginis = np.apply_along_axis(lambda x: np.abs(
        np.subtract.outer(x, x)).sum(), 1, Z)/(2*d)
    L = 1-ginis
    W = (Z-1/d)/norms[:, np.newaxis]

    if X is not None:
        W = np.c_[W, X]

    res = cont_pert_effect(Y, L, W, Y_regression, L_regression,
                           method, L_cond_var_regression, folds, random_state)

    return res


def cdi_unit(Y, Z, Y_regression, L_regression, X=None, method="partially_linear", L_cond_var_regression=None, folds=2, random_state=None):
    if not np.allclose(Z.sum(axis=1), 1):
        raise ValueError("Z is not in the simplex.")

    d = Z.shape[1]
    norms = np.linalg.norm(Z-1/d, axis=1, ord=1)
    L = -norms
    W = (Z-1/d)/norms[:, np.newaxis]

    if X is not None:
        W = np.c_[W, X]

    res = cont_pert_effect(Y, L, W, Y_regression, L_regression,
                           method, L_cond_var_regression, folds, random_state)

    return res


def cfi_unit(Y, Z, j, Y_regression, L_regression, X=None, method="partially_linear", L_cond_var_regression=None, folds=2, random_state=None):
    if not np.allclose(Z.sum(axis=1), 1):
        raise ValueError("Z is not in the simplex.")

    d = Z.shape[1]
    A_z = np.zeros(d)
    A_z[j] = 1
    norms = np.linalg.norm(A_z - Z, axis=1, ord=1)

    L = -norms
    W = (A_z - Z)/norms[:, np.newaxis]

    if X is not None:
        W = np.c_[W, X]

    res = cont_pert_effect(Y, L, W, Y_regression, L_regression,
                           method, L_cond_var_regression, folds, random_state)

    return res


def cfi_mult(Y, Z, j, Y_regression, L_regression, X=None, method="partially_linear", L_cond_var_regression=None, folds=2, random_state=None):
    if not np.allclose(Z.sum(axis=1), 1):
        raise ValueError("Z is not in the simplex.")
    selected_indices = Z[:, j] > 0
    Y = Y[selected_indices]
    Z = Z[selected_indices, :]

    d = Z.shape[1]
    A_z = np.zeros(d)
    A_z[j] = 1
    norms = np.linalg.norm(A_z - Z, axis=1, ord=1)

    L = np.log(Z[:, j]/(1-Z[:, j]))
    W = (A_z - Z)/norms[:, np.newaxis]

    if X is not None:
        W = np.c_[W, X]

    res = cont_pert_effect(Y, L, W, Y_regression, L_regression,
                           method, L_cond_var_regression, folds, random_state)
    res = correct_nonzero(res, selected_indices)
    return res


def cke(Y, Z, j, Y_regression, L_classifier, X=None, method="partially_linear", folds=2, random_state=None):
    if not np.allclose(Z.sum(axis=1), 1):
        raise ValueError("Z is not in the simplex.")
    L = Z[:, j] == 0
    W = Z.copy()
    W[:, j] = 0
    W = W/(1-Z[:, j])[:, np.newaxis]

    if X is not None:
        W = np.c_[W, X]

    if method == "partially_linear":
        res = semi_est.partially_linear_model(
            Y, L, W, Y_regression, L_classifier, folds=folds, random_state=random_state)
    elif method == "nonparametric":
        res = semi_est.average_predictive_effect(
            Y, L, W, Y_regression, L_classifier, folds=folds, random_state=random_state)
    else:
        raise ValueError(
            "method arguments needs to be either `nonparametric` or `partially_linear`.")

    return res
