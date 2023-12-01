from main.derivative_estimation import derivative_estimate
from main.spline_score import spline_score_cv
import numpy as np
from sklearn.model_selection import KFold
import classo

def partially_linear_model(Y, L, W, Y_on_W_regression, L_on_W_regression, folds=2, random_state=None):
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    kappas = np.zeros(kf.get_n_splits())
    Js = np.zeros(kf.get_n_splits())
    nu_1s = np.zeros(kf.get_n_splits())
    nu_2s = np.zeros(kf.get_n_splits())
    nu_3s = np.zeros(kf.get_n_splits())
    
    for i, (train_ind, test_ind) in enumerate(kf.split(Y)):
        
        L_on_W_regression.fit(W[train_ind, :], L[train_ind])
        Y_on_W_regression.fit(W[train_ind, :], Y[train_ind])

        if hasattr(L_on_W_regression, "predict_proba"):
            L_resid = L[test_ind] - L_on_W_regression.predict_proba(W[test_ind,:])[:, 1]
        else:
            L_resid = L[test_ind] - L_on_W_regression.predict(W[test_ind,:])

        if hasattr(Y_on_W_regression, "predict_proba"):
            Y_resid = Y[test_ind] - Y_on_W_regression.predict_proba(W[test_ind,:])[:, 1]
        else:
            Y_resid = Y[test_ind] - Y_on_W_regression.predict(W[test_ind,:])

        Js[i] = np.mean(L_resid**2)
        kappas[i] = np.mean(L_resid*Y_resid)
        nu_1s[i] = np.mean(L_resid**2*Y_resid**2)
        nu_2s[i] = np.mean(L_resid**4)
        nu_3s[i] = np.mean(L_resid**3*Y_resid)
    J = np.mean(Js)
    theta = np.mean(kappas)/J
    result = {"estimate": theta,
              "variance": (np.mean(nu_1s) + theta**2*np.mean(nu_2s) - 2*theta*np.mean(nu_3s))/J**2}
    result["standard_error"] = np.sqrt(result["variance"]/len(Y))
    return result


def partially_linear_model_no_crossfitting(Y, L, W, Y_on_W_regression, L_on_W_regression):
    L_on_W_regression.fit(W, L)
    Y_on_W_regression.fit(W, Y)

    if hasattr(L_on_W_regression, "predict_proba"):
        L_resid = L - L_on_W_regression.predict_proba(W)[:, 1]
    else:
        L_resid = L - L_on_W_regression.predict(W)

    if hasattr(Y_on_W_regression, "predict_proba"):
        Y_resid = Y - Y_on_W_regression.predict_proba(W)[:, 1]
    else:
        Y_resid = Y - Y_on_W_regression.predict(W)

    J = np.mean(L_resid**2)
    kappa = np.mean(L_resid*Y_resid)
    nu_1 = np.mean(L_resid**2*Y_resid**2)
    nu_2 = np.mean(L_resid**4)
    nu_3 = np.mean(L_resid**3*Y_resid)
    theta = kappa/J
    result = {"estimate": theta,
              "variance": (nu_1 + theta**2*nu_2 - 2*theta*nu_3)/J**2}
    result["standard_error"] = np.sqrt(result["variance"]/len(Y))
    return result

def average_partial_effect(Y, L, W, Y_on_LW_regression, L_on_W_regression,          
                           L_cond_var_W_regression, folds=2, random_state=None):

    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    kf_score = KFold(n_splits=2, shuffle=False)
    
    estimates = np.zeros(2*kf.get_n_splits())
    second_moments = np.zeros(2*kf.get_n_splits())

    LW = np.c_[L, W]
    n = 0
    for i, (train_ind, test_ind) in enumerate(kf.split(Y)):
        
        Y_on_LW_regression.fit(LW[train_ind, :], Y[train_ind])

        dummy_f_hat = np.zeros_like(Y)
        if hasattr(Y_on_LW_regression, "predict_proba"):
            dummy_f_hat[train_ind] = Y_on_LW_regression.predict_proba(LW[train_ind, :])[:, 1]
        else:
            dummy_f_hat[train_ind] = Y_on_LW_regression.predict(LW[train_ind, :])

        deriv_mat = derivative_estimate(LW, Y, dummy_f_hat, d=0, train_ind=train_ind, random_state=random_state)
        f_hat = deriv_mat[:, 0]
        df_hat = deriv_mat[:, 1]

        L_on_W_regression.fit(W[train_ind, :], L[train_ind])
        L_cond_var_W_regression.fit(W[train_ind, :], (L[train_ind] - L_on_W_regression.predict(W[train_ind, :]))**2)

        for j, (score_train, score_test) in enumerate(kf_score.split(test_ind)):
            score_train = test_ind[score_train]
            score_test = test_ind[score_test]

            sigma_train = np.sqrt(L_cond_var_W_regression.predict(W[score_train,:]))
            xi_train = (L[score_train]-L_on_W_regression.predict(W[score_train,:]))/sigma_train
            score_func = spline_score_cv(xi_train, rule="1se", random_state=random_state)
            score_func.extrapolate = False

            sigma_test = np.sqrt(L_cond_var_W_regression.predict(W[score_test,:]))
            xi_test = (L[score_test]-L_on_W_regression.predict(W[score_test,:]))/sigma_test

            influence_functions = -score_func(xi_test)/sigma_test*(Y[score_test] - f_hat[score_test])

            n += np.count_nonzero(~np.isnan(influence_functions))
            estimates[2*i+j] = np.nanmean(df_hat[score_test] + influence_functions)
            second_moments[2*i+j] = np.nanmean((df_hat[score_test] + influence_functions)**2)

    result = {"estimate": np.mean(estimates),
              "variance": np.mean(second_moments)-np.mean(estimates)**2,
              }
    result["standard_error"] = np.sqrt(result["variance"]/n)

    return result


def average_partial_effect_no_crossfitting(Y, L, W, Y_on_LW_regression, L_on_W_regression,          
                           L_cond_var_W_regression, random_state=None):


    LW = np.c_[L, W] 
    Y_on_LW_regression.fit(LW, Y)

    dummy_f_hat = np.zeros_like(Y)
    if hasattr(Y_on_LW_regression, "predict_proba"):
        dummy_f_hat = Y_on_LW_regression.predict_proba(LW)[:, 1]
    else:
        dummy_f_hat = Y_on_LW_regression.predict(LW)

    deriv_mat = derivative_estimate(LW, Y, dummy_f_hat, d=0, random_state=random_state)
    f_hat = deriv_mat[:, 0]
    df_hat = deriv_mat[:, 1]

    L_on_W_regression.fit(W, L)
    L_cond_var_W_regression.fit(W, (L - L_on_W_regression.predict(W))**2)

    sigma = np.sqrt(L_cond_var_W_regression.predict(W))
    xi = (L-L_on_W_regression.predict(W))/sigma

    score_func = spline_score_cv(xi, rule="1se", random_state=random_state)

    influence_functions = -score_func(xi)/sigma*(Y - f_hat)

    estimate = np.mean(df_hat + influence_functions)
    second_moment = np.mean((df_hat + influence_functions)**2)

    result = {"estimate": estimate,
              "variance": second_moment-estimate**2,
              }
    result["standard_error"] = np.sqrt(result["variance"]/len(Y))

    return result

def average_partial_effect_true_score(Y, L, W, Y_on_LW_regression, true_score, folds=2, random_state=None):
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)

    estimates = np.zeros(kf.get_n_splits())
    second_moments = np.zeros(kf.get_n_splits())
    
    LW = np.c_[L, W]

    for i, (train_ind, test_ind) in enumerate(kf.split(Y)):
        
        Y_on_LW_regression.fit(LW[train_ind, :], Y[train_ind])

        dummy_f_hat = np.zeros_like(Y)
        if hasattr(Y_on_LW_regression, "predict_proba"):
            dummy_f_hat[train_ind] = Y_on_LW_regression.predict_proba(LW[train_ind, :])[:, 1]
        else:
            dummy_f_hat[train_ind] = Y_on_LW_regression.predict(LW[train_ind, :])
        deriv_mat = derivative_estimate(LW, Y, dummy_f_hat, d=0, train_ind=train_ind, random_state=random_state)
        f_hat = deriv_mat[:, 0]
        df_hat = deriv_mat[:, 1]

        estimates[i] = np.mean(df_hat[test_ind] - true_score(L[test_ind], W[test_ind,:])*(Y[test_ind] - f_hat[test_ind]))
        second_moments[i] = np.mean((df_hat[test_ind] - true_score(L[test_ind], W[test_ind,:])*(Y[test_ind] - f_hat[test_ind]))**2)

    result = {"estimate": np.mean(estimates),
              "variance": np.mean(second_moments)-np.mean(estimates)**2,
              }
    result["standard_error"] = np.sqrt(result["variance"]/len(Y))

    return result



def average_partial_effect_plugin(Y, L, W, Y_on_LW_regression, folds=2, random_state=None):

    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)

    estimates = np.zeros(kf.get_n_splits())
    second_moments = np.zeros(kf.get_n_splits())
    
    LW = np.c_[L, W]

    for i, (train_ind, test_ind) in enumerate(kf.split(Y)):
        
        Y_on_LW_regression.fit(LW[train_ind, :], Y[train_ind])

        dummy_f_hat = np.zeros_like(Y)
        if hasattr(Y_on_LW_regression, "predict_proba"):
            dummy_f_hat[train_ind] = Y_on_LW_regression.predict_proba(LW[train_ind, :])[:, 1]
        else:
            dummy_f_hat[train_ind] = Y_on_LW_regression.predict(LW[train_ind, :])
        deriv_mat = derivative_estimate(LW, Y, dummy_f_hat, d=0, train_ind=train_ind, random_state=random_state)
        df_hat = deriv_mat[:, 1]

        estimates[i] = np.mean(df_hat[test_ind])
        second_moments[i] = np.mean((df_hat[test_ind])**2)

    result = {"estimate": np.mean(estimates),
              "variance": np.mean(second_moments)-np.mean(estimates)**2,
              }
    result["standard_error"] = np.sqrt(result["variance"]/len(Y))
    return result


def average_partial_effect_plugin_no_crossfitting(Y, L, W, Y_on_LW_regression, random_state=None): 
    LW = np.c_[L, W]
        
    Y_on_LW_regression.fit(LW, Y)

    if hasattr(Y_on_LW_regression, "predict_proba"):
        dummy_f_hat = Y_on_LW_regression.predict_proba(LW)[:, 1]
    else:
        dummy_f_hat = Y_on_LW_regression.predict(LW)
    deriv_mat = derivative_estimate(LW, Y, dummy_f_hat, d=0, random_state=random_state)
    df_hat = deriv_mat[:, 1]

    estimate = np.mean(df_hat)
    second_moment = np.mean((df_hat)**2)

    result = {"estimate": estimate,
              "variance": second_moment - estimate**2,
              }
    result["standard_error"] = np.sqrt(result["variance"]/len(Y))
    return result

def average_predictive_effect(Y, L, W, Y_on_LW_regression, L_on_W_classifier, propensity_threshold=1e-4, folds=2, random_state=None):
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    estimates = np.zeros(kf.get_n_splits())
    second_moments = np.zeros(kf.get_n_splits())

    LW = np.c_[L, W]
    oneW = LW.copy()
    oneW[:, 0] = 1
    for i, (train_ind, test_ind) in enumerate(kf.split(Y)):
        Y_on_LW_regression.fit(LW[train_ind, :], Y[train_ind])

        if hasattr(Y_on_LW_regression, "predict_proba"):
            f_hat1 = Y_on_LW_regression.predict_proba(oneW[test_ind, :])[:, 1]
            f_hat = Y_on_LW_regression.predict_proba(LW[test_ind, :])[:, 1]
        else:
            f_hat1 = Y_on_LW_regression.predict(oneW[test_ind, :])
            f_hat = Y_on_LW_regression.predict(LW[test_ind, :])

        L_on_W_classifier.fit(W[train_ind, :], L[train_ind])
        pi_hat = L_on_W_classifier.predict_proba(W[test_ind, :])[:, 1] 
        pi_hat = np.clip(pi_hat, propensity_threshold, 1-propensity_threshold) #bound away from 0 and 1


        estimates[i] =  np.mean(f_hat1-Y[test_ind] + (Y[test_ind]-f_hat)*L[test_ind]/pi_hat)
        second_moments[i] = np.mean((f_hat1-Y[test_ind] + (Y[test_ind]-f_hat)*L[test_ind]/pi_hat)**2)
    p_hat = np.mean(L==0)
    result = {"estimate": np.mean(estimates)/p_hat,
              "variance": (np.mean(second_moments)-np.mean(estimates)**2)/p_hat**2 -  np.mean(estimates)**2*(1-p_hat)/p_hat**3,
              }
    result["standard_error"] = np.sqrt(result["variance"]/len(Y))
    return result



def average_predictive_effect_no_crossfitting(Y, L, W, Y_on_LW_regression, L_on_W_classifier, 
                                              propensity_threshold=1e-4):
    LW = np.c_[L, W]
    oneW = LW.copy()
    oneW[:, 0] = 1

    Y_on_LW_regression.fit(LW, Y)

    if hasattr(Y_on_LW_regression, "predict_proba"):
        f_hat1 = Y_on_LW_regression.predict_proba(oneW)[:, 1]
        f_hat = Y_on_LW_regression.predict_proba(LW)[:, 1]
    else:
        f_hat1 = Y_on_LW_regression.predict(oneW)
        f_hat = Y_on_LW_regression.predict(LW)

    L_on_W_classifier.fit(W, L)
    pi_hat = L_on_W_classifier.predict_proba(W)[:, 1] 
    pi_hat = np.clip(pi_hat, propensity_threshold, 1-propensity_threshold) #bound away from 0 and 1


    estimate =  np.mean(f_hat1-Y + (Y-f_hat)*L/pi_hat)
    second_moment = np.mean((f_hat1-Y + (Y-f_hat)*L/pi_hat)**2)
    p_hat = np.mean(L==0)

    result = {"estimate": estimate/p_hat,
              "variance": (second_moment-estimate**2)/p_hat**2 -  estimate**2*(1-p_hat)/p_hat**3,
              }
    result["standard_error"] = np.sqrt(result["variance"]/len(Y))
    return result

def average_predictive_effect_true_score(Y, L, W, Y_on_LW_regression, true_score, folds=2,
                              random_state=None):
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    estimates = np.zeros(kf.get_n_splits())
    second_moments = np.zeros(kf.get_n_splits())

    LW = np.c_[L, W]
    oneW = LW.copy()
    oneW[:, 0] = 1
    for i, (train_ind, test_ind) in enumerate(kf.split(Y)):
        Y_on_LW_regression.fit(LW[train_ind, :], Y[train_ind])
        
        if hasattr(Y_on_LW_regression, "predict_proba"):
            f_hat1 = Y_on_LW_regression.predict_proba(oneW[test_ind, :])[:, 1]
            f_hat = Y_on_LW_regression.predict_proba(LW[test_ind, :])[:, 1]
        else:
            f_hat1 = Y_on_LW_regression.predict(oneW[test_ind, :])
            f_hat = Y_on_LW_regression.predict(LW[test_ind, :])

        estimates[i] =  np.mean(f_hat1-Y[test_ind] + (Y[test_ind]-f_hat)*L[test_ind]/true_score(W[test_ind]))
        second_moments[i] = np.mean((f_hat1-Y[test_ind] + (Y[test_ind]-f_hat)*L[test_ind]/true_score(W[test_ind]))**2)

    p_hat = np.mean(L==0)
    result = {"estimate": np.mean(estimates)/p_hat,
              "variance": (np.mean(second_moments)-np.mean(estimates)**2)/p_hat**2 -  np.mean(estimates)**2*(1-p_hat)/p_hat**3,
              }
    result["standard_error"] = np.sqrt(result["variance"]/len(Y))
    return result

def average_predictive_effect_plugin(Y, L, W, Y_on_LW_regression, folds=2, random_state=None):
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    estimates = np.zeros(kf.get_n_splits())
    second_moments = np.zeros(kf.get_n_splits())

    LW = np.c_[L, W]
    oneW = LW.copy()
    oneW[:, 0] = 1
    for i, (train_ind, test_ind) in enumerate(kf.split(Y)):
        Y_on_LW_regression.fit(LW[train_ind, :], Y[train_ind])
        
        if hasattr(Y_on_LW_regression, "predict_proba"):
            f_hat1 = Y_on_LW_regression.predict_proba(oneW[test_ind, :])[:, 1]
        else:
            f_hat1 = Y_on_LW_regression.predict(oneW[test_ind, :])

        estimates[i] =  np.mean(f_hat1-Y[test_ind])
        second_moments[i] = np.mean((f_hat1-Y[test_ind])**2)

    p_hat = np.mean(L==0)
    result = {"estimate": np.mean(estimates)/p_hat,
              "variance": (np.mean(second_moments)-np.mean(estimates)**2)/p_hat**2 -  np.mean(estimates)**2*(1-p_hat)/p_hat**3,
              }
    result["standard_error"] = np.sqrt(result["variance"]/len(Y))
    return result

def average_predictive_effect_plugin_no_crossfitting(Y, L, W, Y_on_LW_regression):

    LW = np.c_[L, W]
    oneW = LW.copy()
    oneW[:, 0] = 1

    Y_on_LW_regression.fit(LW, Y)
    
    if hasattr(Y_on_LW_regression, "predict_proba"):
        f_hat1 = Y_on_LW_regression.predict_proba(oneW)[:, 1]
    else:
        f_hat1 = Y_on_LW_regression.predict(oneW)

    p_hat = np.mean(L==0)
    estimate =  np.mean(f_hat1-Y)
    second_moment = np.mean((f_hat1-Y)**2)

    result = {"estimate": estimate/p_hat,
              "variance": (second_moment-estimate**2)/p_hat**2 -  estimate**2*(1-p_hat)/p_hat**3,
              }
    result["standard_error"] = np.sqrt(result["variance"]/len(Y))
    return result



def nonparametric_r2(Y, L, W, Y_on_LW_regression, Y_on_W_regression, folds=2, random_state=None):
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    psis = np.zeros(kf.get_n_splits())

    for i, (train_ind, test_ind) in enumerate(kf.split(Y)):
        
        Y_on_LW_regression.fit(np.c_[L[train_ind], W[train_ind, :]], Y[train_ind])
        Y_on_W_regression.fit(W[train_ind, :], Y[train_ind])

        if hasattr(Y_on_LW_regression, "predict_proba"):
            Y_on_LW_resid = Y[test_ind] - Y_on_LW_regression.predict_proba(np.c_[L[test_ind], W[test_ind, :]])[:, 1]
            Y_on_W_resid = Y[test_ind] - Y_on_W_regression.predict_proba(W[test_ind, :])[:, 1]
        else:
            Y_on_LW_resid = Y[test_ind] - Y_on_LW_regression.predict(np.c_[L[test_ind], W[test_ind, :]])
            Y_on_W_resid = Y[test_ind] - Y_on_W_regression.predict(W[test_ind, :])

        psis[i] = (np.sum(Y_on_W_resid**2)-np.sum(Y_on_LW_resid**2))/np.sum(( Y[test_ind] - np.mean( Y[test_ind]))**2)
    result = {"estimate": np.mean(psis),
              "variance": np.nan}
    result["standard_error"] = np.nan
    return result


def linear_model(X, Y):
    n = X.shape[0]
    d = X.shape[1]

    X = np.c_[np.ones(n), X]

    D = np.linalg.inv(X.T.dot(X)/n)
    beta = D.dot(X.T.dot(Y)/n).reshape(d+1, 1)

    epsilon = Y - X.dot(beta.reshape(d+1, ))
    sigma_hat_2 = epsilon.dot(epsilon)/(n-d)
    vcov_mat = sigma_hat_2*D

    return {"beta": beta.reshape(d+1, ), "vcov_mat": vcov_mat}

def log_contrast(Z, Y):
    """Using restricted least squares (RLS) theory"""
    X = np.log(Z)
    d = X.shape[1]
    n = X.shape[0]
    L = np.ones((1, d))
    r = np.zeros((1, 1))

    D = np.linalg.inv(X.T.dot(X)/n)
    beta_OLS = D.dot(X.T.dot(Y)/n).reshape(d, 1)

    LDL_inv = np.linalg.inv(L.dot(D.dot(L.T)))

    beta_RLS = beta_OLS - D.dot(L.T).dot(LDL_inv).dot(L.dot(beta_OLS)- r)

    epsilon = Y - X.dot(beta_RLS.reshape(d, ))
    sigma_hat_2 = epsilon.dot(epsilon)/(n-d)
    vcov_mat = sigma_hat_2*(D - D.dot(L.T).dot(LDL_inv).dot(L).dot(D))

    return {"beta": beta_RLS.reshape(d, ), "vcov_mat": vcov_mat}


def sparse_log_contrast(Z, Y, pseudo_count, refit=True, seed=None):
    if pseudo_count > 0:
        X = np.log(Z + pseudo_count)
    else:
        raise ValueError("pseudo_count must be positive.")
   
    problem = classo.classo_problem(X, Y)
    problem.formulation.intercept = True
    problem.formulation.concomitant = False
    problem.model_selection.StabSel = False
    problem.model_selection.PATH = True
    problem.model_selection.CV = True
    problem.formulation.classification = False
    problem.model_selection.CVparameters.seed = seed  # 6
    problem.model_selection.CVparameters.oneSE = False
    problem.solve()
    if refit:
        alpha = problem.solution.CV.refit
    else:
        alpha = problem.solution.CV.beta
    return {"beta": alpha[1:], "score":np.min(problem.solution.CV.yGraph)}

def log_contrast_effect(Y, Z_j, Z_rest):
    Z = np.c_[Z_j, Z_rest]
    log_contrast_fit = log_contrast(Z, Y)
    return {"estimate": log_contrast_fit["beta"][0],
            "variance": log_contrast_fit["vcov_mat"][0, 0],
            "standard_error": np.sqrt(log_contrast_fit["vcov_mat"][0, 0]/len(Y))}

def linear_effect(Y, X_j, X_rest):
    X = np.c_[X_j, X_rest]
    linear_fit = linear_model(X, Y)
    return {"estimate": linear_fit["beta"][1],
            "variance": linear_fit["vcov_mat"][1, 1],
            "standard_error": np.sqrt(linear_fit["vcov_mat"][1, 1]/len(Y))}
