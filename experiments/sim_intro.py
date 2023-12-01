import numpy as np
import main.semiparametric_estimators as semi_est 
import scipy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import main.perturbation_effects as pert

expit = lambda x: 1/(1+np.exp(-x))

seed = 31124
rng = np.random.RandomState(seed)

Y_on_W = RandomForestRegressor(n_jobs=-1, random_state=rng)
L_on_W = RandomForestRegressor(n_jobs=-1, random_state=rng)

##### Gini example #####

n = 1000
theta = 1
d = 3

V = rng.normal(0, 1, (n, d))
U = np.matmul(V, np.identity(d) - np.ones((d, 1)).dot(np.ones((1,d)))/d)
W = U/np.linalg.norm(U, axis=1, ord=1)[:, np.newaxis]
SAD_W = np.apply_along_axis(lambda x: np.abs(np.subtract.outer(x, x)).sum(), 1, W)
D = expit(W.dot(np.eye(1, d, 0).flatten()) + rng.normal(0, 1, n))
L = -(d-1)/d**2*D*SAD_W
Y = theta*L + 4*W.dot(np.eye(1, d, 0).flatten()) + rng.normal(0, 1, n)
Z = -W*D[:, np.newaxis] + np.ones(d)/d 


res_L = semi_est.linear_effect(Y, L, np.zeros((n, 0)))
print("Y on L: est = {:.2f}, CI = ({:.2f}, {:.2f})".format(res_L["estimate"], res_L["estimate"] - 1.96*res_L["standard_error"], res_L["estimate"] +1.96*res_L["standard_error"]))

res_LW = pert.cdi_gini(Y, Z, LinearRegression(), LinearRegression(), folds=10, random_state=rng)
print("CDI_Gini with linear regression: est = {:.2f}, CI = ({:.2f}, {:.2f})".format(res_LW["estimate"], res_LW["estimate"] - 1.96*res_LW["standard_error"], res_LW["estimate"] +1.96*res_LW["standard_error"]))

res_LW = pert.cdi_gini(Y, Z, Y_on_W, L_on_W, folds=10, random_state=rng)
print("CDI_Gini with RF: est = {:.2f}, CI = ({:.2f}, {:.2f})".format(res_LW["estimate"], res_LW["estimate"] - 1.96*res_LW["standard_error"], res_LW["estimate"] +1.96*res_LW["standard_error"]))



##### Binary example #####

L_on_W = RandomForestClassifier(n_jobs=-1, random_state=rng, criterion="log_loss")

n = 1000

L = rng.binomial(1, 0.5, n)
W1 = rng.uniform(0, 1, n)
W1[L == 1] = rng.binomial(1, 0.5, np.sum(L))*W1[L==1]
Y = rng.binomial(1, 0.75 + 1/8*L - 1/2*(W1==0))
Z1 = (1-L)*rng.uniform(0, 1, n)
Z = np.c_[Z1, W1*(1-Z1), (1-W1)*(1-Z1)]


T = Y[L==1].mean() - Y[L==0].mean()
#p_asymptotic = 1-scipy.stats.chi2.cdf(T, df=1)
B = 9999
T_perm = np.zeros(B)
for b in np.arange(B):
  Y_perm = rng.permutation(Y)
  T_perm[b] = Y_perm[L==1].mean() - Y_perm[L==0].mean()

p_permutation = (np.sum(T_perm <= T) + 1)/(B+1)


print("Permutation p-value is {}".format(p_permutation))

res_L = semi_est.linear_effect(Y, L, np.zeros((n, 0)))
print("Y on L: est = {:.2f}, CI = ({:.2f}, {:.2f})".format(res_L["estimate"], res_L["estimate"] - 1.96*res_L["standard_error"], res_L["estimate"] +1.96*res_L["standard_error"]))
print("Linear p-value is {}".format(2*scipy.stats.norm.cdf(-np.abs(res_L["estimate"]/res_L["standard_error"]))))

res_LW = pert.cke(Y, Z, 0, Y_on_W, L_on_W, folds=10, random_state=rng)
print("CKE^1 with RF: est = {:.2f}, CI = ({:.2f}, {:.2f})".format(res_LW["estimate"], res_LW["estimate"] - 1.96*res_LW["standard_error"], res_LW["estimate"] +1.96*res_LW["standard_error"]))
print("RF p-value is {}".format(2*scipy.stats.norm.cdf(-np.abs(res_LW["estimate"]/res_LW["standard_error"]))))
