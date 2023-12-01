import pandas as pd
import numpy as np
import main.spline_score as spline_score
import main.derivative_estimation as derivative
from scipy.stats import beta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 16})
plt.rc('text.latex', preamble=r'\usepackage{amsfonts,amssymb,amsthm,amsmath}')
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


##### Continuous example #####

def normal_score(x, mu, sigma):
    return -(x-mu)/sigma**2

n = 1000
rng = np.random.RandomState(211123)

W = rng.dirichlet(np.ones(3), n)
L = W[:, 0] + (1+(W[:, 0] > 1/2))*rng.normal(0, 1, n)
true_score = lambda L, W: normal_score(L, W[:, 0], (1+(W[:, 0] > 1/2)))
Y = (L + 1)*W[:, 0] + rng.normal(0, 1, n)

W_te = rng.dirichlet(np.ones(3), n)
L_te = W_te[:, 0] + (1+(W_te[:, 0] > 1/2))*rng.normal(0, 1, n)
Y_te = (L_te + 1)*W_te[:, 0] + rng.normal(0, 1, n)

m_hat = RandomForestRegressor(random_state=rng, n_jobs=-1).fit(W, L).predict(W_te)
g_hat = RandomForestRegressor(random_state=rng, n_jobs=-1).fit(W, Y).predict(W_te)

f_hat = RandomForestRegressor(random_state=rng, n_jobs=-1).fit(np.c_[L, W], Y).predict(np.c_[L, W])
dummy_f_hat = np.zeros(2*n)
dummy_f_hat[0:n] = f_hat
deriv_mat = derivative.derivative_estimate(np.r_[np.c_[L, W], np.c_[L_te, W_te]], np.r_[Y, Y_te], dummy_f_hat, d=0, train_ind=np.arange(0, n), random_state=rng)
f_hat = deriv_mat[n:, 0]
df_hat = deriv_mat[n:, 1]

v_hat = RandomForestRegressor(random_state=rng, n_jobs=-1).fit(W, (L-m_hat)**2).predict(W_te)
xi = (L-m_hat)/np.sqrt(v_hat)

n_2 = 500
rho_hat = np.zeros(n)
score_func = spline_score.spline_score_cv(xi[n_2:n], rule="1se", random_state=rng)
score_func.extrapolate = False
rho_hat[0:n_2] = score_func(xi[0:n_2])
score_func = spline_score.spline_score_cv(xi[0:n_2], rule="1se", random_state=rng)
score_func.extrapolate = False
rho_hat[n_2:n] = score_func(xi[n_2:n])



fig, axn = plt.subplots(2, 2, figsize=(10, 6))
fig.subplots_adjust(hspace=0.5, top=0.85)

axn[0, 0].scatter(W_te[:, 0], g_hat, c="C3", alpha=0.3)
axn[0, 0].scatter(W_te[:, 0], W_te[:, 0]**2 + W_te[:, 0], c="C0", alpha=0.3)
axn[0, 0].set_xlabel(r"$W^1$")
axn[0, 0].set_title(r"Estimation of $g$")

axn[0, 1].scatter(W_te[:, 0], m_hat, c="C3", alpha=0.3)
axn[0, 1].scatter(W_te[:, 0], W_te[:, 0], c="C0", alpha=0.3)
axn[0, 1].set_xlabel(r"$W^1$")
axn[0, 1].set_title(r"Estimation of $m$")

axn[1, 0].scatter(L_te, df_hat, c="C3", alpha=0.3)
axn[1, 0].scatter(L_te, W_te[:, 0], c="C0", alpha=0.3)
axn[1, 0].set_xlabel(r"$L$")
axn[1, 0].set_title(r"Estimation of $\partial_{\ell} f$")


axn[1, 1].scatter(L_te, rho_hat, c="C3", alpha=0.3)
axn[1, 1].scatter(L_te, true_score(L_te, W_te), c="C0", alpha=0.3)
axn[1, 1].set_xlabel(r"$L$")
axn[1, 1].set_title(r"Estimation of $\rho$")


legend_elements = [
    Line2D([0], [0], marker="o", color="white", markerfacecolor="C0", markersize=10),
    Line2D([0], [0], marker="o", color="white", markerfacecolor="C3", markersize=10)]

fig.legend(legend_elements, ["True function", "Estimate"], loc="upper center", ncol=2)

fig.savefig("plots/semiparametric_continuous_robustness.pdf", bbox_inches="tight")


##### Binary example #####
expit = lambda x: 1/(1+np.exp(-x))
logit = lambda x: np.log(x/(1-x))

pi = lambda W: expit(logit(W[:, 0]))*0.95+0.05

W = rng.dirichlet(np.ones(3), n)
L = rng.binomial(1, pi(W), n)
Y = 2*L*W[:, 0] + rng.normal(0, 1, n)

W_te = rng.dirichlet(np.ones(3), n)
L_te = rng.binomial(1, pi(W_te), n)
Y_te = 2*L_te*W_te[:, 0] + rng.normal(0, 1, n)

m_hat = RandomForestClassifier(random_state=rng, n_jobs=-1).fit(W, L).predict_proba(W_te)[:,1]
g_hat = RandomForestRegressor(random_state=rng, n_jobs=-1).fit(W, Y).predict(W_te)
f_hat = RandomForestRegressor(random_state=rng, n_jobs=-1).fit(np.c_[L, W], Y).predict(np.c_[L_te, W_te])



fig, axn = plt.subplots(2, 2, figsize=(10, 6))
fig.subplots_adjust(hspace=0.5, top=0.85)

axn[0, 0].scatter(W_te[:, 0], g_hat, c="C3", alpha=0.3)
axn[0, 0].scatter(W_te[:, 0], 2*pi(W_te)*W_te[:, 0], c="C0", alpha=0.3)
axn[0, 0].set_xlabel(r"$W^1$")
axn[0, 0].set_title(r"Estimation of $g$")

axn[0, 1].scatter(W_te[:, 0], m_hat, c="C3", alpha=0.3)
axn[0, 1].scatter(W_te[:, 0], pi(W_te), c="C0", alpha=0.3)
axn[0, 1].set_xlabel(r"$W^1$")
axn[0, 1].set_title(r"Estimation of $m$")

axn[1, 0].scatter(W_te[:, 0], f_hat, c="C3", alpha=0.3)
axn[1, 0].scatter(W_te[:, 0], 2*L_te*W_te[:, 0], c="C0", alpha=0.3)
axn[1, 0].set_xlabel(r"$W^1$")
axn[1, 0].set_title(r"Estimation of $f$")

axn[1, 1].scatter(W_te[:, 0], 1/np.clip(m_hat, 1e-2, 1-1e-2), c="C3", alpha=0.3)
axn[1, 1].scatter(W_te[:, 0], 1/pi(W_te), c="C0", alpha=0.3)
axn[1, 1].set_xlabel(r"$W^1$")
axn[1, 1].set_title(r"Estimation of $\pi^{-1}$")

legend_elements = [
    Line2D([0], [0], marker="o", color="white", markerfacecolor="C0", markersize=10),
    Line2D([0], [0], marker="o", color="white", markerfacecolor="C3", markersize=10)]

fig.legend(legend_elements, ["True function", "Estimate"], loc="upper center", ncol=2)

fig.savefig("plots/semiparametric_binary_robustness.pdf", bbox_inches="tight")
