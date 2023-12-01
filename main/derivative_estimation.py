import numpy as np
from sklearn.model_selection import KFold
from regressiontree import DecisionTreeRegressor
from matplotlib import pyplot as plt


# Function that performs penalized local polynomial fit
def penalized_locpol(fval, v, X, W, train_ind, degree,
                     pen=0, penalize_intercept=False):
    v = v.reshape(-1, 1)
    n = X.shape[0]
    fval = fval.reshape(n, 1)
    if train_ind is None:
        train_ind = np.arange(n)
    if penalize_intercept:
        dstart = 0
    else:
        dstart = 1
    dd = degree + 1
    Dmat = np.zeros((n*dd, n*dd))
    YDmat = np.zeros((n*dd, 1))
    Pmats = [np.zeros((n, n*dd)) for kk in range(dstart, dd)]
    fval = fval[train_ind, :]
    for i in range(n):
        Wi = W[train_ind, i].reshape(-1, 1)
        for kk in range(dstart, dd):
            Pmats[kk-dstart][i, i*dd + kk] = 1
        x0 = X[i, :]
        inner_prod = (X[train_ind, :] - x0[None, :]).dot(v)
        Xmat = np.tile(inner_prod, dd)**np.arange(dd) * Wi
        Dmat[(i*dd):((i+1)*dd), (i*dd):((i+1)*dd)] = (Xmat.T).dot(Xmat)
        Ytilde = fval*Wi
        YDmat[(i*dd):((i+1)*dd), :] = (Xmat.T).dot(Ytilde)
    penmat = np.zeros(Dmat.shape)
    # Adjust the smoothing penalty according to train_ind
    Wtmp = W[:, train_ind].dot(np.eye(n)[train_ind])
    for kk in range(dstart, dd):
        # PP = (np.eye(n) - W).dot(Pmats[kk-dstart])
        PP = (np.eye(n) - Wtmp).dot(Pmats[kk-dstart])
        penmat += pen*(np.math.factorial(kk)**2)*((PP.T).dot(PP))
    B = np.linalg.solve(Dmat + penmat, YDmat)
    coefs = B.reshape(n, -1)
    # Extract derivatives from coefficients
    deriv_mat = coefs*np.array([np.math.factorial(k)
                                for k in range(degree+1)])
    return (deriv_mat)


# Auxiliary function to compute weights (this does the honest
# splitting and fitting)
def compute_rf_weights_full(X, Y, Xeval, tree, num_trees=100, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    elif random_state is int:
        random_state = np.random.RandomState(random_state)
    n, d = X.shape
    nn = 0
    if Xeval is not None:
        nn, _ = Xeval.shape
        X = np.r_[X, Xeval]
    weight_mat = np.zeros((n+nn, n+nn))
    s = 0.5
    bn = int(n * s)

    for k in range(num_trees):
        # Draw boostrap sample
        boot_sample = random_state.choice(np.arange(n),
                                          bn, replace=False)
        split1 = boot_sample[:int(bn/2)]
        split2 = np.concatenate([boot_sample[int(bn/2):],
                                 np.arange(nn)+n])
        # Fit tree
        tree.fit(X[split1, :], Y[split1])
        # Extract weights
        weight_mat += compute_tree_weights(
            X, split2, tree)/num_trees
    # Adjust weights to account for subsampling effects
    for ii in range(n):
        weight_mat[ii, ii] *= (s/2)
    weight_mat *= 2/s

    return weight_mat


# Auxiliary function used to compute fits (extracts weights from a tree)
def compute_tree_weights(X, test_ind, tree):
    n, d = X.shape
    weights = np.zeros((n, n))
    # Extract weights
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    # Compute leave nodes by traversing the tree once
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        # If left and right child of a node is not the same split node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and
        # depth to `stack` so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    leaf_nodes = tree.decision_path(X)[:, np.where(is_leaves)[0]]
    for ll in range(leaf_nodes.shape[1]):
        samples_node_ll = leaf_nodes[:, ll].nonzero()[0]
        samples_node_ll2 = [kk for kk in samples_node_ll if kk in test_ind]
        if len(samples_node_ll2) > 0:
            weight = 1/len(samples_node_ll2)
            weights[np.ix_(samples_node_ll2,
                           samples_node_ll2)] += weight
    return weights


# Function that outputs a list of weight matrices for different
# min_impurity_decreases
def rf_weights(X, Y,
               DecisionTreeRegressor,
               Xeval=None,
               min_impurity_decrease=0,
               min_mrss_grid=[0], min_samples_leaf_factor=0.1,
               num_trees=1000,
               criterion="local_linear_grad", random_state=None):
    weight_list = []
    n = len(Y)
    for mrss in min_mrss_grid:
        tree = DecisionTreeRegressor(
            criterion=criterion,
            min_impurity_decrease=min_impurity_decrease,
            min_mrss=mrss, random_state=random_state, min_samples_leaf=int(min_samples_leaf_factor*n))
        weight_list.append(compute_rf_weights_full(
            X, Y, Xeval, tree, num_trees=num_trees, random_state=random_state))
    return weight_list


# Function to performs cross-validation
def weight_pen_cv(X, Y, fval, v,
                  weight_list,
                  pen_grid,
                  folds=5,
                  verbose=0, random_state=None):
    kf = KFold(n_splits=folds, random_state=random_state)
    scores = np.zeros((kf.get_n_splits(X),
                       len(weight_list),
                       len(pen_grid)))
    for ll, pen in enumerate(pen_grid):
        if verbose > 0:
            print(ll)
        for kk, W in enumerate(weight_list):
            for ii, (train_ind, test_ind) in enumerate(kf.split(X)):
                deriv_mat = penalized_locpol(
                    fval, v, X, W, train_ind,
                    degree=3, pen=pen, penalize_intercept=False)
                scores[ii, kk, ll] = np.mean(
                    (Y[test_ind] - deriv_mat[test_ind, 0])**2)

    avg_scores = np.mean(scores, axis=0)
    std_scores = np.std(scores, axis=0)
    return avg_scores, std_scores


def derivative_estimate(X, Y, fhat, d=0, degree=2, train_ind=None, pen_grid=np.array([0]), min_mrss_grid=np.array([0]), random_state=None):

    if min_mrss_grid is None:
        min_mrss_grid = [np.mean((Y[train_ind]-fhat[train_ind])**2)]
    v = np.eye(1, np.shape(X)[1], d)

    # The swapped version is required because rf_weights with firstonly requires the first coordinate to be the one we're computing the influence of

    X_swapped = X.copy()
    X_swapped[:, [d, 0]] = X_swapped[:, [d, 0]]

    weight_list = rf_weights(X_swapped, Y,
                             DecisionTreeRegressor,
                             min_mrss_grid=min_mrss_grid,
                             num_trees=1000,
                             criterion="local_linear_grad", random_state=random_state)

    if len(pen_grid) > 1 or len(min_mrss_grid) > 1:

        avg_scores, std_scores = weight_pen_cv(X, Y, fhat, v,
                                               weight_list,
                                               pen_grid,
                                               folds=5, random_state=random_state)
        ind1 = np.where(
            np.min(avg_scores[:, 0] + std_scores[:, 0]) > avg_scores[:, 0])[0][-1]
        ind2 = np.argmin(avg_scores[ind1, :])
        ind = [ind1, ind2]
    else:
        ind = [0, 0]

    deriv_mat = penalized_locpol(
        fhat, v, X, weight_list[ind[0]], train_ind=train_ind,
        degree=degree, pen=pen_grid[ind[1]], penalize_intercept=False)

    return deriv_mat


##
# Minimal working usage example
##
if __name__ == "main":
    import matplotlib.pyplot as plt
    def f(x): return np.sin(x)

    n = 1000
    X = np.random.uniform(0, 1, n).reshape(n, 1)
    Y = f(X) + np.random.uniform(0, 1, n).reshape(n, 1)
    fhat = f(X) + np.random.uniform(0, 0.05, n).reshape(n, 1)

    deriv_mat = derivative_estimate(X, Y, fhat)
    plt.plot(np.sort(X[:, 0]), f(np.sort(X[:, 0])), c="red")
    plt.scatter(X, fhat, c="orange")
    plt.scatter(X, deriv_mat[:, 0], c="blue")
    plt.show()

    plt.scatter(X, deriv_mat[:, 1], c="blue")
    plt.plot(np.sort(X[:, 0]), np.cos(np.sort(X[:, 0])), c="red")
    plt.show()
