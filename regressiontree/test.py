import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor as TreeOriginal
from regressiontree import DecisionTreeRegressor
from sklearn import tree


n = 1000
# X = np.random.uniform(0, 1, n)
# noise = np.random.normal(0, 0.1, n)
# y = 2 * X * (X < 0.5) + (-2 * X + 2) * (X > 0.5) + noise
# # y = (10*(X-0.25)**2)*(X < 0.5)+(10*(X - 0.75)**2)*(X > 0.5) + noise
# X = X.reshape(-1, 1)


X = np.random.uniform(0, 1, n*2).reshape(-1, 2)
noise = np.random.normal(0, 0.01, n)
# y = 2 * X[:, 0] * (X[:, 0] < 0.5) + (-2 * X[:, 0] + 2) * (X[:, 0] > 0.5) + noise
y = X[:, 0] * (X[:, 1] > 0.5) + noise
# y = X[:, 1] * X[:, 0] + noise

# # tree_original = TreeOriginal(min_impurity_decrease=0.1)
# tree_new = DecisionTreeRegressor(criterion="local_linear_firstonly",
#                                  min_impurity_decrease=np.sum(y**2)/100)
# tree_original = DecisionTreeRegressor(criterion="local_linear",
#                                       min_impurity_decrease=np.sum(y**2)/100)

# tree_original.fit(X, y)
# # print(tree_original.predict(X))
# tree_new.fit(X, y)
# # print(tree_new.predict(X))
# plt.ion()
# plt.clf()
# plt.scatter(X, tree_new.predict(X))
# plt.scatter(X, tree_original.predict(X))
# plt.scatter(X, y)

##
# Understanding tree
##

clf = DecisionTreeRegressor(
    criterion="local_linear_grad",
    min_mrss=0.0)
clf.fit(X, y)
tree.plot_tree(clf)
plt.show()
plt.scatter(X[:, 1], y)
plt.scatter(X[:, 1], clf.predict(X))
plt.scatter(y, clf.predict(X))
plt.scatter(X[:, 1], y-clf.predict(X))
plt.scatter(X[:, 1], noise)
