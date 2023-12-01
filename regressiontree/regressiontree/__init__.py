"""
This file has been modified since it was retrieved as part of
https://github.com/scikit-learn/scikit-learn/archive/0.23.2.tar.gz
License and copyright notice of the original file are retained below.

"""

"""
The :mod:`sklearn.tree` module includes decision tree-based models for
classification and regression.
"""

from ._classes import DecisionTreeRegressor
from sklearn.tree._export import export_graphviz, plot_tree, export_text

__all__ = ["DecisionTreeRegressor",
           "export_graphviz",
           "plot_tree", "export_text"]
