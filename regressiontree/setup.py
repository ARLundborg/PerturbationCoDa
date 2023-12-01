# Constructed based on sklearn building settings
from setuptools import setup
import numpy
import scipy
from Cython.Build import cythonize
import os

cython_enable_debug_directives = (
    os.environ.get("SKLEARN_ENABLE_DEBUG_CYTHON_DIRECTIVES", "0") != "0"
)

compiler_directives = {
    "language_level": 3,
    "boundscheck": cython_enable_debug_directives,
    "wraparound": False,
    "initializedcheck": False,
    "nonecheck": False,
    "cdivision": True,
}

setup(
    name='RegressionTree',
    ext_modules=cythonize(["regressiontree/_tree.pyx"],
                          language="c++",
                          compiler_directives=compiler_directives,
                          language_level=3),
    include_dirs=[numpy.get_include()],
)

setup(
    name='RegressionTree',
    ext_modules=cythonize(["regressiontree/_splitter.pyx",
                           "regressiontree/_criterion.pyx",
                           "regressiontree/_random.pyx",
                           "regressiontree/_quad_tree.pyx",
                           "regressiontree/_utils.pyx"],
                          compiler_directives=compiler_directives,
                          language_level=3),
    include_dirs=[numpy.get_include(), scipy.get_include()],
)
