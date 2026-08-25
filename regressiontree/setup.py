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

# A single setup() call: calling it twice breaks the PEP 517 wheel build, which
# is now the default for both `pip install .` and `pip install -e .`.  The first
# call finalizes and removes build/bdist.*/wheel, so the second fails copying its
# extensions into a directory that no longer exists.  _tree.pyx declares
# `# distutils: language=c++` itself, which is also what Cython recommends over
# passing language="c++" to cythonize().
setup(
    name='RegressionTree',
    ext_modules=cythonize(["regressiontree/_tree.pyx",
                           "regressiontree/_splitter.pyx",
                           "regressiontree/_criterion.pyx",
                           "regressiontree/_random.pyx",
                           "regressiontree/_quad_tree.pyx",
                           "regressiontree/_utils.pyx"],
                          compiler_directives=compiler_directives,
                          language_level=3),
    include_dirs=[numpy.get_include(), scipy.get_include()],
)
