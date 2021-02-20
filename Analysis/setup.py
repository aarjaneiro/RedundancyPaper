from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("*", ["redundancy/*.pyx"],
              include_dirs=[numpy.get_include()], language_level='3')
]
setup(
    name="Redundancy Paper Helper Tools",
    ext_modules=cythonize(extensions),
)
