from setuptools import setup, Extension
import pybind11, sys
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "_lcs_ext",
        ["lcs.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3", "-march=native", "-fopenmp"],
        extra_link_args=["-lgomp"],
    ),
]

setup(
    name="lcs_ext",
    version="0.1",
    author="jekim",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
