from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension("volumes", sources=["src/zmsh/volumes.cpp"], cxx_std=20)
]

setup(
    name="zmsh",
    packages=["zmsh"],
    package_dir={"": "src"},
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
)
