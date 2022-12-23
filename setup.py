from setuptools import setup
import pkgconfig
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "zmsh.volumes",
        sources=["src/zmsh/volumes.cpp"],
        extra_compile_args=[pkgconfig.cflags("eigen3")],
        cxx_std=20,
    )
]

setup(
    name="zmsh",
    packages=["zmsh"],
    package_dir={"": "src"},
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
)
