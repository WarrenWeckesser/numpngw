from setuptools import setup
from os import path


def get_numpngw_version():
    """
    Find the value assigned to __version__ in numpngw.py.

    This function assumes that there is a line of the form

        __version__ = "version-string"

    in numpngw.py.  It returns the string version-string, or None if such a
    line is not found.
    """
    with open("numpngw.py", "r") as f:
        for line in f:
            s = [w.strip() for w in line.split("=", 1)]
            if len(s) == 2 and s[0] == "__version__":
                return s[1][1:-1]


# Get the long description from README.rst.
_here = path.abspath(path.dirname(__file__))
with open(path.join(_here, 'README.rst')) as f:
    _long_description = f.read()

setup(
    name='numpngw',
    version=get_numpngw_version(),
    author='Warren Weckesser',
    description="Write numpy array(s) to a PNG or animated PNG file.",
    long_description=_long_description,
    license="BSD",
    url="https://github.com/WarrenWeckesser/numpngw",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    py_modules=["numpngw"],
    install_requires=[
        'numpy >= 1.6.0',
    ],
    keywords="numpy png matplotlib animation",
)
