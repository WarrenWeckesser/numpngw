from setuptools import setup


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

_long_description = """
This python package defines the function `write_png` that writes a
numpy array to a PNG file, and the function `write_apng` that writes
a sequence of arrays to an animated PNG file.  Also included is the
class `AnimatedPNGWriter` that can be used to save a Matplotlib
animation as an animated PNG file.
"""

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
        "Programming Language :: Python",
    ],
    py_modules=["numpngw"],
    install_requires=[
        'numpy >= 1.6.0',
    ],
)
