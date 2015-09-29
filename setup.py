from setuptools import setup


def get_pngw_version():
    """
    Find the value assigned to __version__ in pngw.py.

    This function assumes that there is a line of the form

        __version__ = "version-string"

    in pngw.py.  It returns the string version-string, or None if such a
    line is not found.
    """
    with open("pngw.py", "r") as f:
        for line in f:
            s = [w.strip() for w in line.split("=", 1)]
            if len(s) == 2 and s[0] == "__version__":
                return s[1][1:-1]


setup(
    name='pngw',
    version=get_pngw_version(),
    author='Warren Weckesser',
    description=("Write a numpy array to a PNG file."),
    license="BSD",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    py_modules=["pngw"],
    install_requires=[
        'numpy >= 1.6.0',
    ],
)
