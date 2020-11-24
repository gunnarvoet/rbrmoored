from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

setup(
    # Project information
    name="rbrmoored",
    version="0.1.0",
    author="Gunnar Voet",
    author_email="gvoet@ucsd.edu",
    url="https://github.com/gunnarvoet/rbrmoored",
    license="MIT License",
    # Description
    description="Data processing for moored RBR sensors",
    long_description=f"{readme}\n\n{history}",
    long_description_content_type="text/x-rst",
    # Requirements
    python_requires=">=3.6",
    install_requires=["numpy", "gsw", "scipy", "matplotlib", "pyrsktools"],
    extras_require={
        "test": ["pytest"],  # install these with: pip install mixsea[test]
    },
    # Packaging
    packages=find_packages(include=["rbrmoored", "rbrmoored.*"], exclude=["*.tests"]),
    package_data={"rbrmoored": ["tests/data/*.rsk"]},
    include_package_data=True,
    zip_safe=False,
    platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
    # Metadata
    # project_urls={"Documentation": "https://mixsea.readthedocs.io"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
