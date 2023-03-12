from setuptools import setup

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

setup(
    name="spatial-casadi",
    description="",
    version="1.0.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cmower/spatial-casadi",
    project_urls={
        "Bug Tracker": "https://github.com/cmower/spatial-casadi/issues",
    },
    author="Christopher E. Mower",
    author_email="christopher.mower@kcl.ac.uk",
    license="Apache License, Version 2.0",
    packages=["spatial_casadi"],
    install_requires=[
        "numpy",
        "scipy",
        "casadi",
    ],
)
