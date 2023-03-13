# spatial-casadi

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This package implements various data structures and helper methods for manipulating spatial transformations using [CasADi](https://web.casadi.org/) variables in Python.
The library interface is partially based on the [Scipy spatial module](https://docs.scipy.org/doc/scipy/reference/spatial.html).

# Install

## From source

In a new terminal:
1. Clone repository:
   - (ssh) `$ git clone git@github.com:cmower/spatial-casadi.git`, or
   - (https) `$ git clone https://github.com/cmower/spatial-casadi.git`
2. Change directory: `$ cd spatial-casadi`
3. Ensure `pip` is up-to-date: `$ python -m pip install --upgrade pip`
3. Install from source:
   - (main library) `$ pip install .`
   - (when you want to also run the test scripts) `$ pip install .[test]`

# Build documentation

In a new terminal:
1. Clone repository:
   - (ssh) `$ git clone git@github.com:cmower/spatial-casadi.git`, or
   - (https) `$ git clone https://github.com/cmower/spatial-casadi.git`
2. Change directory: `$ cd spatial-casadi/doc`
3. Install doxygen: `$ sudo apt install doxygen`
4. Build documentation: `$ doxygen`
5. View documentation:
   - In a browser, open `html/index.html`
   - Build pdf (requires LaTeX)
	 - `$ cd latex`
	 - `$ make`
	 - Open the file called `refman.pdf`

# Citing

If you use `spatial_casadi` in your work, please consider citing the following.

```bibtex
@software{Mower2023
  title="Spatial CasADi: A Compact Library for Manipulating Spatial Transformations",
  author = "Christopher E. Mower",
  year="2023",
  url = {https://github.com/cmower/spatial-casadi},
}
```

# Contributing

If you have any issues with the library, or find inaccuracies in the documentation please [raise an issue](https://github.com/cmower/spatial-casadi/issues/new/choose).
I am happy to consider new features if you [fork the library](https://github.com/cmower/spatial-casadi/fork) and submit a pull request.
