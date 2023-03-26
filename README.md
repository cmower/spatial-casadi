<p align="center">
  <img src="https://raw.githubusercontent.com/cmower/spatial-casadi/master/doc/image/spatial-casadi.png" width="60" align="right">
</p>

# spatial-casadi

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This package implements various data structures and helper methods for manipulating spatial transformations using [CasADi](https://web.casadi.org/) variables in Python.
The library interface is partially based on the [Scipy spatial module](https://docs.scipy.org/doc/scipy/reference/spatial.html).


- Code: [https://github.com/cmower/spatial-casadi](https://github.com/cmower/spatial-casadi)
- Documentation: [https://cmower.github.io/spatial-casadi/](https://cmower.github.io/spatial-casadi/)
- PyPI: [https://pypi.org/project/spatial-casadi](https://pypi.org/project/spatial-casadi/)
- Issues: [https://github.com/cmower/spatial-casadi/issues](https://github.com/cmower/spatial-casadi/issues)

# Examples

There are three main data structures: [Rotation](https://cmower.github.io/spatial-casadi/classspatial__casadi_1_1spatial_1_1Rotation.html), [Translation](https://cmower.github.io/spatial-casadi/classspatial__casadi_1_1spatial_1_1Translation.html), and [Transformation](https://cmower.github.io/spatial-casadi/classspatial__casadi_1_1spatial_1_1Transformation.html).
The following showcases some of the main functionality of the library.

```
>>> import spatial_casadi as sc
>>> import casadi as cs
>>> cs.np.random.seed(10)
>>> euler = cs.SX.sym("euler", 3)
>>> sc.Rotation.from_euler('xyz', euler).as_quat()
SX(@1=2, @2=cos((x_2/@1)), @3=cos((x_1/@1)), @4=sin((x_0/@1)), @5=(@3*@4), @6=sin((x_2/@1)), @7=cos((x_0/@1)), @8=sin((x_1/@1)), @9=(@7*@8), @10=(@3*@7), @11=(@8*@4), [((@2*@5)-(@6*@9)), ((@2*@9)+(@6*@5)), ((@10*@6)-(@2*@11)), ((@2*@10)+(@6*@11))])
>>> r = sc.Rotation.random()
>>> r.as_quat()
DM([0.615982, 0.330883, -0.71489, -0.0038783])
>>> r.as_rotvec()
DM([-1.9304, -1.03694, 2.24037])
>>> r.as_matrix()
DM(
[[-0.241103, 0.40209, -0.883285],
 [0.41318, -0.781003, -0.468312],
 [-0.878152, -0.477867, 0.0221665]])
>>> r.as_euler('xyz')
DM([-1.52444, 1.07199, 2.09902])
>>> r.as_mrp()
DM([-0.613602, -0.329604, 0.712128])
>>> sc.Rotation.from_euler('x', 90, degrees=True).as_matrix()
DM(
[[1, 0, 0],
 [0, 2.22045e-16, -1],
 [0, 1, 2.22045e-16]])
>> r1 = sc.Rotation.random()
>>> r1.as_quat()
DM([0.625459, -0.724863, 0.267273, 0.109269])
>> r2 = sc.Rotation.random()
>>> r2.as_quat()
DM([0.00332548, -0.1353, 0.335557, 0.932247])
>>> (r1 * r2).as_quat()
DM([0.376374, -0.899524, 0.203617, -0.0879736])
```

# Install

## From PyPI

```shell
$ pip install spatial-casadi
$ pip install spatial-casadi[test] # if you want to run the test scripts
```

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

# Running the test scripts

1. Install `spatial-casadi` from source and ensure you install the `test` packages (see previous section).
2. Change directory: `$ cd /path/to/spatial-casadi`
3. Run tests: `pytest`

# Build documentation

The documentation is hosted [here](https://cmower.github.io/spatial-casadi/).
However, if you want to build it yourself, then follow these steps.

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

If you use `spatial-casadi` in your work, please consider citing the following.

```bibtex
@software{Mower2023
  title = "Spatial CasADi: A Compact Library for Manipulating Spatial Transformations",
  author = "Christopher E. Mower",
  year = "2023",
  url = {https://github.com/cmower/spatial-casadi},
}
```

# Contributing

If you have any issues with the library, or find inaccuracies in the documentation please [raise an issue](https://github.com/cmower/spatial-casadi/issues/new/choose).
I am happy to consider new features if you [fork the library](https://github.com/cmower/spatial-casadi/fork) and submit a pull request.
