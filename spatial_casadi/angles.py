import casadi as cs

# The number pi (i.e. 3.141..)
pi = cs.np.pi

def deg2rad(x):
    """Convert degrees to radians.

    Args
      x (array-like) An array containing angles in degrees.

    Returns
      casadi-array
        An array containing angles in radians.

    """
    return (pi / 180.0) * cs.horzcat(x)


def rad2deg(x):
    """Convert radians to degrees.

    Args
      x (array-like) An array containing angles in radians.

    Returns
      casadi-array
        An array containing angles in degrees.
    """
    return (180.0 / pi) * cs.horzcat(x)
