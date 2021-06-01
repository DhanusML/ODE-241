import numpy as np
from matplotlib import pyplot as plt
import System2d as sys2d


# Two positive eigen values 
# Unstable node
def lin_1(x, y):
    xx = 0.1 * x
    yy = 0.3 * y

    return xx, yy


# One positive and one negative eigen value
# Saddle point
def lin_2(x, y):
    xx = 0.1 * x
    yy = -0.3 * y

    return xx, yy


# Two negative eigen values
# Stable node
def lin_3(x, y):
    xx = -0.1 * x
    yy = -0.3 * y

    return xx, yy


# Two equal eigen value with geometric multiplicity 1
# Unstable case
def lin_4(x, y):
    xx = 0.1 * x + y
    yy = 0.1 * y

    return xx, yy


# Two imaginary eigen values
# Periodic orbit
def lin_5(x, y):
    xx = -0.3 * y
    yy = 0.1 * x

    return xx, yy


def main():
    x0 = np.array([1, 2])
    system = sys2d.System2D(lin_4, x0, dt=0.001)

    fig, ax = plt.subplots()

    ax = system.plot_orbits(ax)
    ax = system.plot_quiver(ax)

    plt.show()


if __name__ == "__main__":
    main()
