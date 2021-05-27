import numpy as np
from matplotlib import pyplot as plt


class System2D:
    """
    A class that represents a 2-d autonomous dynamical system.
    *********************************************************
    dynamical system of the form x' = f(x) where x is a 2-d
    vector using the Range-Kutta 4-th order method.
    ....

    Attributes
    ----------
    f: function
        The function is the differential equation.
    x0: np.array
        2-d vector representing the initial condition.
    t0: The initial time.
    T: The final time.
    dt: The step size to be used in RK4 algorithm.
    ....

    Methods
    -------
    plot_phase_diagram(matplotlib.axes, labelX="x", labelY="y"):
        Returns pyplot axis with the phase plot.

    plot_quiver(matplotlib.axes, xMin, xMax, yMin, yMax, separation):
        Plots quiver plot of the orbits.
    """
    def __init__(self,
                 f=lambda a: a,
                 x0=np.array([0, 0]),
                 t0=0.0,
                 T=10.0,
                 dt=0.1):
        """
        :param f: The function f in the Differential equation x' = f(x).
        :type f: function.

        :param x0: The initial condition in the form np.array([a, b]).
        :type f: np.array

        :parm t0: Initial time.
        :type t0: float.

        :param T: Final time.
        type T: float.

        :param dt: Step size to be used in the rk4 algorithm.
        :type dt: float.
        """
        # initializations
        self.f = f
        self.x0 = x0
        self.t0 = t0
        self.T = T
        self.dt = dt
        self.__data = []

    def __f(self, X):  # Same as f, but takes and
        x = X[0]       # returns np.array
        y = X[1]       # used in __generate_data()

        a, b = self.f(x, y)
        return np.array([a, b])

    def __generate_data(self):
        # RK-4 algorithm
        self.__data.clear()
        t = np.arange(self.t0, self.T, self.dt)
        size = len(t)
        self.__data.append(t)  # time
        self.__data.append(np.zeros((size, 1)))  # X values
        self.__data.append(np.zeros((size, 1)))  # Y values
        self.__data[1][0] = self.x0[0]
        self.__data[2][0] = self.x0[1]

        for i in range(size-1):
            X = np.array([self.__data[1][i], self.__data[2][i]])
            k1 = self.__f(X) * self.dt
            k2 = self.__f(X + (0.5 * k1)) * self.dt
            k3 = self.__f(X + (0.5 * k2)) * self.dt
            k4 = self.__f(X + k3) * self.dt
            ans = X + (k1 + k2 + k3 + k4)/6
            self.__data[1][i + 1] = ans[0]
            self.__data[2][i + 1] = ans[1]

    def plot_phase_diagram(self,
                           axis,
                           x0=np.array([0, 0])):
        """
        Returns the axis after plotting the phase portrait.
        (If both time series and phase plot of a data is required,
        generate the two in concecutive steps since both require
        the same data set. This would save some time because of
        the way in which these two functions are implemented.)

        :param axis: matplotlib axes
        :type axis: matplotlib.axes._subplots.AxesSubplot

        :param x0: initial value
        :type x0: np.array

        :return axis: the axis after plotting
        :rtype: matplotlib.axes._subplots.AxesSubplot
        """
        self.x0 = x0
        self.__generate_data()
        axis.plot(self.__data[1], self.__data[2])
        return axis

    def plot_time_series(self,
                         axis,
                         x0=np.array([0, 0]),
                         switch='x'):
        """
        Returns the time series portrait.
        (If both time series and phase plot of a data is required,
        generate the two in concecutive steps since both require
        the same data set.This would save some time because of
        the way in which these functions areimplemented.)

        :param axis: the matplotlib.axes on which the graph is plotted.
        :type axis: matplotlib.axes._subplots.AxesSubplot

        :param x0: the initial condition
        :type x0: np.array


        :param switch: if switch is 'x', time series of x i plotted
                        Otherwise the time series of y is plotted
        :type switch: str
        :param label: the label for the vertical axis
        :type label: str

        :returns axis: the plotted axis
        :rtype: matplotlib.axes._subplots.AxesSubplot
        """
        # If the data is not already present, generate.
        if len(self.__data) == 0 or np.all(x0 - self.x0):
            self.x0 = x0
            self.__generate_data()

        if switch == 'x':
            axis.plot(self.__data[0], self.__data[1])
        else:
            axis.plot(self.__data[0], self.__data[2])

        return axis

    def plot_quiver(self,
                    axis,
                    xMin=-10.0,
                    xMax=10.0,
                    yMin=-10.0,
                    yMax=10.0,
                    spacing=1,
                    scl=None):
        """
        Returns the axis after plotting the quiver.

        :param axis: matplotlib.axes on which the graph is plotted
        :type axis: matplotlib.axes._subplots.AxesSubplot

        :param xMin: minimum value of x
        :type xMin:float

        :param xMax: maximum value of x
        :type xMin: float

        :param yMin: minimum value of y
        :type yMin: float

        :param yMax: maximum value of y
        :type yMax: float

        :param spacing: the spacing between the contours
        :type spacing: float

        :param scl: the unit of the arrow. If not specified, autoscales
        :type scl: float

        :returns axis: the plotted axis
        :rtype: matplotlib.axes._subplots.AxesSubplot
        """

        x, y = np.meshgrid(np.arange(xMin, xMax, spacing),
                           np.arange(yMin, yMax, spacing))
        u, v = self.f(x, y)

        axis.quiver(x, y, u, v, scale=scl)
        return axis

    def plot_orbits(self,
                    axis,
                    xMin=-5.0,
                    xMax=5.0,
                    yMin=-5.0,
                    yMax=5.0,
                    spacing=1.0):
        """
        Returns the axis after ploting a few orbits.
        The initial conditions are taken from
        [xMin: spacing: xMax] * [yMin: spacing: yMax]

        :param axis: matplotlib.axes on which graph is plotted
        :type axis: matplotlib.axes._subplots.AxesSubplot

        :param xMin: Minimum x value of initial condition
        :type xMin: float

        :param xMax: Maximum x value of initial condition
        :type xMax: float

        :param yMin: Minimum y value of initial condition
        :type yMin: float

        :param yMax: Maximum y value of initial condition
        :type yMax: float

        :param spacing: the spacing between the initial values
        :type spacing: float

        :returns axis: The axis with all the orbits corresponding
                        to the initial values plotted
        :rtype: matplotlib.axes._subplots.AxesSubplot
        """
        xVals = np.arange(xMin, xMax, spacing)
        yVals = np.arange(yMin, yMax, spacing)

        for x in xVals:
            for y in yVals:
                self.x0 = np.array([x, y])
                self.__generate_data()
                axis.plot(self.__data[1], self.__data[2])

        return axis
