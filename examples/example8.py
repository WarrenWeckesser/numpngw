
import numpy as np
from scipy.integrate import odeint
from scipy.fftpack import diff as psdiff

import matplotlib.pyplot as plt
from matplotlib import animation
from numpngw import AnimatedPNGWriter


def kdv_exact(x, c):
    """
    Profile of the exact solution to the KdV for a single soliton
    on the real line.
    """
    u = 0.5*c*np.cosh(0.5*np.sqrt(c)*x)**(-2)
    return u


def kdv(u, t, L):
    """
    Differential equations for the KdV equation, discretized in x.
    """
    # Compute the x derivatives using the pseudo-spectral method.
    ux = psdiff(u, period=L)
    uxxx = psdiff(u, period=L, order=3)

    # Compute du/dt.
    dudt = -6*u*ux - uxxx

    return dudt


def kdv_solution(u0, t, L):
    """
    Use odeint to solve the KdV equation on a periodic domain.

    `u0` is initial condition, `t` is the array of time values at which
    the solution is to be computed, and `L` is the length of the periodic
    domain.
    """
    sol = odeint(kdv, u0, t, args=(L,), mxstep=5000)
    return sol


def update_line(num, x, data, line):
    """
    Animation "call back" function for each frame.
    """
    line.set_data(x, data[num, :])
    return line,


# Set the size of the domain, and create the discretized grid.
L = 80.0
N = 256
dx = L / (N - 1.0)
x = np.linspace(0, (1-1.0/N)*L, N)

# Set the initial conditions.
# Not exact for two solitons on a periodic domain, but close enough...
u0 = kdv_exact(x-0.15*L, 0.8) + kdv_exact(x-0.4*L, 0.4)

# Set the time sample grid.
T = 260
t = np.linspace(0, T, 225)

print("Computing the solution.")
sol = kdv_solution(u0, t, L)

print("Generating the animated PNG file.")

fig = plt.figure(figsize=(7.5, 1.5))
ax = fig.gca(title="Korteweg de Vries interacting solitons in a periodic "
                   "domain (L = 80)")

# Plot the initial condition. lineplot is reused in the animation.
lineplot, = ax.plot(x, u0, 'c-', linewidth=3)
plt.tight_layout()

ani = animation.FuncAnimation(fig, update_line, frames=len(t),
                              init_func=lambda : None,
                              fargs=(x, sol, lineplot))
writer = AnimatedPNGWriter(fps=12)
ani.save('example8.png', dpi=60, writer=writer)

plt.close(fig)
