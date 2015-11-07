
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from numpngw import AnimatedPNGWriter


def update_line(num, data, line):
    line.set_data(data[:, :num+1])
    return line,

fig = plt.figure(figsize=(5.75, 5.6))
ax = fig.add_subplot(111, xlim=(-1, 1), ylim=(-1, 1),
                     autoscale_on=False,  aspect='equal',
                     title="Matplotlib Animation")

num_frames = 20

theta = np.linspace(0, 24*np.pi, num_frames)
data = np.exp(1j*theta).view(np.float64).reshape(-1, 2).T

lineplot, = ax.plot([], [], 'c-', linewidth=3)

ani = animation.FuncAnimation(fig, update_line, frames=num_frames,
                              fargs=(data, lineplot))
writer = AnimatedPNGWriter(fps=2)
ani.save('example8.png', dpi=50, writer=writer)
