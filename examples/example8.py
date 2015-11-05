
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from numpngw import AnimatedPNGWriter


def update_line(num, data, line):
    line.set_data(data[:, :num+1])
    return line,

fig = plt.figure(figsize=(4.25, 4.1))

num_frames = 18

theta = np.linspace(0, 24*np.pi, num_frames)
data = np.vstack((np.cos(theta), np.sin(theta)))

lineplot, = plt.plot([], [], 'c-', linewidth=3)
plt.axis('equal')
plt.ylim(-1, 1)
plt.xlim(-1, 1)
plt.title('Matplotlib Animation')
ani = animation.FuncAnimation(fig, update_line, frames=num_frames,
                              fargs=(data, lineplot))
writer = AnimatedPNGWriter(fps=2)
ani.save('example8.png', dpi=50, writer=writer)
