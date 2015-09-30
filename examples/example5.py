import numpy as np
from pngw import write_apng


# Example 5
#
# Create an 8-bit RGB animated PNG file.

height = 20
width = 200
t = np.linspace(0, 10*np.pi, width)
seq = []
for phase in np.linspace(0, 2*np.pi, 25, endpoint=False):
    y = 150*0.5*(1 + np.sin(t - phase))
    a = np.zeros((height, width, 3), dtype=np.uint8)
    a[:, :, 0] = y
    a[:, :, 2] = y
    seq.append(a)

write_apng("example5.png", seq, delay=50, use_palette=True)
