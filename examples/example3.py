import numpy as np
from pngw import write_png


# Example 3
#
# Create a 16-bit RGB image, with (0, 0, 0) indicating a transparent pixel.

# Create some interesting data.
w = 32
nrows = 3*w
ncols = 5*w
kernel = np.exp(-np.linspace(-2, 2, 35)**2)
kernel = kernel/kernel.sum()
np.random.seed(123)
x = np.random.randn(nrows, ncols, 3)
x = np.apply_along_axis(lambda z: np.convolve(z, kernel, mode='same'), 0, x)
x = np.apply_along_axis(lambda z: np.convolve(z, kernel, mode='same'), 1, x)

# Convert to 16 bit unsigned integers.
z = (65535*((x - x.max())/x.ptp())).astype(np.uint16)

# Create two squares containing (0, 0, 0).
z[w:2*w, w:2*w] = 0
z[w:2*w, -2*w:-w] = 0

# Write the PNG file, and indicate that (0, 0, 0) should be transparent.
write_png('example3.png', z, transparent=(0, 0, 0))
