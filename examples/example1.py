import numpy as np
from pngw import write_png


# Example 1
#
# Create an 8-bit RGB image.

img = np.zeros((80, 128, 3), dtype=np.uint8)

grad = np.linspace(0, 255, img.shape[1])

img[:16, :, :] = 127
img[16:32, :, 0] = grad
img[32:48, :, 1] = grad[::-1]
img[48:64, :, 2] = grad
img[64:, :, :] = 127

write_png('example1.png', img)
