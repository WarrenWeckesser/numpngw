import numpy as np
from pngw import write_png


# Example 1
#
# Create an 8-bit RGB image.

img = np.empty((80, 128, 3), dtype=np.uint8)

grad1 = np.linspace(0, 255, img.shape[1])
grad2 = np.linspace(255, 0, img.shape[1])

img[:16, :, :] = 127
img[16:32, :, 0] = grad1
img[32:48, :, 1] = grad2
img[48:64, :, 2] = grad1
img[64:, :, :] = 127

write_png('example1.png', img)
