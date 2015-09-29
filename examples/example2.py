import numpy as np
from pngw import write_png


# Example 2
#
# Create a 1-bit grayscale image.

mask = np.zeros((48, 48), dtype=np.uint8)
mask[:2, :] = 1
mask[:, -2:] = 1
mask[4:6, :-4] = 1
mask[4:, -6:-4] = 1
mask[-16:, :16] = 1
mask[-32:-16, 16:32] = 1

write_png('example2.png', mask, bitdepth=1)
