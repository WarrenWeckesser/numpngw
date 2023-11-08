import numpy as np
from numpngw import write_png


# Example 4
#
# Create an 8-bit indexed RGB image that uses a palette.

img_width = 300
img_height = 200
img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

rng = np.random.default_rng(seed=121263137472525314065)
for _ in range(40):
    width = rng.integers(5, img_width // 5)
    height = rng.integers(5, img_height // 5)
    row = rng.integers(5, img_height - height - 5)
    col = rng.integers(5, img_width - width - 5)
    color = rng.integers(80, 256, size=2)
    img[row:row+height, col:col+width, 1:] = color

write_png('example4.png', img, use_palette=True)
