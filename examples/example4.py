import numpy as np
from pngw import write_png


# Example 4
#
# Create an 8-bit indexed RGB image that uses a palette.

img_width = 200
img_height = 150
img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

np.random.seed(1234)
for _ in range(21):
    width = np.random.randint(5, img_width // 3)
    height = np.random.randint(5, img_height // 3)
    row = np.random.randint(5, img_height - height - 5)
    col = np.random.randint(5, img_width - width - 5)
    color = np.random.randint(40, 256, size=3)
    img[row:row+height, col:col+width, :] = color

write_png('example4.png', img, use_palette=True)
