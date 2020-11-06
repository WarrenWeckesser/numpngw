import numpy as np
from numpngw import write_apng

# Example 7
#
# Create an animated PNG file with nonuniform display times
# of the frames.

bits1 = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    ])

bits2 = np.array([
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    ])

bits3 = np.array([
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
    ])

bits_box1 = np.array([
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    ])

bits_box2 = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    ])

bits_dot = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    ])

bits_zeros = np.zeros((7, 5), dtype=bool)
bits_ones = np.ones((7, 5), dtype=bool)


def bits_to_image(bits, blocksize=32, color=None):
    bits = np.asarray(bits, dtype=np.bool)
    if color is None:
        color = np.array([255, 0, 0], dtype=np.uint8)
    else:
        color = np.asarray(color, dtype=np.uint8)

    x = np.linspace(-1, 1, blocksize)
    X, Y = np.meshgrid(x, x)
    Z = np.sqrt(np.maximum(1 - (X**2 + Y**2), 0))
    # The "on" image:
    img1 = (Z.reshape(blocksize, blocksize, 1)*color)
    # The "off" image:
    img0 = 0.2*img1

    data = np.where(bits[:, None, :, None, None],
                    img1[:, None, :], img0[:, None, :])
    img = data.reshape(bits.shape[0]*blocksize, bits.shape[1]*blocksize, 3)
    return img.astype(np.uint8)


# Create `seq` and `delay`, the sequence of images and the
# corresponding display times.

color = np.array([32, 48, 255])
blocksize = 24
# Images...
im3 = bits_to_image(bits3, blocksize=blocksize, color=color)
im2 = bits_to_image(bits2, blocksize=blocksize, color=color)
im1 = bits_to_image(bits1, blocksize=blocksize, color=color)
im_all = bits_to_image(bits_ones, blocksize=blocksize, color=color)
im_none = bits_to_image(bits_zeros, blocksize=blocksize, color=color)
im_box1 = bits_to_image(bits_box1, blocksize=blocksize, color=color)
im_box2 = bits_to_image(bits_box2, blocksize=blocksize, color=color)
im_dot = bits_to_image(bits_dot, blocksize=blocksize, color=color)

# The sequence of images:
seq = [im3, im2, im1, im_all, im_none, im_all, im_none, im_all, im_none,
       im_box1, im_box2, im_dot, im_none]
# The time duration to display each image, in milliseconds:
delay = [1000, 1000, 1000, 333, 250, 333, 250, 333, 500,
         167, 167, 167, 1000]

# Create the animated PNG file.
write_apng("example7.png", seq, delay=delay, default_image=im_all,
           use_palette=True)
