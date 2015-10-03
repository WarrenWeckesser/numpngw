import numpy as np
from pngw import write_apng


# Example 6
#
# Create an 8-bit RGB animated PNG file.

def smoother(w):
    # Return the periodic convolution of w with a 3-d Gaussian kernel.
    r = np.linspace(-3, 3, 21)
    X, Y, Z = np.meshgrid(r, r, r)
    kernel = np.exp(-0.25*(X*X + Y*Y + Z*Z)**2)
    fw = np.fft.fftn(w)
    fkernel = np.fft.fftn(kernel, w.shape)
    v = np.fft.ifftn(fw*fkernel).real
    return v

height = 40
width = 250
num_frames = 30
np.random.seed(12345)
w = np.random.randn(num_frames, height, width, 3)
for k in range(3):
    w[..., k] = smoother(w[..., k])

seq = (255*(w - w.min())/w.ptp()).astype(np.uint8)

write_apng("example6.png", seq, delay=40)
