pngw
====

This python package defines the function `write_png` that writes a
numpy array to a PNG file.

This is prototype-quality software.  The documentation is sparse, and the API
will likely change.

For packages with more features (including functions for *reading* PNG files),
take a look at `pypng` (https://pypi.python.org/pypi/pypng) or `imageio`
(https://pypi.python.org/pypi/imageio).

This software is released under the BSD 2-clause license.

Example 1
---------

The following script creates this PNG file, an 8-bit RGB image.

![](https://github.com/WarrenWeckesser/pngw/blob/master/examples/example1.png)

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


Example 2
---------

The following script creates this PNG file, a 1-bit grayscale image.

![](https://github.com/WarrenWeckesser/pngw/blob/master/examples/example2.png)

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


Example 3
---------

The following script creates this PNG file, a 16-bit RGB file in which
the value (0, 0, 0) is transparent.  It might not be obvious, but the
two squares are transparent.

![](https://github.com/WarrenWeckesser/pngw/blob/master/examples/example3.png)


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
