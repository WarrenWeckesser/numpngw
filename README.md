pngw
====

This python package defines the function `write_png` that writes a
numpy array to a PNG file.

Capabilities of `write_png` include:

* creation of 8-bit and 16-bit RGB files;
* creation of 1-bit, 2-bit, 4-bit, 8-bit and 16-bit grayscale files;
* creation of RGB and grayscale images with an alpha channel;
* setting a transparent color;
* automatic creation of a palette for an indexed file;
* inclusion of `tEXt` chunks.

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

    img = np.zeros((80, 128, 3), dtype=np.uint8)

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


Example 4
---------

The following script uses the option `use_palette=True` to create this 8-bit
indexed RGB file.

![](https://github.com/WarrenWeckesser/pngw/blob/master/examples/example4.png)

    import numpy as np
    from pngw import write_png


    # Example 4
    #
    # Create an 8-bit indexed RGB image that uses a palette.

    img_width = 300
    img_height = 200
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    np.random.seed(222)
    for _ in range(40):
        width = np.random.randint(5, img_width // 5)
        height = np.random.randint(5, img_height // 5)
        row = np.random.randint(5, img_height - height - 5)
        col = np.random.randint(5, img_width - width - 5)
        color = np.random.randint(80, 256, size=2)
        img[row:row+height, col:col+width, 1:] = color

    write_png('example4.png', img, use_palette=True)
