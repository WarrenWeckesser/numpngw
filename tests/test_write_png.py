from __future__ import division, print_function

import unittest
import io
import struct
import zlib
import numpy as np
try:
    import nose
except ImportError:
    raise ImportError("The 'nose' package must be installed to run the tests.")
from numpy.testing import (assert_, assert_equal, assert_array_equal,
                           assert_raises)
import numpngw


def next_chunk(s):
    chunk_len = struct.unpack("!I", s[:4])[0]
    chunk_type = s[4:8]
    chunk_data = s[8:8+chunk_len]
    crc = struct.unpack("!I", s[8+chunk_len:8+chunk_len+4])[0]
    check = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
    if crc != check:
        raise RuntimeError("CRC not correct, chunk_type=%r" % (chunk_type,))
    return chunk_type, chunk_data, s[8+chunk_len+4:]


def check_signature(s):
    signature = s[:8]
    s = s[8:]
    assert_equal(signature, b'\x89PNG\x0D\x0A\x1A\x0A')
    return s


def check_ihdr(file_contents, width, height, bit_depth, color_type,
               compression_method=0, filter_method=0, interlace=0):
    chunk_type, chunk_data, file_contents = next_chunk(file_contents)
    assert_equal(chunk_type, b"IHDR")
    values = struct.unpack("!IIBBBBB", chunk_data)
    assert_equal(values[:2], (width, height), "wrong width and height")
    assert_equal(values[2], bit_depth, "wrong bit depth")
    assert_equal(values[3], color_type, "wrong color type")
    assert_equal(values[4], compression_method, "wrong compression method")
    assert_equal(values[5], filter_method, "wrong filter method")
    assert_equal(values[6], interlace, "wrong interlace")
    return file_contents


def check_trns(file_contents, color_type, transparent, palette=None):
    if color_type == 3 and palette is None:
        raise ValueError("color_type is 3 but no palette was provided.")

    chunk_type, chunk_data, file_contents = next_chunk(file_contents)
    assert_equal(chunk_type, b"tRNS")
    assert_(color_type not in [4, 6],
            msg='Found tRNS chunk, but color_type is %r' % (color_type,))
    if color_type == 0:
        # Grayscale
        trns = struct.unpack("!H", chunk_data)[0]
        assert_equal(trns, transparent)
    elif color_type == 2:
        # RGB
        trns = struct.unpack("!HHH", chunk_data)
        assert_equal(trns, transparent)
    elif color_type == 3:
        # alphas for the first len(chunk_data) palette indices.
        trns_index = np.fromstring(chunk_data, dtype=np.uint8)[0]
        trns_color = palette[trns_index]
        assert_equal(trns_color, transparent)
    else:
        raise RuntimeError("check_trns called with invalid color_type %r" %
                           (color_type,))
    return file_contents


def check_bkgd(file_contents, color, color_type, palette=None):
    if color_type == 3 and palette is None:
        raise ValueError("color_type is 3 but no palette was provided.")

    chunk_type, chunk_data, file_contents = next_chunk(file_contents)
    assert_equal(chunk_type, b"bKGD")
    if color_type == 0 or color_type == 4:
        clr = struct.unpack("!H", chunk_data)
    elif color_type == 2 or color_type == 6:
        clr = struct.unpack("!HHH", chunk_data)
    else:
        # color_type is 3.
        clr_index = struct.unpack("B", chunk_data)
        clr = palette[clr_index]
    assert_equal(clr, color,
                 "%r != %r  color_type=%r" % (clr, color, color_type))
    return file_contents


def check_phys(file_contents, phys):
    chunk_type, chunk_data, file_contents = next_chunk(file_contents)
    assert_equal(chunk_type, b"pHYs")
    xppu, yppu, unit = struct.unpack("!IIB", chunk_data)
    assert_equal((xppu, yppu, unit), phys)
    return file_contents


def check_text(file_contents, keyword, text_string=None):
    # If text_string is None, this code just checks the keyword.
    chunk_type, chunk_data, file_contents = next_chunk(file_contents)
    assert_equal(chunk_type, b"tEXt")
    assert_(b'\x00' in chunk_data)
    key, text = chunk_data.split(b'\x00', 1)
    assert_equal(key, keyword)
    if text_string is not None:
        assert_equal(text, text_string)
    return file_contents


def check_idat(file_contents, color_type, bit_depth, interlace, img,
               palette=None):
    # This function assumes the entire image is in the chunk.
    chunk_type, chunk_data, file_contents = next_chunk(file_contents)
    assert_equal(chunk_type, b"IDAT")
    decompressed = zlib.decompress(chunk_data)
    stream = np.fromstring(decompressed, dtype=np.uint8)
    height, width = img.shape[:2]
    img2 = stream_to_array(stream, width, height, color_type, bit_depth,
                           interlace)
    if palette is not None:
        img2 = palette[img2]
    assert_array_equal(img2, img)
    return file_contents


def stream_to_array_old(stream, width, height, color_type, bit_depth):
    # `stream` is 1-d numpy array with dytpe np.uint8 containing the
    # data from one or more IDAT or fdAT chunks.
    #
    # This function converts `stream` to a numpy array.

    ncols, rembits = divmod(width*bit_depth, 8)
    ncols += rembits > 0

    if bit_depth < 8:
        bytes_per_pixel = 1  # Not really, but we need 1 here for later.
        data_bytes_per_line = ncols
    else:
        # nchannels is a map from color_type to the number of color
        # channels (e.g. an RGB image has three channels).
        nchannels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
        bytes_per_channel = bit_depth // 8
        bytes_per_pixel = bytes_per_channel * nchannels[color_type]
        data_bytes_per_line = bytes_per_pixel * width

    data_width = data_bytes_per_line / bytes_per_pixel

    lines = stream.reshape(height, data_bytes_per_line + 1)

    prev = np.zeros((data_width, bytes_per_pixel), dtype=np.uint8)
    p = np.empty((height, data_width, bytes_per_pixel), dtype=np.uint8)
    for k in range(lines.shape[0]):
        line_filter_type = lines[k, 0]
        filtered = lines[k, 1:].reshape(-1, bytes_per_pixel)
        if line_filter_type == 0:
            p[k] = filtered
        elif line_filter_type == 1:
            p[k] = numpngw._filter1inv(filtered, prev)
        elif line_filter_type == 2:
            p[k] = numpngw._filter2inv(filtered, prev)
        elif line_filter_type == 3:
            p[k] = numpngw._filter3inv(filtered, prev)
        elif line_filter_type == 4:
            p[k] = numpngw._filter4inv(filtered, prev)
        else:
            raise ValueError('invalid filter type: %i' % (line_filter_type,))
        prev = p[k]

    # As this point, p has data type uint8 and has shape
    # (height, width, bytes_per_pixel).
    # 16 bit components of the pixel are stored in big-endian format.

    uint8to16 = np.array([256, 1], dtype=np.uint16)

    if color_type == 0:
        # grayscale
        if bit_depth == 16:
            img = p.dot(uint8to16)
        elif bit_depth == 8:
            img = p[:, :, 0]
        else:  # bit_depth is 1, 2 or 4.
            img = numpngw._unpack(p.reshape(height, -1),
                                  bitdepth=bit_depth, width=width)

    elif color_type == 2:
        # RGB
        if bit_depth == 16:
            # Combine high and low bytes to 16-bit values.
            img = p.reshape(height, width, 3, 2).dot(uint8to16)
        else:  # bit_depth is 8.
            img = p

    elif color_type == 3:
        # indexed
        img = p[:, :, 0]

    elif color_type == 4:
        # grayscale with alpha
        if bit_depth == 16:
            # Combine high and low bytes to 16-bit values.
            img = p.reshape(height, width, 2, 2).dot(uint8to16)
        else:  # bit_depth is 8.
            img = p

    elif color_type == 6:
        # RGBA
        if bit_depth == 16:
            # Combine high and low bytes to 16-bit values.
            img = p.reshape(height, width, 4, 2).dot(uint8to16)
        else:  # bit_depth is 8.
            img = p

    else:
        raise RuntimeError('invalid color type %r' % (color_type,))

    return img


def img_color_format(color_type, bitdepth):
    """
    Given a color type and a bit depth, return the length
    of the color dimension and the data type of the numpy
    array that will hold an image with those parameters.
    """
    # nchannels is a map from color_type to the number of color
    # channels (e.g. an RGB image has three channels).
    nchannels = {0: 1,  # grayscale
                 2: 3,  # RGB
                 3: 1,  # indexed RGB
                 4: 2,  # grayscale+alpha
                 6: 4}  # RGBA
    if color_type == 3:
        dtype = np.uint8
    else:
        dtype = np.uint8 if bitdepth <= 8 else np.uint16
    return nchannels[color_type], dtype


def stream_to_array(stream, width, height, color_type, bit_depth, interlace=0):
    # `stream` is 1-d numpy array with dytpe np.uint8 containing the
    # data from one or more IDAT or fdAT chunks.
    #
    # This function converts `stream` to a numpy array.

    img_color_dim, img_dtype = img_color_format(color_type, bit_depth)
    if img_color_dim == 1:
        img_shape = (height, width)
    else:
        img_shape = (height, width, img_color_dim)
    img = np.empty(img_shape, dtype=img_dtype)

    if interlace == 1:
        passes = numpngw._interlace_passes(img)
    else:
        passes = [img]

    pass_start_index = 0
    for a in passes:
        if a.size == 0:
            continue
        pass_height, pass_width = a.shape[:2]
        ncols, rembits = divmod(pass_width*bit_depth, 8)
        ncols += rembits > 0

        if bit_depth < 8:
            bytes_per_pixel = 1  # Not really, but we need 1 here for later.
            data_bytes_per_line = ncols
        else:
            # nchannels is a map from color_type to the number of color
            # channels (e.g. an RGB image has three channels).
            nchannels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
            bytes_per_channel = bit_depth // 8
            bytes_per_pixel = bytes_per_channel * nchannels[color_type]
            data_bytes_per_line = bytes_per_pixel * pass_width

        data_width = data_bytes_per_line // bytes_per_pixel

        pass_end_index = (pass_start_index +
                          pass_height * (data_bytes_per_line + 1))
        shp = (pass_height, data_bytes_per_line + 1)
        lines = stream[pass_start_index:pass_end_index].reshape(shp)
        pass_start_index = pass_end_index

        prev = np.zeros((data_width, bytes_per_pixel), dtype=np.uint8)
        shp = (pass_height, data_width, bytes_per_pixel)
        p = np.empty(shp, dtype=np.uint8)
        for k in range(lines.shape[0]):
            line_filter_type = lines[k, 0]
            filtered = lines[k, 1:].reshape(-1, bytes_per_pixel)
            if line_filter_type == 0:
                p[k] = filtered
            elif line_filter_type == 1:
                p[k] = numpngw._filter1inv(filtered, prev)
            elif line_filter_type == 2:
                p[k] = numpngw._filter2inv(filtered, prev)
            elif line_filter_type == 3:
                p[k] = numpngw._filter3inv(filtered, prev)
            elif line_filter_type == 4:
                p[k] = numpngw._filter4inv(filtered, prev)
            else:
                raise ValueError('invalid filter type: %i' %
                                 (line_filter_type,))
            prev = p[k]

        # As this point, p has data type uint8 and has shape
        # (height, width, bytes_per_pixel).
        # 16 bit components of the pixel are stored in big-endian format.

        uint8to16 = np.array([256, 1], dtype=np.uint16)

        if color_type == 0:
            # grayscale
            if bit_depth == 16:
                pass_img = p.dot(uint8to16)
            elif bit_depth == 8:
                pass_img = p[:, :, 0]
            else:  # bit_depth is 1, 2 or 4.
                pass_img = numpngw._unpack(p.reshape(pass_height, -1),
                                           bitdepth=bit_depth,
                                           width=pass_width)

        elif color_type == 2:
            # RGB
            if bit_depth == 16:
                # Combine high and low bytes to 16-bit values.
                shp = (pass_height, pass_width, 3, 2)
                pass_img = p.reshape(shp).dot(uint8to16)
            else:  # bit_depth is 8.
                pass_img = p

        elif color_type == 3:
            # indexed
            if bit_depth < 8:
                pass_img = numpngw._unpack(p[:, :, 0], bitdepth=bit_depth,
                                           width=pass_width)
            else:
                pass_img = p[:, :, 0]

        elif color_type == 4:
            # grayscale with alpha
            if bit_depth == 16:
                # Combine high and low bytes to 16-bit values.
                shp = (pass_height, pass_width, 2, 2)
                pass_img = p.reshape(shp).dot(uint8to16)
            else:  # bit_depth is 8.
                pass_img = p

        elif color_type == 6:
            # RGBA
            if bit_depth == 16:
                # Combine high and low bytes to 16-bit values.
                shp = (pass_height, pass_width, 4, 2)
                pass_img = p.reshape(shp).dot(uint8to16)
            else:  # bit_depth is 8.
                pass_img = p

        else:
            raise RuntimeError('invalid color type %r' % (color_type,))

        a[...] = pass_img

    return img


def check_actl(file_contents, num_frames, num_plays):
    chunk_type, chunk_data, file_contents = next_chunk(file_contents)
    assert_equal(chunk_type, b"acTL")
    values = struct.unpack("!II", chunk_data)
    assert_equal(values, (num_frames, num_plays))
    return file_contents


def check_fctl(file_contents, sequence_number, width, height,
               x_offset=0, y_offset=0, delay_num=0, delay_den=1,
               dispose_op=0, blend_op=0):
    chunk_type, chunk_data, file_contents = next_chunk(file_contents)
    assert_equal(chunk_type, b"fcTL")
    values = struct.unpack("!IIIIIHHBB", chunk_data)
    expected_values = (sequence_number, width, height, x_offset, y_offset,
                       delay_num, delay_den, dispose_op, blend_op)
    assert_equal(values, expected_values)
    return file_contents


def check_time(file_contents, timestamp):
    chunk_type, chunk_data, file_contents = next_chunk(file_contents)
    assert_equal(chunk_type, b"tIME")
    values = struct.unpack("!HBBBBB", chunk_data)
    assert_equal(values, timestamp)
    return file_contents


def check_gama(file_contents, gamma):
    # gamma is the floating point gamma value.
    chunk_type, chunk_data, file_contents = next_chunk(file_contents)
    assert_equal(chunk_type, b"gAMA")
    gama = struct.unpack("!I", chunk_data)[0]
    igamma = int(gamma*100000 + 0.5)
    assert_equal(gama, igamma)
    return file_contents


def check_iend(file_contents):
    chunk_type, chunk_data, file_contents = next_chunk(file_contents)
    assert_equal(chunk_type, b"IEND")
    assert_equal(chunk_data, b"")
    # The IEND chunk is the last chunk, so file_contents should now
    # be empty.
    assert_equal(file_contents, b"")


class TestWritePng(unittest.TestCase):

    def test_write_png_nbit_grayscale(self):
        # Test the creation of grayscale images for bit depths of 1, 2, 4
        # 8 and 16, with or without a `transparent` color selected.
        np.random.seed(123)
        for filter_type in [0, 1, 2, 3, 4, "heuristic", "auto"]:
            for bitdepth in [1, 2, 4, 8, 16]:
                for transparent in [None, 0]:
                    for interlace in [0, 1]:
                        dt = np.uint16 if bitdepth == 16 else np.uint8
                        maxval = 2**bitdepth
                        sz = (3, 11)
                        img = np.random.randint(0, maxval, size=sz).astype(dt)
                        if transparent is not None:
                            img[2:4, 2:] = transparent

                        f = io.BytesIO()
                        numpngw.write_png(f, img, bitdepth=bitdepth,
                                          transparent=transparent,
                                          filter_type=filter_type,
                                          interlace=interlace)

                        file_contents = f.getvalue()

                        file_contents = check_signature(file_contents)

                        file_contents = check_ihdr(file_contents,
                                                   width=img.shape[1],
                                                   height=img.shape[0],
                                                   bit_depth=bitdepth,
                                                   color_type=0,
                                                   interlace=interlace)

                        file_contents = check_text(file_contents,
                                                   b"Creation Time")
                        software = numpngw._software_text().encode('latin-1')
                        file_contents = check_text(file_contents, b"Software",
                                                   software)

                        if transparent is not None:
                            file_contents = check_trns(file_contents,
                                                       color_type=0,
                                                       transparent=transparent)

                        file_contents = check_idat(file_contents, color_type=0,
                                                   bit_depth=bitdepth,
                                                   interlace=interlace,
                                                   img=img)

                        check_iend(file_contents)

    def test_write_png_with_alpha(self):
        # Test creation of grayscale+alpha and RGBA images (color types 4
        # and 6, resp.), with bit depths 8 and 16.
        w = 25
        h = 15
        np.random.seed(12345)
        for filter_type in [0, 1, 2, 3, 4, "heuristic", "auto"]:
            for color_type in [4, 6]:
                num_channels = 2 if color_type == 4 else 4
                for bit_depth in [8, 16]:
                    for interlace in [0, 1]:
                        dt = np.uint8 if bit_depth == 8 else np.uint16
                        sz = (h, w, num_channels)
                        img = np.random.randint(0, 2**bit_depth,
                                                size=sz).astype(dt)
                        f = io.BytesIO()
                        numpngw.write_png(f, img, filter_type=filter_type,
                                          interlace=interlace)

                        file_contents = f.getvalue()

                        file_contents = check_signature(file_contents)

                        file_contents = check_ihdr(file_contents,
                                                   width=w, height=h,
                                                   bit_depth=bit_depth,
                                                   color_type=color_type,
                                                   interlace=interlace)

                        file_contents = check_text(file_contents,
                                                   b"Creation Time")
                        software = numpngw._software_text().encode('latin-1')
                        file_contents = check_text(file_contents, b"Software",
                                                   software)

                        file_contents = check_idat(file_contents,
                                                   color_type=color_type,
                                                   bit_depth=bit_depth,
                                                   interlace=interlace,
                                                   img=img)

                        check_iend(file_contents)

    def test_write_png_RGB(self):
        # Test creation of RGB images (color type 2), with and without
        # a `transparent` color selected, and with bit depth 8 and 16.
        w = 24
        h = 10
        np.random.seed(12345)
        for filter_type in [0, 1, 2, 3, 4, "heuristic", "auto"]:
            for transparent in [None, (0, 0, 0)]:
                for bit_depth in [8, 16]:
                    for interlace in [0, 1]:
                        dt = np.uint16 if bit_depth == 16 else np.uint8
                        maxval = 2**bit_depth
                        img = np.random.randint(0, maxval,
                                                size=(h, w, 3)).astype(dt)
                        if transparent:
                            img[2:4, 2:4] = transparent

                        f = io.BytesIO()
                        numpngw.write_png(f, img, transparent=transparent,
                                          filter_type=filter_type,
                                          interlace=interlace)

                        file_contents = f.getvalue()

                        file_contents = check_signature(file_contents)

                        file_contents = check_ihdr(file_contents,
                                                   width=w, height=h,
                                                   bit_depth=bit_depth,
                                                   color_type=2,
                                                   interlace=interlace)

                        file_contents = check_text(file_contents,
                                                   b"Creation Time")
                        software = numpngw._software_text().encode('latin-1')
                        file_contents = check_text(file_contents, b"Software",
                                                   software)

                        if transparent:
                            file_contents = check_trns(file_contents,
                                                       color_type=2,
                                                       transparent=transparent)

                        file_contents = check_idat(file_contents, color_type=2,
                                                   bit_depth=bit_depth,
                                                   interlace=interlace,
                                                   img=img)

                        check_iend(file_contents)

    def test_write_png_8bit_RGB_palette(self):
        for interlace in [0, 1]:
            for transparent in [None, (0, 1, 2)]:
                for bitdepth in [1, 2, 4, 8]:
                    w = 13
                    h = 4
                    ncolors = min(2**bitdepth, w*h)
                    idx = np.arange(w*h).reshape(h, w) % ncolors
                    colors = np.arange(ncolors*3).reshape(ncolors, 3)
                    colors = colors.astype(np.uint8)
                    img = colors[idx]
                    f = io.BytesIO()
                    numpngw.write_png(f, img, use_palette=True,
                                      transparent=transparent,
                                      interlace=interlace,
                                      bitdepth=bitdepth)

                    file_contents = f.getvalue()

                    file_contents = check_signature(file_contents)

                    file_contents = check_ihdr(file_contents,
                                               width=img.shape[1],
                                               height=img.shape[0],
                                               bit_depth=bitdepth,
                                               color_type=3,
                                               interlace=interlace)

                    file_contents = check_text(file_contents, b"Creation Time")
                    software = numpngw._software_text().encode('latin-1')
                    file_contents = check_text(file_contents, b"Software",
                                               software)

                    # Check the PLTE chunk.
                    chunk_type, chunk_data, file_contents = \
                        next_chunk(file_contents)
                    self.assertEqual(chunk_type, b"PLTE")
                    p = np.fromstring(chunk_data,
                                      dtype=np.uint8).reshape(-1, 3)
                    n = ncolors*3
                    expected = np.arange(n, dtype=np.uint8).reshape(-1, 3)
                    assert_array_equal(p, expected)

                    if transparent is not None:
                        file_contents = check_trns(file_contents,
                                                   color_type=3,
                                                   transparent=transparent,
                                                   palette=p)

                    # Check the IDAT chunk.
                    chunk_type, chunk_data, file_contents = \
                        next_chunk(file_contents)
                    self.assertEqual(chunk_type, b"IDAT")
                    decompressed = zlib.decompress(chunk_data)
                    stream = np.fromstring(decompressed, dtype=np.uint8)
                    height, width = img.shape[:2]
                    img2 = stream_to_array(stream, width, height, color_type=3,
                                           bit_depth=bitdepth,
                                           interlace=interlace)
                    expected = idx
                    assert_array_equal(img2, expected)

                    check_iend(file_contents)

    def test_write_png_max_chunk_len(self):
        # Create an 8-bit grayscale image.
        w = 250
        h = 150
        max_chunk_len = 500
        img = np.random.randint(0, 256, size=(h, w)).astype(np.uint8)
        f = io.BytesIO()
        numpngw.write_png(f, img, max_chunk_len=max_chunk_len)

        file_contents = f.getvalue()

        file_contents = check_signature(file_contents)

        file_contents = check_ihdr(file_contents,
                                   width=w, height=h,
                                   bit_depth=8, color_type=0, interlace=0)

        file_contents = check_text(file_contents, b"Creation Time")
        file_contents = check_text(file_contents, b"Software",
                                   numpngw._software_text().encode('latin-1'))

        zstream = b''
        while True:
            chunk_type, chunk_data, file_contents = next_chunk(file_contents)
            if chunk_type != b"IDAT":
                break
            self.assertEqual(chunk_type, b"IDAT")
            zstream += chunk_data
            self.assertLessEqual(len(chunk_data), max_chunk_len)
        data = zlib.decompress(zstream)
        b = np.fromstring(data, dtype=np.uint8)
        lines = b.reshape(h, w + 1)
        img2 = lines[:, 1:].reshape(h, w)
        assert_array_equal(img2, img)

        # Check the IEND chunk; chunk_type and chunk_data were read
        # in the loop above.
        self.assertEqual(chunk_type, b"IEND")
        self.assertEqual(chunk_data, b"")

        self.assertEqual(file_contents, b"")

    def test_write_png_timestamp_gamma(self):
        np.random.seed(123)
        img = np.random.randint(0, 256, size=(10, 10)).astype(np.uint8)
        f = io.BytesIO()
        timestamp = (1452, 4, 15, 8, 9, 10)
        gamma = 2.2
        numpngw.write_png(f, img, timestamp=timestamp, gamma=gamma)

        file_contents = f.getvalue()

        file_contents = check_signature(file_contents)

        file_contents = check_ihdr(file_contents,
                                   width=img.shape[1], height=img.shape[0],
                                   bit_depth=8, color_type=0, interlace=0)

        file_contents = check_text(file_contents, b"Creation Time")
        file_contents = check_text(file_contents, b"Software",
                                   numpngw._software_text().encode('latin-1'))

        file_contents = check_time(file_contents, timestamp)

        file_contents = check_gama(file_contents, gamma)

        file_contents = check_idat(file_contents, color_type=0, bit_depth=8,
                                   interlace=0, img=img)

        check_iend(file_contents)

    def test_write_png_bkgd(self):
        # Test creation of RGB images (color type 2), with a background color.
        w = 16
        h = 8
        np.random.seed(123)
        for bit_depth in [8, 16]:
            maxval = 2**bit_depth
            bg = (maxval - 1, maxval - 2, maxval - 3)
            dt = np.uint16 if bit_depth == 16 else np.uint8
            img = np.random.randint(0, maxval, size=(h, w, 3)).astype(dt)

            f = io.BytesIO()
            numpngw.write_png(f, img, background=bg, filter_type=0)

            file_contents = f.getvalue()

            file_contents = check_signature(file_contents)

            file_contents = check_ihdr(file_contents, width=w, height=h,
                                       bit_depth=bit_depth, color_type=2,
                                       interlace=0)

            file_contents = check_text(file_contents, b"Creation Time")
            software = numpngw._software_text().encode('latin-1')
            file_contents = check_text(file_contents, b"Software",
                                       software)

            file_contents = check_bkgd(file_contents, color=bg, color_type=2)

            file_contents = check_idat(file_contents, color_type=2,
                                       bit_depth=bit_depth, interlace=0,
                                       img=img)

            check_iend(file_contents)

    def test_write_png_bkgd_palette(self):
        # Test creation of RGB images with a background color
        # when use_palette is True.
        w = 6
        h = 8
        np.random.seed(123)
        for bg_in_img in [True, False]:
            bit_depth = 8
            maxval = 2**bit_depth
            bg = (maxval - 1, maxval - 3, maxval - 2)

            img = np.arange(1, w*h*3 + 1, dtype=np.uint8).reshape(h, w, 3)
            if bg_in_img:
                img[-1, -1] = bg

            f = io.BytesIO()
            numpngw.write_png(f, img, background=bg, use_palette=True)

            file_contents = f.getvalue()

            file_contents = check_signature(file_contents)

            file_contents = check_ihdr(file_contents, width=w, height=h,
                                       bit_depth=bit_depth, color_type=3,
                                       interlace=0)

            file_contents = check_text(file_contents, b"Creation Time")
            software = numpngw._software_text().encode('latin-1')
            file_contents = check_text(file_contents, b"Software",
                                       software)

            # Check the PLTE chunk.
            chunk_type, chunk_data, file_contents = next_chunk(file_contents)
            self.assertEqual(chunk_type, b"PLTE")
            plte = np.fromstring(chunk_data, dtype=np.uint8).reshape(-1, 3)
            expected_palette = np.arange(1, w*h*3+1,
                                         dtype=np.uint8).reshape(-1, 3)
            if bg_in_img:
                expected_palette[-1] = bg
            else:
                expected_palette = np.append(expected_palette,
                                             np.array([bg], dtype=np.uint8),
                                             axis=0)
            assert_array_equal(plte, expected_palette,
                               "unexpected palette %r %r" %
                               (plte[-2], expected_palette[-2]))

            file_contents = check_bkgd(file_contents, color=bg, color_type=3,
                                       palette=expected_palette)

            file_contents = check_idat(file_contents, color_type=3,
                                       bit_depth=bit_depth, interlace=0,
                                       img=img, palette=plte)

            check_iend(file_contents)

    def test_text_and_phys(self):
        img = np.arange(15).reshape(3, 5).astype(np.uint8)
        text_list = [('Monster', 'Godzilla'), ('Creation Time', None)]
        phys = (5, 4, 0)

        f = io.BytesIO()
        numpngw.write_png(f, img, filter_type=0, text_list=text_list,
                          phys=phys)

        file_contents = f.getvalue()

        file_contents = check_signature(file_contents)

        file_contents = check_ihdr(file_contents,
                                   width=img.shape[1],
                                   height=img.shape[0],
                                   bit_depth=8, color_type=0,
                                   interlace=0)

        file_contents = check_text(file_contents, b"Monster", b"Godzilla")
        file_contents = check_text(file_contents, b"Software",
                                   numpngw._software_text().encode('latin-1'))

        file_contents = check_phys(file_contents, phys)

        file_contents = check_idat(file_contents, color_type=0,
                                   bit_depth=8, interlace=0,
                                   img=img)

        check_iend(file_contents)

    def test_bad_text_keyword(self):
        img = np.zeros((5, 10), dtype=np.uint8)

        f = io.BytesIO()

        # keyword too long
        bad_keyword = "X"*90
        text_list = [(bad_keyword, "foo")]
        assert_raises(ValueError, numpngw.write_png, f, img,
                      dict(text_list=text_list))

        # keyword starts with a space
        bad_keyword = " ABC"
        text_list = [(bad_keyword, "foo")]
        assert_raises(ValueError, numpngw.write_png, f, img,
                      dict(text_list=text_list))

        # keyword ends with a space
        bad_keyword = "ABC "
        text_list = [(bad_keyword, "foo")]
        assert_raises(ValueError, numpngw.write_png, f, img,
                      dict(text_list=text_list))

        # keyword contains consecutive spaces
        bad_keyword = "A  BC"
        text_list = [(bad_keyword, "foo")]
        assert_raises(ValueError, numpngw.write_png, f, img,
                      dict(text_list=text_list))

        # keyword contains a nonprintable character (nonbreaking space,
        # in this case)
        bad_keyword = "ABC\xA0XYZ"
        text_list = [(bad_keyword, "foo")]
        assert_raises(ValueError, numpngw.write_png, f, img,
                      dict(text_list=text_list))

        # keyword cannot be encoded as latin-1
        bad_keyword = "ABC\u1234XYZ"
        text_list = [(bad_keyword, "foo")]
        assert_raises(ValueError, numpngw.write_png, f, img,
                      dict(text_list=text_list))

        # text string contains the null character
        bad_keyword = "ABC"
        text_list = [(bad_keyword, "foo\0bar")]
        assert_raises(ValueError, numpngw.write_png, f, img,
                      dict(text_list=text_list))

        # text string cannot be encoded as latin-1
        bad_keyword = "ABC"
        text_list = [(bad_keyword, "foo\u1234bar")]
        assert_raises(ValueError, numpngw.write_png, f, img,
                      dict(text_list=text_list))

    def test_bad_phys(self):
        img = np.zeros((5, 10), dtype=np.uint8)

        f = io.BytesIO()

        # Third value must be 0 or 1.
        phys = (1, 2, 3)
        assert_raises(ValueError, numpngw.write_png, f, img,
                      dict(phys=phys))

        # pixel per unit values must be positive.
        phys = (1, -2, 0)
        assert_raises(ValueError, numpngw.write_png, f, img,
                      dict(phys=phys))

        # pixel per unit values must be positive.
        phys = (0, 2, 0)
        assert_raises(ValueError, numpngw.write_png, f, img,
                      dict(phys=phys))

    def test_too_many_colors_for_palette(self):
        f = io.BytesIO()

        img = np.zeros((4, 4, 3), dtype=np.uint8)
        img[0, 0] = 1
        img[0, 1] = 2
        # img has 3 unique colors.
        assert_raises(ValueError, numpngw.write_png, f, img,
                      use_palette=True, bitdepth=1)


class TestWritePngFilterType(unittest.TestCase):

    def test_basic(self):
        w = 22
        h = 10
        bitdepth = 8
        np.random.seed(123)
        img = np.random.randint(0, 256, size=(h, w)).astype(np.uint8)

        f = io.BytesIO()
        numpngw.write_png(f, img, filter_type=1)

        file_contents = f.getvalue()

        file_contents = check_signature(file_contents)

        file_contents = check_ihdr(file_contents,
                                   width=img.shape[1],
                                   height=img.shape[0],
                                   bit_depth=bitdepth, color_type=0,
                                   interlace=0)

        file_contents = check_text(file_contents, b"Creation Time")
        file_contents = check_text(file_contents, b"Software",
                                   numpngw._software_text().encode('latin-1'))

        file_contents = check_idat(file_contents, color_type=0,
                                   bit_depth=bitdepth, interlace=0,
                                   img=img)

        check_iend(file_contents)


class TestWriteApng(unittest.TestCase):

    def test_write_apng_8bit_RGBA(self):
        num_frames = 4
        w = 25
        h = 15
        np.random.seed(12345)
        seq_size = (num_frames, h, w, 4)
        seq = np.random.randint(0, 256, size=seq_size).astype(np.uint8)
        f = io.BytesIO()
        numpngw.write_apng(f, seq)

        file_contents = f.getvalue()

        file_contents = check_signature(file_contents)

        file_contents = check_ihdr(file_contents, width=w, height=h,
                                   bit_depth=8, color_type=6, interlace=0)

        file_contents = check_text(file_contents, b"Creation Time")
        file_contents = check_text(file_contents, b"Software",
                                   numpngw._software_text().encode('latin-1'))

        file_contents = check_actl(file_contents, num_frames=num_frames,
                                   num_plays=0)

        sequence_number = 0
        file_contents = check_fctl(file_contents,
                                   sequence_number=sequence_number,
                                   width=w, height=h)
        sequence_number += 1

        file_contents = check_idat(file_contents, color_type=6, bit_depth=8,
                                   interlace=0, img=seq[0])

        for k in range(1, num_frames):
            file_contents = check_fctl(file_contents,
                                       sequence_number=sequence_number,
                                       width=w, height=h)
            sequence_number += 1

            # Check the fdAT chunk.
            chunk_type, chunk_data, file_contents = next_chunk(file_contents)
            self.assertEqual(chunk_type, b"fdAT")
            actual_seq_num = struct.unpack("!I", chunk_data[:4])[0]
            self.assertEqual(actual_seq_num, sequence_number)
            sequence_number += 1
            decompressed = zlib.decompress(chunk_data[4:])
            b = np.fromstring(decompressed, dtype=np.uint8)
            lines = b.reshape(h, 4*w+1)
            expected_col0 = np.zeros(h, dtype=np.uint8)
            assert_array_equal(lines[:, 0], expected_col0)
            img2 = lines[:, 1:].reshape(h, w, 4)
            assert_array_equal(img2, seq[k])

        check_iend(file_contents)

    def test_default_image(self):
        num_frames = 2
        w = 16
        h = 8
        np.random.seed(12345)
        seq_size = (num_frames, h, w, 4)
        seq = np.random.randint(0, 256, size=seq_size).astype(np.uint8)
        default_image = np.zeros((h, w, 4), dtype=np.uint8)

        f = io.BytesIO()

        numpngw.write_apng(f, seq, default_image=default_image)

        file_contents = f.getvalue()

        file_contents = check_signature(file_contents)

        file_contents = check_ihdr(file_contents, width=w, height=h,
                                   bit_depth=8, color_type=6, interlace=0)

        file_contents = check_text(file_contents, b"Creation Time")
        file_contents = check_text(file_contents, b"Software",
                                   numpngw._software_text().encode('latin-1'))

        file_contents = check_actl(file_contents, num_frames=num_frames,
                                   num_plays=0)

        sequence_number = 0

        file_contents = check_idat(file_contents, color_type=6, bit_depth=8,
                                   interlace=0, img=default_image)

        for k in range(0, num_frames):
            file_contents = check_fctl(file_contents,
                                       sequence_number=sequence_number,
                                       width=w, height=h)
            sequence_number += 1

            # Check the fdAT chunk.
            chunk_type, chunk_data, file_contents = next_chunk(file_contents)
            self.assertEqual(chunk_type, b"fdAT")
            actual_seq_num = struct.unpack("!I", chunk_data[:4])[0]
            self.assertEqual(actual_seq_num, sequence_number)
            sequence_number += 1
            decompressed = zlib.decompress(chunk_data[4:])
            b = np.fromstring(decompressed, dtype=np.uint8)
            lines = b.reshape(h, 4*w+1)
            expected_col0 = np.zeros(h, dtype=np.uint8)
            assert_array_equal(lines[:, 0], expected_col0)
            img2 = lines[:, 1:].reshape(h, w, 4)
            assert_array_equal(img2, seq[k])

        check_iend(file_contents)

    def test_write_apng_bkgd(self):
        # Test creation of RGB images (color type 2), with a background color.
        w = 16
        h = 8
        np.random.seed(123)
        num_frames = 3
        for bit_depth in [8, 16]:
            maxval = 2**bit_depth
            bg = (maxval - 1, maxval - 2, maxval - 3)
            dt = np.uint16 if bit_depth == 16 else np.uint8
            seq = np.random.randint(0, maxval,
                                    size=(num_frames, h, w, 3)).astype(dt)

            f = io.BytesIO()
            numpngw.write_apng(f, seq, background=bg, filter_type=0)

            file_contents = f.getvalue()

            file_contents = check_signature(file_contents)

            file_contents = check_ihdr(file_contents, width=w, height=h,
                                       bit_depth=bit_depth, color_type=2,
                                       interlace=0)

            file_contents = check_text(file_contents, b"Creation Time")
            software = numpngw._software_text().encode('latin-1')
            file_contents = check_text(file_contents, b"Software",
                                       software)

            file_contents = check_bkgd(file_contents, color=bg, color_type=2)

            file_contents = check_actl(file_contents, num_frames=num_frames,
                                       num_plays=0)

            sequence_number = 0
            file_contents = check_fctl(file_contents,
                                       sequence_number=sequence_number,
                                       width=w, height=h)
            sequence_number += 1

            file_contents = check_idat(file_contents, color_type=2,
                                       bit_depth=bit_depth,
                                       interlace=0, img=seq[0])

            for k in range(1, num_frames):
                file_contents = check_fctl(file_contents,
                                           sequence_number=sequence_number,
                                           width=w, height=h)
                sequence_number += 1

                # Check the fdAT chunk.
                nxt = next_chunk(file_contents)
                chunk_type, chunk_data, file_contents = nxt
                self.assertEqual(chunk_type, b"fdAT")
                actual_seq_num = struct.unpack("!I", chunk_data[:4])[0]
                self.assertEqual(actual_seq_num, sequence_number)
                sequence_number += 1
                decompressed = zlib.decompress(chunk_data[4:])
                b = np.fromstring(decompressed, dtype=np.uint8)
                img2 = stream_to_array(b, w, h, color_type=2,
                                       bit_depth=bit_depth, interlace=0)
                assert_array_equal(img2, seq[k])

            check_iend(file_contents)

    def test_too_many_colors_for_palette(self):
        f = io.BytesIO()

        img1 = np.zeros((4, 4, 3), dtype=np.uint8)
        img1[0, 0] = 1
        img1[0, 1] = 2
        img2 = np.zeros_like(img1)

        # [img1, img2] has 3 unique colors.
        assert_raises(ValueError, numpngw.write_apng, f, [img1, img2],
                      use_palette=True, bitdepth=1)


if __name__ == '__main__':
    unittest.main()
