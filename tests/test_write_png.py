from __future__ import division, print_function

import unittest
import io
import struct
import zlib
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
import pngw


def next_chunk(s):
    chunk_len = struct.unpack("!I", s[:4])[0]
    chunk_type = s[4:8]
    chunk_data = s[8:8+chunk_len]
    crc = struct.unpack("!I", s[8+chunk_len:8+chunk_len+4])[0]
    check = zlib.crc32(chunk_type + chunk_data)
    if crc != check:
        raise RuntimeError("CRC not correct, chunk_type=%r" % (chunk_type,))
    return chunk_type, chunk_data, s[8+chunk_len+4:]


def check_signature(s):
    signature = s[:8]
    s = s[8:]
    assert_equal(signature, b'\x89PNG\x0D\x0A\x1A\x0A')
    return s


def check_ihdr(file_contents, width, height, bit_depth, color_type,
               compression_method=0, filter_method=0, interface_method=0):
    chunk_type, chunk_data, file_contents = next_chunk(file_contents)
    assert_equal(chunk_type, b"IHDR")
    values = parse_ihdr(chunk_data)
    expected = (width, height, bit_depth, color_type, compression_method,
                filter_method, interface_method)
    assert_equal(values, expected)
    return file_contents


def check_iend(file_contents):
    chunk_type, chunk_data, file_contents = next_chunk(file_contents)
    assert_equal(chunk_type, b"IEND")
    assert_equal(chunk_data, b"")
    # The IEND chunk is the last chunk, so file_contents should now
    # be empty.
    assert_equal(file_contents, b"")


def parse_ihdr(chunk_data):
    fmt = "!IIBBBBB"
    values = struct.unpack(fmt, chunk_data)
    # values are:
    #    width, height, bit_depth, color_type,
    #    compression_method, filter_method, interface_method
    return values


class TestWritePng(unittest.TestCase):

    def test_write_png_nbit_grayscale(self):
        np.random.seed(123)
        for bitdepth in [1, 2, 4, 8]:
            maxval = 2**bitdepth
            img = np.random.randint(0, maxval, size=(3, 11)).astype(np.uint8)
            f = io.BytesIO()
            pngw.write_png(f, img, bitdepth=bitdepth)

            file_contents = f.getvalue()

            file_contents = check_signature(file_contents)

            file_contents = check_ihdr(file_contents,
                                       width=img.shape[1], height=img.shape[0],
                                       bit_depth=bitdepth, color_type=0)

            # Check the IDAT chunk.
            chunk_type, chunk_data, file_contents = next_chunk(file_contents)
            self.assertEqual(chunk_type, b"IDAT")
            decompressed = zlib.decompress(chunk_data)
            b = np.fromstring(decompressed, dtype=np.uint8)
            ncols, rembits = divmod(img.shape[1]*bitdepth, 8)
            ncols += rembits > 0
            lines = b.reshape(img.shape[0], ncols + 1)
            expected_col0 = np.zeros(img.shape[0], dtype=np.uint8)
            assert_array_equal(lines[:, 0], expected_col0)
            p = lines[:, 1:]
            img2 = pngw._unpack(p, bitdepth=bitdepth, width=img.shape[1])
            assert_array_equal(img2, img)

            check_iend(file_contents)

    def test_write_png_4x5_8bit_RGB_palette(self):
        img = np.arange(4*5*3, dtype=np.uint8).reshape(4, 5, 3)
        f = io.BytesIO()
        pngw.write_png(f, img, use_palette=True)

        file_contents = f.getvalue()

        file_contents = check_signature(file_contents)

        file_contents = check_ihdr(file_contents,
                                   width=img.shape[1], height=img.shape[0],
                                   bit_depth=8, color_type=3)

        # Check the PLTE chunk.
        chunk_type, chunk_data, file_contents = next_chunk(file_contents)
        self.assertEqual(chunk_type, b"PLTE")
        p = np.fromstring(chunk_data, dtype=np.uint8).reshape(-1, 3)
        assert_array_equal(p, np.arange(4*5*3, dtype=np.uint8).reshape(-1, 3))

        # Check the IDAT chunk.
        chunk_type, chunk_data, file_contents = next_chunk(file_contents)
        self.assertEqual(chunk_type, b"IDAT")
        decompressed = zlib.decompress(chunk_data)
        b = np.fromstring(decompressed, dtype=np.uint8)
        lines = b.reshape(img.shape[0], img.shape[1]+1)
        img2 = lines[:, 1:].reshape(img.shape[:2])
        expected = np.arange(20, dtype=np.uint8).reshape(img.shape[:2])
        assert_array_equal(img2, expected)

        check_iend(file_contents)

    def test_write_png_max_chunk_len(self):
        # Create an 8-bit grayscale image.
        w = 250
        h = 150
        max_chunk_len = 500
        img = np.random.randint(0, 256, size=(h, w)).astype(np.uint8)
        f = io.BytesIO()
        pngw.write_png(f, img, max_chunk_len=max_chunk_len)

        file_contents = f.getvalue()

        file_contents = check_signature(file_contents)

        file_contents = check_ihdr(file_contents,
                                   width=w, height=h,
                                   bit_depth=8, color_type=0)

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


if __name__ == '__main__':
    unittest.main()
