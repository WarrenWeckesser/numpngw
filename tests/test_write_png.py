from __future__ import division, print_function

import unittest
import io
import struct
import zlib
import numpy as np
from numpy.testing import assert_, assert_equal, assert_array_equal
import pngw


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
               compression_method=0, filter_method=0, interface_method=0):
    chunk_type, chunk_data, file_contents = next_chunk(file_contents)
    assert_equal(chunk_type, b"IHDR")
    values = struct.unpack("!IIBBBBB", chunk_data)
    expected = (width, height, bit_depth, color_type, compression_method,
                filter_method, interface_method)
    assert_equal(values, expected)
    return file_contents


def check_trns(file_contents, color_type, transparent):
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
        trns = np.fromstring(chunk_data, dtype=np.uint8)
        # TODO: Write a test for the case use_palette=True and
        #       transparent is not None.
        assert_(False, msg="This test is not complete!")
    else:
        raise RuntimeError("check_trns called with invalid color_type %r" %
                           (color_type,))
    return file_contents


def check_actl(file_contents, num_frames, num_plays):
    chunk_type, chunk_data, file_contents = next_chunk(file_contents)
    assert_equal(chunk_type, b"acTL")
    values = struct.unpack("!II", chunk_data)
    assert_equal(values, (num_frames, num_plays))
    return file_contents


def check_fctl(file_contents, sequence_number, width, height,
               x_offset=0, y_offset=0, delay_num=0, delay_den=1,
               dispose_op=0, blend_op=1):
    chunk_type, chunk_data, file_contents = next_chunk(file_contents)
    assert_equal(chunk_type, b"fcTL")
    values = struct.unpack("!IIIIIHHBB", chunk_data)
    expected_values = (sequence_number, width, height, x_offset, y_offset,
                       delay_num, delay_den, dispose_op, blend_op)
    assert_equal(values, expected_values)
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
        # and 8, with or without a `transparent` color selected.
        np.random.seed(123)
        for bitdepth in [1, 2, 4, 8]:
            for transparent in [None, 0]:
                maxval = 2**bitdepth
                img = np.random.randint(0, maxval, size=(3, 11))
                img = img.astype(np.uint8)
                if transparent is not None:
                    img[2:4, 2:] = transparent

                f = io.BytesIO()
                pngw.write_png(f, img, bitdepth=bitdepth,
                               transparent=transparent)

                file_contents = f.getvalue()

                file_contents = check_signature(file_contents)

                file_contents = check_ihdr(file_contents,
                                           width=img.shape[1],
                                           height=img.shape[0],
                                           bit_depth=bitdepth, color_type=0)

                if transparent is not None:
                    file_contents = check_trns(file_contents, color_type=0,
                                               transparent=transparent)

                # Check the IDAT chunk.
                chunk_type, chunk_data, file_contents = \
                    next_chunk(file_contents)
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

    def test_write_png_8bit_RGBA(self):
        w = 25
        h = 15
        np.random.seed(12345)
        img = np.random.randint(0, 256, size=(h, w, 4)).astype(np.uint8)
        f = io.BytesIO()
        pngw.write_png(f, img)

        file_contents = f.getvalue()

        file_contents = check_signature(file_contents)

        file_contents = check_ihdr(file_contents,
                                   width=img.shape[1], height=img.shape[0],
                                   bit_depth=8, color_type=6)

        # Check the IDAT chunk.
        chunk_type, chunk_data, file_contents = next_chunk(file_contents)
        self.assertEqual(chunk_type, b"IDAT")
        decompressed = zlib.decompress(chunk_data)
        b = np.fromstring(decompressed, dtype=np.uint8)
        lines = b.reshape(img.shape[0], 4*img.shape[1]+1)
        expected_col0 = np.zeros(img.shape[0], dtype=np.uint8)
        assert_array_equal(lines[:, 0], expected_col0)
        img2 = lines[:, 1:].reshape(img.shape)
        assert_array_equal(img2, img)

        check_iend(file_contents)

    def test_write_png_8bit_RGB(self):
        # Test creation of an 8 bit RGB file (color type 2), with and without
        # a `transparent` color selected.
        w = 24
        h = 10
        np.random.seed(12345)
        for transparent in [None, (0, 0, 0)]:
            img = np.random.randint(0, 256, size=(h, w, 3)).astype(np.uint8)
            if transparent:
                img[2:4, 2:4] = transparent

            f = io.BytesIO()
            pngw.write_png(f, img, transparent=transparent)

            file_contents = f.getvalue()

            file_contents = check_signature(file_contents)

            file_contents = check_ihdr(file_contents,
                                       width=img.shape[1], height=img.shape[0],
                                       bit_depth=8, color_type=2)

            if transparent:
                file_contents = check_trns(file_contents, color_type=2,
                                           transparent=transparent)

            # Check the IDAT chunk.
            chunk_type, chunk_data, file_contents = next_chunk(file_contents)
            self.assertEqual(chunk_type, b"IDAT")
            decompressed = zlib.decompress(chunk_data)
            b = np.fromstring(decompressed, dtype=np.uint8)
            lines = b.reshape(img.shape[0], 3*img.shape[1]+1)
            expected_col0 = np.zeros(img.shape[0], dtype=np.uint8)
            assert_array_equal(lines[:, 0], expected_col0)
            img2 = lines[:, 1:].reshape(img.shape)
            assert_array_equal(img2, img)

            check_iend(file_contents)

    def test_write_png_8bit_RGB_palette(self):
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


class TestWriteApng(unittest.TestCase):

    def test_write_apng_8bit_RGBA(self):
        num_frames = 4
        w = 25
        h = 15
        np.random.seed(12345)
        seq_size = (num_frames, h, w, 4)
        seq = np.random.randint(0, 256, size=seq_size).astype(np.uint8)
        f = io.BytesIO()
        pngw.write_apng(f, seq)

        file_contents = f.getvalue()

        file_contents = check_signature(file_contents)

        file_contents = check_ihdr(file_contents, width=w, height=h,
                                   bit_depth=8, color_type=6)

        file_contents = check_actl(file_contents, num_frames=4, num_plays=0)

        sequence_number = 0
        file_contents = check_fctl(file_contents,
                                   sequence_number=sequence_number,
                                   width=w, height=h)
        sequence_number += 1

        # Check the IDAT chunk.
        chunk_type, chunk_data, file_contents = next_chunk(file_contents)
        self.assertEqual(chunk_type, b"IDAT")
        decompressed = zlib.decompress(chunk_data)
        b = np.fromstring(decompressed, dtype=np.uint8)
        lines = b.reshape(h, 4*w+1)
        expected_col0 = np.zeros(h, dtype=np.uint8)
        assert_array_equal(lines[:, 0], expected_col0)
        img2 = lines[:, 1:].reshape(h, w, 4)
        assert_array_equal(img2, seq[0])

        for k in range(1, 4):
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


if __name__ == '__main__':
    unittest.main()
