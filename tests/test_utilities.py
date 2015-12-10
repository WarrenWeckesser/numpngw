from __future__ import division, print_function

import unittest
import numpy as np
try:
    import nose
except ImportError:
    raise ImportError("The 'nose' package must be installed to run the tests.")
from numpy.testing import assert_equal, assert_array_equal
import numpngw


class TestUtilities(unittest.TestCase):

    def test_palettize(self):
        colors = np.array([[10, 20, 30],
                           [20, 30, 40],
                           [20, 30, 50],
                           [40, 40, 40],
                           [90, 90,  0]], dtype=np.uint8)
        index_x = np.array([[2, 2, 0],
                            [1, 4, 4]])
        index_y = np.array([[4, 3, 3],
                            [1, 2, 3]])
        index_z = np.array([[0, 0, 0],
                            [0, 2, 0]])
        x = colors[index_x]
        y = colors[index_y]
        z = colors[index_z]

        index_seq, palette, trans = numpngw._palettize([x, y, z])
        assert_array_equal(index_seq[0], index_x)
        assert_array_equal(index_seq[1], index_y)
        assert_array_equal(index_seq[2], index_z)
        assert_array_equal(palette, colors)
        assert_equal(trans, None)

        colors = np.array([[10, 20, 30, 255],
                           [20, 30, 40, 128],
                           [20, 30, 50, 255],
                           [40, 40, 40, 128],
                           [40, 40, 40, 255],
                           [90, 90,  0, 255]], dtype=np.uint8)
        index_x = np.array([[2, 2, 0],
                            [1, 5, 4]])
        index_y = np.array([[3, 3, 3],
                            [3, 3, 3]])
        index_z = np.array([[0, 0, 0],
                            [5, 2, 0]])
        x = colors[index_x]
        y = colors[index_y]
        z = colors[index_z]

        index_seq, palette, trans = numpngw._palettize_seq(np.array([x, y, z]))
        assert_array_equal(index_seq[0], index_x)
        assert_array_equal(index_seq[1], index_y)
        assert_array_equal(index_seq[2], index_z)
        assert_array_equal(palette, colors[:, :3])
        assert_equal(trans, colors[:, 3])

    def test_palettize_seq(self):
        colors = np.array([[10, 20, 30],
                           [20, 30, 40],
                           [20, 30, 50],
                           [40, 40, 40],
                           [90, 90,  0]], dtype=np.uint8)
        index_x = np.array([[2, 2, 0],
                            [1, 4, 4]])
        index_y = np.array([[4, 3],
                            [1, 2]])
        index_z = np.array([[0, 0, 0, 0],
                            [0, 2, 0, 4]])
        x = colors[index_x]
        y = colors[index_y]
        z = colors[index_z]

        index_seq, palette, trans = numpngw._palettize_seq([x, y, z])
        assert_array_equal(index_seq[0], index_x)
        assert_array_equal(index_seq[1], index_y)
        assert_array_equal(index_seq[2], index_z)
        assert_array_equal(palette, colors)
        assert_equal(trans, None)

        colors = np.array([[10, 20, 30, 255],
                           [20, 30, 40, 128],
                           [20, 30, 50, 255],
                           [40, 40, 40, 128],
                           [40, 40, 40, 255],
                           [90, 90,  0, 255]], dtype=np.uint8)
        index_x = np.array([[2, 2, 0],
                            [1, 5, 4]])
        index_y = np.array([[3, 3, 3],
                            [3, 3, 3],
                            [3, 3, 3]])
        index_z = np.array([[0, 0, 0, 0],
                            [5, 2, 0, 4]])
        x = colors[index_x]
        y = colors[index_y]
        z = colors[index_z]

        index_seq, palette, trans = numpngw._palettize_seq([x, y, z])
        assert_array_equal(index_seq[0], index_x)
        assert_array_equal(index_seq[1], index_y)
        assert_array_equal(index_seq[2], index_z)
        assert_array_equal(palette, colors[:, :3])
        assert_equal(trans, colors[:, 3])

    def test_pack(self):
        a = np.array([[0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
                      [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                      [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]], dtype=np.uint8)
        b = numpngw._pack(a, 1)
        expected_b = np.array([[0b01001000, 0b01000000],
                               [0b11110000, 0b00000000],
                               [0b10101010, 0b10000000]], dtype=np.uint8)
        assert_array_equal(b, expected_b)

        a = np.array([[0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1],
                      [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]], dtype=np.uint8)
        b = numpngw._pack(a, 1)
        expected_b = np.array([[0b01001000, 0b01100000],
                               [0b11111111, 0b00100000],
                               [0b10101010, 0b10100000]], dtype=np.uint8)

        a = np.array([[0, 1, 0, 0, 1, 0, 0, 0],
                      [1, 1, 1, 1, 0, 0, 0, 0],
                      [1, 0, 1, 0, 1, 0, 1, 0]], dtype=np.uint8)
        b = numpngw._pack(a, 1)
        expected_b = np.array([[0b01001000],
                               [0b11111111],
                               [0b10101010]], dtype=np.uint8)

        a = np.array([[2, 1, 2, 1, 0],
                      [1, 2, 2, 3, 2],
                      [3, 3, 2, 2, 1],
                      [0, 0, 0, 0, 0]], dtype=np.uint8)
        b = numpngw._pack(a, 2)
        expected_b = np.array([[0b10011001, 0b00000000],
                               [0b01101011, 0b10000000],
                               [0b11111010, 0b01000000],
                               [0b00000000, 0b00000000]], dtype=np.uint8)

        a = np.array([[2, 1, 2, 1],
                      [1, 2, 2, 3],
                      [3, 3, 2, 2],
                      [0, 0, 0, 0]], dtype=np.uint8)
        b = numpngw._pack(a, 2)
        expected_b = np.array([[0b10011001],
                               [0b01101011],
                               [0b11111010],
                               [0b00000000]], dtype=np.uint8)

        a = np.array([[0xD, 0xE, 0xA, 0xD],
                      [0xB, 0xE, 0xE, 0xF],
                      [0x4, 0x3, 0x2, 0x1],
                      [0x0, 0x0, 0x0, 0x0]], dtype=np.uint8)
        b = numpngw._pack(a, 4)
        expected_b = np.array([[0xDE, 0xAD],
                               [0xBE, 0xEF],
                               [0x43, 0x21],
                               [0x00, 0x00]], dtype=np.uint8)

        a = np.array([[0xC, 0xA, 0xF, 0xE, 0x0],
                      [0x4, 0x3, 0x2, 0x1, 0xF],
                      [0x0, 0x0, 0x0, 0x0, 0x1]], dtype=np.uint8)
        b = numpngw._pack(a, 4)
        expected_b = np.array([[0xCA, 0xFE, 0x00],
                               [0x43, 0x21, 0xF0],
                               [0x00, 0x00, 0x10]], dtype=np.uint8)


if __name__ == '__main__':
    unittest.main()
