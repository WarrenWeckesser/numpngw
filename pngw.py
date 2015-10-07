"""
write_png(...) writes a numpy array to a PNG file.
write_apng(...) writes a sequence of arrays to an APNG file.

This code has no dependencies other than numpy and the python standard
libraries.


Limitations:

* Only filter type 0 (i.e. no line filter) is implemented.
* Only tested with Python 2.7 and 3.4 (but it definitely requires
  at least 2.6).
* _write_text requires the text string to be ASCII.  This might
  be too strong of a requirement.
* Channel bit depths of 1, 2, or 4 are supported for input arrays
  with dtype np.uint8, but this could be made more flexible.
  Only color_type 0 allows smaller bit depths.
* The values in the input array(s) are assumed to be within the
  range of the given bit depth.  Higher bits are ignored.

-----
Copyright (c) 2015, Warren Weckesser
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import (division as _division,
                        print_function as _print_function)

import struct as _struct
import zlib as _zlib
from fractions import Fraction
import numpy as _np


__version__ = "0.0.2-dev0"


def _create_stream(a):
    """
    Convert the data in `a` into a python string.

    The string is formatted as the "scan lines" of the array.
    """
    # `a` is expected to be a 2D or 3D array of unsigned integers.
    lines = []
    for row in a:
        # Convert the row to big-endian (i.e. network byte order).
        row_be = row.astype('>' + row.dtype.str[1:])
        # Create the string, with '\0' prepended.  The extra 0 is the
        # filter type byte; 0 means no filtering of this scan line.
        lines.append(b'\0' + row_be.tostring())
    stream = b''.join(lines)
    return stream


def _write_chunk(f, chunk_type, chunk_data):
    """
    Write a chunk to the file `f`.  This function wraps the chunk_type and
    chunk_data with the length and CRC field, and writes the result to `f`.
    """
    content = chunk_type + chunk_data
    length = _struct.pack("!I", len(chunk_data))
    crc = _struct.pack("!I", _zlib.crc32(content) & 0xFFFFFFFF)
    f.write(length + content + crc)


def _write_ihdr(f, width, height, nbits, color_type):
    """Write an IHDR chunk to `f`."""
    fmt = "!IIBBBBB"
    chunk_data = _struct.pack(fmt, width, height, nbits, color_type, 0, 0, 0)
    _write_chunk(f, b"IHDR", chunk_data)


def _write_text(f, keyword, text_string):
    """Write a tEXt chunk to `f`.

    keyword and test_string are expected to be strings (not bytes).
    The function encodes them as ASCII before writing to the file.
    """
    data = keyword.encode('ascii') + b'\0' + text_string.encode('ascii')
    _write_chunk(f, b'tEXt', data)


def _write_time(f, timestamp):
    """Write a tIME chunk to `f`."""
    chunk_data = _struct.pack('!HBBBBB', *timestamp)
    _write_chunk(f, b'tIME', chunk_data)


def _write_gama(f, gamma):
    """Write a gAMA chunk to `f`."""
    gama = int(gamma*100000 + 0.5)
    chunk_data = _struct.pack('!I', gama)
    _write_chunk(f, b'gAMA', chunk_data)


def _write_plte(f, palette):
    _write_chunk(f, b"PLTE", palette.tostring())


def _write_trns(f, trans):
    trans_be = trans.astype('>' + trans.dtype.str[1:])
    _write_chunk(f, b"tRNS", trans_be.tostring())


def _write_idat(f, data):
    """Write an IDAT chunk to `f`."""
    _write_chunk(f, b"IDAT", data)


def _write_iend(f):
    """Write an IEND chunk to `f`."""
    _write_chunk(f, b"IEND", b"")


def _write_actl(f, num_frames, num_plays):
    """Write an acTL chunk to `f`."""
    if num_frames < 1:
        raise ValueError("Attempt to create acTL chunk with num_frames (%i) "
                         "less than 1." % (num_frames,))
    chunk_data = _struct.pack("!II", num_frames, num_plays)
    _write_chunk(f, b"acTL", chunk_data)


def _write_fctl(f, sequence_number, width, height, x_offset, y_offset,
                delay_num, delay_den, dispose_op=0, blend_op=0):
    """Write an fcTL chunk to `f`."""
    if width < 1:
        raise ValueError("width must be greater than 0")
    if height < 1:
        raise ValueError("heigt must be greater than 0")
    if x_offset < 0:
        raise ValueError("x_offset must be nonnegative")
    if y_offset < 0:
        raise ValueError("y_offset must be nonnegative")

    fmt = "!IIIIIHHBB"
    chunk_data = _struct.pack(fmt, sequence_number, width, height,
                              x_offset, y_offset, delay_num, delay_den,
                              dispose_op, blend_op)
    _write_chunk(f, b"fcTL", chunk_data)


def _write_fdat(f, sequence_number, data):
    """Write an fdAT chunk to `f`."""
    seq = _struct.pack("!I", sequence_number)
    _write_chunk(f, b"fdAT", seq + data)


def _write_data(f, a, bitdepth, max_chunk_len=None, sequence_number=None):
    """
    Write the image data in the array `a` to the file, using IDAT chunks
    if sequence_number is None and fdAT chunks otherwise.

    `f` must be a writable file object.
    `a` must be a numpy array to be written to the file `f`.
    If `sequence_number` is None, 'IDAT' chunks are written.
    If `sequence_number` is not None, `fdAT` chunks are written.

    Returns the number of chunks written to the file.
    """
    if bitdepth is not None and bitdepth < 8:
        data = _pack(a, bitdepth)
    else:
        data = a
    stream = _create_stream(data)
    zstream = _zlib.compress(stream)

    # zstream is a string containing the packed, compressed version of the
    # data from the array `a`.  This will be written to the file in one or
    # more IDAT or fdAT chunks.

    if max_chunk_len is None:
        # Put the whole thing in one chunk.
        max_chunk_len = len(zstream)
    elif max_chunk_len < 1:
        raise ValueError("max_chunk_len must be at least 1.")

    num_data_chunks = (len(zstream) + max_chunk_len - 1) // max_chunk_len
    for k in range(num_data_chunks):
        start = k * max_chunk_len
        end = min(start + max_chunk_len, len(zstream))
        data = zstream[start:end]
        if sequence_number is None:
            _write_idat(f, data)
        else:
            _write_fdat(f, sequence_number, data)
            sequence_number += 1

    return num_data_chunks


def _validate_text(text_list):
    if text_list is None:
        return
    for keyword, text_string in text_list:
        if not (0 < len(keyword) < 80):
            raise ValueError("length of keyword must greater than 0 and less "
                             "than 80.")
        if '\0' in text_string:
            raise ValueError("text_string contains a null character.")
        kw_check = all([(31 < ord(c) < 127) or (160 < ord(c) < 256)
                        for c in keyword])
        if not kw_check:
            raise ValueError("keyword %r contains non-printable characters." %
                             (keyword,))


def _palettize(a):
    # `a` must be a numpy array with dtype `np.uint8` and shape (m, n, 3) or
    # (m, n, 4).
    a = _np.ascontiguousarray(a)
    depth = a.shape[-1]
    dt = ','.join(['u1'] * depth)
    b = a.view(dt).reshape(a.shape[:-1])
    colors, inv = _np.unique(b, return_inverse=True)
    index = inv.astype(_np.uint8).reshape(a.shape[:-1])
    # palette is the RGB values of the unique RGBA colors.
    palette = colors.view(_np.uint8).reshape(-1, depth)[:, :3]
    if depth == 3:
        trans = None
    else:
        # trans is the 1-d array of alpha values of the unique RGBA colors.
        # trans is the same length as `palette`.
        trans = colors['f3']
    return index, palette, trans


def _pack(a, bitdepth):
    """
    Pack the values in `a` into bitfields of a smaller array.

    `a` must be a 2-d numpy array with dtype `np.uint8`
    bitdepth must be either 1, 2, 4 or 8.
    (bitdepth=8 is a trivial case, for which the return value is simply `a`.)
    """
    if a.dtype != _np.uint8:
        raise ValueError('Input array must have dtype uint8')
    if a.ndim != 2:
        raise ValueError('Input array must be two dimensional')

    if bitdepth == 8:
        return a

    ncols, rembits = divmod(a.shape[1]*bitdepth, 8)
    if rembits > 0:
        ncols += 1
    b = _np.zeros((a.shape[0], ncols), dtype=_np.uint8)
    for row in range(a.shape[0]):
        bcol = 0
        pos = 8
        for col in range(a.shape[1]):
            val = (2**bitdepth - 1) & a[row, col]
            pos -= bitdepth
            if pos < 0:
                bcol += 1
                pos = 8 - bitdepth
            b[row, bcol] |= (val << pos)

    return b


def _unpack(p, bitdepth, width):
    powers = _np.arange(bitdepth-1, -1, -1)
    up = _np.unpackbits(p).reshape(p.shape[0], -1, bitdepth).dot(2**powers)
    a = up[:, :width]
    return a


def _validate_array(a):
    if a.ndim != 2:
        if a.ndim != 3 or a.shape[2] > 4 or a.shape[2] == 0:
            raise ValueError("array must be 2D, or 3D with shape "
                             "(m, n, d) with 1 <= d <= 4.")
    itemsize = a.dtype.itemsize
    if not _np.issubdtype(a.dtype, _np.unsignedinteger) or itemsize > 2:
        raise ValueError("array must be an array of 8- or 16-bit "
                         "unsigned integers")


def _get_color_type(a, use_palette):
    if a.ndim == 2:
        color_type = 0
    else:
        depth = a.shape[2]
        if depth == 1:
            # Grayscale
            color_type = 0
        elif depth == 2:
            # Grayscale and alpha
            color_type = 4
        elif depth == 3:
            # RGB
            if a.dtype == _np.uint8 and use_palette:
                # Indexed color (create a palette)
                color_type = 3
            else:
                # RGB colors
                color_type = 2
        elif depth == 4:
            # RGB and alpha
            if a.dtype == _np.uint8 and use_palette:
                color_type = 3
            else:
                color_type = 6
    return color_type


def _validate_bitdepth(bitdepth, a, color_type):
    if bitdepth is not None:
        if color_type != 0:
            raise ValueError('bitdepth may only be specified for grayscale '
                             'images with no alpha channel')
        if bitdepth not in [1, 2, 4, 8, 16]:
            raise ValueError('bitdepth %i is not valid.  Valid values are '
                             '1, 2, 4, 8 or 16' % (bitdepth,))
        if bitdepth == 16:
            if a.dtype != _np.uint16:
                raise ValueError('Input array must have dtype uint16 when '
                                 'bitdepth=16 is given.')
        else:
            if a.dtype != _np.uint8:
                raise ValueError('Input array must have dtype uint8 when '
                                 'bitdepth < 8 is given.')


def _validate_timestamp(timestamp):
    if timestamp is None:
        return None
    if len(timestamp) != 6:
        raise ValueError("timestamp must have length 6")
    return timestamp


def write_png(fileobj, a, text_list=None, use_palette=False,
              transparent=None,  bitdepth=None, max_chunk_len=None,
              timestamp=None, gamma=None):
    """
    Write a numpy array to a PNG file.

    Parameters
    ----------
    fileobj : string or file object
        If fileobj is a string, it is the name of the PNG file to be created.
        Otherwise fileobj must be a file opened for writing.
    a : numpy array
        Must be an array of 8- or 16-bit unsigned integers.  The shape of `a`
        must be (m, n) or (m, n, d) with 1 <= d <= 4.
    text_list : list of (keyword, text) tuples, optional
        Each tuple is written to the file as a 'tEXt' chunk.
    use_palette : bool, optional
        If True, *and* the data type of `a` is `numpy.uint8`, *and* the size
        of `a` is (m, n, 3), then a PLTE chunk is created and an indexed color
        image is created.  (If the conditions on `a` are not true, this
        argument is ignored and a palette is not created.)  There must not be
        more than 256 distinct colors in `a`.  If the conditions on `a` are
        true but the array has more than 256 colors, a ValueError exception
        is raised.
    transparent : integer or 3-tuple of integers (r, g, b), optional
        If the colors in `a` do not include an alpha channel (i.e. the shape
        of `a` is (m, n), (m, n, 1) or (m, n, 3)), the `transparent` argument
        can be used to specify a single color that is to be considered the
        transparent color.  This argument is ignored if `a` includes an
        alpha channel.
    bitdepth : integer, optional
        Bit depth of the output image.  Valid values are 1, 2, 4 and 8.
        Only valid for grayscale images with no alpha channel with an input
        array having dtype numpy.uint8.  If not given, the bit depth is
        inferred from the data type of the input array `a`.
    max_chunk_len : integer, optional
        The data in a PNG file is stored in records called IDAT chunks.
        `max_chunk_len` sets the maximum number of data bytes to stored in
        each IDAT chunk.  The default is None, which means that all the data
        is written to a single IDAT chunk.
    timestamp : tuple with length 6, optional
        If this argument is not None, a 'tIME' chunk is included in the
        PNG file.  The value must be a tuple of six integers: (year, month,
        day, hour, minute, second).
    gamma : float, optional
        If this argument is not None, a 'gAMA' chunk is included in the
        PNG file.  The argument is expected to be a floating point value.
        The value written in the 'gAMA' chunk is int(gamma*100000 + 0.5).

    Notes
    -----
    If `a` is three dimensional (i.e. `a.ndim == 3`), the size of the last
    dimension determines how the values in the last dimension are interpreted,
    as follows:

        a.shape[2]     Interpretation
        ----------     --------------------
            1          grayscale
            2          grayscale and alpha
            3          RGB
            4          RGB and alpha
    """

    _validate_array(a)

    _validate_text(text_list)

    timestamp = _validate_timestamp(timestamp)

    # Determine color_type:
    #
    #  color_type   meaning                    tRNS chunk contents (optional)
    #  ----------   ------------------------   --------------------------------
    #      0        grayscale                  Single gray level value, 2 bytes
    #      2        RGB                        Single RGB, 2 bytes per channel
    #      3        8 bit indexed RGB or RGBA  Series of 1 byte alpha values
    #      4        Grayscale and alpha
    #      6        RGBA

    color_type = _get_color_type(a, use_palette)

    trans = None
    if color_type == 3:
        # The array is 8 bit RGB or RGBA, and a palette is to be created.

        # Note that this replaces `a` with the index array.
        a, palette, trans = _palettize(a)
        # `a` has the same shape as before, but now it is an array of indices
        # into the array `palette`, which contains the colors.  `trans` is
        # either None (if there was no alpha channel), or an array the same
        # length as `palette` containing the alpha values of the colors.
        if len(palette) > 256:
            raise ValueError("The array has %d colors.  No more than 256 "
                             "colors are allowed when using a palette." %
                             len(palette))

        if trans is None and transparent is not None:
            # The array does not have an alpha channel.  The caller has given
            # a color value that should be considered to be transparent.
            # We construct an array `trans` of alpha values, and set the
            # alpha of the color that is to be transparent to 0.  All other
            # alpha values are set to 255 (fully opaque).
            # `trans` only has entries for colors in the palette up to the
            # given `transparent` color, so `trans` is not the same length as
            # `palette` (unless the transparent color happens to be the last
            # color in the pallete).
            pal_index = _np.nonzero((palette == transparent).all(axis=1))[0]
            if pal_index.size > 0:
                if pal_index.size > 1:
                    raise ValueError("Only one transparent color may "
                                     "be given.")
                trans = _np.zeros(pal_index[0]+1, dtype=_np.uint8)
                trans[:-1] = 255

    elif (color_type == 0 or color_type == 2) and transparent is not None:
        # XXX Should do some validation of `transparent`...
        trans = _np.asarray(transparent, dtype=_np.uint16)

    if bitdepth == 8 and a.dtype == _np.uint8:
        bitdepth = None

    _validate_bitdepth(bitdepth, a, color_type)

    if hasattr(fileobj, 'write'):
        # Assume it is a file-like object with a write method.
        f = fileobj
    else:
        # Assume it is a filename.
        f = open(fileobj, "wb")

    # Write the PNG header.
    png_header = b"\x89PNG\x0D\x0A\x1A\x0A"
    f.write(png_header)

    # Write the chunks...

    # IHDR chunk
    if bitdepth is not None:
        nbits = bitdepth
    else:
        nbits = a.dtype.itemsize*8
    _write_ihdr(f, a.shape[1], a.shape[0], nbits, color_type)

    # tEXt chunks, if any.
    if text_list is not None:
        for keyword, text_string in text_list:
            _write_text(f, keyword, text_string)

    if timestamp is not None:
        _write_time(f, timestamp)

    if gamma is not None:
        _write_gama(f, gamma)

    # PLTE chunk, if requested.
    if color_type == 3:
        _write_plte(f, palette)

    # tRNS chunk, if there is one.
    if trans is not None:
        _write_trns(f, trans)

    # _write_data(...) writes the IDAT chunk(s).
    _write_data(f, a, bitdepth, max_chunk_len=max_chunk_len)

    # IEND chunk
    _write_iend(f)

    if f != fileobj:
        f.close()


def _msec_to_numden(delay):
    """
    delay is the time delay in milliseconds.

    Return value is the tuple (delay_num, delay_den) representing
    the delay in seconds as the fraction delay_num/delay_den.
    Each value in the tuple is an integer less than 65536.
    """
    if delay == 0:
        return (0, 1)
    # Convert delay to seconds.
    delay = delay/1000.0
    if delay > 1:
        f = Fraction.from_float(1.0/delay).limit_denominator(65535)
        num = f.denominator
        den = f.numerator
    else:
        f = Fraction.from_float(delay).limit_denominator(65535)
        num = f.numerator
        den = f.denominator
    if (num, den) == (1, 0):
        raise ValueError("delay=%r is too large to convert to "
                         "delay_num/delay_den" % (delay,))
    if (num, den) == (0, 1):
        raise ValueError("delay=%r is too small to convert to "
                         "delay_num/delay_den" % (delay,))
    return num, den


def write_apng(fileobj, seq, delay=None, num_plays=0, include_first_frame=True,
               text_list=None, use_palette=False,
               transparent=None, bitdepth=None,
               max_chunk_len=None, timestamp=None, gamma=None):
    """
    Write an APNG file from a sequence of numpy arrays.

    Warning:
    * This API is experimental, and will likely change.
    * The function has not been thoroughly tested.

    Parameters
    ----------
    seq : sequence of numpy arrays
        All the arrays must have the same shape and dtype.
    delay : scalar, optional
        The time delay between frames, in milliseconds.
        If `delay` is None (the default) or 0, the frames are played as
        fast as possible.
    num_plays : int
        The number of times to repeat the animation.  If 0, the animation
        is repeated indefinitely.
    include_first_frame : bool
        The first frame is the image that is displayed by a renderer that does
        not support APNG.  If `include_first_frame` is True, the first frame
        in `seq` is also included in the animation.  If it is False, the first
        frame is not part of the animation, so it will only be seen by users
        of renderers that do not support APNG.
    text_list : list of (keyword, text) tuples, optional
        Each tuple is written to the file as a 'tEXt' chunk.
    use_palette : bool, optional
        If True, *and* the data type of the arrays in `seq` is `numpy.uint8`,
        *and* the size of each array is (m, n, 3), then a PLTE chunk is
        created and an indexed color image is created.  (If the conditions
        on the arrays are not true, this argument is ignored and a palette
        is not created.)  There must not be more than 256 distinct colors in
        the arrays.  If the above conditions are true but the arrays have
        more than 256 colors, a ValueError exception is raised.
    transparent : integer or 3-tuple of integers (r, g, b), optional
        If the colors in the input arrays do not include an alpha channel
        (i.e. the shape of each array is (m, n), (m, n, 1) or (m, n, 3)),
        the `transparent` argument can be used to specify a single color that
        is to be considered the transparent color.  This argument is ignored
        if the arrays have an alpha channel.
    bitdepth : integer, optional
        Bit depth of the output image.  Valid values are 1, 2, 4 and 8.
        Only valid for grayscale images with no alpha channel with an input
        array having dtype numpy.uint8.  If not given, the bit depth is
        inferred from the data type of the input arrays.
    max_chunk_len : integer, optional
        The data in a APNG file is stored in records called IDAT and fdAT
        chunks.  `max_chunk_len` sets the maximum number of data bytes to
        stored in each chunk.  The default is None, which means that all the
        data from a frame is written to a single IDAT or fdAT chunk.
    timestamp : tuple with length 6, optional
        If this argument is not None, a 'tIME' chunk is included in the
        PNG file.  The value must be a tuple of six integers: (year, month,
        day, hour, minute, second).
    gamma : float, optional
        If this argument is not None, a 'gAMA' chunk is included in the
        PNG file.  The argument is expected to be a floating point value.
        The value written in the 'gAMA' chunk is int(gamma*100000 + 0.5).
    """
    num_frames = len(seq)
    if num_frames == 0:
        raise ValueError("no frames given in `seq`")

    if delay is None:
        delay = 0

    if type(seq) == _np.ndarray:
        # seq is a single numpy array containing the frames.
        _validate_array(seq[0])
    else:
        # seq is not a numpy array, so it must be a sequence of numpy arrays,
        # all with the same dtype and shape.
        for a in seq:
            _validate_array(a)
        if any(a.dtype != seq[0].dtype for a in seq[1:]):
            raise ValueError("all arrays in `seq` must have the same dtype.")
        if any(a.shape != seq[0].shape for a in seq[1:]):
            raise ValueError("all arrays in `seq` must have the same shape.")

    _validate_text(text_list)

    timestamp = _validate_timestamp(timestamp)

    # Get the array for the first frame.
    a = seq[0]

    color_type = _get_color_type(a, use_palette)

    trans = None
    if color_type == 3:
        # The array is 8 bit RGB or RGBA, and a palette is to be created.

        # Put all the input arrays into a single numpy array, so we have
        # a single array to pass to _palettize().
        seq = _np.array(seq)
        seq, palette, trans = _palettize(seq)
        # seq has the same shape as before, but now it is an array of indices
        # into the array `palette`, which contains the colors.  `trans` is
        # either None (if there was no alpha channel), or an array the same
        # length as `palette` containing the alpha values of the colors.
        a = seq[0]
        if len(palette) > 256:
            raise ValueError("The input has %d colors.  No more than 256 "
                             "colors are allowed when using a palette." %
                             len(palette))

        if trans is None and transparent is not None:
            # The array does not have an alpha channel.  The caller has given
            # a color value that should be considered to be transparent.
            pal_index = _np.nonzero((palette == transparent).all(axis=1))[0]
            if pal_index.size > 0:
                if pal_index.size > 1:
                    raise ValueError("Only one transparent color may "
                                     "be given.")
                trans = _np.zeros(pal_index[0]+1, dtype=_np.uint8)
                trans[:-1] = 255

    elif (color_type == 0 or color_type == 2) and transparent is not None:
        # XXX Should do some validation of `transparent`...
        trans = _np.asarray(transparent, dtype=_np.uint16)

    if bitdepth == 8 and a.dtype == _np.uint8:
        bitdepth = None

    _validate_bitdepth(bitdepth, a, color_type)

    # --- Open and write the file ---------

    if hasattr(fileobj, 'write'):
        # Assume it is a file-like object with a write method.
        f = fileobj
    else:
        # Assume it is a filename.
        f = open(fileobj, "wb")

    # Write the PNG header.
    png_header = b"\x89PNG\x0D\x0A\x1A\x0A"
    f.write(png_header)

    # Write the chunks...

    # IHDR chunk
    if bitdepth is not None:
        nbits = bitdepth
    else:
        nbits = a.dtype.itemsize*8
    _write_ihdr(f, a.shape[1], a.shape[0], nbits, color_type)

    # tEXt chunks, if any.
    if text_list is not None:
        for keyword, text_string in text_list:
            _write_text(f, keyword, text_string)

    if timestamp is not None:
        _write_time(f, timestamp)

    if gamma is not None:
        _write_gama(f, gamma)

    # PLTE chunk, if requested.
    if color_type == 3:
        _write_plte(f, palette)

    # tRNS chunk, if there is one.
    if trans is not None:
        _write_trns(f, trans)

    # acTL chunk
    _write_actl(f, num_frames, num_plays)

    # Convert delay (which is milliseconds) to the number of
    # seconds expressed as the fraction delay_num/delay_den.
    delay_num, delay_den = _msec_to_numden(delay)

    sequence_number = 0

    if include_first_frame:
        # fcTL chunk for the first frame
        _write_fctl(f, sequence_number=sequence_number,
                    width=a.shape[1], height=a.shape[0],
                    x_offset=0, y_offset=0,
                    delay_num=delay_num, delay_den=delay_den,
                    dispose_op=0, blend_op=1)
        sequence_number += 1

    # IDAT chunk(s) for the first frame (no sequence_number)
    _write_data(f, a, bitdepth, max_chunk_len=max_chunk_len)

    for a in seq[1:]:
        # fcTL chunk for the next frame
        _write_fctl(f, sequence_number=sequence_number,
                    width=a.shape[1], height=a.shape[0],
                    x_offset=0, y_offset=0,
                    delay_num=delay_num, delay_den=delay_den,
                    dispose_op=0, blend_op=1)
        sequence_number += 1

        # fdAT chunk(s) for the next frame
        num_chunks = _write_data(f, a, bitdepth, max_chunk_len=max_chunk_len,
                                 sequence_number=sequence_number)
        sequence_number += num_chunks

    # IEND chunk
    _write_iend(f)

    if f != fileobj:
        f.close()
