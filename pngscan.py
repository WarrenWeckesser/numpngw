"""
The function pngscan reads a PNG file and prints some of the information
in the file to stdout.  It is used for testing and debugging.
"""

from __future__ import print_function, division

import struct
import zlib


color_types = {0: "grayscale",
               2: "RGB",
               3: "indexed color",
               4: "grayscale+alpha",
               6: "RBG+alpha"}

interlaces = {0: "no interlace",
              1: "Adam7"}

png_signature = b'\x89PNG\x0D\x0A\x1A\x0A'


def pngscan(filename, chunk_info_only=False, print_palette=False,
            print_trans=False):
    """
    Print some of the information found in a PNG file.

    This function is used for testing and debugging.
    """
    with open(filename, 'rb') as f:
        start = f.read(8)
        if start != png_signature:
            print("ERROR: First 8 bytes %r are not the standard PNG signature."
                  % (start,))
        # compressed_contents will hold the contents of the IDAT chunk(s).
        compressed_contents = []
        while True:
            lenstr = f.read(4)
            if len(lenstr) == 0:
                break
            chunk_type = f.read(4)
            length = struct.unpack("!I", lenstr)[0]
            content = f.read(length)
            crcstr = f.read(4)
            crc = struct.unpack("!I", crcstr)[0]
            check = zlib.crc32(chunk_type + content) & 0xFFFFFFFF
            if chunk_info_only:
                print("length: %5i   chunk_type: %s  CRC: %11i  (%i)" %
                      (length, chunk_type, crc, check))
                continue

            if crc != check:
                raise RuntimeError("CRC mismatch in chunk type %r" %
                                   (chunk_type,))

            if chunk_type == b'IHDR':
                fmt = "!IIBBBBB"
                values = struct.unpack(fmt, content)
                (width, height, nbits, color_type, compression, _,
                    interlace) = values
                print("%s: width=%i  height=%i  nbits=%i  "
                      "color_type=%i (%s)  interlace=%i (%s)" %
                      (chunk_type, width, height, nbits,
                       color_type, color_types[color_type],
                       interlace, interlaces[interlace]))
                if compression != 0:
                    print("ERROR: Unknown compression method %i" % compression)
            elif chunk_type == b'acTL':
                num_frames, num_plays = struct.unpack("!II", content)
                print("%s: num_frames=%i  num_plays=%i" %
                      (chunk_type, num_frames, num_plays))
            elif chunk_type == b'fcTL':
                fmt = "!IIIIIHHBB"
                values = struct.unpack(fmt, content)
                (sequence_number, width, height, x_offset, y_offset,
                 delay_num, delay_den, dispose_op, blend_op) = values
                print(("%s: sequence_number=%i  width=%i  height=%i  "
                       "x_offset=%i  y_offset=%i  delay_num=%i delay_den=%i") %
                      (chunk_type, sequence_number, width, height,
                       x_offset, y_offset, delay_num, delay_den))
            elif chunk_type == b'IDAT':
                compressed_contents.append(content)
                descr = ">u" + ('1' if nbits == 8 else '2')
                # if color_type == 2:
                #     row_data_len = 3*width
                # else:
                #     row_data_len = width
                # row = np.frombuffer(u, dtype=descr).reshape(height,
                #                                             row_data_len+1)
                # if color_type == 2:
                #     row = row[:, 1:].reshape(height, width, 3)
                print("%s: (%i bytes)" % (chunk_type, len(content)))
                # print(row)
            elif chunk_type == b'fdAT':
                seqstr = content[:4]
                sequence_number = struct.unpack("!I", seqstr)[0]
                u = zlib.decompress(content[4:])
                descr = ">u" + ('1' if nbits == 8 else '2')
                # if color_type == 2:
                #     row_data_len = 3*width
                # else:
                #     row_data_len = width
                # row = np.frombuffer(u, dtype=descr).reshape(height,
                #                                             row_data_len+1)
                # if color_type == 2:
                #     row = row[:, 1:].reshape(height, width, 3)
                print("%s: sequence_number=%i (%i bytes)" %
                      (chunk_type, sequence_number, len(content)))
                # print(row)
            elif chunk_type == b'bKGD':
                if color_type == 0 or color_type == 4:
                    value = struct.unpack('!H', content)[0]
                    descr = "grayscale"
                elif color_type == 2 or color_type == 6:
                    value = struct.unpack('!HHH', content)
                    descr = "RGB"
                else:
                    value = struct.unpack('B', content)[0]
                    descr = "index"
                print("%s: value=%r (%s)" % (chunk_type, value, descr))
            elif chunk_type == b'pHYs':
                values = struct.unpack('!IIB', content)
                print("%s: pixels per unit: x=%i  y=%i;  unit=%i" %
                      ((chunk_type,) + values))
            elif chunk_type == b'tIME':
                values = struct.unpack('!HBBBBB', content)
                print("%s: year=%i  month=%i  day=%i  "
                      "hour=%i  minute=%i  second=%i" %
                      ((chunk_type,) + values))
            elif chunk_type == b'sBIT':
                # Mapping from color_type to required length of sbit:
                len_sbit = {0: 1, 2: 3, 3: 3, 4: 2, 6: 4}[color_type]
                values = struct.unpack('BBBB'[:len_sbit], content)
                print('%s: %s' % (chunk_type, values,))
            elif chunk_type == b'gAMA':
                gama = struct.unpack('!I', content)[0]
                print("%s: gama=%i  (%g)" % (chunk_type, gama, gama/100000.))
            elif chunk_type == b'cHRM':
                chrm = struct.unpack('!IIIIIIII', content)
                print('%s: ' % (chunk_type,), end='')
                print('white point:', chrm[:2], end='  ')
                print('red:', chrm[2:4], end='  ')
                print('green:', chrm[4:6], end='  ')
                print('blue:', chrm[6:])
            elif chunk_type == b'tRNS':
                if color_type == 0:
                    value = struct.unpack("!H", content)[0]
                    msg = "value=%d" % value
                elif color_type == 2:
                    value = struct.unpack("!HHH", content)
                    msg = "value=%r" % (value,)
                elif color_type == 3:
                    values = tuple(content)
                    if print_trans:
                        msg = ",".join("%i" % t for t in values)
                    else:
                        msg = "(%i indexed alpha values)" % len(values)
                else:
                    msg = ("ERROR: Found a tRNS chunk, but color_type=%r" %
                           (color_type,))
                print("%s: %s" % (chunk_type, msg))
            elif chunk_type == b'PLTE':
                print("%s: " % (chunk_type,), end='')
                plte = [struct.unpack("BBB", content[3*k:3*(k+1)])
                        for k in range(len(content)//3)]
                if print_palette:
                    print("colors:")
                    for k, clr in enumerate(plte):
                        print("     %3i (%3i, %3i, %3i)" % ((k,) + clr))
                else:
                    print("%s colors" % len(plte))
            elif chunk_type == b'tEXt':
                print("%s: " % (chunk_type,), end='')
                if b'\0' not in content:
                    print("ERROR!  content is missing a null character")
                else:
                    keyword, text = content.split(b'\x00', 1)
                    if len(keyword) > 79:
                        print("ERROR! len(keyword) > 79")
                    else:
                        print("keyword: %r  " % keyword)
                    print("         text: %r" % text)
            else:
                print("%s: (%i bytes)" % (chunk_type, len(content)))

        if len(compressed_contents) == 0:
            print("ERROR: No IDAT chunks found.")
        compressed_datastream = b''.join(compressed_contents)
        print("Compressed image datastream has", len(compressed_datastream),
              "bytes.")
        try:
            u = zlib.decompress(compressed_datastream)
        except Exception:
            print("ERROR: Failed to decompress the image data.")
        else:
            print("Decompressed image datastream has", len(u), " bytes.")
