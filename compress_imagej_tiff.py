#!/usr/bin/env python
import sys
import tifffile 


in_file = sys.argv[1]
out_file = sys.argv[2]

with tifffile.TiffFile(in_file) as tiffin:
    tifffile.imwrite(out_file, tiffin.asarray(), imagej=True, metadata=tiffin.imagej_metadata, compression='zlib', compressionargs={'level':9})
