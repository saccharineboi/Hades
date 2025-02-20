# Copyright 2025 Omar Huseynov
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import chess
import chess.svg
import cairosvg
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io
import sys

if len(sys.argv) < 2:
    print(f'Usage: {sys.argv[0]} <SYMBOL>')
    sys.exit(0)

svg_string = chess.svg.piece(chess.Piece.from_symbol(sys.argv[1]))

png_image = cairosvg.svg2png(bytestring=svg_string.encode(), scale=2)

matplotlib.use('TkAgg')
img = mpimg.imread(io.BytesIO(png_image))
plt.imshow(img)
plt.axis('off')
plt.show()

