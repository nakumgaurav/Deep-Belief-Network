# from google.colab import drive
# drive.mount('/content/drive/')

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt


##### Some matplotlib custom settings for plotting good images
# plt.rcParams['axes.axisbelow'] = 'line'
plt.rcParams['axes.grid'] = False
# plt.rcParams['figure.figsize'] = [6.4, 4.8]
# plt.rcParams['figure.subplot.bottom'] = 0.11
plt.rcParams['image.cmap'] = 'gray'


#%matplotlib inline
# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2

# dataset - binary 3x3 images of vertical or horizontal lines (Line width = 1)
num_columns = 3
num_rows = 3

# vertical lines
vertical_lines = []
for image_id in range(num_columns):
    white_line_image = []   # line is represented by 1
    black_line_image = []   # line is represented by 0
    for row_index in range(num_rows):
        white_image_row = []
        black_image_row = []
        for col_index in range(num_columns):
            if image_id==col_index:
                white_image_row.append(1)
                black_image_row.append(0)
            else:
                white_image_row.append(0)
                black_image_row.append(1)
            white_line_image.append(white_image_row)
black_line_image.append(black_image_row)
    vertical_lines.append(white_line_image)
    vertical_lines.append(black_line_image)

# vertical_lines = np.array(vertical_lines)
# print(vertical_lines)

# horizontal lines - same procedure as vertical lines with rows and columns swapped. Then take transpose
horizontal_lines = []
for image_id in range(num_columns):
    white_line_image = []   # line is represented by 1
    black_line_image = []   # line is represented by 0
    for row_index in range(num_columns):
        white_image_row = []
        black_image_row = []
        for col_index in range(num_rows):
            if image_id==col_index:
                white_image_row.append(1)
                black_image_row.append(0)
            else:
                white_image_row.append(0)
                black_image_row.append(1)
            white_line_image.append(white_image_row)
black_line_image.append(black_image_row)
    horizontal_lines.append(np.transpose(white_line_image))
    horizontal_lines.append(np.transpose(black_line_image))

# horizontal_lines = np.array(horizontal_lines)
# print(horizontal_lines)

def show_images(lines_array):
    for i, arr in enumerate(lines_array):
        plt.imshow(np.array(arr))
        plt.show()

show_images(vertical_lines)
show_images(horizontal_lines)
