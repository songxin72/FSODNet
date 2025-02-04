# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 16:50:27 2022

@author: song'xin
"""

import matplotlib.pyplot as plt
import numpy as np

# plt.imshow(img) img形状（h,w,c）

# 格式 (1, 128, 450, 620)
x = np.load('./tno_show/w1.npy')
x = x[0]

# visualize the first conv layer filters
# plt.figure(figsize=(35, 35))
# for i, filter in enumerate(x):
# plt.figure(figsize=(8, 8))
# plt.subplot(8, 8, i+1) # we have 5x5 filters and total of 16 (see printed shapes)
# plt.imshow(filter, cmap='gray')  # seismic, viridis
# plt.axis('off')
# plt.savefig('filter1.png')
# plt.show()

# n_feaures = x.shape[0]
# images_per_row = 10
# n_cols = n_feaures // images_per_row
# size = x.shape[1]
# grid = np.zeros((size * n_cols, images_per_row * size))
# for col in range(n_cols):
#     for row in range(images_per_row):
#         channel = x[col * images_per_row + row, :, :]
#         grid[col * size:(col + 1) * size, row * size:(row + 1) * size] = channel

# scale = 1. / size

for i in range(0, x.shape[0]):
    grid = x[i]
    plt.figure(figsize=(grid.shape[1]/100, grid.shape[0]/100))  # scale * grid.shape[1], scale * grid.shape[0]
    plt.grid(False)
    plt.imshow(grid, cmap='viridis')