# MIT License

# Copyright (c) 2021 alxyok

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np

# no condition, no loop, pure matrix.

def grid_2d_connectivity_matrix(shape: tuple) -> np.ndarray:
    
    x_dim, y_dim = shape
    
    array = np.reshape(np.arange(x_dim * y_dim), (x_dim, y_dim))
    
    columnar = (-1, 1)
    
    left = np.reshape(arr[..., :-1], columnar)
    right = np.reshape(arr[..., 1:], columnar)
    up = np.reshape(arr[:-1, ...], columnar)
    left = np.reshape(arr[1:, ...], columnar)
    
    connectivity = np.concatenate(
        (np.hstack((left, right)),
         np.hstack((right, left)),
         np.hstack((up, down)),
         np.hstack((down, up))),)
    
    return connectivity


def grid_3d_connectivity_matrix(shape: tuple) -> np.ndarray:
    
    x_dim, y_dim, z_dim = shape
    
    array = np.reshape(np.arange(x_dim * y_dim * z_dim), (x_dim, y_dim, z_dim))
    
    columnar = (-1, 1)
    
    left = np.reshape(array[..., :-1], columnar)
    right = np.reshape(array[..., 1:], columnar)
    up = np.reshape(array[:, :-1, :], columnar)
    down = np.reshape(array[:, 1:, :], columnar)
    sup = np.reshape(array[1:, ...], columnar)
    inf = np.reshape(array[:-1, ...], columnar)
    
    connectivity = np.concatenate(
        (np.hstack((left, right)),
         np.hstack((right, left)),
         np.hstack((up, down)),
         np.hstack((down, up)),
         np.hstack((sup, inf)),
         np.hstack((inf, sup))),)
    
    return connectivity