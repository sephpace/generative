
import math
import os

import matplotlib.pyplot as plt

SAVE_DIR = os.path.abspath('analysis')


def visualize_input_output(io, **kwargs):
    """
    Displays the given input and output images from the autoencoder.

    Args:
        io (list):       A list of tuples containing the input image data and output image data.

    Keyword Args:
        figsize (tuple): The figure size.
        rows (int):      The amount of rows in the figure (default: half of io size rounded up).
        cols (int):      The amount of columns in the figure (default: 4, two for each input and output).
        save (bool):     If True, the figure will be saved in the analysis folder.
        file_name (str): The name of the saved image file.
    """
    figsize = kwargs.get('figsize', (8, 8))
    rows = kwargs.get('rows', math.ceil(len(io) / 2))
    cols = kwargs.get('cols', 4)
    save = kwargs.get('save', False)
    file_name = kwargs.get('file_name', 'io_visualization.png')

    fig = plt.figure(figsize=figsize)
    for i, (x, y) in enumerate(io):
        if 0 <= i < cols // 2:
            fig.add_subplot(rows, cols, i * 2 + 1, title='input')
            plt.imshow(x)
            fig.add_subplot(rows, cols, i * 2 + 2, title='output')
            plt.imshow(y)
        else:
            fig.add_subplot(rows, cols, i * 2 + 1)
            plt.imshow(x)
            fig.add_subplot(rows, cols, i * 2 + 2)
            plt.imshow(y)
    if save:
        plt.savefig(os.path.join(SAVE_DIR, file_name))
    plt.show()
