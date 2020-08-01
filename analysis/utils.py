
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


class Logger:
    """
    Prints data to stdout and writes it into a log file in the analysis folder.

    Attributes:
        __last (int):           The previous amount of bytes written to the log.
        __total (int):          The total amount of bytes written to the log.
        file (TextIOWrapper):   The log file to write the data to.
        template (LogTemplate): The template for each line in the log.
    """

    def __init__(self, file_name, template):
        """
        Constructor.

        Args:
            file_name (str):        The name of the log file (e.g. 'train.log').
            template (LogTemplate): The template for each line in the log.
        """
        self.__last = 0
        self.__total = 0

        self.file = open(os.path.join(SAVE_DIR, file_name), 'w')
        self.template = template

    def __del__(self):
        """
        Closes the log file when the logger is deleted.
        """
        self.close()

    def close(self):
        """
        Closes the log file.
        """
        self.file.close()

    def write(self, ctx, overwrite=False):
        """
        Writes the given data to the log file.

        Args:
            ctx (dict):                 The context data to write to the log.
            overwrite (Optional[bool]): True if the previous line should be overwritten. Defaults to False.
        """
        data = self.template.render(**ctx)
        if overwrite:
            print(data, end='\r')
            self.file.seek(self.__total - self.__last)
            self.__total -= self.__last
        else:
            print(data)
        self.__last = self.file.write(f'{data}\n')
        self.__total += self.__last


class LogTemplate:
    """
    Contains a template string for variable mapping.

    Attributes:
        template_str (str): A template string to map variables to.
    """

    def __init__(self, template_str):
        """
        Constructor.

        Args:
            template_str (str): A template string to map variables to.
        """
        self.template_str = template_str

    def render(self, **ctx):
        """
        Renders the template with the given variables.

        Args:
            **ctx (dict): Any key word arguments to be rendered in the template.
        """
        return self.template_str.format_map(ctx)


class TestingLogTemplate(LogTemplate):
    """
    A log template with default template str for model testing.
    """

    def __init__(self):
        """
        Constructor.
        """
        template_str = 'Item: {step}/{data_len}  Loss: {loss:.4f}'
        LogTemplate.__init__(self, template_str)

    def render(self, step, loss, data_len):
        """
        Renders the template with the given variables.

        Args:
            step (int):     The current step in the epoch.
            loss (int):     The current loss.
            data_len (int): The length of the dataset.
        """
        ctx = {
            'step': step,
            'loss': loss,
            'data_len': data_len,
        }
        return LogTemplate.render(self, **ctx)


class TrainingLogTemplate(LogTemplate):
    """
    A log template with default template str for model training.
    """

    def __init__(self):
        """
        Constructor.
        """
        template_str = 'Epoch: {epoch:{epochs_str_len}d}/{epochs}  Item: {step}/{data_len}  Loss: {loss:.4f}'
        LogTemplate.__init__(self, template_str)

    def render(self, epoch, epochs, step, loss, data_len):
        """
        Renders the template with the given variables.

        Args:
            epoch (int):    The current epoch.
            epochs (int):   The total number of epochs.
            step (int):     The current step in the epoch.
            loss (int):     The current loss.
            data_len (int): The length of the dataset.
        """
        ctx = {
            'epoch': epoch,
            'epochs': epochs,
            'epochs_str_len': len(str(epochs)),
            'step': step,
            'loss': loss,
            'data_len': data_len,
        }
        return LogTemplate.render(self, **ctx)
