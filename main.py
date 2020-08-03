
import argparse
from importlib import import_module
import os

ROOT_DIR = os.path.abspath('')


def get_module(path, model_name):
    """
    Returns the python module for the model with the given name.

    Args:
        path (str):       The path to the directory containing the model.
        model_name (str): The name of the model.

    Returns:
        (module): The module for the model with the given name.
    """
    path = path.replace('/', '.')
    module = import_module(f'{path}.{model_name}')
    return module


def train(model_name):
    """
    Calls the train function for the given module.

    Args:
        model_name (str): The name of the model to train.
    """
    module = get_module('training', model_name)
    module.train()


def test(model_name):
    """
    Calls the test function for the given module.

    Args:
        model_name (str): The name of the model to test.
    """
    module = get_module('testing', model_name)
    module.test()


if __name__ == '__main__':
    # Set up commands
    command_map = {
        'train': train,
        'test': test,
    }

    # Parse model choices
    files = os.listdir(os.path.join(ROOT_DIR, 'models'))
    files.remove('__init__.py')
    files.remove('__pycache__')
    models = [os.path.splitext(file)[0] for file in files]

    # Set up parser
    parser = argparse.ArgumentParser(description='Train or test a model.')
    parser.add_argument('command', choices=command_map.keys(), help='The command to call.')
    parser.add_argument('model_name', choices=models, help='The name of the module.')

    # Parse the arguments and call the command
    args = parser.parse_args()
    command = command_map.get(args.command, None)
    if command is None:
        parser.print_help()
    else:
        command(args.model_name)
