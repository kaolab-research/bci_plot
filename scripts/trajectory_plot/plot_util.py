import numpy as np
import data_util
import matplotlib.pyplot as plt
import pathlib
import argparse
import ast
import re

def get_readme(data_path):
    """ converts readme.txt into a dictionary """

    readme_file_path = data_path / 'README.txt'

    # Initialize an empty dictionary to store key-value pairs
    readme_data = {}

    with open(readme_file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        # Split each line into key and value using the ':' as a separator
        split_result = line.rsplit(':', 1) # doing rsplit instead of split would be a better option

        # check if the split result has at least two elements
        if len(split_result) >= 2:
            key, value = map(str.strip, split_result)
            readme_data[key] = value

    return readme_data

def get_target_pos_dia(data_path):
    """ extracts target position and diameter from the readme.txt """
    readme = get_readme(data_path)
    array = np.array # for eval to work
    # Extract target_pos and target_size
    target_pos = eval(readme['Target info'])[0][0]
    target_dia = eval(readme['Target info'])[1][1]

    if target_pos == -0.7 or target_pos == 0.7:
        # Define positions for 8 targets
        positions = [
            (-0.7, 0),    # Left
            (0.7, 0),     # Right
            (0, 0.7),     # Up
            (0, -0.7),    # Down
            (-0.495, 0.495),  # LeftUp
            (-0.495, -0.495), # LeftDown
            (0.495, 0.495),   # RightUp
            (0.495, -0.495)   # RightDown
        ]

    return positions, target_dia