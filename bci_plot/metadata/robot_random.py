

resample_start_end = {
     'exp_2024-03-18_12-59-19': {
         'exp_2024-03-18_12-34-06.log': [(26824, 228607)],
         'exp_2024-03-18_12-44-36.log': [(23827, 184485)],
         'exp_2024-03-18_13-00-03.log': [
             (24707, 153713),
             (206799, 314380),
         ],
     },
    'exp_2024-05-01_12-21-58': {
        'exp_2024-05-01_12-25-05.log': [(33086, 216896)],
        'exp_2024-05-01_12-35-34.log': [(25611, 227160)],
        'exp_2024-05-01_12-45-59.log': [(27524, 427245)],
    },
}

trajectory_info = ('S2', 'exp_2024-03-18_12-59-19', 'exp_2024-03-18_13-00-03.log', 1)
trajectory_offset = 206799
trajectory_split = [item - trajectory_offset for item in [206799, 214583, 230101, 235195, 248181, 277478, 286660, 294148, 314380]]

# aligned to the robot_logs file
sequence_start_end = {
    # S2
    'S2': {
        'exp_2024-03-18_12-59-19': {
            'exp_2024-03-18_12-34-06.log': [0],
            'exp_2024-03-18_12-44-36.log': [0],
            'exp_2024-03-18_13-00-03.log': [0, 1],
        },
        'exp_2024-05-01_12-21-58': {
            'exp_2024-05-01_12-25-05.log': [0],
            'exp_2024-05-01_12-35-34.log': [0],
            'exp_2024-05-01_12-45-59.log': [0],
        },
    },
    
    # H1
    'H1': {
        'exp_2024-06-18_14-29-38': {
            'exp_2024-06-18_14-30-06.log': [0, 1],
            'exp_2024-06-18_14-43-05.log': [],
        },
        'exp_2024-06-18_14-47-00': {
            'exp_2024-06-18_14-47-22.log': [0],
            'exp_2024-06-18_14-51-01.log': [0, 1],
        },

        'exp_2024-06-18_15-16-07': {
            'exp_2024-06-18_15-16-44.log': [0, 1, 2, 3, 4, 6, 7],
            'exp_2024-06-18_15-43-19.log': [2, 3, 4],
        },
    },

    
    # H4
    'H4': {
        'exp_2024-07-11_14-09-52': {
            'exp_2024-07-11_14-10-12.log': [0, 1, 2, 3],
        },
        'exp_2024-07-11_14-34-14': {
            'exp_2024-07-11_14-34-24.log': [0, 1, 2, 4, 5],
            'exp_2024-07-11_14-59-33.log': [0],
        },
    },
    
    # H2
    'H2': {
        'exp_2024-07-16_15-02-42': {
            'exp_2024-07-16_15-03-50.log': [0, 1, 2, 3, 4],
            'exp_2024-07-16_15-41-45.log': [0, 1, 2, 3, 4],
        }, 
    },
    
    
}

# 0: error or missing (at end)
# 1: correct
# 2: error correction
# 3: out of order correct
sequence_actions = {
    'H1': [
        ('yPbOgYrB', 'yPbOgYrB', '11111111'),
        ('bOyYgPrB', 'bOyYgPrB', '11111111'),
        ('bPgOyBrY', 'bPgOyBrY', '11111111'),
        ('bOyYrBgP', 'bOyYrBgP', '11111111'),
        ('bByOrPgY', 'bByOrPgY', '11111111'),
        ('gOrByPbY', 'gOrByPbY', '11111111'),
        ('bYyOrBgP', 'bYyOrBgP', '11111111'),
        ('rYgOyPbB', 'rYgOyPbB', '11111111'),
        ('bYrPyBgO', 'bYrPyByBgO', '1111110211'),
        ('rOgPyBbY', 'rOgPyBbY', '11111111'),
        ('bYyOrBgP', 'bYyOrBgP', '11111111'),
        ('rYyBgObP', 'rYyBgObP', '11111111'),
        ('bYgOyPrB', 'bYgOyPrB', '11111111'),
        ('yPbYgOrB', 'yPbYgOrB', '11111111'),
        ('bPrByYgO', 'bPrByYgO', '11111111'),
    ],
    
    'H4': [
        ('bPrByYgO', 'bbbPrByYgO', '1021111111'),
        ('rPyObBgY', 'rPyObBgggY', '1111111021'),
        ('gYrBbPyO', 'gYrPrBbPyO', '1110211111'),
        ('yOgPbBrY', 'yOgPbBrY', '11111111'),
        ('bByPrYgO', 'bByPrYgO', '11111111'),
        ('bOyPrBgY', 'bOyPrBgY', '11111111'),
        ('yOrYgBbP', 'yOyOrYgBbP', '1102111111'),
        ('gOyPbYrB', 'gOyPrrbYrB', '1111021111'),
        ('gYrBbOyP', 'gggggYrBbObOyP', '10202111110211'),
        ('rOyPbYgB', 'rOyPbYgB', '11111111'),
    ],
    
    'H2': [
        ('gOyBrPbY', 'gOyBrPbY', '11111111'),
        ('yBgPrObY', 'yBgPrObY', '11111111'),
        ('yObBgYrP', 'yObBgYrP', '11111111'),
        ('gOrPyBbY', 'gOrrrrrPyBbY', '111020211111'),
        ('rYgByPbO', 'rYgByPbbbO', '1111111021'),
        ('rPbOyYgB', 'rPbOyYgB', '11111111'),
        ('gYrObPyB', 'gYrObPyB', '11111111'),
        ('bOgYyBrP', 'bbbbbOgYyBrP', '102021111111'),
        ('rPyYbBgO', 'rPyYbBgO', '11111111'),
        ('bYrBgOyP', 'bBbYrBgOyP', '1021111111'),
    ],
    
    'S2': [
        ('yYrBbOgP', 'yYrrrOrBbPbO', '11102021102100'),
        ('bPyBrYgO', 'bObPgOyBrY', '1021031111'),
        ('rYgObByP', 'rYbBgggOyP', '1103102111'),
        ('yObPrBgY', 'yObPrBgY', '11111111'),
        ('bOrYyPgB', 'rYgBgByPbO', '0303020311'),
        ('yYbPrOgB', 'yYgOrrgBbPrO', '110002221111'),
        ('rPgYbOyB', 'rPgggYgYbbbbbObO', '111021021020210200'),
    ]
}
