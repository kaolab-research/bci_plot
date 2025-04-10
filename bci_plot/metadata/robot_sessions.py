
logs_meta = {}

logs_meta['exp_2024-05-30_14-48-57'] = [
    'exp_2024-05-30_14-52-44.log',
    'exp_2024-05-30_14-53-28.log',
    'exp_2024-05-30_14-54-38.log',
    'exp_2024-05-30_14-58-37.log',
    'exp_2024-05-30_15-00-02.log',
    'exp_2024-05-30_15-05-31.log',
    'exp_2024-05-30_15-08-42.log',
    'exp_2024-05-30_15-10-13.log',
    'exp_2024-05-30_15-17-47.log',
    'exp_2024-05-30_15-23-20.log',
    'exp_2024-05-30_15-25-04.log',
    'exp_2024-05-30_15-26-22.log',
    'exp_2024-05-30_15-27-22.log',
    'exp_2024-05-30_15-29-49.log',
    'exp_2024-05-30_15-31-20.log',
    'exp_2024-05-30_15-35-55.log',
    'exp_2024-05-30_15-47-33.log',
    'exp_2024-05-30_15-52-54.log',
    'exp_2024-05-30_15-59-19.log',
    'exp_2024-05-30_16-03-37.log',
    'exp_2024-05-30_16-09-51.log',
    'exp_2024-05-30_16-14-58.log',
    'exp_2024-05-30_16-20-04.log',
    'exp_2024-05-30_16-24-46.log',
]

hover_ab_meta = {}

# H2 hover AB
hover_ab_meta['exp_2024-06-06_12-37-30'] = [
    'exp_2024-06-06_12-38-21.log',
    'exp_2024-06-06_12-47-53.log',
    'exp_2024-06-06_12-50-54.log',
    'exp_2024-06-06_12-54-25.log',
    'exp_2024-06-06_12-59-56.log',
    'exp_2024-06-06_13-13-25.log',
    'exp_2024-06-06_13-18-19.log',
    'exp_2024-06-06_13-26-34.log',
    'exp_2024-06-06_13-31-45.log',
]
# H1 hover AB
hover_ab_meta['exp_2024-06-10_14-41-45'] = [
    'exp_2024-06-10_14-43-05.log',
    'exp_2024-06-10_14-50-01.log',
    'exp_2024-06-10_14-55-49.log',
    'exp_2024-06-10_15-00-54.log',
    'exp_2024-06-10_15-08-18.log',
    'exp_2024-06-10_15-13-55.log',
    'exp_2024-06-10_15-19-40.log',
    'exp_2024-06-10_15-28-00.log',
]
hover_ab_meta['exp_2024-06-10_15-53-08'] = [
    'exp_2024-06-10_15-53-53.log',
    'exp_2024-06-10_16-01-07.log',
    'exp_2024-06-10_16-05-57.log',
    'exp_2024-06-10_16-10-48.log',
    'exp_2024-06-10_16-19-07.log',
    'exp_2024-06-10_16-26-47.log',
    'exp_2024-06-10_16-32-13.log',
    'exp_2024-06-10_16-40-07.log',
]

# H4 hover AB
hover_ab_meta['exp_2024-06-13_15-08-39'] = [
    'exp_2024-06-13_15-09-31.log',
    'exp_2024-06-13_15-14-02.log',
    'exp_2024-06-13_15-23-01.log',
    'exp_2024-06-13_15-28-25.log',
    'exp_2024-06-13_15-36-55.log',
    'exp_2024-06-13_15-44-29.log',
    'exp_2024-06-13_15-49-58.log',
    'exp_2024-06-13_15-56-12.log',
]
hover_ab_meta['exp_2024-06-13_16-05-13'] = [
    'exp_2024-06-13_16-05-59.log',
    'exp_2024-06-13_16-10-18.log',
    'exp_2024-06-13_16-19-12.log',
    'exp_2024-06-13_16-23-49.log',
    'exp_2024-06-13_16-30-58.log',
    'exp_2024-06-13_16-38-54.log',
    'exp_2024-06-13_16-44-17.log',
    'exp_2024-06-13_16-48-26.log',
]

# S2 hover AB
hover_ab_meta['exp_2024-06-26_12-45-37'] = [
    'exp_2024-06-26_12-48-30.log',
    'exp_2024-06-26_12-57-12.log',
    'exp_2024-06-26_13-03-21.log',
    'exp_2024-06-26_13-11-29.log',
    'exp_2024-06-26_13-19-26.log',
    'exp_2024-06-26_13-29-43.log',
    'exp_2024-06-26_13-37-58.log',
    'exp_2024-06-26_13-43-09.log',
]

hover_ab_meta['exp_2024-07-09_15-01-12'] = [
    'exp_2024-07-09_15-03-15.log',
    'exp_2024-07-09_15-11-06.log',
    'exp_2024-07-09_15-17-03.log',
    'exp_2024-07-09_15-26-22.log',
    'exp_2024-07-09_15-33-55.log',
    'exp_2024-07-09_15-39-49.log',
    'exp_2024-07-09_15-49-24.log',
    'exp_2024-07-09_15-54-04.log',
]

#((block, x), is_hover, (pick_status, place_status))
# pick_status:  1: correct block
#              -1: incorrect block
#               0: no pick
# place_status: 1: on correct x (subjective for hover)
#              -1: not on correct x
#               0: no place
#              99: no pick
outcomes_meta = {
    'exp_2024-05-30_14-48-57': [
        (('r', 'y'), 1, (0, 99)),
        (('r', 'p'), 1, (0, 99)),
        (('g', 'b'), 1, (1, 1)),
        (('y', 'p'), 1, (1, -1)),
        (('r', 'y'), 0, (1, 1)),
        (('r', 'p'), 0, (1, 1)),
        (('g', 'b'), 0, (1, 1)),
        (('y', 'p'), 0, (1, 1)), #
        (('g', 'p'), 0, (1, 1)),
        (('b', 'o'), 0, (1, 1)),
        (('y', 'y'), 0, (1, 1)),
        (('g', 'y'), 0, (1, 1)),
        (('g', 'p'), 1, (0, 99)),
        (('b', 'o'), 1, (0, 99)),
        (('y', 'y'), 1, (0, 99)),
        (('g', 'y'), 1, (1, -1)), #
        (('b', 'y'), 1, (1, -1)),
        (('y', 'b'), 1, (0, 99)),
        (('r', 'b'), 1, (1, 1)),
        (('r', 'o'), 1, (1, 1)),
        (('b', 'y'), 0, (1, 1)),
        (('y', 'b'), 0, (1, 1)),
        (('r', 'b'), 0, (1, 1)),
        (('r', 'o'), 0, (-1, 1)),
        (('b', 'p'), 1, (1, -1)),
        (('b', 'b'), 1, (0, 99)),
        (('g', 'o'), 1, (1, -1)),
        (('y', 'o'), 1, (0, 99)),
        (('b', 'p'), 0, (1, 1)),
        (('b', 'b'), 0, (1, 1)),
        (('g', 'o'), 0, (1, 1)),
        (('y', 'o'), 0, (1, 1)),
        #
        (('b', 'y'), 1, (1, 1)),
        (('g', 'y'), 1, (0, 99)),
        (('r', 'b'), 1, (0, 99)),
        (('r', 'p'), 1, (0, 99)),
        (('b', 'y'), 0, (1, 1)),
        (('g', 'y'), 0, (1, 1)),
        (('r', 'p'), 0, (1, 1)), # originally r,b but user misheard and performed r,p instead.
        (('r', 'b'), 0, (1, 1)),
        (('g', 'b'), 0, (1, 1)),
        (('b', 'o'), 0, (1, 1)),
        (('b', 'b'), 0, (-1, -1)),
        (('b', 'p'), 0, (1, 1)),
        (('g', 'b'), 1, (0, 99)),
        (('b', 'o'), 1, (0, 99)),
        (('b', 'b'), 1, (1, 1)),
        (('b', 'p'), 1, (1, 1)),
        (('r', 'o'), 1, (1, -1)),
        (('g', 'o'), 1, (0, 99)),
        (('y', 'p'), 1, (0, 99)),
        (('y', 'b'), 1, (1, 1)),
        (('r', 'o'), 0, (1, 1)),
        (('g', 'o'), 0, (1, 1)),
        (('y', 'p'), 0, (1, 1)),
        (('y', 'b'), 0, (1, 1)),
        (('y', 'y'), 1, (0, 99)),
        (('g', 'p'), 1, (0, 99)),
        (('r', 'y'), 1, (0, 99)),
        (('y', 'o'), 1, (0, 99)),
        (('y', 'y'), 0, (1, 1)),
        (('g', 'p'), 0, (1, 1)),
        (('r', 'y'), 0, (1, 1)),
        (('y', 'o'), 0, (1, 1)),
    ],
    # positions of
    # COLORblock COLORx
    # Gb Yx
    # Bb Ox
    # Yb Px
    # Rb Bx
    'exp_2024-06-06_12-37-30': [
        (('y', 'b'), 1, (0, 99)),
        (('y', 'y'), 1, (0, 99)),
        (('b', 'b'), 1, (0, 99)),
        (('g', 'o'), 1, (1, -1)),
        (('y', 'b'), 0, (0, 99)),
        (('y', 'y'), 0, (-1, 1)),
        (('b', 'b'), 0, (1, -1)),
        (('g', 'o'), 0, (1, 1)),
        (('y', 'p'), 0, (0, 99)),
        (('b', 'o'), 0, (1, 1)),
        (('r', 'o'), 0, (1, 1)),
        (('y', 'o'), 0, (1, 1)),
        (('y', 'p'), 1, (1, 1)), # off by a bit
        (('b', 'o'), 1, (0, 99)),
        (('r', 'o'), 1, (1, -1)), # close but still off.
        (('y', 'o'), 1, (0, 99)),
        (('g', 'p'), 0, (1, -1)),
        (('g', 'y'), 0, (1, 1)),
        (('b', 'y'), 0, (-1, 1)),
        (('r', 'p'), 0, (1, 1)),
        (('g', 'p'), 1, (1, 1)),
        (('g', 'y'), 1, (1, 1)),
        (('b', 'y'), 1, (0, 99)),
        (('r', 'p'), 1, (0, 99)),
        (('b', 'p'), 0, (-1, 1)),
        (('r', 'b'), 0, (1, -1)),
        (('r', 'y'), 0, (1, 1)),
        (('g', 'b'), 0, (1, 1)),
        (('b', 'p'), 1, (0, 99)),
        (('r', 'b'), 1, (0, 99)), # red block possibly pushed out of bounds
        (('r', 'y'), 1, (1, -1)),
        (('g', 'b'), 1, (1, -1)), # in between blue and pink
        #
    ],
    'exp_2024-06-10_14-41-45': [
        (('y', 'p'), 1, (1, 1)),
        (('y', 'b'), 1, (0, 99)),
        (('y', 'o'), 1, (1, -1)),
        (('r', 'o'), 1, (1, -1)),
        (('y', 'p'), 0, (1, 1)),
        (('y', 'b'), 0, (1, 1)),
        (('y',' o'), 0, (1, 1)),
        (('r', 'o'), 0, (1, 1)),
        (('r', 'y'), 0, (1, 1)),
        (('b', 'y'), 0, (1, 1)),
        (('g', 'b'), 0, (1, -1)),
        (('b', 'p'), 0, (1, 1)),
        (('r', 'y'), 1, (1, -1)),
        (('b', 'y'), 1, (1, -1)),
        (('g', 'b'), 1, (0, 99)),
        (('b', 'p'), 1, (1, -1)),
        (('g', 'y'), 1, (1, -1)),
        (('b', 'o'), 1, (1, -1)),
        (('r', 'b'), 1, (1, -1)),
        (('r', 'p'), 1, (1, 1)),
        (('g', 'y'), 0, (1, 1)),
        (('b', 'o'), 0, (1, 1)),
        (('r', 'b'), 0, (1, 1)),
        (('r', 'p'), 0, (1, 1)),
        (('g', 'p'), 1, (1, 1)),
        (('g', 'o'), 1, (1, -1)),
        (('b', 'b'), 1, (1, -1)),
        (('y', 'y'), 1, (1, -1)),
        (('g', 'p'), 0, (1, 1)),
        (('g', 'o'), 0, (1, 1)),
        (('b', 'b'), 0, (0, 99)),
        (('y', 'y'), 0, (1, 1)),
    ],
    'exp_2024-06-10_15-53-08': [
        (('r', 'b'), 1, (1 ,1)),
        (('r', 'y'), 1, (1, 1)),
        (('r', 'p'), 1, (1, -1)),
        (('g', 'b'), 1, (1, 1)),
        (('r', 'b'), 0, (1, 1)),
        (('r', 'y'), 0, (1, -1)),
        (('r', 'p'), 0, (1, 1)),
        (('g', 'b'), 0, (1, -1)),
        (('g', 'o'), 0, (1, 1)),
        (('b', 'y'), 0, (-1, 1)),
        (('y', 'o'), 0, (1, 1)),
        (('y', 'y'), 0, (1, 1)),
        (('g', 'o'), 1, (1, -1)),
        (('b', 'y'), 1, (1, 1)),
        (('y', 'o'), 1, (1, 1)),
        (('y', 'y'), 1, (0, 99)),
        (('b', 'o'), 1, (1, 1)),
        (('g', 'p'), 1, (0, 99)),
        (('r', 'o'), 1, (1, -1)),
        (('y', 'p'), 1, (0, 99)),
        (('b', 'o'), 0, (1, 1)),
        (('g', 'p'), 0, (1, -1)),
        (('r', 'o'), 0, (1, 1)),
        (('y', 'p'), 0, (1, -1)), # sneezed
        (('y', 'b'), 1, (0, 99)),
        (('b', 'p'), 1, (0, 99)),
        (('g', 'y'), 1, (1, 1)),
        (('b', 'b'), 1, (0, 99)),
        (('y', 'b'), 0, (1, 1)),
        (('b', 'p'), 0, (1, 1)),
        (('g', 'y'), 0, (1, 1)),
        (('b', 'b'), 0, (1, 1)),
    ],
    'exp_2024-06-13_15-08-39': [
        (('b', 'b'), 0, (1, 1)),
        (('y', 'y'), 0, (1, 1)),
        (('r', 'b'), 0, (1, 1)),
        (('g', 'o'), 0, (1, 1)),
        (('b', 'b'), 1, (1, -1)),
        (('y', 'y'), 1, (1, 1)),
        (('r', 'b'), 1, (1, -1)),
        (('g', 'o'), 1, (0, 99)),
        (('g', 'b'), 0, (1, 0)),
        (('y', 'b'), 0, (1, 1)),
        (('r', 'o'), 0, (1, 1)),
        (('b', 'p'), 0, (1, 1)),
        (('g', 'b'), 1, (1, 1)),
        (('y', 'b'), 1, (1, -1)),
        (('r', 'o'), 1, (0, 99)),
        (('b', 'p'), 1, (1, 1)),
        (('y', 'o'), 1, (0, 99)),
        (('r', 'y'), 1, (1, 1)),
        (('r', 'p'), 1, (0, 99)),
        (('y', 'p'), 1, (1, -1)),
        (('y', 'o'), 0, (1, 1)),
        (('r', 'y'), 0, (1, 1)),
        (('r', 'p'), 0, (1, 1)),
        (('y', 'p'), 0, (1, 1)),
        (('b', 'y'), 1, (0, 99)),
        (('b', 'o'), 1, (1, 1)),
        (('g', 'p'), 1, (1, 1)),
        (('g', 'y'), 1, (0, 99)),
        (('b', 'y'), 0, (1, 1)),
        (('b', 'o'), 0, (1, 1)),
        (('g', 'p'), 0, (1, 1)),
        (('g', 'y'), 0, (1, 1)),
    ],
    'exp_2024-06-13_16-05-13': [
        (('r', 'b'), 0, (1, 1)),
        (('r', 'y'), 0, (1, 1)),
        (('g', 'b'), 0, (1, 1)),
        (('b', 'b'), 0, (1, 1)),
        (('r', 'b'), 1, (1, 1)),
        (('r', 'y'), 1, (1, -1)),
        (('g', 'b'), 1, (1, -1)),
        (('b', 'b'), 1, (1, -1)),
        (('y', 'b'), 0, (1, -1)),
        (('y', 'p'), 0, (1, 1)),
        (('r', 'o'), 0, (1, 1)),
        (('b', 'p'), 0, (1, 1)),
        (('y', 'b'), 1, (1, 1)),
        (('y', 'p'), 1, (1, 1)),
        (('r', 'o'), 1, (1, -1)),
        (('b', 'p'), 1, (1, 1)),
        (('g', 'o'), 1, (1, -1)),
        (('g', 'p'), 1, (1, -1)),
        (('y', 'o'), 1, (0, 99)),
        (('y', 'y'), 1, (0, 99)),
        (('g', 'o'), 0, (1, 1)),
        (('g', 'p'), 0, (1, 1)),
        (('y', 'o'), 0, (1, 1)),
        (('y', 'y'), 0, (1, 1)),
        (('b', 'y'), 0, (1, 1)),
        (('b', 'o'), 0, (1, 1)),
        (('g', 'y'), 0, (1, 1)),
        (('r', 'p'), 0, (1, 1)),
        (('b', 'y'), 1, (1, 1)),
        (('b', 'o'), 1, (1, -1)),
        (('g', 'y'), 1, (1, -1)),
        (('r', 'p'), 1, (1, 1)),
    ],
    'exp_2024-06-26_12-45-37': [
        (('y', 'y'), 1, (0, 99)),
        (('g', 'o'), 1, (0, 99)),
        (('g', 'y'), 1, (1, -1)),
        (('r', 'y'), 1, (1, -1)),
        (('y', 'y'), 0, (1, 0)),
        (('g', 'o'), 0, (1, -1)),
        (('g', 'y'), 0, (1, 1)),
        (('r', 'y'), 0, (1, 1)),
        (('r', 'b'), 1, (0, 99)),
        (('y', 'o'), 1, (1, -1)),
        (('r', 'o'), 1, (1, -1)),
        (('g', 'b'), 1, (0, 99)),
        (('r', 'b'), 0, (1, 1)),
        (('y', 'o'), 0, (1, 0)),
        (('r', 'o'), 0, (-1, 0)),
        (('g', 'b'), 0, (1, 0)),
        (('b', 'p'), 0, (0, 99)),
        (('b', 'o'), 0, (0, 99)),
        (('g', 'p'), 0, (1, 0)),
        (('y', 'b'), 0, (1, 0)),
        (('b', 'p'), 1, (-1, -1)),
        (('b', 'o'), 1, (0, 99)),
        (('g', 'p'), 1, (1, -1)),
        (('y', 'b'), 1, (0, 99)),
        (('y', 'p'), 0, (1, 1)),
        (('b', 'b'), 0, (0, 99)),
        (('r', 'p'), 0, (1, 1)),
        (('b', 'y'), 0, (-1, -1)),
        (('y', 'p'), 1, (0, 99)),
        (('b', 'b'), 1, (0, 99)),
        (('r', 'p'), 1, (1, -1)),
        (('b', 'y'), 1, (-1, -1)),
    ],
    'exp_2024-07-09_15-01-12': [
        (('b', 'o'), 1, (-1, -1)),
        (('b', 'b'), 1, (0, 99)),
        (('b', 'y'), 1, (1, -1)),
        (('y', 'o'), 1, (0, 99)),
        (('b', 'o'), 0, (1, 1)),
        (('b', 'b'), 0, (1, 1)),
        (('b', 'y'), 0, (-1, 1)),
        (('y', 'o'), 0, (1, 1)),
        (('y', 'y'), 1, (0, 99)),
        (('b', 'p'), 1, (0, 99)),
        (('r', 'o'), 1, (0, 99)),
        (('g', 'y'), 1, (0, 99)),
        (('y', 'y'), 0, (1, -1)),
        (('b', 'p'), 0, (0, 99)),
        (('r', 'o'), 0, (-1, 0)),
        (('g', 'y'), 0, (1, 1)),
        (('y', 'p'), 0, (1, -1)),
        (('g', 'b'), 0, (1, -1)),
        (('g', 'p'), 0, (1, -1)),
        (('r', 'b'), 0, (-1, 1)),
        (('y', 'p'), 1, (0, 99)),
        (('g', 'b'), 1, (0, 99)),
        (('g', 'p'), 1, (0, 99)),
        (('r', 'b'), 1, (0, 99)),
        (('y', 'b'), 0, (0, 99)),
        (('r', 'p'), 0, (0, 99)),
        (('g', 'o'), 0, (-1, -1)),
        (('r', 'y'), 0, (1, 1)),
        (('y', 'b'), 1, (0, 99)),
        (('r', 'p'), 1, (0, 99)),
        (('g', 'o'), 1, (0, 99)),
        (('r', 'y'), 1, (0, 99)),
    ]
}