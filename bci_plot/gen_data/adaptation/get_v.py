import os
import sys
import pathlib

import numpy as np

from bci_plot.utils import data_util
datadir = pathlib.Path('/data/raspy')


def get_v(session):
    task_data = data_util.load_data(datadir / f'{session}' / 'task.bin')
    state_task = task_data['state_task'].flatten().copy()
    
    N = len(state_task)
    
    start_inds = np.nonzero(state_task[1:None] != state_task[0:-1])[0][1:None] + 1 # start tick of new condition.
    end_inds = np.hstack([start_inds[1:None], [len(state_task)]])
    trial_tick = np.full(len(state_task), np.nan, dtype=np.float64)
    for s, e in zip(start_inds, end_inds):
        trial_tick[s:e] = np.arange(e-s)
    
    start = np.nonzero(task_data['eeg_step'] >= 1000)[0][0]
    end = N
    
    valid_ticks = (task_data['eeg_step'] >= 1000)*(trial_tick >= 20)*(state_task != 99)
    
    out = {}
    out['valid_kf_state'] = task_data['kf_state'][valid_ticks]
    out['valid_kf_vel'] = task_data['kf_state'][valid_ticks, 2:6]
    out['valid_rlud_decoder_output'] = task_data['decoder_output'][:, [1, 0, 2, 3]][valid_ticks]
    return out