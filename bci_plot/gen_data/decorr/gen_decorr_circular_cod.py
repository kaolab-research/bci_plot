
'''
Processes sessions and outputs the generated statistics.
'''

import numpy as np
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from bci_plot.utils import data_util
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

from bci_plot.gen_data.decorr import mlp
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  #ignore Lazy Module warnings
#from bci_plot.metadata import decorr_info
import tqdm

def calculate_cod(y_pred, y_tgt, y_train, axis=0):
    mu = y_train.mean(axis=axis, keepdims=True)
    naive_variance = (mu - y_tgt).var(axis=axis)
    res = y_pred - y_tgt
    res_variance = res.var(axis=axis)
    out = 1 - res_variance/naive_variance
    return out



datadir = pathlib.Path('/data/raspy/')
datadir_replayed = pathlib.Path('/data/raspy/replayed/')

def gen_decorr_circular_cod(session, device='cpu', n_repetition = 1000, delay_low=None, delay_high=None):
    """ 
    handle multiple session 
    delay low = integer in seconds, None means, delay is the size of training set
    delay high = integer in seconds, None means, delay is the size of training set
    """

    if isinstance(session, list): sessions = session
    else: sessions = [session]

    X_raw = np.empty((0,4))
    Y_raw = np.empty((0,16))
    X_index = np.empty(0,dtype=int)
    Y_index = np.empty(0,dtype=int)
    for session in sessions:
        task_data = data_util.load_data(datadir / session / 'task.bin')
        task_data_replayed = data_util.load_data(datadir_replayed / (session + '_replay-decorr') / 'task.bin')
        gaze_data = data_util.load_data(datadir / session / 'gaze.bin')
        
        avg_gaze = 0.5*gaze_data['gaze_buffer'][:, 2:4] + 0.5*gaze_data['gaze_buffer'][:, 17:19]
        
        n_gaze = len(avg_gaze)
        #f = scipy.interpolate.interp1d(np.nonzero(~gaze_is_nan)[0], avg_gaze[~gaze_is_nan], fill_value='extrapolate', axis=0)
        #avg_gaze_interpolated = f(np.arange(len(avg_gaze)))
        
        cursor_pos = task_data['decoded_pos']
        target_pos = task_data['target_pos']
        render_angle = task_data['render_angle']
        rendered_cursor_pos = np.array([cp.reshape((1, 2))@np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).T for \
                                        cp, theta in zip(cursor_pos, render_angle/180*np.pi)])
        rendered_cursor_pos = rendered_cursor_pos[:, 0, 0, :]
        rendered_target_pos = np.array([tp.reshape((1, 2))@np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).T for \
                                        tp, theta in zip(target_pos, render_angle/180*np.pi)])
        rendered_target_pos = rendered_target_pos[:, 0, 0, :]
        
        # topleft at xy (492, 91)
        # bottom right at xy (1426, 1025)
        # resolution is (1920, 1080)
        xcent = 0.5*(492 + 1426) / 1920
        xl = 1426 / 1920 - xcent
        ycent = 0.5*(91 + 1025) / 1080
        yl = 91 / 1080 - ycent
        normalized_gaze = np.zeros_like(avg_gaze)
        normalized_gaze[:, 0] = (avg_gaze[:, 0] - xcent)/xl
        normalized_gaze[:, 1] = (avg_gaze[:, 1] - ycent)/yl
        
        dgaze = np.diff(normalized_gaze, axis=0)*60
        dgaze = np.concatenate([dgaze[0:1, :], dgaze], axis=0)
        normalized_gaze_kinematics = np.concatenate([normalized_gaze, dgaze], axis=1) # x, y, dx/dt, dy/dt
        gaze_is_nan = np.isnan(normalized_gaze_kinematics).any(-1)
        
        #M = 20*2 # 20 fps * 2 seconds
        #delays = np.arange(-M, M+1, 10) # in ticks
        fs = 60
        M = fs*5 # 60 Hz * 2 seconds
        n_per_second = 5
        # Positive delay: current h corresponds to past gaze data (h lags behind gaze)
        # Negative delay: current h corresponds to future gaze data (h leads gaze)
        # Zero delay: as recorded.
    
    
        #task_has_gaze = (task_data['gaze_step'] >= 0)
        #task_valid_steps = np.nonzero((task_data['gaze_step'] >= M)*(len(avg_gaze) - task_data['gaze_step'] >= M))[0]
        #print(normalized_gaze_kinematics.shape)
        gaze_valid_steps = np.arange(M, n_gaze-M)
        gaze_valid_steps = gaze_valid_steps[~gaze_is_nan[gaze_valid_steps]]

        
        # create X_raw and Y_raw
        X_raw_ = np.array([normalized_gaze_kinematics[i] for i in gaze_valid_steps])
        X_index_ = np.arange(len(gaze_valid_steps))
        Y_raw_ = np.array([task_data_replayed['decoder_h'][gaze_data['task_step'][i]] for i in gaze_valid_steps])
        Y_index_ = np.arange(len(gaze_valid_steps))

        # consolidate
        X_raw = np.concatenate((X_raw, X_raw_))
        Y_raw = np.concatenate((Y_raw, Y_raw_))
        X_index = np.concatenate((X_index, X_index_ + len(X_index)))
        Y_index = np.concatenate((Y_index, Y_index_ + len(Y_index)))

    out = {
        'h_from_gaze': {
            'linear': [],
        },
        'gaze_from_h': {
            'linear': [],
        },
        'delays': [],
        'delays_s': [],
    }

    np.random.seed(0) # to get consistent result
    kfold = KFold(n_splits=5, shuffle=False) # Do not shuffle so as to reduce time-correlated effects which artificially increase validation performance.
    for j, (train_index, val_index) in enumerate(kfold.split(X_index)):

        out['h_from_gaze']['linear'].append([]) # by fold
        out['gaze_from_h']['linear'].append([])
        out['delays'].append([])
        out['delays_s'].append([])

        for _ in tqdm.tqdm(range(n_repetition), desc=f"Fold {j}"):
            n_train = len(train_index)
            delay = np.random.randint(low=-n_train if delay_low is None else delay_low * fs,
                                      high=n_train if delay_high is None else delay_high * fs) # integer from -n_train to +n_train OR (9*fs, high=18*fs)

            regular_integer_index = np.arange(n_train)
            delayed_integer_index = (np.arange(n_train) + n_train + delay) % n_train # to keep positive circular
            
            # regular_train_index = gaze_valid_steps[train_index[regular_integer_index]]
            # delayed_train_index = gaze_valid_steps[train_index[delayed_integer_index]]
            # regular_val_index = gaze_valid_steps[val_index]
            
            # X_train = normalized_gaze_kinematics[regular_train_index]
            # Y_train = task_data_replayed['decoder_h'][gaze_data['task_step'][delayed_train_index]]
            # X_val = normalized_gaze_kinematics[regular_val_index]
            # Y_val = task_data_replayed['decoder_h'][gaze_data['task_step'][regular_val_index]]

            X_train_index = X_index[train_index[regular_integer_index]]
            Y_train_index = Y_index[train_index[delayed_integer_index]]
            X_val_index = X_index[val_index]
            Y_val_index = Y_index[val_index]

            X_train = X_raw[X_train_index]
            Y_train = Y_raw[Y_train_index]
            X_val = X_raw[X_val_index]
            Y_val = Y_raw[Y_val_index]
                    
            ### Predict decoder_h from gaze
            # Linear
            model = LinearRegression()
            model.fit(X_train, Y_train)
            Y_train_hat = model.predict(X_train)
            Y_val_hat = model.predict(X_val)
            cods = calculate_cod(Y_val_hat, Y_val, Y_train, axis=(0,))
            out['h_from_gaze']['linear'][-1].append(cods)
            
            
            X_train, Y_train, X_val, Y_val = Y_train, X_train, Y_val, X_val # swap X and Y
            ### Predict gaze from decoder_h
            # Linear
            model = LinearRegression()
            model.fit(X_train, Y_train)
            Y_train_hat = model.predict(X_train)
            Y_val_hat = model.predict(X_val)
            cods = calculate_cod(Y_val_hat, Y_val, Y_train, axis=(0,))
            out['gaze_from_h']['linear'][-1].append(cods)

            # delay
            out['delays'].append(delay)
            out['delays_s'].append(delay/fs)
    
    return out