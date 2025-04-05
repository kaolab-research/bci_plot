
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
from sklearn.linear_model import LinearRegression, Ridge
from scipy.stats import pearsonr

import matplotlib.pyplot as plt

from bci_plot.gen_data.decorr import mlp
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  #ignore Lazy Module warnings

import tqdm

def calculate_cod(y_pred, y_tgt, y_train, axis=0):
    mu = y_train.mean(axis=axis, keepdims=True)
    naive_sq_sum = ((mu - y_tgt)**2).sum(axis=axis)
    res = y_pred - y_tgt
    res_sq_sum = (res**2).sum(axis=axis)
    out = 1 - res_sq_sum/naive_sq_sum
    return out

datadir = pathlib.Path('/data/raspy/')
datadir_replayed = pathlib.Path('/data/raspy/replayed/')

def gen_decorr_cod(session, device='cpu', run_mlp=True):

    task_data = data_util.load_data(datadir / session / 'task.bin')
    task_data_replayed = data_util.load_data(datadir_replayed / (session + '_replay-decorr') / 'task.bin')
    gaze_data = data_util.load_data(datadir / session / 'gaze.bin')
    
    avg_gaze = 0.5*gaze_data['gaze_buffer'][:, 2:4] + 0.5*gaze_data['gaze_buffer'][:, 17:19]
    
    n_gaze = len(avg_gaze)
    
    
    # used to filter valid data
    task_fs = 20.0
    state_task = task_data['state_task'].flatten()
    trial_starts = np.nonzero((state_task[1:None] != -1)*(state_task[0:-1] == -1))[0] + 1
    trial_ends   = np.nonzero((state_task[1:None] == -1)*(state_task[0:-1] != -1))[0] + 1
    if trial_ends[0] < trial_starts[0]:
        trial_ends = trial_ends[1:None]
    trial_starts = trial_starts[0:len(trial_ends)]
    trial_ticks = np.full(len(state_task), -np.inf)
    for start, end in zip(trial_starts, trial_ends):
        trial_ticks[start:end] = np.arange(end-start)
    
    cursor_pos = task_data['decoded_pos']
    target_pos = task_data['target_pos']
    render_angle = task_data['render_angle']
    rendered_cursor_pos = np.array([cp.reshape((1, 2))@np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).T for \
                                    cp, theta in zip(cursor_pos, render_angle/180*np.pi)])
    rendered_cursor_pos = rendered_cursor_pos[:, 0, 0, :]
    rendered_target_pos = np.array([tp.reshape((1, 2))@np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).T for \
                                    tp, theta in zip(target_pos, render_angle/180*np.pi)])
    rendered_target_pos = rendered_target_pos[:, 0, 0, :]
    rendered_target_loc = (4*np.arctan2(rendered_target_pos[:, 1], rendered_target_pos[:, 0])/np.pi).astype('int') % 8
    
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
    
    fs = 60
    M = fs*5 # 60 Hz * 2 seconds
    n_per_second = 5
    # Positive delay: current h corresponds to past gaze data (h lags behind gaze)
    # Negative delay: current h corresponds to future gaze data (h leads gaze)
    # Zero delay: as recorded.
    delays = np.arange(-M, M+1, int(fs/n_per_second)) # in samples @ fs Hz. 5 times per second. must be int.
    
    out = {
        'h_from_gaze': {
            'linear': [],
            'mlp': [],
            'mlp_hist': [],
            'linear_by_state': [],
        },
        'gaze_from_h': {
            'linear': [],
            'mlp': [],
            'mlp_hist': [],
            'linear_by_state': [],
        },
        'delays': delays,
        'delays_s': delays/fs
    }
    
    for i, delay in tqdm.tqdm(list(enumerate(delays))):
        gaze_valid_steps = np.arange(M, n_gaze-M) - delay
        gaze_valid_steps = gaze_valid_steps[~gaze_is_nan[gaze_valid_steps]]
        task_valid_steps = gaze_data['task_step'][gaze_valid_steps + delay]
        
        task_step_filter = (trial_ticks[task_valid_steps] >= 0*task_fs)
        
        Y_raw = task_data_replayed['decoder_h'][task_valid_steps][task_step_filter]
        X_raw = normalized_gaze_kinematics[gaze_valid_steps][task_step_filter]
        task_steps_filtered = task_valid_steps[task_step_filter] # this is the actual task tick corresponding to the labels.
        
        kfold = KFold(n_splits=5, shuffle=False) # Do not shuffle so as to reduce time-correlated effects which artificially increase validation performance.
        
        out['h_from_gaze']['linear'].append([])
        out['h_from_gaze']['linear_by_state'].append([])
        out['h_from_gaze']['mlp'].append([])
        out['h_from_gaze']['mlp_hist'].append([])
        out['gaze_from_h']['linear'].append([])
        out['gaze_from_h']['linear_by_state'].append([])
        out['gaze_from_h']['mlp'].append([])
        out['gaze_from_h']['mlp_hist'].append([])
        for j, (train_index, val_index) in enumerate(kfold.split(X_raw)):
            val_index = val_index[int(fs*5):-int(fs*5)] # cut 5 seconds from each end.
            X_train = X_raw[train_index]
            Y_train = Y_raw[train_index]
            state_train = state_task[task_steps_filtered[train_index]]
            loc_train = rendered_target_loc[task_steps_filtered[train_index]]
            X_val = X_raw[val_index]
            Y_val = Y_raw[val_index]
            state_val = state_task[task_steps_filtered[val_index]]
            loc_val = rendered_target_loc[task_steps_filtered[val_index]]
            
            ### Predict decoder_h from gaze
            # Linear
            model = LinearRegression()
            model.fit(X_train, Y_train)
            Y_train_hat = model.predict(X_train)
            Y_val_hat = model.predict(X_val)
            cods = calculate_cod(Y_val_hat, Y_val, Y_train, axis=(0,))
            out['h_from_gaze']['linear'][-1].append(cods)
            
            tmp = {}
            for state in np.unique(state_train):
                state = int(state)
                model = LinearRegression()
                model.fit(X_train[state_train==state], Y_train[state_train==state])
                Y_val_hat = model.predict(X_val)
                cods = calculate_cod(Y_val_hat[state_val==state], Y_val[state_val==state], Y_val[state_val==state], axis=(0,))
                tmp[state] = cods
            out['h_from_gaze']['linear_by_state'][-1].append(tmp)
            
            # MLP
            if run_mlp:
                batch_size = 512
                max_epochs = 10 # around 10 epochs until plateau for batch_size=512
                model = mlp.MLP(odim=Y_train.shape[-1], hdim=128, device=device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                mseloss = nn.MSELoss()
                X_train_pt = torch.tensor(X_train, dtype=torch.float32, device=device)
                X_val_pt = torch.tensor(X_val, dtype=torch.float32, device=device)
                Y_train_pt = torch.tensor(Y_train, dtype=torch.float32, device=device)
                Y_val_pt = torch.tensor(Y_val, dtype=torch.float32, device=device)
                train_dataset = torch.utils.data.TensorDataset(X_train_pt, Y_train_pt)
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                cods_all = []
                for epoch_no in range(max_epochs):
                    model.train()
                    for x, y in train_dataloader:
                        optimizer.zero_grad()
                        yh = model(x)
                        loss = mseloss(yh, y)
                        loss.backward()
                        optimizer.step()
                    model.eval()
                    Y_val_hat = model.predict(X_val_pt).detach().cpu().numpy()
                    cods = calculate_cod(Y_val_hat, Y_val, Y_train, axis=(0,))
                    cods_all.append(cods)
                out['h_from_gaze']['mlp'][-1].append(cods)
                out['h_from_gaze']['mlp_hist'][-1].append(cods_all)
            
            '''
            '''
            X_train, Y_train, X_val, Y_val = Y_train, X_train, Y_val, X_val # swap X and Y
            ### Predict gaze from decoder_h
            # Linear
            model = LinearRegression()
            model.fit(X_train, Y_train)
            Y_train_hat = model.predict(X_train)
            Y_val_hat = model.predict(X_val)
            cods = calculate_cod(Y_val_hat, Y_val, Y_train, axis=(0,))
            train_cods = calculate_cod(Y_train_hat, Y_train, Y_train, axis=(0,))
            out['gaze_from_h']['linear'][-1].append(cods)
            
            tmp = {}
            for state in np.unique(state_train):
                state = int(state)
                model = LinearRegression()
                model.fit(X_train[state_train==state], Y_train[state_train==state])
                Y_val_hat = model.predict(X_val)
                
                cods = calculate_cod(Y_val_hat[state_val==state], Y_val[state_val==state], Y_val[state_val==state], axis=(0,))
                tmp[state] = cods
            out['gaze_from_h']['linear_by_state'][-1].append(tmp)
            
            # MLP
            if run_mlp:
                batch_size = 512
                max_epochs = 10 # around 10 epochs until plateau for batch_size=512
                model = mlp.MLP(odim=Y_train.shape[-1], hdim=128, device=device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                mseloss = nn.MSELoss()
                X_train_pt = torch.tensor(X_train, dtype=torch.float32, device=device)
                X_val_pt = torch.tensor(X_val, dtype=torch.float32, device=device)
                Y_train_pt = torch.tensor(Y_train, dtype=torch.float32, device=device)
                Y_val_pt = torch.tensor(Y_val, dtype=torch.float32, device=device)
                train_dataset = torch.utils.data.TensorDataset(X_train_pt, Y_train_pt)
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                cods_all = []
                for epoch_no in range(max_epochs):
                    model.train()
                    for x, y in train_dataloader:
                        optimizer.zero_grad()
                        yh = model(x)
                        loss = mseloss(yh, y)
                        loss.backward()
                        optimizer.step()
                    model.eval()
                    Y_val_hat = model.predict(X_val_pt).detach().cpu().numpy()
                    cods = calculate_cod(Y_val_hat, Y_val, Y_train, axis=(0,))
                    cods_all.append(cods)
                out['gaze_from_h']['mlp'][-1].append(cods)
                out['gaze_from_h']['mlp_hist'][-1].append(cods_all)
        pass
    
    return out

def gen_decorr_corrs(session, device='cpu', run_mlp=True):

    task_data = data_util.load_data(datadir / session / 'task.bin')
    task_data_replayed = data_util.load_data(datadir_replayed / (session + '_replay-decorr') / 'task.bin')
    gaze_data = data_util.load_data(datadir / session / 'gaze.bin')
    
    avg_gaze = 0.5*gaze_data['gaze_buffer'][:, 2:4] + 0.5*gaze_data['gaze_buffer'][:, 17:19]
    
    n_gaze = len(avg_gaze)
    
    
    # used to filter valid data
    task_fs = 20.0
    state_task = task_data['state_task'].flatten()
    trial_starts = np.nonzero((state_task[1:None] != -1)*(state_task[0:-1] == -1))[0] + 1
    trial_ends   = np.nonzero((state_task[1:None] == -1)*(state_task[0:-1] != -1))[0] + 1
    if trial_ends[0] < trial_starts[0]:
        trial_ends = trial_ends[1:None]
    trial_starts = trial_starts[0:len(trial_ends)]
    trial_ticks = np.full(len(state_task), -np.inf)
    for start, end in zip(trial_starts, trial_ends):
        trial_ticks[start:end] = np.arange(end-start)
    
    cursor_pos = task_data['decoded_pos']
    target_pos = task_data['target_pos']
    render_angle = task_data['render_angle']
    rendered_cursor_pos = np.array([cp.reshape((1, 2))@np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).T for \
                                    cp, theta in zip(cursor_pos, render_angle/180*np.pi)])
    rendered_cursor_pos = rendered_cursor_pos[:, 0, 0, :]
    rendered_target_pos = np.array([tp.reshape((1, 2))@np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).T for \
                                    tp, theta in zip(target_pos, render_angle/180*np.pi)])
    rendered_target_pos = rendered_target_pos[:, 0, 0, :]
    rendered_target_loc = (4*np.arctan2(rendered_target_pos[:, 1], rendered_target_pos[:, 0])/np.pi).astype('int') % 8
    
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
    
    fs = 60
    M = fs*5 # 60 Hz * 2 seconds
    n_per_second = 5
    # Positive delay: current h corresponds to past gaze data (h lags behind gaze)
    # Negative delay: current h corresponds to future gaze data (h leads gaze)
    # Zero delay: as recorded.
    delays = np.arange(-M, M+1, int(fs/n_per_second)) # in samples @ fs Hz. 5 times per second. must be int.
    
    out = {
        'rendered_x': [],
        'rendered_y': [],
        'target_x': [],
        'target_y': [],
        'rendered_target_x': [],
        'rendered_target_y': [],
    }
    
    for i, delay in tqdm.tqdm(list(enumerate(delays))):
        gaze_valid_steps = np.arange(M, n_gaze-M) - delay
        gaze_valid_steps = gaze_valid_steps[~gaze_is_nan[gaze_valid_steps]]
        task_valid_steps = gaze_data['task_step'][gaze_valid_steps + delay]
        
        task_step_filter = (trial_ticks[task_valid_steps] >= 0*task_fs)
        
        Y_raw = task_data_replayed['decoder_h'][task_valid_steps][task_step_filter]
        X_raw = normalized_gaze_kinematics[gaze_valid_steps][task_step_filter]
        task_steps_filtered = task_valid_steps[task_step_filter] # this is the actual task tick corresponding to the labels.
        rendered_cursor_pos_filtered = rendered_cursor_pos[task_valid_steps][task_step_filter]
        rendered_target_pos_filtered = rendered_target_pos[task_valid_steps][task_step_filter]
        
        out['rendered_x'].append(pearsonr(rendered_cursor_pos_filtered[:, 0], X_raw[:, 0])[0])
        out['rendered_y'].append(pearsonr(rendered_cursor_pos_filtered[:, 1], X_raw[:, 1])[0])
        
        out['rendered_target_x'].append(pearsonr(rendered_target_pos_filtered[:, 0], X_raw[:, 0])[0])
        out['rendered_target_y'].append(pearsonr(rendered_target_pos_filtered[:, 1], X_raw[:, 1])[0])
        pass
    
    return out

def gen_decorr_cod_list(sessions:list, device='cpu'):
    """
    list version of gen_decorr_cod
    """

    fs = 60
    M = fs*5 # 60 Hz * 2 seconds
    n_per_second = 5
    # Positive delay: current h corresponds to past gaze data (h lags behind gaze)
    # Negative delay: current h corresponds to future gaze data (h leads gaze)
    # Zero delay: as recorded.
    delays = np.arange(-M, M+1, int(fs/n_per_second)) # in samples @ fs Hz. 5 times per second. must be int.

    out = {
        'h_from_gaze': {
            'linear': [],
            'mlp': [],
            'mlp_hist': [],
        },
        'gaze_from_h': {
            'linear': [],
            'mlp': [],
            'mlp_hist': [],
        },
        'delays': delays,
        'delays_s': delays/fs
    }

    # create two sets of X and Y data. 
    # total 0123456789 0123456789
    # valid     4567       4567
    # delay can shift both to
    # delayed     6789       6789    
    for i, delay in tqdm.tqdm(list(enumerate(delays))):
 
        X_raw = np.empty((0,4))
        Y_raw = np.empty((0,16))

        for session in sessions:

            task_data = data_util.load_data(datadir / session / 'task.bin')
            task_data_replayed = data_util.load_data(datadir_replayed / (session + '_replay-decorr') / 'task.bin')
            gaze_data = data_util.load_data(datadir / session / 'gaze.bin')
            
            avg_gaze = 0.5*gaze_data['gaze_buffer'][:, 2:4] + 0.5*gaze_data['gaze_buffer'][:, 17:19]
            
            n_gaze = len(avg_gaze)
            
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
            
            # create dataset and valid_index
            gaze_valid_steps = np.arange(M, n_gaze-M) - delay
            gaze_valid_steps = gaze_valid_steps[~gaze_is_nan[gaze_valid_steps]]
            task_valid_steps = gaze_data['task_step'][gaze_valid_steps + delay]
            X_raw_ = normalized_gaze_kinematics[gaze_valid_steps]
            Y_raw_ = task_data_replayed['decoder_h'][task_valid_steps]

            # consolidate dataset
            X_raw = np.concatenate((X_raw, X_raw_))
            Y_raw = np.concatenate((Y_raw, Y_raw_))
            

        kfold = KFold(n_splits=5, shuffle=False) # Do not shuffle so as to reduce time-correlated effects which artificially increase validation performance.
        
        out['h_from_gaze']['linear'].append([])
        out['h_from_gaze']['mlp'].append([])
        out['h_from_gaze']['mlp_hist'].append([])
        out['gaze_from_h']['linear'].append([])
        out['gaze_from_h']['mlp'].append([])
        out['gaze_from_h']['mlp_hist'].append([])
        for j, (train_index, val_index) in enumerate(kfold.split(X_raw)):
            val_index = val_index[int(fs*5):-int(fs*5)] # cut 5 seconds from each end.
            X_train = X_raw[train_index]
            Y_train = Y_raw[train_index]
            X_val = X_raw[val_index]
            Y_val = Y_raw[val_index]
            
            ### Predict decoder_h from gaze
            # Linear
            model = LinearRegression()
            model.fit(X_train, Y_train)
            Y_train_hat = model.predict(X_train)
            Y_val_hat = model.predict(X_val)
            cods = calculate_cod(Y_val_hat, Y_val, Y_train, axis=(0,))
            out['h_from_gaze']['linear'][-1].append(cods)
            
            # MLP
            batch_size = 512
            max_epochs = 10 # around 10 epochs until plateau for batch_size=512
            model = mlp.MLP(odim=Y_train.shape[-1], hdim=128, device=device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            mseloss = nn.MSELoss()
            X_train_pt = torch.tensor(X_train, dtype=torch.float32, device=device)
            X_val_pt = torch.tensor(X_val, dtype=torch.float32, device=device)
            Y_train_pt = torch.tensor(Y_train, dtype=torch.float32, device=device)
            Y_val_pt = torch.tensor(Y_val, dtype=torch.float32, device=device)
            train_dataset = torch.utils.data.TensorDataset(X_train_pt, Y_train_pt)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            cods_all = []
            for epoch_no in range(max_epochs):
                model.train()
                for x, y in train_dataloader:
                    optimizer.zero_grad()
                    yh = model(x)
                    loss = mseloss(yh, y)
                    loss.backward()
                    optimizer.step()
                model.eval()
                Y_val_hat = model.predict(X_val_pt).detach().cpu().numpy()
                cods = calculate_cod(Y_val_hat, Y_val, Y_train, axis=(0,))
                cods_all.append(cods)
            out['h_from_gaze']['mlp'][-1].append(cods)
            out['h_from_gaze']['mlp_hist'][-1].append(cods_all)
            
            X_train, Y_train, X_val, Y_val = Y_train, X_train, Y_val, X_val # swap X and Y
            ### Predict gaze from decoder_h
            # Linear
            model = LinearRegression()
            model.fit(X_train, Y_train)
            Y_train_hat = model.predict(X_train)
            Y_val_hat = model.predict(X_val)
            cods = calculate_cod(Y_val_hat, Y_val, Y_train, axis=(0,))
            out['gaze_from_h']['linear'][-1].append(cods)
            
            # MLP
            batch_size = 512
            max_epochs = 10 # around 10 epochs until plateau for batch_size=512
            model = mlp.MLP(odim=Y_train.shape[-1], hdim=128, device=device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            mseloss = nn.MSELoss()
            X_train_pt = torch.tensor(X_train, dtype=torch.float32, device=device)
            X_val_pt = torch.tensor(X_val, dtype=torch.float32, device=device)
            Y_train_pt = torch.tensor(Y_train, dtype=torch.float32, device=device)
            Y_val_pt = torch.tensor(Y_val, dtype=torch.float32, device=device)
            train_dataset = torch.utils.data.TensorDataset(X_train_pt, Y_train_pt)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            cods_all = []
            for epoch_no in range(max_epochs):
                model.train()
                for x, y in train_dataloader:
                    optimizer.zero_grad()
                    yh = model(x)
                    loss = mseloss(yh, y)
                    loss.backward()
                    optimizer.step()
                model.eval()
                Y_val_hat = model.predict(X_val_pt).detach().cpu().numpy()
                cods = calculate_cod(Y_val_hat, Y_val, Y_train, axis=(0,))
                cods_all.append(cods)
            out['gaze_from_h']['mlp'][-1].append(cods)
            out['gaze_from_h']['mlp_hist'][-1].append(cods_all)
        pass
    
    return out