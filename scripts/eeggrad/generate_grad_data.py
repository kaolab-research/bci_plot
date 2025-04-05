import os
import sys
sys.path.append('../eeg')
import pathlib

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import ast
    
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm

cm = plt.get_cmap('bwr').copy()
cm.set_bad('k', alpha=0.1)

import torch
import torch.nn as nn
import torch.nn.functional as F


from bci_plot.utils import data_util
from bci_plot.metadata import sessions_info_w_day
from bci_plot.metadata import layouts
layout = layouts.layouts['wet64']

from bci_plot.gen_data.grad import instantiate

datadir = pathlib.Path('/data/raspy')
replay_datadir = pathlib.Path('/data/raspy/replayed')
sessions, decoders, folds, subjects, days = list(zip(*sessions_info_w_day.sessions_info))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=str, default='cpu')
    parser.add_argument('--rerun', action='store_true', help='whether to rerun existing data. flag for true.')
    args = parser.parse_args()
    rerun = args.rerun
    try:
        device_id = args.device_id
        device = torch.device(device_id)
    except:
        device_id = ast.literal_eval(args.device_id)
        device = torch.device(device_id)
    
    for (session, decoder, fold, subject, day) in sessions_info_w_day.sessions_info:
        if os.path.exists(f'../../data/grad_vals/{session}.npz'):
            if not rerun:
                continue
        print(session, decoder, fold)
        td0 = data_util.load_data(datadir / f'{session}' / 'task.bin')
        td1 = data_util.load_data(replay_datadir / f'{session}-replay-inputs' / 'task.bin') # same data but with inputs to CNN saved.
        if (td1['decoder_h'] != td0['decoder_h']).sum() != 0:
            print('mismatching decoder_h', session)
        decoder_inputs = td1['decoder_inputs']
        N = len(decoder_inputs)
        M2 = td0['kf_M2']
        
        state_task = td0['state_task'].flatten().copy()
        state_task[td0['numCompletedBlocks'].flatten() < 0] = 99
        state_task[(td0['numCompletedBlocks'].flatten() < 0).sum()] = -1
        start_inds = np.nonzero(state_task[1:None] != state_task[0:-1])[0][1:None] + 1
        end_inds = np.hstack([start_inds[1:None], [len(state_task)]])
        trial_tick = np.full(len(state_task), np.nan, dtype=np.float64)
        for s, e in zip(start_inds, end_inds):
            trial_tick[s:e] = np.arange(e-s)
        
        # Load the experiment's model .yaml file.
        expdir = replay_datadir / f'{session}-replay-inputs' / 'models'
        yaml_name = list(filter(lambda x: x[-5:None] == '.yaml', os.listdir(expdir)))[0]
        with open(expdir / yaml_name, 'r') as f:
            yaml_data = yaml.load(f, Loader=Loader)
        params = yaml_data['modules']['decoder_hidden_inputs']['params']
        with open(pathlib.Path(params['path']) / 'config.yaml', 'r') as f:
            config = yaml.load(f, Loader=Loader)
        # array of indices in [0, 66) to keep. corresponding to channels.
        keep = np.array([i for i, (channel_name, channel_inds) in enumerate(zip(*layout)) if channel_name not in config['data_preprocessor']['ch_to_drop'] + ['REF', 'GND']])
        
        
        # Compile the data from valid ticks
        start = np.nonzero(td0['eeg_step'] >= 1000)[0][0]
        end = N
        valid_ticks = (td0['eeg_step'] >= 1000)*(trial_tick >= 20)*(state_task != 99)
        L = valid_ticks.sum()
        xx = torch.tensor(td1['decoder_inputs'][valid_ticks].copy())
        xx = xx.permute((0, 2, 1))[:, :, :, None]
        m2 = torch.tensor(M2[valid_ticks].copy(), dtype=torch.float, device=device)
        
        # Instantiate the model.
        model = instantiate.instantiate(params, model_architecture='../../bci_plot/gen_data/grad/EEGNet_grad.py')
        model = model.to(device)
        model.eval()
        C = model._depthwise.weight.shape[2]
        F1 = model._conv1.weight.shape[0]
        D = model._depthwise.weight.shape[0] / F1
        if D % 1 != 0:
            raise ValueError('confused about F1 and D')
        D = int(D)
        
        # Calculate gradient averages.
        batch_size = 128
        csgs = [] # channel_scale_grad values
        fsgs = [] # filter_scale_grad values
        for k in range(4):
            channel_scale_grad = np.zeros((C,))
            filter_scale_grad = np.zeros((F1*D, C))
            
            for i in np.arange(len(xx)/batch_size, dtype='int'):
                j1 = i*batch_size
                j2 = min(end, (i+1)*batch_size)
                
                x = xx[j1:j2].detach().clone().to(device)
                x.requires_grad_(True)
                scale = torch.ones((1, C, 1, 1), dtype=torch.float, device=device)
                scale.requires_grad_(True)
                outputs = model(x*scale, return_dataclass='extended')
                proj = (m2[j1:j2]@outputs[2][:, :, None])[:, :, 0]
                projv = proj[:, 2:6]
                projv[:, k].sum().backward()

                channel_scale_grad += scale.grad.detach().cpu().numpy().flatten() # shape (C,)
                filter_scale_grad += (outputs[4]*outputs[4].grad).sum((-1, 0)).detach().cpu().numpy() # shape (D*F1, C)
            
            channel_scale_grad /= L
            filter_scale_grad  /= L
            csgs.append(channel_scale_grad)
            fsgs.append(filter_scale_grad)
            
            torch.cuda.empty_cache()
        csgs = np.array(csgs) # shape (4, C)
        fsgs = np.array(fsgs) # shape (4, D*F1, C)
        
        save_dict = {'L': L, 'channel_scale_grad': csgs, 'filter_scale_grad': fsgs,
                    'conv_weights': model._conv1.weight.detach().cpu().numpy(),
                    'depthwise_weights': model._depthwise.weight.detach().cpu().numpy()}
        fname = f'../../data/grad_vals/{session}.npz'
        with open(fname, 'wb') as f:
            np.savez(f, **save_dict)
    pass
