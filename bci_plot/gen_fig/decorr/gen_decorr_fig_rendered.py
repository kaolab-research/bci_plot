
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import scipy.stats

def hex2float(v):
    v = v.lstrip('#')
    out = tuple(int(v[i:i+2], 16)/255.0 for i in (0, 2, 4))
    return out
color_cycle = [hex2float(item) for item in plt.rcParams['axes.prop_cycle'].by_key()['color']]

def gen_decorr_fig(cods_list, titles=None, n_cols=1):
    '''
    cods: list of dictionaries for each session.
    '''
    n_sessions = len(cods_list)
    n_rows = int(np.ceil(n_sessions/n_cols))
    
    fig = plt.figure(figsize=(16, 16))
    gs = plt.GridSpec(n_rows, n_cols, width_ratios=[1]*n_cols, height_ratios=[1]*n_rows)
    
    for m, cods in enumerate(cods_list):
        i, j = m//n_cols, m%n_cols
        ax = fig.add_subplot(gs[i, j])
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.xlim(cods['delays_s'][0], cods['delays_s'][-1])
        
        mu = np.array(cods['gaze_from_h']['linear']).mean(1)
        sem = scipy.stats.sem(np.array(cods['gaze_from_h']['linear']), axis=1)
        for k in range(2):
            color=color_cycle[k]
            _ = plt.plot(cods['delays_s'], mu[:, k], '-', color=color)
            _ = plt.fill_between(cods['delays_s'], mu[:, k]-sem[:, k], mu[:, k]+sem[:, k], alpha=0.1, color=color, zorder=-4, label='_nolegend_')
            _ = plt.plot(cods['delays_s'], np.array(cods['gaze_from_h']['linear'])[:, :, k], '-', color=color, alpha=0.1, label='_nolegend_')
        
        #mu = np.array(cods['gaze_from_h']['mlp']).mean(1)
        #sem = scipy.stats.sem(np.array(cods['gaze_from_h']['mlp']), axis=1)
        #for k in range(2):
        #    color = color_cycle[2+k]
        #    _ = plt.plot(cods['delays_s'], mu[:, k], '-', color=color)
        #    _ = plt.fill_between(cods['delays_s'], mu[:, k]-sem[:, k], mu[:, k]+sem[:, k], alpha=0.1, color=color, zorder=-4, label='_nolegend_')
        if titles is not None:
            plt.title(titles[m])
        
        if m==0:
            #plt.legend(['x (linear)', 'y (linear)', 'x (MLP)', 'y (MLP)'])
            plt.legend(['x (linear)', 'y (linear)'])
            plt.xlabel('Lag (s)')
            plt.ylabel('Coefficient of Determination (+/- s.e.m.)')
    ylims = [ax.get_ylim() for ax in fig.get_axes()]
    print(ylims)
    #new_ylim = (max(-0.1, min([item[0] for item in ylims])), max([item[1] for item in ylims]))
    ylim_upper = max([np.array(cods['gaze_from_h']['linear']).max() for cods in cods_list])
    ylim_upper = 0.1*np.ceil(ylim_upper / 0.1)
    new_ylim = (-0.1, ylim_upper)
    [ax.set_ylim(*new_ylim) for ax in fig.get_axes()]
    return fig

def gen_decorr_fig2(cods_list, titles=None, n_cols=1):
    '''
    cods: list of dictionaries for each session.
    '''
    n_sessions = len(cods_list)
    n_rows = int(np.ceil(n_sessions/n_cols))
    
    fig = plt.figure(figsize=(16, 16))
    gs = plt.GridSpec(n_rows, n_cols, width_ratios=[1]*n_cols, height_ratios=[1]*n_rows)
    
    vmax = -np.inf
    
    for m, cods in enumerate(cods_list):
        i, j = m//n_cols, m%n_cols
        ax = fig.add_subplot(gs[i, j])
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.xlim(cods['delays_s'][0], cods['delays_s'][-1])
        
#         mu = np.array(cods['gaze_from_h']['linear']).mean(1)
#         sem = scipy.stats.sem(np.array(cods['gaze_from_h']['linear']), axis=1)
#         for k in range(2):
#             color=color_cycle[k]
#             _ = plt.plot(cods['delays_s'], mu[:, k], '-', color=color)
#             _ = plt.fill_between(cods['delays_s'], mu[:, k]-sem[:, k], mu[:, k]+sem[:, k], alpha=0.1, color=color, zorder=-4, label='_nolegend_')
        
        #np.array([item[state][k] for item in cods['gaze_from_h']['linear_by_state']])
        #mu = np.array(cods['gaze_from_h']['linear']).mean(1)
        #sem = scipy.stats.sem(np.array(cods['gaze_from_h']['linear']), axis=1)
        
        data = np.array([[[item[fold][state] for item in cods['gaze_from_h']['linear_by_state']] for state in [0, 1, 2, 3]] for fold in range(5)])
        print(data.shape, np.sum(np.isnan(data)))
        
        for k in range(2):
            direction = ['x', 'y'][k]
            linestyle = ['-', '--'][k]
            data_means = np.nanmean(data, axis=0)
            plt.plot(cods['delays_s'], data_means.max(0)[:, k], label=f'{direction}', linestyle=linestyle, color='k')
            for l in [1, 0, 2, 3]:
                plt.plot(cods['delays_s'], data_means[l, ..., k], label='_nolegend_', linestyle=linestyle, color=color_cycle[l], alpha=0.2)
            #plt.plot(cods['delays_s'], np.nanmean(data, axis=0)[..., k].T, label=f'{direction}', linestyle=linestyle)
            #plt.plot(cods['delays_s'], np.nanmax(data, axis=0).max(0)[:, k], label=f'{direction}', linestyle='--')
            
#             for state in [0, 1, 2, 3]:
#                 data = np.array([[item[fold][state] for item in cods['gaze_from_h']['linear_by_state']] for fold in range(5)])
#                 vmax = max(np.max(data), vmax)
#                 mu = np.nanmean(data, axis=0)
#                 
#                 state_name = ['L', 'R', 'U', 'D'][state]
#                 plt.plot(cods['delays_s'], mu[:, k], color=color_cycle[4*k+state], label=f'{direction}, state={state_name}')
#                 plt.plot(cods['delays_s'], data[:, :, k].T, color=color_cycle[4*k+state], label='_nolegend_', alpha=0.1)
                #for fold in range(5):
                #    plt.plot(cods['delays_s'], data[fold, :, k], color=color_cycle[k])
                #print(data.shape)
                #plt.plot(data)
            #color=color_cycle[k]
            #_ = plt.plot(cods['delays_s'], mu[:, k], '-', color=color)
            #_ = plt.fill_between(cods['delays_s'], mu[:, k]-sem[:, k], mu[:, k]+sem[:, k], alpha=0.1, color=color, zorder=-4, label='_nolegend_')
            #_ = plt.plot(cods['delays_s'], np.array(cods['gaze_from_h']['linear'])[:, :, k], '-', color=color, alpha=0.1, label='_nolegend_')
        
        if titles is not None:
            plt.title(titles[m])
        
        if m==0:
            #plt.legend(['x (linear)', 'y (linear)', 'x (MLP)', 'y (MLP)'])
            plt.legend(['x (linear)', 'y (linear)'])
            plt.xlabel('Lag (s)', size=12)
            plt.ylabel('Coefficient of Determination', size=12)
        if m == 0:
            plt.legend(loc = 'upper left').set_zorder(-5)
        plt.yticks([-0.4, 0.0, 0.4])
        
        _ = plt.gca().twinx()
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.yticks([])
        for l in [1, 0, 2, 3]:
            state_name = ['L', 'R', 'U', 'D'][l]
            plt.plot([], [], color=color_cycle[l], label=state_name)
        if m == 0:
            plt.legend(loc='upper right').set_zorder(-5)
    ylims = [ax.get_ylim() for ax in fig.get_axes()]
    print(ylims)
    #new_ylim = (max(-0.1, min([item[0] for item in ylims])), max([item[1] for item in ylims]))
    ylim_upper = max([np.array(cods['gaze_from_h']['linear']).max() for cods in cods_list])
    ylim_upper = 0.1*np.ceil(ylim_upper / 0.1)
    new_ylim = (-0.1, ylim_upper)
    #[ax.set_ylim(*new_ylim) for ax in fig.get_axes()]
    #[ax.set_ylim(-0.1, ax.get_ylim()[1]) for ax in fig.get_axes()]
    #[ax.set_ylim(-0.1, vmax) for ax in fig.get_axes()]
    [ax.set_ylim(-0.4, 0.4) for ax in fig.get_axes()]
    #[ax.set_yticks([-0.4, 0.0, 0.4]) for ax in fig.get_axes()]
    
    plt.tight_layout()
    return fig
    
