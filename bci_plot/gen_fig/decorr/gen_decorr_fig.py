
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
        
        mu = np.array(cods['gaze_from_h']['mlp']).mean(1)
        sem = scipy.stats.sem(np.array(cods['gaze_from_h']['mlp']), axis=1)
        for k in range(2):
            color = color_cycle[2+k]
            _ = plt.plot(cods['delays_s'], mu[:, k], '-', color=color)
            _ = plt.fill_between(cods['delays_s'], mu[:, k]-sem[:, k], mu[:, k]+sem[:, k], alpha=0.1, color=color, zorder=-4, label='_nolegend_')
        if titles is not None:
            plt.title(titles[m])
        
        if m==0:
            plt.legend(['x (linear)', 'y (linear)', 'x (MLP)', 'y (MLP)'])
            plt.xlabel('Lag (s)')
            plt.ylabel('Coefficient of Determination (+/- s.e.m.)')
    ylims = [ax.get_ylim() for ax in fig.get_axes()]
    new_ylim = (min([item[0] for item in ylims]), max([item[1] for item in ylims]))
    [ax.set_ylim(*new_ylim) for ax in fig.get_axes()]
    return fig
    
