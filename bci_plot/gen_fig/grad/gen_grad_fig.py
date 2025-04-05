import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm
cm = plt.get_cmap('bwr').copy()
cm.set_bad('k', alpha=0.1)
cm2 = plt.get_cmap('bwr').copy()
cm2.set_bad('k', alpha=0.1)

from bci_plot.metadata import layouts
layout = layouts.layouts['wet64']
ch_to_drop = ['M1', 'M2', 'Fpz', 'Fp1', 'Fp2', 'EOG', 'TRGR', 'CNT'] + ['REF', 'GND']
keep = np.array([i for i, (channel_name, channel_inds) in enumerate(zip(*layout)) if channel_name not in ch_to_drop])
keep_names = [channel_name for i, (channel_name, channel_inds) in enumerate(zip(*layout)) if channel_name not in ch_to_drop]
ch_names = keep_names

layout_inds = np.array([channel_inds for (channel_name, channel_inds) in zip(*layout) if channel_name not in ch_to_drop])
layout_gridsize = np.array([9, 11])


import mne
template_montage = mne.channels.make_standard_montage('standard_1020')
offset = 3
dig = [template_montage.dig[offset+template_montage.ch_names.index(name)] for name in ch_names]
montage = mne.channels.DigMontage(dig=dig, ch_names=ch_names)
info = mne.create_info(ch_names, 1000.0, ch_types='eeg', verbose=False)
_ = info.set_montage(montage)

#border = 'mean'
border = 0.0
#extrapolate = 'head'
extrapolate = 'local'

def hex2float(v):
    v = v.lstrip('#')
    out = tuple(int(v[i:i+2], 16)/255.0 for i in (0, 2, 4))
    return out
color_cycle = [hex2float(item) for item in plt.rcParams['axes.prop_cycle'].by_key()['color']]

def make_grid(vals, idx=layout_inds, gridsize=layout_gridsize):
    out = np.full(gridsize, np.nan, dtype=np.float64)
    out[idx[:, 0], idx[:, 1]] = vals
    return out

def gen_fig_partial(channel_scale_grad, path):
    fig = plt.figure(figsize=(20, 5))
    gs = plt.GridSpec(2, 5, width_ratios=[1]*4 + [0.2], height_ratios=[0.2] + [1])
    
    lim0 = np.max(np.abs(channel_scale_grad))
    
    for k in range(4):
        ax = fig.add_subplot(gs[0, k])
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.text(0, 0, ['Right', 'Left', 'Up', 'Down'][k], size=24, ha='center', va='top')
        _ = plt.xticks([])
        _ = plt.yticks([])
        plt.gca().spines[['left', 'top', 'right', 'bottom']].set_visible(False)
    for k in range(4):
        ax = fig.add_subplot(gs[1, k])
        grid = make_grid(channel_scale_grad[k])
        ax.imshow(grid, vmin=-lim0, vmax=lim0, origin='upper', cmap=cm)
        _ = plt.xticks([])
        _ = plt.yticks([])
    ax = fig.add_subplot(gs[1, 4])
    plt.imshow(np.linspace(-lim0, lim0, 257)[:, None], cmap=cm, aspect='auto', origin='lower')
    _ = plt.yticks([0, 128, 256], [-np.round(lim0, 3), 0, np.round(lim0, 3)], size=20)
    ax.yaxis.tick_right()
    _ = plt.xticks([])
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return

def gen_topo_partial(channel_scale_grad, path):
    fig = plt.figure(figsize=(20, 5))
    gs = plt.GridSpec(2, 5, width_ratios=[1]*4 + [0.2], height_ratios=[0.2] + [1])
    
    lim0 = np.max(np.abs(channel_scale_grad))
    
    for k in range(4):
        ax = fig.add_subplot(gs[0, k])
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.text(0, 0, ['Right', 'Left', 'Up', 'Down'][k], size=24, ha='center', va='top')
        _ = plt.xticks([])
        _ = plt.yticks([])
        plt.gca().spines[['left', 'top', 'right', 'bottom']].set_visible(False)
    for k in range(4):
        ax = fig.add_subplot(gs[1, k])
        #grid = make_grid(channel_scale_grad[k])
        im, _ = mne.viz.plot_topomap(channel_scale_grad[k], info, ch_type='eeg', 
                                 vlim=(-lim0, lim0),
                                 axes=ax, cmap=cm, extrapolate=extrapolate, border=border, contours=0, show=False, size=None, image_interp='cubic')
    ax = fig.add_subplot(gs[1, 4])
    plt.imshow(np.linspace(-lim0, lim0, 257)[:, None], cmap=cm, aspect='auto', origin='lower')
    _ = plt.yticks([0, 128, 256], [-np.round(lim0, 3), 0, np.round(lim0, 3)], size=20)
    ax.yaxis.tick_right()
    _ = plt.xticks([])
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return
def gen_topo_seq(csgs, path, row_bkgnd_color=None, row_labels=None):
    N = len(csgs)
    
    fig = plt.figure(figsize=(20, 5*N))
    gs = plt.GridSpec(1+N, 1+4+1, width_ratios=[0.2] + [1]*4 + [0.2], height_ratios=[0.2] + [1]*N)
    
    for k in range(4):
        subfig = fig.add_subfigure(gs[0, 1+k])
        ax = subfig.subplots()
        plt.text(0, 0, ['Right', 'Left', 'Up', 'Down'][k], size=24, ha='center', va='top')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        _ = plt.xticks([])
        _ = plt.yticks([])
        plt.gca().spines[['left', 'top', 'right', 'bottom']].set_visible(False)
    for i in range(N):
        subfig = fig.add_subfigure(gs[1+i, 0])
        ax = subfig.subplots()
        if row_labels is not None:
            plt.text(0, 0, row_labels[i], size=24, ha='center', va='center', rotation=90)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        _ = plt.xticks([])
        _ = plt.yticks([])
        plt.gca().spines[['left', 'top', 'right', 'bottom']].set_visible(False)
    
    lim0 = np.max(np.abs(csgs))
    for i, channel_scale_grad in enumerate(csgs):
        for k in range(4):
            subfig = fig.add_subfigure(gs[1+i, 1+k])
            ax = subfig.subplots()
            im, _ = mne.viz.plot_topomap(channel_scale_grad[k], info, ch_type='eeg', 
                                     vlim=(-lim0, lim0),
                                     axes=ax, cmap=cm, extrapolate=extrapolate, contours=0, show=False, size=None)
            if row_bkgnd_color is not None:
                subfig.set_facecolor(row_bkgnd_color[i])
    
    subfig = fig.add_subfigure(gs[1, 1+4])
    ax = subfig.subplots()
    ax.imshow(np.linspace(-lim0, lim0, 257)[:, None], cmap=cm, aspect='auto', origin='lower')
    _ = plt.yticks([0, 128, 256], [-np.round(lim0, 3), 0, np.round(lim0, 3)], size=20)
    ax.yaxis.tick_right()
    _ = plt.xticks([])
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return

def gen_fig(channel_scale_grad, filter_scale_grad, conv_weights, path,  D=2):
    lim0 = np.max(np.abs(channel_scale_grad))
    lim = np.max(np.abs(filter_scale_grad))
    lim_w = np.max(np.abs(conv_weights))
    F1 = conv_weights.shape[0]

    fig = plt.figure(figsize=(24, 18))

    gs = plt.GridSpec(1+1+F1+1, 2+4*D+1, width_ratios=[1.5] + [1.5] + [1]*4*D + [0.25], height_ratios=[0.2] + [2] + F1*[1] + [0.2])

    bkgnds = [
        (0.0, 0.0, 0.0, 0.05),
        (1.0, 1.0, 0.0, 0.05),
        (0.0, 0.0, 1.0, 0.05),
        (0.0, 1.0, 1.0, 0.05),
    ]

    for k in range(4):
        subfig = fig.add_subfigure(gs[0, 2+2*k:2+2+2*k])
        subfig.set_facecolor(bkgnds[k])
        ax = subfig.subplots()
        ax.text(0, 0, ['Right', 'Left', 'Up', 'Down'][k], ha='center', va='top', size=24)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        _ = plt.xticks([])
        _ = plt.yticks([])
        plt.gca().spines[['left', 'top', 'right', 'bottom']].set_visible(False)
        plt.gca().set_facecolor((0, 0, 0, 0))


    for k in range(4):
        subfig = fig.add_subfigure(gs[1, 2+2*k:2+2+2*k])
        subfig.set_facecolor(bkgnds[k])
        ax = subfig.subplots()
        grid = make_grid(channel_scale_grad[k])
        ax.imshow(grid, vmin=-lim0, vmax=lim0, origin='upper', cmap=cm)
        _ = plt.xticks([])
        _ = plt.yticks([])

    subfig = fig.add_subfigure(gs[1, -1])
    ax = subfig.subplots()
    plt.imshow(np.linspace(-lim0, lim0, 257)[:, None], cmap=cm, aspect='auto', origin='lower')
    _ = plt.yticks([0, 128, 256], [-np.round(lim0, 3), 0, np.round(lim0, 3)], size=20)
    ax.yaxis.tick_right()
    _ = plt.xticks([])



    for i in range(F1):
        for j in range(D):
            for k in range(4):
                subfig = fig.add_subfigure(gs[2+i, 2+j+2*k])
                ax = subfig.subplots()
                grid = make_grid(filter_scale_grad[k][D*i+j])
                ax.imshow(grid, vmin=-lim, vmax=lim, origin='upper', cmap=cm)
                plt.xticks([])
                plt.yticks([])
                subfig.set_facecolor(bkgnds[k])

        subfig = fig.add_subfigure(gs[2+i, 0])
        ax = subfig.subplots()
        w = conv_weights.shape[-1]
        ax.plot(np.arange(w)/100, conv_weights[i, 0, 0, :], color=(0.3, 0.3, 0.3))
        plt.gca().spines[['top', 'right']].set_visible(False)

        plt.ylim(-lim_w, lim_w)
        plt.xlim(0, (w-1)/100)
        if i < F1-1:
            _ = plt.xticks([])
        else:
            plt.xlabel('time within kernel (s)', size=16)
            plt.ylabel('value (a.u.)', size=16)
        plt.xticks(size=14)
        plt.yticks(size=14)
        
        subfig = fig.add_subfigure(gs[2+i, 1])
        ax = subfig.subplots()
        w = conv_weights.shape[-1]
        ax.plot(np.arange(w//2+1)/51*100, np.abs(np.fft.fft(conv_weights[i, 0, 0, :]))[0:w//2+1]**2, color=(0.3, 0.3, 0.3))
        plt.gca().spines[['top', 'right']].set_visible(False)

        plt.xlim(0, w//2+1)
        if i < F1-1:
            _ = plt.xticks([])
        else:
            plt.xlabel('frequency (Hz)', size=16)
            plt.ylabel('kernel spectrogram (a.u.)', size=16)
        plt.xticks(size=14)
        plt.yticks([0, 10], size=14)
        plt.ylim(0, plt.ylim()[1])

    subfig = fig.add_subfigure(gs[2:, -1])
    ax = subfig.subplots()
    plt.imshow(np.linspace(-lim, lim, 257)[:, None], cmap=cm, aspect='auto', origin='lower')
    _ = plt.yticks([0, 128, 256], [-np.round(lim, 3), 0, np.round(lim, 3)], size=20)
    ax.yaxis.tick_right()
    _ = plt.xticks([])


    subfig = fig.add_subfigure(gs[-1, 1:1+4*D])
    ax = subfig.subplots()
    ax.text(0.0, 0, 'Online scaling gradient average', size=20, ha='center', va='top')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    _ = plt.xticks([])
    _ = plt.yticks([])
    plt.gca().spines[['left', 'top', 'right', 'bottom']].set_visible(False)
    plt.gca().set_facecolor((0, 0, 0, 0))

    plt.savefig(path, bbox_inches='tight')
    
    return
def gen_topo(channel_scale_grad, filter_scale_grad, conv_weights, path,  D=2):
    lim0 = np.max(np.abs(channel_scale_grad))
    lim = np.max(np.abs(filter_scale_grad))
    lim_w = np.max(np.abs(conv_weights))
    F1 = conv_weights.shape[0]

    fig = plt.figure(figsize=(24, 18))

    gs = plt.GridSpec(1+1+F1+1, 2+4*D+1, width_ratios=[1.5] + [1.5] + [1]*4*D + [0.25], height_ratios=[0.2] + [2] + F1*[1] + [0.2])

    bkgnds = [
        (0.0, 0.0, 0.0, 0.05),
        (1.0, 1.0, 0.0, 0.05),
        (0.0, 0.0, 1.0, 0.05),
        (0.0, 1.0, 1.0, 0.05),
    ]
    
    # text labeling each direction
    for k in range(4):
        subfig = fig.add_subfigure(gs[0, 2+2*k:2+2+2*k])
        subfig.set_facecolor(bkgnds[k])
        ax = subfig.subplots()
        ax.text(0, 0, ['Right', 'Left', 'Up', 'Down'][k], ha='center', va='top', size=24)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        _ = plt.xticks([])
        _ = plt.yticks([])
        plt.gca().spines[['left', 'top', 'right', 'bottom']].set_visible(False)
        plt.gca().set_facecolor((0, 0, 0, 0))

    # topomaps for each direction
    for k in range(4):
        subfig = fig.add_subfigure(gs[1, 2+2*k:2+2+2*k])
        subfig.set_facecolor(bkgnds[k])

        ax = subfig.subplots()
        im, _ = mne.viz.plot_topomap(channel_scale_grad[k], info, ch_type='eeg', 
                                 vlim=(-lim0, lim0), 
                                 #names=ch_names,
                                 axes=ax, cmap=cm, extrapolate=extrapolate, border=border, contours=0, show=False, size=None)
    # Add labels for the bottom left
    subfig = fig.add_subfigure(gs[1, -1])
    ax = subfig.subplots()
    plt.imshow(np.linspace(-lim0, lim0, 257)[:, None], cmap=cm, aspect='auto', origin='lower')
    _ = plt.yticks([0, 128, 256], [-np.round(lim0, 3), 0, np.round(lim0, 3)], size=20)
    ax.yaxis.tick_right()
    _ = plt.xticks([])


    # i: which temporal filter
    # j: which spatial filter (for each temporal filter)
    # k: which class (right, left, up, down)
    for i in range(F1):
        for j in range(D):
            for k in range(4):
                subfig = fig.add_subfigure(gs[2+i, 2+j+2*k])
                ax = subfig.subplots()
                im, _ = mne.viz.plot_topomap(filter_scale_grad[k][D*i+j], info, ch_type='eeg', 
                                         vlim=(-lim0, lim0),
                                         axes=ax, cmap=cm, extrapolate=extrapolate, border=border, contours=0, show=False, size=None)
                subfig.set_facecolor(bkgnds[k])
        
        # Plot the temporal filter
        subfig = fig.add_subfigure(gs[2+i, 0])
        ax = subfig.subplots()
        w = conv_weights.shape[-1]
        ax.plot(np.arange(w)/100, conv_weights[i, 0, 0, :], color=(0.3, 0.3, 0.3))
        plt.gca().spines[['top', 'right']].set_visible(False)

        plt.ylim(-lim_w, lim_w)
        plt.xlim(0, (w-1)/100)
        if i < F1-1:
            _ = plt.xticks([])
        else:
            plt.xlabel('time within kernel (s)', size=16)
            plt.ylabel('value (a.u.)', size=16)
        plt.xticks(size=14)
        plt.yticks(size=14)
        
        # Plot spectrum of the temporal filter
        subfig = fig.add_subfigure(gs[2+i, 1])
        ax = subfig.subplots()
        w = conv_weights.shape[-1]
        ax.plot(np.arange(w//2+1)/51*100, np.abs(np.fft.fft(conv_weights[i, 0, 0, :]))[0:w//2+1]**2, color=(0.3, 0.3, 0.3))
        plt.gca().spines[['top', 'right']].set_visible(False)

        plt.xlim(0, w//2+1)
        if i < F1-1:
            _ = plt.xticks([])
        else:
            plt.xlabel('frequency (Hz)', size=16)
            plt.ylabel('kernel spectrogram (a.u.)', size=16)
        plt.xticks(size=14)
        plt.yticks([0, 10], size=14)
        plt.ylim(0, plt.ylim()[1])
    
    # Plot the colorbar
    subfig = fig.add_subfigure(gs[2:, -1])
    ax = subfig.subplots()
    plt.imshow(np.linspace(-lim, lim, 257)[:, None], cmap=cm, aspect='auto', origin='lower')
    _ = plt.yticks([0, 128, 256], [-np.round(lim, 3), 0, np.round(lim, 3)], size=20)
    ax.yaxis.tick_right()
    _ = plt.xticks([])

    # Add a label.
    subfig = fig.add_subfigure(gs[-1, 1:1+4*D])
    ax = subfig.subplots()
    ax.text(0.0, 0, 'Online scaling gradient average', size=20, ha='center', va='top')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    _ = plt.xticks([])
    _ = plt.yticks([])
    plt.gca().spines[['left', 'top', 'right', 'bottom']].set_visible(False)
    plt.gca().set_facecolor((0, 0, 0, 0))

    plt.savefig(path, bbox_inches='tight')
    return

def gen_topo_weights(channel_scale_grad, filter_scale_grad, conv_weights, depthwise_weights, path,  D=2):
    lim0 = np.max(np.abs(channel_scale_grad))
    lim_d = np.max(np.abs(depthwise_weights))
    lim = np.max(np.abs(filter_scale_grad))
    lim_w = np.max(np.abs(conv_weights))
    F1 = conv_weights.shape[0]

    fig = plt.figure(figsize=(24, 18))

    gs = plt.GridSpec(1+2+F1+1, 2+2+4*D+1, width_ratios=[1.5] + [1.5] + [1]*2 + [1]*4*D + [0.25], height_ratios=[0.2] + [1.7, 0.3] + F1*[1] + [0.2])

    bkgnds = [
        (0.0, 0.0, 0.0, 0.05),
        (1.0, 1.0, 0.0, 0.05),
        (0.0, 0.0, 1.0, 0.05),
        (0.0, 1.0, 1.0, 0.05),
    ]
    
    # text labeling each direction
    for k in range(4):
        subfig = fig.add_subfigure(gs[0, 2+2+2*k:2+2+2+2*k])
        subfig.set_facecolor(bkgnds[k])
        ax = subfig.subplots()
        ax.text(0, 0, ['Right', 'Left', 'Up', 'Down'][k], ha='center', va='top', size=24)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        _ = plt.xticks([])
        _ = plt.yticks([])
        plt.gca().spines[['left', 'top', 'right', 'bottom']].set_visible(False)
        plt.gca().set_facecolor((0, 0, 0, 0))
    # text labeling weights
    subfig = fig.add_subfigure(gs[1, 2:4])
    ax = subfig.subplots()
    ax.text(0, -0.6, 'Weights', ha='center', va='top', size=24)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    _ = plt.xticks([])
    _ = plt.yticks([])
    plt.gca().spines[['left', 'top', 'right', 'bottom']].set_visible(False)
    plt.gca().set_facecolor((0, 0, 0, 0))
    

    # topomaps for each direction
    for k in range(4):
        subfig = fig.add_subfigure(gs[1:3, 2+2+2*k:2+2+2+2*k])
        subfig.set_facecolor(bkgnds[k])

        ax = subfig.subplots()
        im, _ = mne.viz.plot_topomap(channel_scale_grad[k], info, ch_type='eeg', 
                                 vlim=(-lim0, lim0), 
                                 #names=ch_names,
                                 axes=ax, cmap=cm, extrapolate=extrapolate, border=border, contours=0, show=False, size=None)
    # Add a colorbar for the top row.
    subfig = fig.add_subfigure(gs[1:3, -1])
    ax = subfig.subplots()
    plt.imshow(np.linspace(-lim0, lim0, 257)[:, None], cmap=cm, aspect='auto', origin='lower')
    _ = plt.yticks([0, 128, 256], [-np.round(lim0, 3), 0, np.round(lim0, 3)], size=20)
    ax.yaxis.tick_right()
    _ = plt.xticks([])
    
    # Add a colorbar for the depthwise weights
    subfig = fig.add_subfigure(gs[2, 2:4])
    ax = subfig.subplots()
    plt.imshow(np.linspace(-lim_d, lim_d, 257)[None, :], cmap=cm, aspect='auto', origin='lower')
    _ = plt.xticks([0, 128, 256], [-np.round(lim_d, 3), 0, np.round(lim_d, 3)], size=20)
    ax.xaxis.tick_top()
    _ = plt.yticks([])


    # i: which temporal filter
    # j: which spatial filter (for each temporal filter)
    # k: which class (right, left, up, down)
    for i in range(F1):
        # topomaps for weights
        for j in range(D):
            subfig = fig.add_subfigure(gs[3+i, 2+j])
            ax = subfig.subplots()
            im, _ = mne.viz.plot_topomap(depthwise_weights[D*i+j, 0, :, 0], info, ch_type='eeg', 
                                     vlim=(-lim_d, lim_d),
                                     axes=ax, cmap=cm2, extrapolate=extrapolate, border=border, contours=0, show=False, size=None)
        # topomaps for grads
        for j in range(D):
            for k in range(4):
                subfig = fig.add_subfigure(gs[3+i, 2+2+j+2*k])
                ax = subfig.subplots()
                im, _ = mne.viz.plot_topomap(filter_scale_grad[k][D*i+j], info, ch_type='eeg', 
                                         vlim=(-lim0, lim0),
                                         axes=ax, cmap=cm, extrapolate=extrapolate, border=border, contours=0, show=False, size=None)
                subfig.set_facecolor(bkgnds[k])
        
        # Plot the temporal filter
        subfig = fig.add_subfigure(gs[3+i, 0])
        ax = subfig.subplots()
        w = conv_weights.shape[-1]
        ax.plot(np.arange(w)/100, conv_weights[i, 0, 0, :], color=(0.3, 0.3, 0.3))
        plt.gca().spines[['top', 'right']].set_visible(False)

        plt.ylim(-lim_w, lim_w)
        plt.xlim(0, (w-1)/100)
        if i < F1-1:
            _ = plt.xticks(plt.xticks()[0], ['']*len(plt.xticks()[0]))
        else:
            plt.xlabel('time within kernel (s)', size=16)
            plt.ylabel('value (a.u.)', size=16)
        plt.xticks(size=14)
        plt.yticks(size=14)
        
        # Plot spectrum of the temporal filter
        subfig = fig.add_subfigure(gs[3+i, 1])
        ax = subfig.subplots()
        w = conv_weights.shape[-1]
        ax.plot(np.arange(w//2+1)/51*100, np.abs(np.fft.fft(conv_weights[i, 0, 0, :]))[0:w//2+1]**2, color=(0.3, 0.3, 0.3))
        plt.gca().spines[['top', 'right']].set_visible(False)

        plt.xlim(0, (w//2+1)/51*100)
        if i < F1-1:
            _ = plt.xticks([0, 10, 20, 30, 40, 50], ['']*6)
        else:
            _ = plt.xticks([0, 10, 20, 30, 40, 50])
            plt.xlabel('frequency (Hz)', size=16)
            plt.ylabel('kernel spectrogram (a.u.)', size=16)
        plt.xticks(size=14)
        plt.yticks([0, 10], size=14)
        plt.ylim(0, plt.ylim()[1])
    
    # Plot the colorbar
    subfig = fig.add_subfigure(gs[3:, -1])
    ax = subfig.subplots()
    plt.imshow(np.linspace(-lim, lim, 257)[:, None], cmap=cm, aspect='auto', origin='lower')
    _ = plt.yticks([0, 128, 256], [-np.round(lim, 3), 0, np.round(lim, 3)], size=20)
    ax.yaxis.tick_right()
    _ = plt.xticks([])

    # Add a label.
    subfig = fig.add_subfigure(gs[-1, 1:1+4*D])
    ax = subfig.subplots()
    ax.text(0.0, 0, 'Online scaling gradient average', size=20, ha='center', va='top')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    _ = plt.xticks([])
    _ = plt.yticks([])
    plt.gca().spines[['left', 'top', 'right', 'bottom']].set_visible(False)
    plt.gca().set_facecolor((0, 0, 0, 0))

    plt.savefig(path, bbox_inches='tight')
    return