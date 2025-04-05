
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import ast
import scipy.stats
import copy
import itertools

from bci_plot.utils import data_util
from bci_plot.utils import eeg_stats
from bci_plot.utils import kernel_smoothing

def hex2float(v):
    v = v.lstrip('#')
    out = tuple(int(v[i:i+2], 16)/255.0 for i in (0, 2, 4))
    return out
color_cycle = [hex2float(item) for item in plt.rcParams['axes.prop_cycle'].by_key()['color']]
color_cycle.pop(3) # remove red


def gen_adaptation_fig(stats, smooth_std=0.5):
    '''
    stats: list of dictionaries.
        Each dictionary: 
            subject:
                day_bps: list of arrays for each day. Fitts bits per second for each block.
                day_success: list of arrays for each day. Success percentage for each block.
                day_ttt: list of arrays for each day. Time to target for each block.
    '''
    alpha = 0.3
    
    stats = copy.deepcopy(stats) # prevent modification in place
    n_sets = len(stats)
    for stats_i in stats:
        for subject, subject_stats in stats_i.items():
            subject_stats['day_x'] = [day_no + (0.5+np.arange(len(day_bps_i)))/len(day_bps_i) for day_no, day_bps_i in enumerate(subject_stats['day_bps'])]
            subject_stats['block_x'] = np.hstack(subject_stats['day_x']) # position on plot
            subject_stats['block_bps'] = np.hstack(subject_stats['day_bps'])
            subject_stats['block_success'] = np.hstack(subject_stats['day_success'])
            subject_stats['block_ttt'] = np.hstack(subject_stats['day_ttt'])
            subject_stats['block_bps_ftt'] = np.hstack(subject_stats['day_bps_ftt'])
            subject_stats['block_is_eval'] = np.hstack(subject_stats['day_is_eval'])
            subject_stats['block_td'] = np.hstack(subject_stats['day_td'])
            subject_stats['block_decoder_name'] = np.hstack(subject_stats['decoder_name'])
    
    #d
    fig = plt.figure(figsize=(16, 22))
    n_col_1 = 12
    n_rows_per_set = 2
    gs = plt.GridSpec(1+n_rows_per_set*n_sets+2+1, n_col_1+1, width_ratios=[1/n_col_1]*n_col_1 + [0.1], height_ratios=[0.4] + ([1]*n_rows_per_set)*n_sets + [1]*2 + [1.4])
    
    n_days = max([max([len(subject_stats['day_bps']) for _, subject_stats in stats_i.items()]) for stats_i in stats])
    n_subjects_per = [len(stats_i) for stats_i in stats]
    n_subjects_cum_left = [0] + [int(item) for item in np.cumsum(n_subjects_per)[0:-1]]
    n_subjects = sum(n_subjects_per)
    
    for set_no, stats_i in enumerate(stats):
        xsmooth = np.linspace(0, n_days, 1+n_days*50)
        
        ################################################################################
        ax = fig.add_subplot(gs[1+set_no*n_rows_per_set, 0:-1])
        plt.gca().spines[['top', 'right', 'left']].set_visible(False)
        plt.xlim(0, n_days)
        plt.ylim(0, 1.1)
        for j, (subject, subject_stats) in enumerate(stats_i.items()):
            sj = n_subjects_cum_left[set_no] + j
            sel = subject_stats['block_is_eval']==0
            plt.scatter(subject_stats['block_x'][sel], 
                        subject_stats['block_success'][sel], 
                       s=625*subject_stats['block_td'][sel]**2,
                       marker='o', 
                        facecolors=color_cycle[sj] + (alpha,), 
                        edgecolors=color_cycle[sj] + (alpha,),
                       label='_nolegend_')
            plt.scatter(subject_stats['block_x'][~sel], 
                        subject_stats['block_success'][~sel], 
                       s=625*subject_stats['block_td'][~sel]**2,
                       marker='o', 
                        facecolors=(0.0, 0.0, 0.0, 0.0), 
                        edgecolors=color_cycle[sj] + (alpha,),
                       label='_nolegend_')
            smoother = kernel_smoothing.GaussianKernelSmoother(
                subject_stats['block_x'],
                subject_stats['block_success'],
                smooth_std,
            )
            ysmooth = smoother(xsmooth)
            for m, bps_m in enumerate(subject_stats['day_success']):
                if len(bps_m) == 0:
                    ysmooth[(xsmooth > m)*(xsmooth < m+1)] = np.nan
            plt.plot(xsmooth, ysmooth, lw=3.0, color=color_cycle[sj])
            ####
            ylims = plt.ylim(plt.ylim())
            decoder_change_blocks = np.nonzero(subject_stats['block_decoder_name'][1:None] != subject_stats['block_decoder_name'][0:-1])[0]
            if len(decoder_change_blocks) > 0:
                decoder_change_blocks += 1
                for change_idx, block_no in enumerate(decoder_change_blocks):
                    xtmp = subject_stats['block_x'][block_no] - 0.5*(subject_stats['block_x'][block_no+1] - subject_stats['block_x'][block_no])
                    plt.plot(
                        [xtmp, xtmp],
                        [ylims[0], ylims[0] + 0.10*(ylims[1] - ylims[0])],
                        color='purple',
                        label='_nolegend_',
                        lw=6,
                    )
                    if change_idx == 0:
                        plt.text(xtmp - 0.01*n_days, ylims[0] + 0.05*(ylims[1] - ylims[0]), 'decoder retraining', va='bottom', ha='right', color='purple', size=14)
            if 'omitted_days' in subject_stats:
                for ii, day_no in enumerate(subject_stats['omitted_days']):
                    plt.plot([day_no, day_no], ylims, 'c', lw=4, zorder=-4)
                    if ii == 0:
                        plt.text(day_no + 0.01*n_days, ylims[0] + 0.15*(ylims[1] - ylims[0]), 'omitted day', color='c', size=14, ha='left', va='center')
            ####
        plt.legend(list(stats_i.keys()), prop={'size': 10}, loc='lower left').set_zorder(-5)
        plt.ylabel('Success %', size=14)
        plt.plot([0, 0], [0, 1], 'k', label='_nolegend_')
        _ = plt.yticks([0, 0.5, 1.0], [0, 50, 100])
        plt.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=False)
        plt.yticks(size=10)
        
        if set_no == 0:
            plt.gca().twinx()
            plt.gca().spines[['top', 'right']].set_visible(False)
            _ = plt.scatter([], [], s=625*0.4**2, marker='o', facecolors=(0.0, 0.0, 0.0, 1.0), edgecolors=(0.0, 0.0, 0.0, 1.0))
            _ = plt.scatter([], [], s=625*0.4**2, marker='o', facecolors=(0.0, 0.0, 0.0, 0.0), edgecolors=(0.0, 0.0, 0.0, 1.0))
            _ = plt.yticks([])
            plt.legend(['Adaptation', 'Evaluation'], loc='lower right', prop={'size': 10})
        
        ################################################################################
        ax = fig.add_subplot(gs[2+set_no*n_rows_per_set, 0:-1])
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.xlim(0, n_days)
        for j, (subject, subject_stats) in enumerate(stats_i.items()):
            sj = n_subjects_cum_left[set_no] + j
            sel = subject_stats['block_is_eval']==0
            plt.scatter(subject_stats['block_x'][sel], 
                        subject_stats['block_ttt'][sel], 
                       s=625*subject_stats['block_td'][sel]**2,
                       marker='o', 
                        facecolors=color_cycle[sj] + (alpha,), 
                        edgecolors=color_cycle[sj] + (alpha,),
                       label='_nolegend_')
            plt.scatter(subject_stats['block_x'][~sel], 
                        subject_stats['block_ttt'][~sel], 
                       s=625*subject_stats['block_td'][~sel]**2,
                       marker='o', 
                        facecolors=(0.0, 0.0, 0.0, 0.0), 
                        edgecolors=color_cycle[sj] + (alpha,),
                       label='_nolegend_')
            smoother = kernel_smoothing.GaussianKernelSmoother(
                subject_stats['block_x'],
                subject_stats['block_ttt'],
                smooth_std,
            )
            ysmooth = smoother(xsmooth)
            for m, bps_m in enumerate(subject_stats['day_bps']):
                if len(bps_m) == 0:
                    ysmooth[(xsmooth > m)*(xsmooth < m+1)] = np.nan
            plt.plot(xsmooth, ysmooth, lw=3.0, color=color_cycle[sj])
        plt.ylabel('Time to acquisition (seconds)', size=14)
        plt.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=False)
        plt.yticks(size=10)
        #
        ax = plt.gca()
        ax.xaxis.set_ticks(np.arange(0, n_days)+0.5, minor=True)
        ax.xaxis.set_ticklabels(np.arange(0, n_days)+1, minor=True)
        plt.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=False)
        plt.tick_params(axis='x', which='minor', length=6, bottom=False, top=False, labelbottom=True)
        plt.xlabel('Adaptation day', size=14)
        plt.xticks(size=10)
    #
    
    # Target diameter
    ax = fig.add_subplot(gs[1, -1])
    plt.gca().spines[['left', 'top', 'right', 'bottom']].set_visible(False)
    szs = np.array([0.25, 0.3, 0.4, 0.6])
    ys = np.linspace(0.1, 0.9, len(szs))
    plt.scatter(np.zeros(len(szs)), ys, s=625*szs**2, color='k')
    plt.xlim(-1, 1.0)
    plt.ylim(0, 1)
    for i, sz in enumerate(szs):
        plt.text(-1.3, ys[i], str(sz), color='k', ha='left', va='center', size=10)
    _ = plt.xticks([])
    _ = plt.yticks([])
    plt.text(1.1, (ys[0] + ys[-1])/2.0, 'Target diameter (a.u.)', color='k', ha='right', va='center', rotation=90, size=12)
    
    # Used for summary statistics
    subjects_list = list(itertools.chain(*[[subject for subject, subject_stats in stats_i.items()] for stats_i in stats]))
    stats_list = list(itertools.chain(*[[subject_stats for subject, subject_stats in stats_i.items()] for stats_i in stats]))
    
    # xy velocity violin plots
    ax = fig.add_subplot(gs[1+n_rows_per_set*n_sets:1+n_rows_per_set*n_sets+2, 0:6])
    plt.gca().spines[['top', 'right']].set_visible(False)
    violin_x_scale = 0.25
    for sj, subject_stats in enumerate(stats_list):
        x_vals_kf = np.hstack([np.hstack([-violin_x_scale*day_info['vals'] + day_no + 0.5, [np.nan]]) for day_no, day_info in enumerate(subject_stats['violin_info']['x_info_kf'])])
        x_coords_kf = np.hstack([np.hstack([day_info['coords'], [np.nan]]) for day_info in subject_stats['violin_info']['x_info_kf']])
        x_vals_en = np.hstack([np.hstack([violin_x_scale*day_info['vals'] + day_no + 0.5, [np.nan]]) for day_no, day_info in enumerate(subject_stats['violin_info']['x_info_eegnet'])])
        x_coords_en = np.hstack([np.hstack([day_info['coords'], [np.nan]]) for day_info in subject_stats['violin_info']['x_info_eegnet']])
        
        plt.plot(x_vals_kf, x_coords_kf, color=color_cycle[sj])
        plt.plot(x_vals_en, x_coords_en, '--', color=color_cycle[sj], label='_nolegend_')
    ax = plt.gca()
    ax.xaxis.set_ticks(np.arange(0, n_days)+0.5, minor=True)
    ax.xaxis.set_ticklabels(np.arange(0, n_days)+1, minor=True)
    plt.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=False)
    plt.tick_params(axis='x', which='minor', length=6, bottom=False, top=False, labelbottom=True)
    plt.xlabel('Adaptation day', size=12)
    plt.xticks(size=10)
    plt.ylabel('x velocity distribution')
    plt.plot([0, n_days], [0, 0], 'k', zorder=-4, label='_nolegend_', lw=1, alpha=0.3)
    plt.xlim(0, n_days)
    plt.legend(subjects_list, prop={'size': 10}, loc='lower left').set_zorder(-5)
    plt.ylim(-3.5, 2)
    
    # make legend.
    plt.gca().twinx()
    plt.gca().spines[['top', 'right']].set_visible(False)
    _ = plt.plot([], [], 'k')
    _ = plt.plot([], [], 'k--')
    _ = plt.yticks([])
    plt.legend(['EEGNet-KF', 'EEGNet'], loc='lower right', prop={'size': 10})
    
    # y
    ax = fig.add_subplot(gs[1+n_rows_per_set*n_sets:1+n_rows_per_set*n_sets+2, 6:12])
    plt.gca().spines[['top', 'right']].set_visible(False)
    violin_x_scale = 0.2
    for sj, subject_stats in enumerate(stats_list):
        y_vals_kf = np.hstack([np.hstack([-violin_x_scale*day_info['vals'] + day_no + 0.5, [np.nan]]) for day_no, day_info in enumerate(subject_stats['violin_info']['y_info_kf'])])
        y_coords_kf = np.hstack([np.hstack([day_info['coords'], [np.nan]]) for day_info in subject_stats['violin_info']['y_info_kf']])
        y_vals_en = np.hstack([np.hstack([violin_x_scale*day_info['vals'] + day_no + 0.5, [np.nan]]) for day_no, day_info in enumerate(subject_stats['violin_info']['y_info_eegnet'])])
        y_coords_en = np.hstack([np.hstack([day_info['coords'], [np.nan]]) for day_info in subject_stats['violin_info']['y_info_eegnet']])
        
        plt.plot(y_vals_kf, y_coords_kf, color=color_cycle[sj])
        plt.plot(y_vals_en, y_coords_en, '--', color=color_cycle[sj])
    ax = plt.gca()
    ax.xaxis.set_ticks(np.arange(0, n_days)+0.5, minor=True)
    ax.xaxis.set_ticklabels(np.arange(0, n_days)+1, minor=True)
    plt.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=False)
    plt.tick_params(axis='x', which='minor', length=6, bottom=False, top=False, labelbottom=True)
    plt.xlabel('Adaptation day', size=12)
    plt.xticks(size=10)
    plt.ylabel('y velocity distribution')
    plt.plot([0, n_days], [0, 0], 'k', zorder=-4, label='_nolegend_', lw=1, alpha=0.3)
    plt.xlim(0, n_days)
    plt.ylim(-3.5, 2)
    
    ########
    
    boxplot_width = 0.3
    median_color = 'c'
    
    # average success percentage over all days
    ax = fig.add_subplot(gs[-1, 0:3])
    plt.gca().spines[['top', 'right']].set_visible(False)
    data_pts = [subject_stats['block_success'] for subject_stats in stats_list]
    bplot = plt.boxplot(data_pts, positions=np.arange(n_subjects), widths=boxplot_width, vert=True, patch_artist=True, showfliers=True)
    for median in bplot['medians']:
        median.set_color(median_color)
    for patch in bplot['boxes']:
        patch.set_facecolor('grey')
    _ = plt.xticks(np.arange(n_subjects), subjects_list)
    plt.ylabel('Success %')
    plt.ylim(0, 1)
    
    # average time to target over all days
    ax = fig.add_subplot(gs[-1, 3:6])
    plt.gca().spines[['top', 'right']].set_visible(False)
    data_pts = [subject_stats['block_ttt'] for subject_stats in stats_list]
    print('time to target median', subjects_list, [np.median(item) for item in data_pts])
    bplot = plt.boxplot(data_pts, positions=np.arange(n_subjects), widths=boxplot_width, vert=True, patch_artist=True, showfliers=True)
    for median in bplot['medians']:
        median.set_color(median_color)
    for patch in bplot['boxes']:
        patch.set_facecolor('grey')
    _ = plt.xticks(np.arange(n_subjects), subjects_list)
    plt.ylabel('Time to acquisition (s)')
    plt.ylim(0, 18)
    
    plt.tight_layout()
    return fig







####
####
####



def gen_adaptation_supp_fig(stats, smooth_std=0.5):
    '''
    stats: list of dictionaries.
        Each dictionary: 
            subject:
                day_bps: list of arrays for each day. Fitts bits per second for each block.
                day_success: list of arrays for each day. Success percentage for each block.
                day_ttt: list of arrays for each day. Time to target for each block.
    '''
    alpha = 0.3
    
    stats = copy.deepcopy(stats) # prevent modification in place
    n_sets = len(stats)
    for stats_i in stats:
        for subject, subject_stats in stats_i.items():
            subject_stats['day_x'] = [day_no + (0.5+np.arange(len(day_bps_i)))/len(day_bps_i) for day_no, day_bps_i in enumerate(subject_stats['day_bps'])]
            subject_stats['block_x'] = np.hstack(subject_stats['day_x']) # position on plot
            subject_stats['block_bps'] = np.hstack(subject_stats['day_bps'])
            subject_stats['block_success'] = np.hstack(subject_stats['day_success'])
            subject_stats['block_ttt'] = np.hstack(subject_stats['day_ttt'])
            subject_stats['block_ftts'] = np.hstack(subject_stats['day_ftts'])
            subject_stats['block_bps_ftt'] = np.hstack(subject_stats['day_bps_ftt'])
            subject_stats['block_is_eval'] = np.hstack(subject_stats['day_is_eval'])
            subject_stats['block_td'] = np.hstack(subject_stats['day_td'])
            subject_stats['block_decoder_name'] = np.hstack(subject_stats['decoder_name'])
    
    #d
    fig = plt.figure(figsize=(16, 30))
    n_col_1 = 12
    n_rows_per_set = 3
    gs = plt.GridSpec(1+n_rows_per_set*n_sets+2+1, n_col_1+1, width_ratios=[1/n_col_1]*n_col_1 + [0.1], height_ratios=[0.4] + ([1]*n_rows_per_set)*n_sets + [1]*2 + [1.4])
    
    n_days = max([max([len(subject_stats['day_bps']) for _, subject_stats in stats_i.items()]) for stats_i in stats])
    n_subjects_per = [len(stats_i) for stats_i in stats]
    n_subjects_cum_left = [0] + [int(item) for item in np.cumsum(n_subjects_per)[0:-1]]
    n_subjects = sum(n_subjects_per)
    
    for set_no, stats_i in enumerate(stats):
        xsmooth = np.linspace(0, n_days, 1+n_days*50)
        
        ################################################################################
        ax = fig.add_subplot(gs[1+set_no*n_rows_per_set, 0:-1])
        plt.gca().spines[['top', 'right', 'left']].set_visible(False)
        plt.xlim(0, n_days)
        #plt.ylim(0, 1.1)
        for j, (subject, subject_stats) in enumerate(stats_i.items()):
            sj = n_subjects_cum_left[set_no] + j
            sel = subject_stats['block_is_eval']==0
            plt.scatter(subject_stats['block_x'][sel], 
                        subject_stats['block_bps'][sel], 
                       s=625*subject_stats['block_td'][sel]**2,
                       marker='o', 
                        facecolors=color_cycle[sj] + (alpha,), 
                        edgecolors=color_cycle[sj] + (alpha,),
                       label='_nolegend_')
            plt.scatter(subject_stats['block_x'][~sel], 
                        subject_stats['block_bps'][~sel], 
                       s=625*subject_stats['block_td'][~sel]**2,
                       marker='o', 
                        facecolors=(0.0, 0.0, 0.0, 0.0), 
                        edgecolors=color_cycle[sj] + (alpha,),
                       label='_nolegend_')
            smoother = kernel_smoothing.GaussianKernelSmoother(
                subject_stats['block_x'],
                subject_stats['block_bps'],
                smooth_std,
            )
            ysmooth = smoother(xsmooth)
            for m, bps_m in enumerate(subject_stats['day_bps']):
                if len(bps_m) == 0:
                    ysmooth[(xsmooth > m)*(xsmooth < m+1)] = np.nan
            plt.plot(xsmooth, ysmooth, lw=3.0, color=color_cycle[sj])
            ####
            ylims = plt.ylim(plt.ylim())
            decoder_change_blocks = np.nonzero(subject_stats['block_decoder_name'][1:None] != subject_stats['block_decoder_name'][0:-1])[0]
            if len(decoder_change_blocks) > 0:
                decoder_change_blocks += 1
                for change_idx, block_no in enumerate(decoder_change_blocks):
                    xtmp = subject_stats['block_x'][block_no] - 0.5*(subject_stats['block_x'][block_no+1] - subject_stats['block_x'][block_no])
                    plt.plot(
                        [xtmp, xtmp],
                        [ylims[0], ylims[0] + 0.10*(ylims[1] - ylims[0])],
                        color='purple',
                        label='_nolegend_',
                        lw=6,
                    )
                    if change_idx == 0:
                        plt.text(xtmp - 0.01*n_days, ylims[0] + 0.05*(ylims[1] - ylims[0]), 'decoder retraining', va='bottom', ha='right', color='purple', size=14)
            if 'omitted_days' in subject_stats:
                for ii, day_no in enumerate(subject_stats['omitted_days']):
                    plt.plot([day_no, day_no], ylims, 'c', lw=4, zorder=-4)
                    if ii == 0:
                        plt.text(day_no + 0.01*n_days, ylims[0] + 0.15*(ylims[1] - ylims[0]), 'omitted day', color='c', size=14, ha='left', va='center')
            ####
        plt.legend(list(stats_i.keys()), prop={'size': 10}, loc='lower left').set_zorder(-5)
        plt.ylabel('Fitts ITR\n(acquisition T - hold T)', size=14)
        plt.plot([0, 0], [0, 1], 'k', label='_nolegend_')
        #_ = plt.yticks([0, 0.5, 1.0], [0, 50, 100])
        plt.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=False)
        plt.yticks(size=10)
        
        if set_no == 0:
            plt.gca().twinx()
            plt.gca().spines[['top', 'right']].set_visible(False)
            _ = plt.scatter([], [], s=625*0.4**2, marker='o', facecolors=(0.0, 0.0, 0.0, 1.0), edgecolors=(0.0, 0.0, 0.0, 1.0))
            _ = plt.scatter([], [], s=625*0.4**2, marker='o', facecolors=(0.0, 0.0, 0.0, 0.0), edgecolors=(0.0, 0.0, 0.0, 1.0))
            _ = plt.yticks([])
            plt.legend(['Adaptation', 'Evaluation'], loc='lower right', prop={'size': 10})
            
        ################################################################################
        ax = fig.add_subplot(gs[2+set_no*n_rows_per_set, 0:-1])
        plt.gca().spines[['top', 'right', 'left']].set_visible(False)
        plt.xlim(0, n_days)
        #plt.ylim(0, 1.1)
        for j, (subject, subject_stats) in enumerate(stats_i.items()):
            sj = n_subjects_cum_left[set_no] + j
            sel = subject_stats['block_is_eval']==0
            plt.scatter(subject_stats['block_x'][sel], 
                        subject_stats['block_bps_ftt'][sel], 
                       s=625*subject_stats['block_td'][sel]**2,
                       marker='o', 
                        facecolors=color_cycle[sj] + (alpha,), 
                        edgecolors=color_cycle[sj] + (alpha,),
                       label='_nolegend_')
            plt.scatter(subject_stats['block_x'][~sel], 
                        subject_stats['block_bps_ftt'][~sel], 
                       s=625*subject_stats['block_td'][~sel]**2,
                       marker='o', 
                        facecolors=(0.0, 0.0, 0.0, 0.0), 
                        edgecolors=color_cycle[sj] + (alpha,),
                       label='_nolegend_')
            smoother = kernel_smoothing.GaussianKernelSmoother(
                subject_stats['block_x'],
                subject_stats['block_bps_ftt'],
                smooth_std,
            )
            ysmooth = smoother(xsmooth)
            for m, bps_m in enumerate(subject_stats['day_bps']):
                if len(bps_m) == 0:
                    ysmooth[(xsmooth > m)*(xsmooth < m+1)] = np.nan
            plt.plot(xsmooth, ysmooth, lw=3.0, color=color_cycle[sj])
            ####
            ylims = plt.ylim(plt.ylim())
            decoder_change_blocks = np.nonzero(subject_stats['block_decoder_name'][1:None] != subject_stats['block_decoder_name'][0:-1])[0]
            if len(decoder_change_blocks) > 0:
                decoder_change_blocks += 1
                for change_idx, block_no in enumerate(decoder_change_blocks):
                    xtmp = subject_stats['block_x'][block_no] - 0.5*(subject_stats['block_x'][block_no+1] - subject_stats['block_x'][block_no])
                    plt.plot(
                        [xtmp, xtmp],
                        [ylims[0], ylims[0] + 0.10*(ylims[1] - ylims[0])],
                        color='purple',
                        label='_nolegend_',
                        lw=6,
                    )
                    if change_idx == 0:
                        plt.text(xtmp - 0.01*n_days, ylims[0] + 0.05*(ylims[1] - ylims[0]), 'decoder retraining', va='bottom', ha='right', color='purple', size=14)
            if 'omitted_days' in subject_stats:
                for ii, day_no in enumerate(subject_stats['omitted_days']):
                    plt.plot([day_no, day_no], ylims, 'c', lw=4, zorder=-4)
                    if ii == 0:
                        plt.text(day_no + 0.01*n_days, ylims[0] + 0.15*(ylims[1] - ylims[0]), 'omitted day', color='c', size=14, ha='left', va='center')
            ####
        plt.legend(list(stats_i.keys()), prop={'size': 10}, loc='lower left').set_zorder(-5)
        plt.ylabel('Fitts ITR (ftt)', size=14)
        plt.plot([0, 0], [0, 1], 'k', label='_nolegend_')
        #_ = plt.yticks([0, 0.5, 1.0], [0, 50, 100])
        plt.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=False)
        plt.yticks(size=10)
        
        if set_no == 0:
            plt.gca().twinx()
            plt.gca().spines[['top', 'right']].set_visible(False)
            _ = plt.scatter([], [], s=625*0.4**2, marker='o', facecolors=(0.0, 0.0, 0.0, 1.0), edgecolors=(0.0, 0.0, 0.0, 1.0))
            _ = plt.scatter([], [], s=625*0.4**2, marker='o', facecolors=(0.0, 0.0, 0.0, 0.0), edgecolors=(0.0, 0.0, 0.0, 1.0))
            _ = plt.yticks([])
            plt.legend(['Adaptation', 'Evaluation'], loc='lower right', prop={'size': 10})
        
        ################################################################################
        ax = fig.add_subplot(gs[3+set_no*n_rows_per_set, 0:-1])
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.xlim(0, n_days)
        for j, (subject, subject_stats) in enumerate(stats_i.items()):
            sj = n_subjects_cum_left[set_no] + j
            sel = subject_stats['block_is_eval']==0
            plt.scatter(subject_stats['block_x'][sel], 
                        subject_stats['block_ftts'][sel], 
                       s=625*subject_stats['block_td'][sel]**2,
                       marker='o', 
                        facecolors=color_cycle[sj] + (alpha,), 
                        edgecolors=color_cycle[sj] + (alpha,),
                       label='_nolegend_')
            plt.scatter(subject_stats['block_x'][~sel], 
                        subject_stats['block_ftts'][~sel], 
                       s=625*subject_stats['block_td'][~sel]**2,
                       marker='o', 
                        facecolors=(0.0, 0.0, 0.0, 0.0), 
                        edgecolors=color_cycle[sj] + (alpha,),
                       label='_nolegend_')
            smoother = kernel_smoothing.GaussianKernelSmoother(
                subject_stats['block_x'],
                subject_stats['block_ftts'],
                smooth_std,
            )
            ysmooth = smoother(xsmooth)
            for m, bps_m in enumerate(subject_stats['day_bps']):
                if len(bps_m) == 0:
                    ysmooth[(xsmooth > m)*(xsmooth < m+1)] = np.nan
            plt.plot(xsmooth, ysmooth, lw=3.0, color=color_cycle[sj])
        plt.ylabel('First touch time (seconds)', size=14)
        plt.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=False)
        plt.yticks(size=10)
        #
        ax = plt.gca()
        ax.xaxis.set_ticks(np.arange(0, n_days)+0.5, minor=True)
        ax.xaxis.set_ticklabels(np.arange(0, n_days)+1, minor=True)
        plt.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=False)
        plt.tick_params(axis='x', which='minor', length=6, bottom=False, top=False, labelbottom=True)
        plt.xlabel('Adaptation day', size=14)
        plt.xticks(size=10)
    #
    
    # Target diameter
    ax = fig.add_subplot(gs[1, -1])
    plt.gca().spines[['left', 'top', 'right', 'bottom']].set_visible(False)
    szs = np.array([0.25, 0.3, 0.4, 0.6])
    ys = np.linspace(0.1, 0.9, len(szs))
    plt.scatter(np.zeros(len(szs)), ys, s=625*szs**2, color='k')
    plt.xlim(-1, 1.0)
    plt.ylim(0, 1)
    for i, sz in enumerate(szs):
        plt.text(-1.3, ys[i], str(sz), color='k', ha='left', va='center', size=10)
    _ = plt.xticks([])
    _ = plt.yticks([])
    plt.text(1.1, (ys[0] + ys[-1])/2.0, 'Target diameter (a.u.)', color='k', ha='right', va='center', rotation=90, size=12)
    
    # Used for summary statistics
    subjects_list = list(itertools.chain(*[[subject for subject, subject_stats in stats_i.items()] for stats_i in stats]))
    stats_list = list(itertools.chain(*[[subject_stats for subject, subject_stats in stats_i.items()] for stats_i in stats]))
    
#     # xy velocity violin plots
#     ax = fig.add_subplot(gs[1+n_rows_per_set*n_sets:1+n_rows_per_set*n_sets+2, 0:6])
#     plt.gca().spines[['top', 'right']].set_visible(False)
#     violin_x_scale = 0.25
#     for sj, subject_stats in enumerate(stats_list):
#         x_vals_kf = np.hstack([np.hstack([-violin_x_scale*day_info['vals'] + day_no + 0.5, [np.nan]]) for day_no, day_info in enumerate(subject_stats['violin_info']['x_info_kf'])])
#         x_coords_kf = np.hstack([np.hstack([day_info['coords'], [np.nan]]) for day_info in subject_stats['violin_info']['x_info_kf']])
#         x_vals_en = np.hstack([np.hstack([violin_x_scale*day_info['vals'] + day_no + 0.5, [np.nan]]) for day_no, day_info in enumerate(subject_stats['violin_info']['x_info_eegnet'])])
#         x_coords_en = np.hstack([np.hstack([day_info['coords'], [np.nan]]) for day_info in subject_stats['violin_info']['x_info_eegnet']])
        
#         plt.plot(x_vals_kf, x_coords_kf, color=color_cycle[sj])
#         plt.plot(x_vals_en, x_coords_en, '--', color=color_cycle[sj], label='_nolegend_')
#     ax = plt.gca()
#     ax.xaxis.set_ticks(np.arange(0, n_days)+0.5, minor=True)
#     ax.xaxis.set_ticklabels(np.arange(0, n_days), minor=True)
#     plt.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=False)
#     plt.tick_params(axis='x', which='minor', length=6, bottom=False, top=False, labelbottom=True)
#     plt.xlabel('Adaptation day', size=12)
#     plt.xticks(size=10)
#     plt.ylabel('x velocity distribution')
#     plt.plot([0, n_days], [0, 0], 'k', zorder=-4, label='_nolegend_', lw=1, alpha=0.3)
#     plt.xlim(0, n_days)
#     plt.legend(subjects_list, prop={'size': 10}, loc='lower left').set_zorder(-5)
#     plt.ylim(-3.5, 2)
    
#     # make legend.
#     plt.gca().twinx()
#     plt.gca().spines[['top', 'right']].set_visible(False)
#     _ = plt.plot([], [], 'k')
#     _ = plt.plot([], [], 'k--')
#     _ = plt.yticks([])
#     plt.legend(['EEGNet-KF', 'EEGNet'], loc='lower right', prop={'size': 10})
    
#     # y
#     ax = fig.add_subplot(gs[1+n_rows_per_set*n_sets:1+n_rows_per_set*n_sets+2, 6:12])
#     plt.gca().spines[['top', 'right']].set_visible(False)
#     violin_x_scale = 0.2
#     for sj, subject_stats in enumerate(stats_list):
#         y_vals_kf = np.hstack([np.hstack([-violin_x_scale*day_info['vals'] + day_no + 0.5, [np.nan]]) for day_no, day_info in enumerate(subject_stats['violin_info']['y_info_kf'])])
#         y_coords_kf = np.hstack([np.hstack([day_info['coords'], [np.nan]]) for day_info in subject_stats['violin_info']['y_info_kf']])
#         y_vals_en = np.hstack([np.hstack([violin_x_scale*day_info['vals'] + day_no + 0.5, [np.nan]]) for day_no, day_info in enumerate(subject_stats['violin_info']['y_info_eegnet'])])
#         y_coords_en = np.hstack([np.hstack([day_info['coords'], [np.nan]]) for day_info in subject_stats['violin_info']['y_info_eegnet']])
        
#         plt.plot(y_vals_kf, y_coords_kf, color=color_cycle[sj])
#         plt.plot(y_vals_en, y_coords_en, '--', color=color_cycle[sj])
#     ax = plt.gca()
#     ax.xaxis.set_ticks(np.arange(0, n_days)+0.5, minor=True)
#     ax.xaxis.set_ticklabels(np.arange(0, n_days), minor=True)
#     plt.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=False)
#     plt.tick_params(axis='x', which='minor', length=6, bottom=False, top=False, labelbottom=True)
#     plt.xlabel('Adaptation day', size=12)
#     plt.xticks(size=10)
#     plt.ylabel('y velocity distribution')
#     plt.plot([0, n_days], [0, 0], 'k', zorder=-4, label='_nolegend_', lw=1, alpha=0.3)
#     plt.xlim(0, n_days)
#     plt.ylim(-3.5, 2)
    
    ########
    
    boxplot_width = 0.3
    median_color = 'c'
    
#     # average success percentage over all days
#     ax = fig.add_subplot(gs[-1, 0:3])
#     plt.gca().spines[['top', 'right']].set_visible(False)
#     data_pts = [subject_stats['block_success'] for subject_stats in stats_list]
#     bplot = plt.boxplot(data_pts, positions=np.arange(n_subjects), widths=boxplot_width, vert=True, patch_artist=True, showfliers=True)
#     for median in bplot['medians']:
#         median.set_color(median_color)
#     for patch in bplot['boxes']:
#         patch.set_facecolor('grey')
#     _ = plt.xticks(np.arange(n_subjects), subjects_list)
#     plt.ylabel('Success %')
#     plt.ylim(0, 1)
    
#     # average time to target over all days
#     ax = fig.add_subplot(gs[-1, 3:6])
#     plt.gca().spines[['top', 'right']].set_visible(False)
#     data_pts = [subject_stats['block_ttt'] for subject_stats in stats_list]
#     print('time to target median', subjects_list, [np.median(item) for item in data_pts])
#     bplot = plt.boxplot(data_pts, positions=np.arange(n_subjects), widths=boxplot_width, vert=True, patch_artist=True, showfliers=True)
#     for median in bplot['medians']:
#         median.set_color(median_color)
#     for patch in bplot['boxes']:
#         patch.set_facecolor('grey')
#     _ = plt.xticks(np.arange(n_subjects), subjects_list)
#     plt.ylabel('Time to acquisition (s)')
#     plt.ylim(0, 18)
    
    plt.tight_layout()
    return fig