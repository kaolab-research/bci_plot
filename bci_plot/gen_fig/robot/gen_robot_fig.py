import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pathlib
import ast
import scipy.stats
import copy
import itertools

from PIL import Image

def hex2float(v):
    v = v.lstrip('#')
    out = tuple(int(v[i:i+2], 16)/255.0 for i in (0, 2, 4))
    return out
color_cycle = [hex2float(item) for item in plt.rcParams['axes.prop_cycle'].by_key()['color']]
color_cycle.pop(3) # remove red

median_color = 'c'

def format_boxplot(bplot, median_color, median_label, boxes_color, boxes_label):
    for median in bplot['medians']:
        median.set_color(median_color)
        median.set_label(median_label)
    for patch in bplot['boxes']:
        patch.set_facecolor(boxes_color)
        patch.set_label(boxes_label)
    return

def plot_circle(ax, center, radius, n=41, **kwargs):
    pts = np.stack([center[0] + radius*np.cos(2*np.pi/(n-1)*np.arange(n)), center[1] + radius*np.sin(2*np.pi/(n-1)*np.arange(n))], axis=1)
    circ = ax.plot(*pts.T, **kwargs)
    return circ

def plot_x(ax, center, width, **kwargs):
    half_width = 0.5*width
    pts_x = [center[0] - half_width, center[0] + half_width, np.nan, center[0] - half_width, center[0] + half_width]
    pts_y = [center[1] - half_width, center[1] + half_width, np.nan, center[1] + half_width, center[1] - half_width]
    x = ax.plot(pts_x, pts_y, **kwargs)
    return x

def plot_box(ax, center, width, **kwargs):
    half_width = 0.5*width
    pts_x = [center[0] - half_width, center[0] + half_width, center[0] + half_width, center[0] - half_width]
    pts_y = [center[1] - half_width, center[1] - half_width, center[1] + half_width, center[1] + half_width]
    box = ax.fill(pts_x, pts_y, **kwargs)
    return box

def plot_line_cmap(ax, pos, data, **kwargs):
    points = pos.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    data_plot = data
    norm = plt.Normalize(np.min(data_plot), np.max(data_plot))
    lc = LineCollection(segments, norm = norm, **kwargs)
    lc.set_array(data_plot)
    ax.add_collection(lc)
    #plt.plot()
    return lc

def proportion_final_correct(target, actions):
    i = 0 # traverse target
    #j = 0 # traverse actions
    blocks = [target[n] for n in range(0, len(target), 2)]
    exes   = [target[n] for n in range(1, len(target), 2)]
    target_locs = {block: ex for block, ex in zip(blocks, exes)}
    
    locs = {block: block for block in blocks}
    for j, c in enumerate(actions):
        # pick
        if j % 2 == 0:
            block = c
        # place
        else:
            locs[block] = c
    out = np.mean([locs[block] == target_locs[block] for block in blocks])
    return out

def has_same_final_state(target, actions):
    i = 0 # traverse target
    #j = 0 # traverse actions
    blocks = [target[n] for n in range(0, len(target), 2)]
    exes   = [target[n] for n in range(1, len(target), 2)]
    target_locs = {block: ex for block, ex in zip(blocks, exes)}
    
    locs = {}
    for j, c in enumerate(actions):
        # pick
        if j % 2 == 0:
            block = c
        # place
        else:
            locs[block] = c
    for k in target_locs:
        try:
            if target_locs[k] != locs[k]:
                return False
        except KeyError:
            return False
    for k in locs:
        try:
            if target_locs[k] != locs[k]:
                return False
        except KeyError:
            return False
    return True

def levenshtein(src, tgt, insertion_cost=1.0, deletion_cost=1.0, substitution_cost=1.0):
    '''returns (src-tgt distance, distance matrix)'''
    l1 = len(src)
    l2 = len(tgt)

    d = np.zeros((l1+1, l2+1)) # distance matrix
    d[0, :] = insertion_cost*np.arange(l2+1)
    d[:, 0] = deletion_cost*np.arange(l1+1)

    for j in range(1, l2+1):
        for i in range(1, l1+1):
            sub_cost = substitution_cost*(src[i-1] != tgt[j-1]) # substitution cost, plus inherited                                                                                                                                          cost.
            del_cost = d[i-1, j] + deletion_cost # deletion cost, plus inherited cost.
            ins_cost = d[i, j-1] + insertion_cost # insertion cost, plus inherited cost.
            d[i, j] = np.min([del_cost, ins_cost, d[i-1, j-1] + sub_cost])
    return d[-1, -1], d

def gen_robot_fig(stats, results, images, trajs_info, subject_names):
    '''
    stats for ab
    results for sequences
    '''
    
    
    fig = plt.figure(figsize=(18, 12), dpi=300)
    n_cols = 4
    n_rows = 3
    #gs = plt.GridSpec(n_rows, n_cols, width_ratios=[4] + [2]*3, height_ratios=[1]*n_rows)
    gs = plt.GridSpec(4, 20, width_ratios=[1]*20, height_ratios=[1, 0.5, 0.5, 0.6])
    
    
    # ab image
    ax = fig.add_subplot(gs[0, 0:8])
    plt.gca().spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    #ab_cropped = images['ab_annotated'].crop(box=(400, 0, 960, 660))
    ab_cropped = images['ab_annotated_rot'].crop(box=(320, 140, 880, 800))
    ab_im = Image.new('RGB', (1440, 800), (255, 255, 255))
    ab_im.paste(images['ab_raw'], (0, 0))
    ab_im.paste(ab_cropped, (ab_im.size[0] - ab_cropped.size[0], ab_im.size[1] - ab_cropped.size[1]))
    ax.imshow(ab_im)
    
    # sequences image
    ax = fig.add_subplot(gs[1:3, 0:8])
    plt.gca().spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    #sequences_cropped = images['sequences_annotated'].crop(box=(400, 0, 960, 660))
    sequences_cropped = images['sequences_annotated_rot'].crop(box=(320, 140, 880, 800))
    sequences_im = Image.new('RGB', (1440, 800), (255, 255, 255))
    sequences_im.paste(images['sequences_raw'], (0, 0))
    sequences_im.paste(sequences_cropped, (sequences_im.size[0] - sequences_cropped.size[0], sequences_im.size[1] - sequences_cropped.size[1]))
    ax.imshow(sequences_im)
    
    
    # ab success
    ax = fig.add_subplot(gs[0, 8:12])
    plt.gca().spines[['top', 'right']].set_visible(False)
    hover_all = []
    cv_all = []
    for subject_idx, subject_name in enumerate(subject_names):
        hover_success_mean = np.mean([stats[subject_name][k]['hover_success_percentage'] for k in sorted(stats[subject_name].keys())])
        cv_success_mean    = np.mean([stats[subject_name][k]['cv_success_percentage'   ] for k in sorted(stats[subject_name].keys())])
        hover_all.append(hover_success_mean)
        cv_all.append(cv_success_mean)
        plt.plot([0, 1], [hover_success_mean, cv_success_mean], label=subject_name, color=color_cycle[subject_idx])
    _ = plt.bar([0], [np.mean(hover_all)], zorder=-1, color='grey')
    _ = plt.bar([1], [np.mean(cv_all)], zorder=-1, color='r')
    _ = plt.yticks([0, 1], [0, 100])
    _ = plt.xticks([0, 1], ['hover', 'CV'])
    plt.ylabel('Success percentage')
    ax.yaxis.set_label_coords(-0.07, 0.5)
    plt.legend()
    
    ax = fig.add_subplot(gs[0, 12:20])
    plt.gca().spines[['top', 'right']].set_visible(False)
    for subject_idx, subject_name in enumerate(subject_names):
        data = np.hstack([stats[subject_name][k]['hover_successful_times'] for k in sorted(stats[subject_name].keys())])
        bplot = plt.boxplot([data], positions=[subject_idx-0.1], vert=True, patch_artist=True, showfliers=True)
        format_boxplot(bplot, median_color, '_nolegend_', 'grey', 'hover' if subject_idx == 0 else '_nolegend_')
        
        data = np.hstack([stats[subject_name][k]['cv_successful_times'] for k in sorted(stats[subject_name].keys())])
        bplot = plt.boxplot([data], positions=[subject_idx+0.1], vert=True, patch_artist=True, showfliers=True)
        format_boxplot(bplot, median_color, '_nolegend_', 'r', 'CV' if subject_idx == 0 else '_nolegend_')
    _ = plt.xticks(np.arange(len(subject_names)), subject_names)
    plt.legend()
    plt.ylabel('Total time (seconds, successful trials only)')
    #ax.yaxis.set_label_coords(-0.25, 0.5)
    
    
    
    ax = fig.add_subplot(gs[3, 0:5])
    plt.gca().spines[['top', 'right']].set_visible(False)
    for i, subject in enumerate(subject_names):
        final_correctness_percentage = np.mean([proportion_final_correct(item[0], item[1]) for item in results['sequence_actions'][subject]])
        print(subject, final_correctness_percentage)
        plt.bar(i, 100*final_correctness_percentage, label=subject, color='grey')
    _ = plt.xticks(np.arange(len(subject_names)), subject_names)
    _ = plt.yticks([0, 50, 100])
    plt.ylabel('% blocks at correct location')
    
    ax = fig.add_subplot(gs[3, 5:10])
    plt.gca().spines[['top', 'right']].set_visible(False)
    for i, subject in enumerate(subject_names):
        bplot_data = [item['trial_time'] for item in results[subject]]
        bplot = plt.boxplot(bplot_data, positions=[i], widths=0.7, vert=True, patch_artist=True, showfliers=True)
        print(subject, np.median(bplot_data))
        format_boxplot(bplot, median_color, '_nolegend_', 'grey', '_nolegend_')
    _ = plt.xticks(np.arange(len(subject_names)), subject_names)
    plt.ylabel('Sequence total time (s)')
    plt.ylim(0, plt.ylim()[1])
    
    ax = fig.add_subplot(gs[3, 10:20])
    plt.gca().spines[['top', 'right']].set_visible(False)
    colors = ['grey', 'red', 'm', 'pink']
    for i, subject in enumerate(subject_names):
        action_correctness = np.array([int(c) for c in ''.join(list(zip(*results['sequence_actions'][subject]))[2])])

        bottom = 0
        for j in [1, 3, 0, 2]:
            v = (action_correctness==j).mean()
            label = '_nolegend_' if i != 0 else ['error', 'correct', 'error correction', 'out of order placement'][j]
            plt.bar(i, 100*v, bottom=100*bottom, color=colors[j], label=label)

            bottom += v
    _ = plt.xticks(np.arange(len(subject_names)), subject_names)
    _ = plt.xlim(-2.5, len(subject_names)-0.5)
    plt.legend()
    plt.ylabel('action distribution')
    
    
    # trajectories
    GRASP_RADIUS = 0.0254
    (trajs, objs_locs, objs_ids, objs_colors, trajs_description) = trajs_info
    for j, ((s, e), tt, xy) in enumerate(trajs):
        row = 1 + j // 4
        col = 8 + 3*(j % 4)
        ax = fig.add_subplot(gs[row, col:col+3])
        #plt.plot(*xy.T)
        plot_line_cmap(ax, xy, tt - tt[0], cmap='cool', alpha=1.0, lw=1)
        plt.xlim(0.1, 0.9)
        plt.ylim(-0.4, 0.4)
        _ = plt.xticks([0.1, 0.3, 0.6, 0.9])
        _ = plt.yticks([-0.4, 0.0, 0.4])
        
        locs = objs_locs[j]
        ids = objs_ids[j]
        for k, (loc, obj_id, color) in enumerate(zip(locs, ids, objs_colors)):
            #plt.plot(loc[0], loc[1], 'r.')
            plot_circle(ax, loc, GRASP_RADIUS, n=41, lw=0.1, c='k', zorder=2)
            if obj_id[0] == 'bin':
                plot_x(ax, loc, 0.054, lw=3, zorder=-4, alpha=0.3, color=color)
            else:
                plot_box(ax, loc, 0.029, color=color, alpha=1.0, zorder=4)
        segment_time = np.round(tt[-1] - tt[0], 1)
        action = ['pick', 'place'][j%2]
        plt.title(f'{j+1}, {trajs_description[j]}: {segment_time}s')
        pass
    
    
    
    plt.tight_layout()
    return fig


def gen_robot_supp_fig(stats, results, images, subject_names):
    fig = plt.figure(figsize=(24, 18), dpi=300)
    #n_cols = 4
    n_cols = 20
    n_rows = 3
    gs = plt.GridSpec(n_rows, n_cols, width_ratios=[1]*n_cols, height_ratios=[1]*n_rows)
    
    #
    ax = fig.add_subplot(gs[0, 0:4])
    plt.gca().spines[['top', 'right']].set_visible(False)
    for subject_idx, subject_name in enumerate(subject_names):
        data = np.hstack([stats[subject_name][k]['hover_moving_times'] for k in sorted(stats[subject_name].keys())])
        bplot = plt.boxplot([data], positions=[subject_idx-0.1], vert=True, patch_artist=True, showfliers=True)
        format_boxplot(bplot, median_color, '_nolegend_', 'grey', 'hover' if subject_idx == 0 else '_nolegend_')
        
        data = np.hstack([stats[subject_name][k]['cv_moving_times'] for k in sorted(stats[subject_name].keys())])
        print('cv_moving_times', subject_name, np.median(data), data.mean())
        bplot = plt.boxplot([data], positions=[subject_idx+0.1], vert=True, patch_artist=True, showfliers=True)
        format_boxplot(bplot, median_color, '_nolegend_', 'r', 'CV' if subject_idx == 0 else '_nolegend_')
    _ = plt.xticks(np.arange(len(subject_names)), subject_names)
    plt.legend()
    plt.ylabel('Moving time (seconds, successful trials only)')
    ax.yaxis.set_label_coords(-0.25, 0.5)
    
    #
    ax = fig.add_subplot(gs[0, 4:8])
    plt.gca().spines[['top', 'right']].set_visible(False)
    for subject_idx, subject_name in enumerate(subject_names):
        data = np.hstack([stats[subject_name][k]['hover_moving_times_all'] for k in sorted(stats[subject_name].keys())])
        bplot = plt.boxplot([data], positions=[subject_idx-0.1], vert=True, patch_artist=True, showfliers=True)
        format_boxplot(bplot, median_color, '_nolegend_', 'grey', 'hover' if subject_idx == 0 else '_nolegend_')
        
        data = np.hstack([stats[subject_name][k]['cv_moving_times_all'] for k in sorted(stats[subject_name].keys())])
        print('cv_moving_times_all', subject_name, np.median(data), data.mean())
        bplot = plt.boxplot([data], positions=[subject_idx+0.1], vert=True, patch_artist=True, showfliers=True)
        format_boxplot(bplot, median_color, '_nolegend_', 'r', 'CV' if subject_idx == 0 else '_nolegend_')
    _ = plt.xticks(np.arange(len(subject_names)), subject_names)
    plt.legend()
    plt.ylabel('Moving time (seconds, all trials)')
    ax.yaxis.set_label_coords(-0.25, 0.5)

    #
    PICKPLACE_TIME = 4.0 #in seconds
    ax = fig.add_subplot(gs[0, 8:12])
    plt.gca().spines[['top', 'right']].set_visible(False)
    for subject_idx, subject_name in enumerate(subject_names):
        data = np.hstack([stats[subject_name][k]['hover_moving_times'] for k in sorted(stats[subject_name].keys())])
        data += PICKPLACE_TIME*np.hstack([stats[subject_name][k]['hover_n_actions'] for k in sorted(stats[subject_name].keys())])
        bplot = plt.boxplot([data], positions=[subject_idx-0.1], vert=True, patch_artist=True, showfliers=True)
        format_boxplot(bplot, median_color, '_nolegend_', 'grey', 'hover' if subject_idx == 0 else '_nolegend_')
        
        data = np.hstack([stats[subject_name][k]['cv_moving_times'] for k in sorted(stats[subject_name].keys())])
        data += PICKPLACE_TIME*np.hstack([stats[subject_name][k]['cv_n_actions'] for k in sorted(stats[subject_name].keys())])
        bplot = plt.boxplot([data], positions=[subject_idx+0.1], vert=True, patch_artist=True, showfliers=True)
        format_boxplot(bplot, median_color, '_nolegend_', 'r', 'CV' if subject_idx == 0 else '_nolegend_')
    _ = plt.xticks(np.arange(len(subject_names)), subject_names)
    plt.legend()
    plt.ylabel('Modeled time (seconds, successful trials only)')
    ax.yaxis.set_label_coords(-0.25, 0.5)
    
    #
    ax = fig.add_subplot(gs[0, 12:16])
    plt.gca().spines[['top', 'right']].set_visible(False)
    for subject_idx, subject_name in enumerate(subject_names):
        data = np.hstack([stats[subject_name][k]['hover_n_actions'] for k in sorted(stats[subject_name].keys())])
        bplot = plt.boxplot([data], positions=[subject_idx-0.1], vert=True, patch_artist=True, showfliers=True)
        format_boxplot(bplot, median_color, '_nolegend_', 'grey', 'hover' if subject_idx == 0 else '_nolegend_')
        
        data = np.hstack([stats[subject_name][k]['cv_n_actions'] for k in sorted(stats[subject_name].keys())])
        bplot = plt.boxplot([data], positions=[subject_idx+0.1], vert=True, patch_artist=True, showfliers=True)
        format_boxplot(bplot, median_color, '_nolegend_', 'r', 'CV' if subject_idx == 0 else '_nolegend_')
        pass
    _ = plt.xticks(np.arange(len(subject_names)), subject_names)
    plt.legend()
    plt.ylabel('# actions, successful trials only')
    plt.ylim(0, plt.ylim()[1])
    
    #
    ax = fig.add_subplot(gs[0, 16:20])
    plt.gca().spines[['top', 'right']].set_visible(False)
    for subject_idx, subject_name in enumerate(subject_names):
        data = np.hstack([stats[subject_name][k]['hover_n_actions_all'] for k in sorted(stats[subject_name].keys())])
        bplot = plt.boxplot([data], positions=[subject_idx-0.1], vert=True, patch_artist=True, showfliers=True)
        format_boxplot(bplot, median_color, '_nolegend_', 'grey', 'hover' if subject_idx == 0 else '_nolegend_')
        
        data = np.hstack([stats[subject_name][k]['cv_n_actions_all'] for k in sorted(stats[subject_name].keys())])
        bplot = plt.boxplot([data], positions=[subject_idx+0.1], vert=True, patch_artist=True, showfliers=True)
        format_boxplot(bplot, median_color, '_nolegend_', 'r', 'CV' if subject_idx == 0 else '_nolegend_')
        pass
    _ = plt.xticks(np.arange(len(subject_names)), subject_names)
    plt.legend()
    plt.ylabel('# actions, all trials')
    plt.ylim(0, plt.ylim()[1])
    
    # 
    ax = fig.add_subplot(gs[1, 0:4])
    plt.gca().spines[['top', 'right']].set_visible(False)
    for i, subject in enumerate(subject_names):
        bplot = plt.boxplot([item['total_movement_time'] for item in results[subject]], positions=[i], widths=0.7, vert=True, patch_artist=True, showfliers=True)
        format_boxplot(bplot, median_color, '_nolegend_', 'grey', '_nolegend_')
    _ = plt.xticks(np.arange(len(subject_names)), subject_names)
    plt.ylabel('Sequence moving time (s)')
    
    # 
    ax = fig.add_subplot(gs[1, 4:8])
    plt.gca().spines[['top', 'right']].set_visible(False)
    for i, subject in enumerate(subject_names):
        data = [item['total_movement_time']/item['n_actions'] for item in results[subject]]
        print(subject, 'moving_time_per_action', np.median(data), np.mean(data))
        bplot = plt.boxplot(data, positions=[i], widths=0.7, vert=True, patch_artist=True, showfliers=True)
        format_boxplot(bplot, median_color, '_nolegend_', 'grey', '_nolegend_')
    _ = plt.xticks(np.arange(len(subject_names)), subject_names)
    plt.ylabel('Moving time per executed action (s)')
    
    # 
    ax = fig.add_subplot(gs[1, 8:12])
    plt.gca().spines[['top', 'right']].set_visible(False)
    for i, subject in enumerate(subject_names):
        vs = [levenshtein(item[0], item[1], insertion_cost=1.0, deletion_cost=1.0, substitution_cost=1.0)[0] for item in results['sequence_actions'][subject]]
        print(subject, vs)
        bplot = plt.boxplot(vs, positions=[i], widths=0.7, vert=True, patch_artist=True, showfliers=True)
        format_boxplot(bplot, median_color, '_nolegend_', 'grey', '_nolegend_')
    _ = plt.xticks(np.arange(len(subject_names)), subject_names)
    plt.ylabel('Levenshtein distance of actions')
    
    #
    ax = fig.add_subplot(gs[1, 12:16])
    plt.gca().spines[['top', 'right']].set_visible(False)
    for i, subject in enumerate(subject_names):
        final_same_percentage = np.mean([has_same_final_state(item[0], item[1]) for item in results['sequence_actions'][subject]])
        plt.bar(i, 100*final_same_percentage, label=subject, color='grey')
    _ = plt.xticks(np.arange(len(subject_names)), subject_names)
    _ = plt.yticks([0, 50, 100])
    plt.ylabel('Final state correct %')
    
    ax = fig.add_subplot(gs[1, 16:20])
    plt.gca().spines[['top', 'right']].set_visible(False)
    for i, subject in enumerate(subject_names):
        vs = [len(item[2]) for item in results['sequence_actions'][subject]]
        print(vs)
        bplot = plt.boxplot(vs, positions=[i], widths=0.7, vert=True, patch_artist=True, showfliers=True)
        format_boxplot(bplot, median_color, '_nolegend_', 'grey', '_nolegend_')
    _ = plt.xticks(np.arange(len(subject_names)), subject_names)
    plt.ylabel('Total actions')
    
    #
    ax = fig.add_subplot(gs[1, 12:16])
    plt.gca().spines[['top', 'right']].set_visible(False)
    for i, subject in enumerate(subject_names):
        final_same_percentage = np.mean([has_same_final_state(item[0], item[1]) for item in results['sequence_actions'][subject]])
        plt.bar(i, 100*final_same_percentage, label=subject, color='grey')
    _ = plt.xticks(np.arange(len(subject_names)), subject_names)
    _ = plt.yticks([0, 50, 100])
    plt.ylabel('Final state correct %')
    
    ax = fig.add_subplot(gs[2, 0:4])
    plt.gca().spines[['top', 'right']].set_visible(False)
    for i, subject in enumerate(subject_names):
        vs = [((item['trial_time'] - item['total_movement_time'])/item['n_actions']) for item in results[subject]]
        bplot = plt.boxplot(vs, positions=[i], widths=0.7, vert=True, patch_artist=True, showfliers=True)
        format_boxplot(bplot, median_color, '_nolegend_', 'grey', '_nolegend_')
    _ = plt.xticks(np.arange(len(subject_names)), subject_names)
    plt.ylabel('Non-movement time per action (s)')
    plt.ylim(0.5*np.floor(plt.ylim()[0]/0.5), 0.5*np.ceil(plt.ylim()[1]/0.5))

    
    plt.tight_layout()
    return fig