import numpy as np
import matplotlib.pyplot as plt
import pathlib
import ast
import scipy.stats

from bci_plot.utils import data_util

HOLD_TIME = 0.500

def parse_line(line):
    return tuple(item.strip() for item in line.rsplit(':', 1))
def parse_readme(session):
    datadir = pathlib.Path('/data/raspy/')
    readme_path = datadir / session / 'README.txt'
    with open(readme_path, 'r') as f:
        readme_contents = f.read().strip()
    readme_tuples = [parse_line(line) for line in readme_contents.split('\n')]
    readme_dict = {'other': []}
    for tup in readme_tuples:
        try:
            readme_dict[tup[0]] = tup[1]
        except:
            readme_dict['other'].append(tup)
    return readme_dict

def get_stats(session, use_ftt=None, timeout_time=24):
    '''
    session: name of session within /data/raspy/
    use_ftt: deprecated
    timeout_time: what time to set unsuccessful trials to (in seconds).
    '''
    datadir = pathlib.Path('/data/raspy/')
    task_data = data_util.load_data(datadir / session / 'task.bin')
    readme = parse_readme(session)
    target_diameter = ast.literal_eval(readme['Target info'].replace('array', ''))[1][0]
    target_radius = 0.5*target_diameter
    
    state_task = task_data['state_task'].flatten().copy()
    state_task[task_data['numCompletedBlocks'].flatten() < 0] = 99
    state_task[(task_data['numCompletedBlocks'].flatten() < 0).sum()] = -1
    start_inds = np.nonzero(state_task[1:None] != state_task[0:-1])[0][1:None] + 1
    end_inds = np.hstack([start_inds[1:None], [len(state_task)]]) # +1?
    sync_ticks = np.nonzero(task_data['allow_kf_sync'])[0]
    adaptation_ticks_list = []
    synced_ticks_list = []
    ebs_list = []
    ebs_synced = []
    starts = []
    raw_acqs = []
    acqs = []
    ftts = []
    walls = []
    is_eval = []
    is_success = []
    ticks_since_sync = []
    for i, (start, end) in enumerate(zip(start_inds, end_inds)):
        block = i // 16
        xy = task_data['decoded_pos'][start:end]
        target = state_task[start]
        if target != 99:
            q = np.digitize(start, sync_ticks) - 1
            if q < 0 or len(sync_ticks) == 0:
                synced_ticks = 0
                ebs_synced.append(task_data['kf_EBS'][0])
            else:
                synced_ticks = task_data['allow_kf_adapt'][0:sync_ticks[q]].sum()
                ebs_synced.append(task_data['kf_EBS'][max(0, sync_ticks[q]-1)])
            adaptation_ticks = task_data['allow_kf_adapt'][0:max(0, start-1)].sum()
            ebs = task_data['kf_EBS'][max(0, start-1)]
            trial_ticks = end - start # length of trial, in ticks
            
            # sync should occur at start-1 (if applicable) and/or at tick end-1
            if task_data['allow_kf_sync'][start:end].any():
                is_eval.append(False)
            else:
                is_eval.append(True)
            
            trial_wall = (task_data['time_ns'][end] - task_data['time_ns'][start])/1e9
            walls.append(trial_wall)
            
            cursor_pos = task_data['decoded_pos'][start:end]
            target_pos = task_data['target_pos'][start]
            
            dist = np.linalg.norm(cursor_pos - target_pos, axis=-1)
            
            if (dist <= target_radius).any():
                ftt = np.nonzero(dist <= target_radius)[0][0] + 1
            else:
                ftt = timeout_time*20
                trial_ticks = timeout_time*20
            
            try:
                val = start - np.nonzero(task_data['allow_kf_sync'][0:start+1])[0][-1]
            except:
                val = start
            ticks_since_sync.append(val)
            
            is_success_trial = bool(task_data['hitRate'][end-1, 0] - task_data['hitRate'][start, 0])
            
            ebs_list.append(ebs)
            adaptation_ticks_list.append(adaptation_ticks) # check if correct
            synced_ticks_list.append(synced_ticks) # check if correct
            starts.append(start)
            acqs.append(trial_ticks)
            ftts.append(ftt)
            raw_acqs.append(end - start)
            is_success.append(is_success_trial)
            
    stats = {}
    stats['adaptation_ticks_list'] = np.array(adaptation_ticks_list)
    stats['synced_ticks_list'] = np.array(synced_ticks_list)
    stats['ebs_list'] = np.array(ebs_list)
    stats['ebs_synced'] = np.array(ebs_synced)
    stats['total_adaptation_ticks'] = np.sum(task_data['allow_kf_adapt'])
    stats['acqs'] = acqs
    stats['raw_acqs'] = raw_acqs
    stats['ftts'] = ftts
    stats['is_success'] = is_success
    stats['is_eval'] = is_eval
    stats['block_success'] = np.array(is_success).reshape((-1, 8)).mean(1)
    stats['block_acqs'] = np.array(acqs).reshape((-1, 8)).mean(1)
    stats['block_ftts'] = np.array(ftts).reshape((-1, 8)).mean(1)
    stats['block_walls'] = np.array(walls).reshape((-1, 8)).mean(1)
    stats['block_is_eval'] = np.array(is_eval).astype('float').reshape((-1, 8)).mean(1)
    stats['ticks_since_sync'] = ticks_since_sync
    #stats['block_raw_acqs'] = np.array(acqs).reshape((-1, 8)).mean(1)
    stats['readme'] = readme
    
    stats['dt'] = 0.050 # tick time (seconds). HARDCODED
    stats['block_ttt'] = stats['block_acqs']*stats['dt']
    stats['target_diameter'] = target_diameter
    stats['block_td'] = np.full(stats['block_acqs'].shape, target_diameter)
    stats['target_distance'] = np.linalg.norm(ast.literal_eval(readme['Target info'].replace('array', ''))[0])
    stats['iod'] = np.log2(1 + stats['target_distance']/stats['target_diameter'])
    stats['block_bps'] = stats['iod']/(stats['block_acqs']*stats['dt'] - HOLD_TIME)
    stats['block_bps_ftt'] = stats['iod']/(stats['block_ftts']*stats['dt'] - HOLD_TIME)
    stats['gilja_iod'] = np.log2(1 + (stats['target_distance'] - 0.5*stats['target_diameter'])/stats['target_diameter'])
    stats['block_bps_gilja'] = stats['gilja_iod']/(stats['block_acqs']*stats['dt'] - HOLD_TIME)
    
    return stats