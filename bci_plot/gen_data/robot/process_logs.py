
import pickle
import json
import numpy as np

def process_logs(path):
    logs_lines = as_lines(path)
    out = {}
    for key in ['timestamp', 'qdot', 'action', 'movement_cum_dt']:
        try:
            out[key] = np.array([log_i[key] for log_i in logs_lines])
        except KeyError:
            pass
    #out['timestamp'] = np.array([log_i['timestamp'] for log_i in logs_lines])
    #out['qdot'] = np.array([log_i['qdot'] for log_i in logs_lines])
    #out['action'] = np.array([log_i['action'] for log_i in logs_lines])
    #, 'movement_cum_dt'
    for key in ['z_state', 'finished_reset', 'gripper', 'gripper_retval', 'gripper_state_at_ret', 'start_press']:
        try:
            out[key] = np.array([log_i['state'][key] for log_i in logs_lines])
        except KeyError:
            pass
    #out['finished_reset'] = np.array([log_i['state']['finished_reset'] for log_i in logs_lines])
    #out['gripper_retval'] = np.array([log_i['state']['gripper_retval'] for log_i in logs_lines])
    #out['gripper_state_at_ret'] = np.array([log_i['state']['gripper_state_at_ret'] for log_i in logs_lines])
    #out['start_press'] = np.array([log_i['state']['start_press'] for log_i in logs_lines])
    try:
        out['hover_state'] = np.array([log_i['hover_state'] for log_i in logs_lines])
    except KeyError:
        pass
    out['xyz'] = np.array([log_i['state']['ee_mat'][12:15] for log_i in logs_lines])
    
    try:
        out['trial_start_idx'], out['trial_end_idx'], out['trial_end_condition'] = partition_processed_logs(out)
    except Exception as e:
        if 'finished_reset' in out:
            out['trial_start_idx'] = np.nonzero((out['finished_reset'][1:None] == 0)*(out['finished_reset'][0:-1] == 1))[0] + 1
            # stop is different from end
            out['trial_stop_idx'] = np.nonzero((out['finished_reset'][1:None] == 1)*(out['finished_reset'][0:-1] == 0))[0] + 1
        out['exception'] = str(e)
    
    
    all_objs_list = []
    all_objs_idx = []
    all_objs_ids = []
    prev_all_objs = None
    prev_all_ids = None
    for idx, log_i in enumerate(logs_lines):
        try:
            #if (np.array(log_i['all_objs']) != prev_all_objs).any():
            if (log_i['all_objs'] != prev_all_objs) or (log_i['all_ids'] != prev_all_ids):
                raise Exception('')
        except:
            all_objs_list.append(np.array(log_i['all_objs']))
            all_objs_idx.append(idx)
            all_objs_ids.append(log_i['all_ids'])
            prev_all_objs = log_i['all_objs']
            prev_all_ids = log_i['all_ids']
    out['all_objs'] = [log_i['all_objs'] for log_i in logs_lines]
    out['all_ids'] = [log_i['all_ids'] for log_i in logs_lines]
    out['all_objs_list'] = all_objs_list
    out['all_objs_idx'] = np.array(all_objs_idx)
    out['all_objs_ids'] = all_objs_ids
    out['all_objs_len'] = np.array([len(log_i['all_objs']) for log_i in logs_lines])
    return out

def as_lines(path):
    with open(path, 'r') as f:
        logs = [json.loads(line) for line in f.readlines()]
    return logs

def partition_processed_logs(logs):
    '''
    Get the start and end idxs of trials, as well as which one it is
      * start is when the start button is pressed
      * end is either when the last successful action was completed or when there was a timeout (qdot[0] == -1)
    '''
    out = []
    trial_start_idx = np.nonzero((logs['finished_reset'][1:None] == 0)*(logs['finished_reset'][0:-1] == 1))[0] + 1
    
    '''
    Trial ends when
      Block is placed (gripper 3->-1)
        OR
      Timeout (qdot[-1] == -1)
    '''
    # pickplace_is_valid (anything -> -1)
    timeout_idx = np.nonzero((logs['qdot'][:, -1][1:None] == -1)*(logs['qdot'][:, -1][0:-1] != -1))[0] + 1
    startstop_idx = np.nonzero(logs['start_press'])[0]
    stop_idx = startstop_idx[~np.isin(startstop_idx, trial_start_idx)]
    #last_successful_action_idx = [np.nonzero(logs['gripper_retval'][0:idx] == 1)[0][-1] for idx in stop_idx]
    last_successful_action_idx = []
    for idx in stop_idx:
        tmp = np.nonzero(logs['gripper_retval'][0:idx] == 1)[0]
        if len(tmp) >= 1:
            last_successful_action_idx.append(tmp[-1] + 1)
        #else:
        #    last_successful_action_idx.append()
    end_condition_stacked = np.hstack([np.zeros(len(timeout_idx)), np.ones(len(last_successful_action_idx))]) # 0 for timeout, 1 for non-timeout
    trial_end_stacked = np.hstack([timeout_idx, last_successful_action_idx])
    trial_end_order = np.argsort(trial_end_stacked)
    trial_end_idx = trial_end_stacked[trial_end_order].astype('int64')
    end_condition = end_condition_stacked[trial_end_order]
    return trial_start_idx, trial_end_idx, end_condition