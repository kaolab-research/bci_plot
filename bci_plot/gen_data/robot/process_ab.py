

def get_moving_time(logs_i, threshold=0.010):
    '''
    returns arrays of the cumulative moving time for each trial in a log file
    threshold: delta time greater than threshold are not counted.
    '''
    out = []
    for i, j in zip(logs_i['trial_start_idx'], logs_i['trial_end_idx']):
        moving_ticks = i + np.nonzero(logs_i['z_state'][i:j+1] == 0)[0]
        moving_ticks = moving_ticks[moving_ticks>=1]
        dts = (logs_i['timestamp'][moving_ticks] - logs_i['timestamp'][moving_ticks-1])
        dts = dts[dts < threshold]
        out.append(dts.sum())
    out = np.array(out)
    return out

def process_ab_logs(session):
    for log_fname in robot_sessions.logs_meta[session]:
        pickle_fname = log_fname.replace('.log', '.pickle')
        with open(f'../../data/robot/{session}/{pickle_fname}', 'rb') as f:
            logs[session].append(pickle.load(f))

    trial_length = np.hstack([item['timestamp'][item['trial_end_idx']] - item['timestamp'][item['trial_start_idx']] for item in logs[session]])
    #movement_cum_dt = np.hstack([item
    trial_end_condition = np.hstack([item['trial_end_condition'] for item in logs[session]])
    trial_hover_state = np.hstack([item['hover_state'][item['trial_start_idx']] for item in logs[session]])
    n_successful_actions = np.hstack([[(item['gripper_retval'][i:j+1]==1).sum() for i,j in zip(item['trial_start_idx'], item['trial_end_idx'])] for item in logs[session]])
    n_unsuccessful_actions = np.hstack([[(item['gripper_retval'][i:j+1]==0).sum() for i,j in zip(item['trial_start_idx'], item['trial_end_idx'])] for item in logs[session]])
    moving_time = np.hstack([get_moving_time(item) for item in logs[session]])

    #((block, x), is_hover, (pick_status, place_status))
    # pick_status:  1: correct block
    #              -1: incorrect block
    #               0: no pick
    # place_status: 1: on correct x (subjective for hover)
    #              -1: not on correct x
    #               0: no place
    #              99: no pick
    outcomes = robot_sessions.outcomes_meta[session]
    (tmp0, hs, tmp1) = list(zip(*outcomes))
    (block_color, x_color) = list(zip(*tmp0))
    (pick_result, place_result) = list(zip(*tmp1))


    session_meta = {
        'trial_length': trial_length,
        'trial_end_condition': trial_end_condition,
        'trial_hover_state': trial_hover_state,
        'n_successful_actions': n_successful_actions,
        'n_unsuccessful_actions': n_unsuccessful_actions,
        'outcomes': outcomes,
        'block_color': block_color,
        'x_color': x_color,
        'pick_result': np.array(pick_result),
        'place_result': np.array(place_result),
        'moving_time': np.array(moving_time),
    }
    return session_meta