import numpy as np



def resample_logs(logs, trial_starts_ends=()):
    logs['trial_start_idx'], logs['trial_end_idx'] = list(zip(*trial_starts_ends))
    logs['trial_start_idx'], logs['trial_end_idx'] = np.array(logs['trial_start_idx']), np.array(logs['trial_end_idx'])
    objs_list, objs_ids, objs_idx = resample_objs(logs, trial_starts_ends)
    logs['all_objs_list'], logs['all_objs_ids'], logs['all_objs_idx'] = objs_list, objs_ids, objs_idx
    return

def resample_objs(logs, trial_starts_ends=()):
    objs_list = []
    objs_ids = []
    objs_idx = []
    
    # For compatibility to prevent indexing errors due to digitize operation
    objs_list.append(logs['all_objs_list'][0])
    objs_ids.append(logs['all_objs_ids'][0])
    objs_idx.append(logs['all_objs_idx'][0])
    
    for trial_start, trial_end in trial_starts_ends:
        original_objs_idx = np.digitize(trial_start, logs['all_objs_idx']) - 1
        original_objs = logs['all_objs_list'][original_objs_idx]
        original_ids = logs['all_objs_ids'][original_objs_idx]
        
        objs_list.append(original_objs.copy())
        objs_ids.append(original_ids.copy())
        objs_idx.append(trial_start)
        
        current_objs = original_objs.copy()
        current_ids = original_ids.copy()
         # these should not change
        block_idxs = np.nonzero([o_id[0] == 'obj' for o_id in current_ids])[0] # these should not change
        bin_idxs = np.nonzero([o_id[0] == 'bin' for o_id in current_ids])[0] # these should not change
        
        gripper_idxs = trial_start + np.nonzero(np.isin(logs['gripper'][trial_start:trial_end], [2, 3]))[0]
        
        gs = -1
        for idx in gripper_idxs:
            if logs['gripper'][idx] == 2:
                xy = logs['xyz'][idx, 0:2]
                closest_block_idx_idx = np.argmin(np.linalg.norm(current_objs[block_idxs] - xy, axis=-1))
                current_block_idx = block_idxs[closest_block_idx_idx]
                current_ids[current_block_idx] = ['obj', 'picked']
                
                bin_dists = np.linalg.norm(current_objs[bin_idxs] - current_objs[current_block_idx], axis=-1)
                if np.min(bin_dists) < 0.001:
                    # pick from bin
                    closest_bin_idx_idx = np.argmin(bin_dists)
                    current_bin_idx = bin_idxs[closest_bin_idx_idx]
                    current_ids[current_bin_idx] = ['bin', 'empty']
                    pass
                pass
            elif logs['gripper'][idx] == 3:
                xy = logs['xyz'][idx, 0:2]
                dists = np.linalg.norm(current_objs[bin_idxs] - xy, axis=-1)
                if np.min(dists) < 0.0254:
                    # place at an empty bin
                    closest_bin_idx_idx = np.argmin(dists)
                    current_bin_idx = bin_idxs[closest_bin_idx_idx]
                    current_ids[current_bin_idx] = ['bin', 'full']
                    
                    current_ids[current_block_idx] = ['obj', 'placed']
                    current_objs[current_block_idx] = current_objs[current_bin_idx]
                else:
                    current_ids[current_block_idx] = ['obj', 'placed']
                    pass
                pass
            objs_list.append(current_objs.copy())
            objs_ids.append(current_ids.copy())
            objs_idx.append(idx)
            pass
        
    return objs_list, objs_ids, objs_idx