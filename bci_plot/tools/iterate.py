import yaml
from pathlib import Path
import numpy as np
import string
import os
this_dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
copilot_info_path = this_dir_path / '..' / 'metadata' / 'info_copilot_session.yaml'

# function that loads all the data from raspy folder
def get_copilot_session_names(random=None, label=False):
    """
    returns list of all copilot sessions both ABA
    Arguments:
        random (None, int) = uniformly select integer number of sessionNames
        label = returns list of tuples with A,B label on it depending on copilot or not
    return
        list of copilot_session_names
    """

    with open(copilot_info_path) as file:
        copilot_session = yaml.safe_load(file)['subjects']
        

    expNames = []
    labels = []
    for subject in copilot_session.keys():
        for date in copilot_session[subject].keys():
            for exp in copilot_session[subject][date]:
                suffix = exp[1]
                useCopiot = exp[0]
                if len(exp) > 2: detail = exp[2]
                expName = date.replace('/','-') + '_' + subject + '_' + suffix
                expNames.append(expName)
                labels.append(useCopiot)

    for expName in expNames:
        dataPath = Path('/data/raspy') / expName
        if not dataPath.exists(): 
            print(expName, 'Not Exists')

    if label:
        expNames = list(zip(expNames, labels))
    
    if random is not None:
        expNames = np.random.choice(expNames, random)

    return expNames

def get_copilot_session_by_ABA(random=None, copilotType=None, noRepeat=None):
    """
    returns list of all copilot sessions by ABA
    Arguments:
        random (None, int) = uniformly select integer number of sessionNames
        copilotType = filter by name (lstm, trajectory, 0.7 etc)
    return
        list of ABA with ABA
    """

    with open(copilot_info_path) as file:
        copilot_session_ABA = yaml.safe_load(file)['ABA']

        # filter by copilot type
        if copilotType is not None:
            copilot_session_ABA = [ABA for ABA in copilot_session_ABA if copilotType in ABA['detail']]

        copilot_session = [ABA['session'] for ABA in copilot_session_ABA]
    
    if random is not None:
        copilot_session = np.random.choice(copilot_session, random)

    return copilot_session



# get trial end index and begin index from session name
def get_trial_beginning_end(taskData, skipCenterIn=True):
    """ 
    returns trials begining and end index
    Arguments:
        taskData: dictionary result of load_data(sessionPath)
    returns
        begin and end time for each trial
    """

    # keys
    ['timer_tick_time_ns', 'state_task', 'decoder_output', 'decoded_pos', 'target_pos', 'target_size', 'game_state', 'kf_state', 'kf_inf_state', 'kf_update_flag', 'allow_kf_adapt', 'allow_kf_sync', 'decoder_h', 'kf_R', 'kf_S', 'kf_T', 'kf_Tinv', 'kf_EBS', 'kf_C', 'kf_Q', 'kf_Qinv', 'kf_S_k', 'kf_K_k', 'kf_M1', 'kf_M2', 'kf_effective_vel', 'kf_ole_rlud', 'sessionLength', 'activeLength', 'cursorVel', 'ignoreWrongTarget', 'cursorMoveInCorretDirectionOnly', 'assistValue', 'assistMode', 'softmaxThres', 'holdTimeThres', 'kfCopilotAlpha', 'hitRate', 'missRate', 'timeoutRate', 'trialCount', 'render_angle', 'numCompletedBlocks', 'enableKfSyncInternal', 'eeg_step', 'gaze_step', 'time_ns', 'name', 'labels', 'dtypes']

    # get trial begin, end time
    oneHot = taskData['trialCount'][1:] - taskData['trialCount'][:-1]
    begin = np.where(oneHot == 1)[0] + 1 
    end = np.append(begin[1:]-1,len(taskData['trialCount'])-1)

    if skipCenterIn:
        end = end[(taskData['target_pos'][end] == np.zeros(2)).sum(1) != 2]
        begin = begin[(taskData['target_pos'][begin] == np.zeros(2)).sum(1) != 2]
            
    # for b,e in zip(begin,end):
    #     print(b,e,taskData['trialCount'][b],taskData['trialCount'][e],chr(taskData['game_state'][e].item()))

    return list(zip(begin,end))


# get first touch time, dial in time, total trial time in seconds
def get_dial_in_time(taskData,skipUnsuccessfulForDialInTime=True):
    """
    skipUnsuccessfulForDialInTime : if this trial is not a success ('H'). then it doesn't count toward dial in time
    returns 
        - first touch time, 
        - dial in time, 
        - total trial time for each trial in session (all in nanoseconds)
    note, first 4 trial is ignored because they are for calibration (do not have touch time)
    """

    beginEnd = get_trial_beginning_end(taskData)
    gameState = taskData['game_state'] #.astype(np.uint8).tostring().decode("ascii")
    ns = taskData['time_ns']
    holdTime = 0.5 # 500ms

    timeToTouch_s = []
    dialInTime_s = []
    trialTime_s = []
    for b,e in beginEnd[4:]:
        touch = np.where(gameState[b:e]==ord('h'))[0]
        if len(touch) > 0:
            firstTouch = touch[0]
            index = b + firstTouch
            timeToTouch = (ns[index] - ns[b]) / 10**9
            dialInTime = (ns[e] - ns[index]) / 10**9 - holdTime
            trialTime = (ns[e] - ns[b]) / 10**9
        else:
            timeToTouch = None
            dialInTime = None
            trialTime = (ns[e] - ns[b]) / 10**9
        
        # save
        timeToTouch_s.append(timeToTouch)
        trialTime_s.append(trialTime)
        if skipUnsuccessfulForDialInTime and gameState[e] == ord('H'): dialInTime_s.append(dialInTime)
        else: dialInTime_s.append(dialInTime)

    return timeToTouch_s, dialInTime_s, trialTime_s

if __name__ == '__main__':
    session = get_copilot_session_names()
    print(session)

    # session = get_copilot_session_by_ABA(copilotType='trajectory')
    # print(session)
    # exit()

    from data_util import load_data
    dataPath = Path('/data/raspy')
    taskData = load_data(dataPath / session[0] / 'task.bin')
    # eegData = load_data(dataPath / session[0] / 'eeg.bin')
    # gazeData = load_data(dataPath / session[0] / 'gaze.bin')

    trialInfo = get_trial_beginning_end(taskData)
    print(trialInfo)

    _,dialInTime,_ = get_dial_in_time(taskData)
    print(dialInTime)

    
