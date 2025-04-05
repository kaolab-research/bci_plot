
import matplotlib.pyplot as plt
import numpy as np

def gen_p_value_fig(sessions, all_cods, all_circular_cods, 
                    MEAN_OR_MAX='max',
                    EFFECT_SIZE=(-1,1),
                    verbose=False,
                    ):

    """
    MEAN_OR_MAX = 'max' # 'max' or 'mean'
    EFFECT_SIZE = (-1,1) # -5 to +5 seconds in unit of 0.2 second
    """
    # constants
    pvalues = {'px':[],'py':[], 'xcod':[], 'ycod':[]}
    for session_name, cods, circular_cods in zip(sessions, all_cods, all_circular_cods):

        # find index
        start, end = np.array(EFFECT_SIZE) * 5 + 25 
        start = int(start)
        end = int(end) + 1
        
        if MEAN_OR_MAX == 'mean':
            mu = np.array(cods['gaze_from_h']['linear']).mean(1)[start:end].mean(0)
        elif MEAN_OR_MAX == 'max':
            mu = np.array(cods['gaze_from_h']['linear']).mean(1)[start:end].max(0)
        else:
            raise Exception('Incorrect value for MEAN_OR_MAX')
        
        x_mu, y_mu = mu[:2]

        # extract all 5000 datapoint for comparison    
        x_circular_cods = np.array(circular_cods['gaze_from_h']['linear'])[:,:,0].flatten()
        y_circular_cods = np.array(circular_cods['gaze_from_h']['linear'])[:,:,1].flatten()

        # find cods bigger than mean/max
        x_pvalue = (x_mu < x_circular_cods).sum() / len(x_circular_cods)
        y_pvalue = (y_mu < y_circular_cods).sum() / len(y_circular_cods)
        if verbose: print(session_name, 'px=', '{:.3f}'.format(x_pvalue), 'py=', '{:.3f}'.format(y_pvalue), '    \      mu x=', '{:.3f}'.format(x_mu), 'mu y=', '{:.3f}'.format(y_mu))
        pvalues['px'].append(x_pvalue)
        pvalues['py'].append(y_pvalue)
        pvalues['xcod'].append(x_mu)
        pvalues['ycod'].append(y_mu)


    """ plot """

    n_cols = 3
    n_sessions = len(sessions)
    n_rows = int(np.ceil(n_sessions/n_cols))

    fig = plt.figure(figsize=(16, 16))
    gs = plt.GridSpec(n_rows, n_cols, width_ratios=[1]*n_cols, height_ratios=[1]*n_rows)


    for m, session in enumerate(sessions):
        i, j = m//n_cols, m%n_cols
        ax = fig.add_subplot(gs[i, j])
        
        x = np.array([f"px {'{:.3f}'.format(pvalues['px'][m])}", f"py {'{:.3f}'.format(pvalues['py'][m])}"])
        y = np.array([pvalues['px'][m],pvalues['py'][m]])
        plt.bar(x[0],y[0], color='red' if y[0] < 0.05 else 'blue')
        plt.bar(x[1],y[1], color='red' if y[1] < 0.05 else 'blue')
        plt.title(session)
        
        # if m==0:
        plt.legend([f"x cod {'{:.3f}'.format(pvalues['xcod'][m])}", f"y cod {'{:.3f}'.format(pvalues['ycod'][m])}"])
        # plt.xlabel('Fold (n)')
        plt.ylabel('pvalue')

    ylims = [ax.get_ylim() for ax in fig.get_axes()]
    new_ylim = (0, 1.0)
    [ax.set_ylim(*new_ylim) for ax in fig.get_axes()]
    
    return pvalues