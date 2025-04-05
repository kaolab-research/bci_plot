
import scipy
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def signficant_star(p_value:float):
    stars = ""
    if 0.001 <= p_value < 0.05:
        stars = "*"
    if 0.0001 <= p_value < 0.001:
        stars = "**"
    if p_value < 0.0001:
        stars = "***"
    # stars = '*' * (int(-np.log10(p_value)) - 1) if p_value < 0.05 else ''
    # if len(stars) > 3: stars = stars[:3]
    return stars

def plotABAbar(session1,session2,session3,ylabel='Path Efficiency (%)',ylim=(0, 100),manual_y_ticks=None,title=None,savePath=None):
    """
    session1: 1d np array with values
    session2: 1d np array with values
    session3: 1d np array with values
    """

    # these are all calucated variables
    t_stat1_2, p_value1_2 = ttest_ind(session1, session2)
    t_stat1_3, p_value1_3 = ttest_ind(session1, session3)
    t_stat2_3, p_value2_3 = ttest_ind(session2, session3)

    mean_session1 = np.mean(session1)
    mean_session2 = np.mean(session2)
    mean_session3 = np.mean(session3)
    std_dev_session1 = np.std(session1)
    std_dev_session2 = np.std(session2)
    std_dev_session3 = np.std(session3)
    significant_difference_stars1_2 = signficant_star(p_value1_2)
    significant_difference_stars1_3 = signficant_star(p_value1_3)
    significant_difference_stars2_3 = signficant_star(p_value2_3)

    fig, ax = plt.subplots(figsize= (5, 5))
    x1, x2, x3 = 0.1, 0.25, 0.4

    # Bar plots for mean efficiency
    ax.bar(x1, mean_session1, color = 'grey', width = 0.1, label = 'A: No copilot')
    ax.bar(x2, mean_session2, color = 'red', width = 0.1, label = 'B: Copilot')
    ax.bar(x3, mean_session3, color = 'grey', width = 0.1)

    ax.scatter(np.ones(len(session1)) * x1, session1, color = 'black', marker = 'o', alpha = 0.5)
    ax.scatter(np.ones(len(session2)) * x2, session2, color = 'black', marker='o', alpha = 0.5)
    ax.scatter(np.ones(len(session3)) * x3, session3, color = 'black', marker='o', alpha = 0.5)

    ax.errorbar(x1, mean_session1, yerr = std_dev_session1, color = 'black', capsize = 5, capthick = 2, alpha = 0.6)
    ax.errorbar(x2, mean_session2, yerr = std_dev_session2, color = 'red', capsize = 5, capthick = 2, alpha = 0.6)
    ax.errorbar(x3, mean_session3, yerr = std_dev_session3, color = 'black', capsize = 5, capthick = 2, alpha = 0.6)

    ax.annotate(significant_difference_stars1_2, xy=(x1 + 0.08, mean_session2 + 3.0), fontsize=12, ha = 'center', va = 'center', color = 'black')
    ax.annotate(significant_difference_stars1_3, xy=(x2, mean_session3 + 3.0), fontsize=12, ha = 'center', va = 'center', color = 'black')
    ax.annotate(significant_difference_stars2_3, xy=(x2 + 0.05, mean_session2 + 3.0), fontsize=12, ha = 'center', va = 'center', color = 'black')

    ax.set_xticks([x1, x2, x3])
    ax.set_xticklabels(['A', 'B', 'A'], fontsize=16)
    ax.set_ylabel(ylabel, fontsize = 16)
    ax.set_ylim(*ylim)
    ax.tick_params(axis='y', labelsize=14)


    # manual ticks
    if manual_y_ticks is not None:
        y_ticks = np.array(manual_y_ticks)
        ax.set_yticks(y_ticks)

    if title is not None: ax.set_title(title, fontsize = 16)
    ax.legend(loc='upper left', fontsize=16, frameon=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['pdf.fonttype'] = 42

    # plt.title(f'Path Efficiency Comparison')
    plt.tight_layout()
    if savePath is None: plt.show()
    else: plt.savefig(savePath)


def plotABAmeanbar(v1,v2,v3,ylabel='Path Efficiency (%)',ylim=(0, 100),manual_y_ticks=None,title=None,savePath=None):
    """
    v1: single float with mean value (i.e 0.5)
    v2: single float with mean value (i.e 0.5)
    v3: single float with mean value (i.e 0.5)
    """
    
    plt.subplots_adjust(left=0.1, right=0.9, top=3.9, bottom=0.1)
    fig, ax = plt.subplots(figsize= (5, 5))
    x1, x2, x3 = 0.1, 0.25, 0.4

    # Bar plots for mean efficiency
    ax.bar(x1, v1, color = 'grey', width = 0.1, label = 'A: No copilot')
    ax.bar(x2, v2, color = 'red', width = 0.1, label = 'B: Copilot')
    ax.bar(x3, v3, color = 'grey', width = 0.1)

    ax.set_xticks([x1, x2, x3])
    ax.set_xticklabels(['A', 'B', 'A'], fontsize=16)
    ax.set_ylabel(ylabel, fontsize = 16)
    ax.set_ylim(*ylim)
    ax.tick_params(axis='y', labelsize=14)
    
    # manual ticks
    if manual_y_ticks is not None:
        y_ticks = np.array(manual_y_ticks)
        ax.set_yticks(y_ticks)

    if title is not None: ax.set_title(title, fontsize = 16)
    ax.legend(loc='upper left', fontsize=16, frameon=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['pdf.fonttype'] = 42
    
    # plt.title(f'Path Efficiency Comparison')
    plt.tight_layout()
    if savePath is None: plt.show()
    else: 
        plt.savefig(savePath)
        plt.show()

if __name__ == '__main__':

    # only data needed is this:
    session1 = np.random.random(20)*25 + 50
    session2 = np.random.random(20)*25 + 50
    session3 = np.random.random(20)*25 + 50
    plotABAbar(session1,session2,session3,ylabel='Path Efficiency (%)',ylim=(0, 100), title='Sample', savePath='fig.png')

    v1 = np.mean(session1)
    v2 = np.mean(session2)
    v3 = np.mean(session3)
    plotABAmeanbar(v1,v2,v3,ylabel='Path Efficiency (%)',ylim=(0, 100),savePath='figmean.png')
