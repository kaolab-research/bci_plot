import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy

def significant_star(p_value:float):
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

def plotABbox(CollectionA, CollectionB, ylabel='Path Efficiency (%)', ylim=(0, 100), manual_y_ticks=None, title='Plot AB', savePath=None, useStar=False, useP=False, useLegends=True, others={}, medianColor='cyan'):
    """ 
    plotting A and B only with other parameter 
    CollectionA: np.array(1D)
    CollectionB: np.array(1D)

    useful parameter
        useStar, useP = True if you want to see them. only one can be set to True!
    """
    # constants
    figsize = (5, 5) # (0.8858268, 0.8858268) = 22.5mm
    barpos = [0,1]
    ylabelFontSize = 16
    titleFontSize = 16

    for k, v in others.items():
        if k=='figsize': figsize=v
        if k=='barpos': barpos=v
        if k=='ylabelFontSize': ylabelFontSize=v
        if k=='titleFontSize': titleFontSize=v

    fig, axs = plt.subplots(figsize=figsize)
    bp0 = axs.boxplot(CollectionA, positions=[barpos[0]], widths=0.5, vert=True, patch_artist=True, showfliers=False)
    bp1 = axs.boxplot(CollectionB, positions=[barpos[1]], widths=0.5, vert=True, patch_artist=True, showfliers=False)

    for median in bp0['medians']: median.set_color(medianColor)
    for median in bp1['medians']: median.set_color(medianColor)

    colors = ['grey', 'red']
    for bplot, color in zip((bp0, bp1), colors):
        for patch in bplot['boxes']:
            patch.set_facecolor(color)

    # p line between
    if useStar or useP:
        pval = scipy.stats.ranksums(CollectionA, CollectionB)[1]
        
        if pval < 0.05:
            mv = np.maximum(bp0['whiskers'][1].get_ydata()[1], bp1['whiskers'][1].get_ydata()[1])
            yscale = ylim[1]-ylim[0]
            liney = mv+yscale*0.08
            lineh = yscale*0.02
            texty = liney + yscale*0.04
            stary = liney + yscale*0.04
            plt.plot([barpos[0], barpos[0], barpos[1], barpos[1]], [liney, liney + lineh, liney + lineh, liney], 'k') # line
            if useP: plt.text(sum(barpos)/2, texty, f'p = {pval:.3}', va='bottom', ha='center') # text
            sigStars = significant_star(pval)
            if useStar: plt.annotate(sigStars, xy=(sum(barpos)/2, stary), fontsize=12, ha = 'center', va = 'center', color = 'black') # stars


    # plt.xticks([0.1, 0.25, 0.4], )
    plt.xticks(fontsize=16)
    plt.ylabel(ylabel, fontsize = ylabelFontSize)
    plt.ylim(*ylim)
    plt.tick_params(axis='y', labelsize=ylabelFontSize)
    axs.set_xticklabels(['A', 'B'], fontsize=ylabelFontSize)
    plt.title(title, fontsize = titleFontSize)
    if useLegends: axs.legend([bp0["boxes"][0], bp1["boxes"][0]], ['A: No Copilot', 'B: Copilot'], loc='upper left', fontsize=14, frameon=False)

    # manual ticks
    if manual_y_ticks is not None:
        y_ticks = np.array(manual_y_ticks)
        axs.set_yticks(y_ticks)
    
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    # plt.gca().yaxis.grid(True)

    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()
    
    if savePath is None: plt.show()
    else: plt.savefig(savePath)


def plotABboxfor4(CollectionA, CollectionB, ylabel='Path Efficiency (%)', ylim=(0, 100), manual_y_ticks=None, title='Plot AB', savePath=None, useStar=False, useP=False, medianColor='cyan'):
    """ 
    plotting A and B for all 4 subjects at once!! 
    CollectionA: [np.array(1D),np.array(1D),np.array(1D),np.array(1D)]
    CollectionB: [np.array(1D),np.array(1D),np.array(1D),np.array(1D)]
    useful parameter
        useStar, useP = True if you want to see them. only one can be set to True!
    """
    subjects = ['H1','H2','H4','S2']

    fig, axs = plt.subplots(figsize= (10, 5))
    bp0 = axs.boxplot(CollectionA, positions=[0,3,6,9], widths=0.5, vert=True, patch_artist=True, showfliers=False)
    bp1 = axs.boxplot(CollectionB, positions=[1,4,7,10], widths=0.5, vert=True, patch_artist=True, showfliers=False)

    # p line between
    if useStar or useP:
        for i, (x1,x2) in enumerate(zip([0,3,6,9],[1,4,7,10])):
            pval = scipy.stats.ranksums(CollectionA[i], CollectionB[i])[1]
            
            if pval < 0.05:
                mv = np.maximum(bp0['whiskers'][i*2+1].get_ydata()[1], bp1['whiskers'][i*2+1].get_ydata()[1])
                yscale = ylim[1]-ylim[0]
                liney = mv+yscale*0.08
                lineh = yscale*0.02
                texty = liney + yscale*0.04
                stary = liney + yscale*0.04
                midx = (x2+x1)/2
                plt.plot([x1, x1, x2, x2], [liney, liney + lineh, liney + lineh, liney], 'k') # line
                if useP: plt.text(midx, texty, f'p = {pval:.3}', va='bottom', ha='center') # text
                sigStars = significant_star(pval)
                if useStar: plt.annotate(sigStars, xy=(midx, stary), fontsize=12, ha = 'center', va = 'center', color = 'black') # stars




    for median in bp0['medians']: median.set_color(medianColor)
    for median in bp1['medians']: median.set_color(medianColor)

    colors = ['grey', 'red', 'grey']
    for bplot, color in zip((bp0, bp1), colors):
        for patch in bplot['boxes']:
            patch.set_facecolor(color)

    # plt.xticks([0.1, 0.25, 0.4], )
    plt.title(title, fontsize = 16)
    plt.ylabel(ylabel, fontsize = 16)
    plt.ylim(*ylim)
    plt.tick_params(axis='y', labelsize=16)
    axs.set_xticklabels([' ' * 10 + subjects[0],
                        ' ' * 10 + subjects[1],
                        ' ' * 10 + subjects[2],
                        ' ' * 10 + subjects[3],
                        '','','',''], fontsize=16)
    
    axs.legend([bp0["boxes"][0], bp1["boxes"][0]], ['A: No Copilot', 'B: Copilot'], loc='upper left', fontsize=14, frameon=False)

    # manual ticks
    if manual_y_ticks is not None:
        y_ticks = np.array(manual_y_ticks)
        axs.set_yticks(y_ticks)

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    # plt.gca().yaxis.grid(True)
    
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()
    
    if savePath is None: pass #plt.show()
    else: plt.savefig(savePath)


def plotABboxfor2(CollectionA, CollectionB, ylabel='Path Efficiency (%)', ylim=(0, 100), manual_y_ticks=None, title='Plot AB', savePath=None, useStar=False, useP=False):
    """ 
    plotting A and B for all 4 subjects at once!! 
    CollectionA: [np.array(1D),np.array(1D)]
    CollectionB: [np.array(1D),np.array(1D)]
    useful parameter
        useStar, useP = True if you want to see them. only one can be set to True!
    """
    pass