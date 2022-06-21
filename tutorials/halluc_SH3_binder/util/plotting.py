def box_off(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def deduplicate_legend_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    return (newHandles, newLabels)

def set_matplotlib_font_sizes(
    small_size = 12,
    medium_size = 14,
    large_size = 16,):
    import matplotlib.pyplot as plt

    plt.rc('font', size=small_size)          # controls default text sizes
    plt.rc('axes', titlesize=large_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=small_size)    # legend fontsize
    plt.rc('figure', titlesize=large_size)  # fontsize of the figure title

def set_jue_plotting_defaults():
    import matplotlib.pyplot as plt
    set_matplotlib_font_sizes()
    plt.style.use('default')
    plt.style.use('default')
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
def savefig(fig, filename, **kwargs):
    args = {'dpi':150,
            'bbox_inches':'tight'
           }
    args.update(kwargs)
    fig.savefig(filename, **args)

def outside_legend(ax, deduplicate=True, **kwargs):
    if deduplicate:
        ax.legend(*deduplicate_legend_labels(ax),bbox_to_anchor=(1.01,0.5), loc='center left', **kwargs)
    else:
        ax.legend(bbox_to_anchor=(1.01,0.5), loc='center left', **kwargs)

def outside_axis_labels(ax, axs, xlabel='', ylabel=''):
    if any(ax==axs[-1,:]):
        ax.set_xlabel(xlabel)
    if any(ax==axs[:,0]):
        ax.set_ylabel(ylabel)

def square_axes(ax, draw_diagonal=True):
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    lim = [min(xl[0],yl[0]), max(xl[1],yl[1])]
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    if draw_diagonal: ax.plot(lim,lim,'k:')

def reorder_df(df, column, values):
    df2 = None
    for v in values:
        if df2 is None:
            df2 = df[df[column]==v]
        else:
            df2 = df2.append(df[df[column]==v])
    return df2
