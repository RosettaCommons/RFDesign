from icecream import ic
import argparse
import numpy as np 
import pandas as pd 

from collections import defaultdict


# loss_s columns are here: 
# c6d1, c6d2, c6d3, c6d4, aa_CCE, acc1, acc2, acc3, str_loss (x10), lddt_loss, ca_lddt (x10), dih_loss, blen_loss, bang_loss
cols = ['c6d1', 'c6d2', 'c6d3', 'c6d4', 'aa_CCE', 'acc1', 'acc2', 'acc3'] + [f'str_loss_{i}' for i in range(1,11)] + ['lddt_loss'] + [f'ca_lddt_{i}' for i in range(1,11)] + ['dih_loss', 'blen_loss', 'bang_loss']
cols = ['total'] + cols 
# plotting params
BATCH_SIZE = 128
N_PRINT_TRAIN = 8
CONV = 1
EPOCH_SIZE = 25600
FIGSIZE=(5,4)

loss_expl = {
    'c6d': '2d distance and angle cross-entropy, 0: distogram, 1,2: angles (360 degrees in bins of 10), 3: (180 degrees in bins of 10)',
    'aa_CCE': 'sequence cross-entropy',
    'acc': 'top-sequence logit accuracy', # TOP 1,2,3 logits, not part of total loss
    'str_loss': 'FAPE loss for last recycle, one for each SE3 layer', # More local than global
    'lddt_loss': 'predicted lddt vs true lddt '
}

get_loss = lambda x: float( x.split('|')[1].split()[-1] )

def convolve(x,w):
    return np.convolve(x, np.ones(w)/w, mode='valid')

def parse_line_task(l):
    tot_loss = [0]
    task_loss = [float(a) for a in l.strip().split('|')[1].split()]
    return tot_loss + task_loss

def parse_line_all(l):
    tot_loss = [float( l.split('|')[1].split()[-1] )]
    task_loss = [float(a) for a in l.strip().split('|')[2].split()]
    return tot_loss + task_loss

def parse_logs(log_path):
    # get lines 
    with open(log_path, 'r') as fp:
        lines = fp.readlines()
        
    assert lines
    logs = {}
    for prefix, stage in [('Local', 'train'), ('Valid', 'valid')]:
        prefix_lines = (l for l in lines if l.startswith(prefix))
        task_logs = defaultdict(list)
        for l in prefix_lines:
            parse = parse_line_task
            w1, w2 = l.split(' ')[:2]
            task = w2[:-1]
            total_line = w1[-1] == ':'
            if total_line:
                task = 'all'
                parse = parse_line_all
            task_logs[task].append(parse(l))
        
        for task, data in task_logs.items():
            task_logs[task] = pd.DataFrame(data=data, columns=cols)
        logs[stage] = task_logs
    return logs

def plot_specific_loss(ax, df, loss_key, smoothing_width = 50, show_unsmoothed=True):
    y = convolve(df[loss_key], CONV)
    x = np.arange(len(y)) * BATCH_SIZE

    if show_unsmoothed:
        ax.plot(x,y,label=loss_key,alpha=1)

    if smoothing_width:
        y2 = convolve(y, smoothing_width)
        x2 = np.arange(len(y2)) * BATCH_SIZE + smoothing_width
        ax.plot(x2,y2, label=f'mean {loss_key}', alpha=0.9)
        plt.xlabel('n_train')
        plt.ylabel('loss')

def plot_all_loss(ax, logs, stage, task):
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    for loss_key in cols:
        plot_specific_loss(ax, logs[stage][task], loss_key, smoothing_width=None, show_unsmoothed=True)
    # plt.legend(bbox_to_anchor=(1.04, 1))
    plt.legend(bbox_to_anchor=(1.10, 1), prop={'size': 3})

def plot_total_loss(ax, logs, smoothing_width=10):
    local_losses = logs['train']['all']['total']

    # y = convolve(local_losses, CONV)
    y = local_losses
    x = np.arange(len(y)) * N_PRINT_TRAIN
    ic(len(x))
    ic(len(y))

    ax.scatter(x,y, label='total loss', alpha=0.8)

    y2 = convolve(local_losses, smoothing_width)
    x2 = (np.arange(len(y2)) + smoothing_width//2) * N_PRINT_TRAIN
    ax.plot(x2,y2,c='red', label='mean', alpha=0.9)

    ic(len(y2))
    ic(len(x2))

    ax.set_title('Combined loss function')
    ax.set_xlabel('# training examples')
    ax.set_ylabel('Total loss')


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TKAgg')
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--log', type=str,
                        help='path to log file')
    args = parser.parse_args()

    logs = parse_logs(args.log)

    fig, ax = plt.subplots(2, dpi=150, figsize=(8,6))
    plot_total_loss(ax[0], logs)
    plot_all_loss(ax[1], logs, 'train', 'seq2str')
    plt.subplots_adjust(right=0.75)
    # plt.tight_layout()

    plt.show()
else:
    import matplotlib
    import matplotlib.pyplot as plt


    # fig, ax = plt.subplots(dpi=150, figsize=FIGSIZE)
    # plot_specific_loss(ax, logs['train']['seq2str'], 'aa_CCE', smoothing_width=10, show_unsmoothed=True)
    # plt.legend()

    # plot_all_loss('train', 'seq2str')
