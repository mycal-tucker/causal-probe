import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

"""
Plotting script for visualizing the probe performance metrics. The metrics are actually calculated during training, so
this is just parsing those metrics out of files and plotting them.:
"""

# Create these global variables for scooping up all metrics; only some will be plotted.
# Distance probes use uuas; depth probes use root_acc; both use dspr.
uuas_scores = np.zeros(25)
dspr_scores = np.zeros(25)
root_accs = np.zeros(25)

root_dir = 'saved_models/example/'
dir_prefix = 'model_depth'

# Pull out the actual metrics from the saved files.
for subdir in os.scandir(root_dir):
    path = subdir.path
    if dir_prefix not in path:
        print("Skipping results dir", path)
        continue
    prefix_idx = path.index(dir_prefix) + len(dir_prefix)
    layer_id = int(path[prefix_idx:])
    # Now iterate through all the folders in this directory; we choose the most recent folder so that we're plotting
    # the latest results.
    model_dirs = glob.glob(os.path.join(path, '*'))
    model_dir = sorted(model_dirs)[-1]
    for results_file in os.scandir(model_dir):
        if 'spearmanr-5_50' in results_file.path:
            with open(results_file, 'r') as spearman_file:
                for line in spearman_file:
                    spearman_score = float(line)
                    dspr_scores[layer_id] = spearman_score
        elif 'uuas' in results_file.path:
            with open(results_file, 'r') as uuas_file:
                for line in uuas_file:
                    uuas_score = float(line)
                    uuas_scores[layer_id] = uuas_score
        elif 'root_acc' in results_file.path:
            with open(results_file, 'r') as acc_file:
                for line in acc_file:
                    parsed = line.split('\t')
                    root_acc_score = float(parsed[0])
                    root_accs[layer_id] = root_acc_score


# Basic plotting functionality takes two metrics and puts them in the same plot, by layer.
def plot_two_metrics(metric1_vals, metric2_vals, metric1_name, metric2_name):
    matplotlib.rcParams.update({'font.size': 12})
    fig, ax1 = plt.subplots()
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    x_axis = [i + 1 for i in range(24)]
    color = 'tab:red'
    ax1.set_xlabel('Layer idx')
    ax1.set_ylabel(metric1_name, color=color)
    ax1.plot(x_axis, metric1_vals[1:], color=color, marker='s')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel(metric2_name, color=color)
    ax2.plot(x_axis, metric2_vals[1:], color=color, linestyle='--', marker='s')
    plt.tight_layout()
    plt.show()


plot_two_metrics(uuas_scores, dspr_scores, 'UUAS', 'DSpr.')
plot_two_metrics(root_accs, dspr_scores, 'Root Acc.', 'DSpr.')

