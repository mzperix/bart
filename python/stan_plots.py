## STAN DIAGNOSTICS

import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import numpy as np

def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*0.5, 0.25*height,
                '%.3g' % height,
                ha='center', va='bottom')

def parameter_plot(parameter_samples, chains, parameter_name, ax = None, show = True, true_value = None):
    ind = np.arange(chains)  # the x locations for the groups
    width = 0.75             # the width of the bars

    if ax is None:
        fig, ax = plt.subplots()
    parameter_means = []
    parameter_stds  = []
    l = int(len(parameter_samples)/chains)
    for c in range(chains):
        parameter_means.append(np.mean(parameter_samples[c*(l):(c+1)*l]))
        parameter_stds.append(2*np.std(parameter_samples[c*(l):(c+1)*l]))
    rects = ax.bar(ind, parameter_means, width, color='#598a8a', yerr=parameter_stds)
    if true_value is not None:
        ax.axhline(y=true_value, xmin=0, xmax=chains, linewidth=2, color = '#2c4545')
    # add some text for labels, title and axes ticks
    ax.set_title(parameter_name)
    autolabel(rects, ax)
    if show: 
        plt.show()


def chain_plot(parameter_samples, chains, parameter_name, ax = None, show = True, true_value = None):
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(parameter_samples, color = '#7aa1a1')
    ax.set_title(parameter_name)
    if true_value is not None:
        ax.axhline(y=true_value, xmin=0, xmax=len(parameter_samples), linewidth=2, color = '#2c4545')
    if show:
        plt.show()

def sample_plots(samples, chains, true_values = dict()):
    rcParams['figure.figsize'] = chains*1.5*len(samples), 7
    fig, axes = plt.subplots(2, len(samples))
    p = 0
    for key, value in samples.items():
        if key in true_values:
            true_value = true_values[key]
        else:
            true_value = None
        chain_plot(value, chains, key, axes[1,p], False, true_value)
        parameter_plot(value, chains, key, axes[0,p], False, true_value)
        p += 1
    plt.show()