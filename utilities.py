import math
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt

import seaborn as sn


def create_confusion_matrix(confusion_m, categories, dir, y_lim_value=5, title="cm", show_plots=False, save_plots=False, 
                            xlabel="Predicted Label", ylabel="True Label", method="TRAINING", fig_size=(16,9)):
    '''
    Creates a confusion matrix
    '''
    plt.figure(figsize = fig_size, dpi=150)
    sn.set(font_scale=2.5) #label size
    hm = sn.heatmap(confusion_m, annot=True, fmt='g', annot_kws={"size": 32}) #font size
    hm.set_ylim(y_lim_value, 0)
    hm.set(xticklabels = categories, yticklabels = categories)
    hm.set_yticklabels(hm.get_yticklabels(), rotation=0)
    plt.title(title + ' Confusion Matrix')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if show_plots:
        plt.show()
    if save_plots:
        hm.figure.savefig(dir+ method + "_" + title + '_CM' + '.png', figsize = (16, 9), dpi=150, bbox_inches="tight")
    plt.close()


def print_distribution(l):
    '''
    Get distribution from list
    '''
    print(Counter(l)) 


def split_into_parts(number, n_parts, dtype=int):
    '''
    Create range of equal parts
    '''
    return np.linspace(0, number, n_parts+1, dtype=dtype)[1:]


def softmax(x):
    '''
    Normalize vector between [0, 1], using pnorm approximation 
    from (https://mobile.twitter.com/CBuschNotes/status/667505348962418688)
    '''
    return np.round(1.0/(1.0+np.exp(-1.69897*(x-x.mean())/x.std())), 3)


def tanh(x):
    '''
    Normalize vector using tanh()
    '''
    return (1/2)*(np.tanh(0.01*((x - x.mean())/ x.std())) + 1)


def normalize_0_1(x):
    '''
    Normalize vector between [0, 1]
    '''
    return (x-min(x))/(max(x)-min(x)) 


def normalize_min1_1(x):
    '''
    Normalize vector between [-1, 1]
    '''
    return 2*(normalize_0_1(x))-1


def standardize(x):
    '''
    Standardize vector
    '''
    return (x-x.mean())/x.std()