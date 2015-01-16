import os
import sys
import cPickle as pickle
import csv
from collections import defaultdict
from operator import itemgetter
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import sys
import random

# The first set of colors is good for colorbind people:
colors = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E', '#E6AB02', '#A6761D', '#666666'] + matplotlib.colors.cnames.values()

######################################################################

def plot_js_distances(
        data,
        group1='living',
        group2='departed',
        logscale=True, 
        sampsize=None, 
        title="Linguistic distance from one's email interlocutors",
        ylabel="Jensen-Shannon distance"): 
    """data should be a dict mapping the keys group1 and group2 to dicts mapping usernames to JS values.
    if sampsize is an int, then sampsize values are chosen from each class. logscale=True can help with 
    plotting and does not affect the Mann-Whitney U test done by boxplot_with_datapoints."""
    living = data[group1].values()
    departed = data[group2].values()
    if sampsize:
        # Warning about sizes:
        for vals, label in ((grp1, group1), (grp2, group2)):
            if sampsize > len(vals):
                print "Warning: sampsize %s is larger than len(%s) = %s" % (sampsize, label, len(vals))            
        # Get the sample:
        random.shuffle(grp1)
        grp1 = grp1[ : sampsize]
        random.shuffle(grp2)
        grp2 = grp2[ :sampsize]            
    boxplot_with_datapoints([grp1, grp2], 
                            title=title, 
                            ylabel=ylabel, 
                            xlabels=('%s (n=%s)' % (group1, len(living)), 'Departed (n=%s)' % (group2, len(departed))),
                            logscale=logscale)

######################################################################

def boxplot_with_datapoints(vals, title="", ylabel="", xlabels=[], logscale=True):
    """Generic boxplot function used throughout."""    
    # Dummy labels if none were provided:
    if not xlabels:
        xlabels = ["X%s" % (i+1) for i in range(len(vals))]
    # Optional log-scale transformation:
    if logscale:
        vals = [np.log(v) for v in vals]    
        if ylabel:
            ylabel = "log(%s)" % ylabel
    # The main boxplot:
    fig = plt.figure(figsize=(6, 6)) 
    xlocs = np.arange(1.0, len(vals)+1, 1.0)
    bp = plt.boxplot(vals, 
                    notch=True, 
                    positions=xlocs, 
                    widths=0.75,
                    sym="") # Hide outliers; use . or o to show them.
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['medians'], color='black')
    # Add data points with horizontal jitter for visibility:
    for i, cat_vals in enumerate(vals):
        plt.plot(jitter(np.repeat(xlocs[i], len(cat_vals))), cat_vals, marker='.', markersize=8, linestyle="", color=colors[i])    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("")
    _ = plt.xticks(xlocs, xlabels)
    # Test stats:
    ustat, pval = scipy.stats.mannwhitneyu(vals[0], vals[1])
    pval = np.round(pval, 4)
    # Avoid printing spurious and impossible 0.0 p values:
    pval_report = "p = %s" % pval
    if pval == 0.0:
        pval_report = "p < 0.001"    
    test_report = "Mann-Whitney U = %s; %s" % (ustat, pval_report)
    x1, x2, y1, y2 = plt.axis()
    plt.text(x1, y1, test_report, verticalalignment='bottom', horizontalalignment='left')
    
def jitter(x, sd=0.1):
    mu = 0.0    
    j = x + np.random.normal(mu, sd, len(x))
    return j

###############################################################################

def plot_trajectories(data, min_months=4, max_months=12, zscore=False, logscale=False):
    """JS distances over time."""
    fig = plt.figure(figsize=(24, 10)) 
    all_dates = sorted(set(date for u, date_dict in data.items() for date in date_dict.keys()))
    color_index = 0
    for user, date_dict in data.items():
        if len(date_dict) >= min_months and len(date_dict) <= max_months:
            date_pairs = sorted(date_dict.items())
            # Remove the final month, which might be short and so misleading:
            date_pairs = date_pairs[ :-1]                                
            dates, vals = zip(*date_pairs)
            # Logscale (done first for compatibility with zscore):
            if logscale:
                vals = np.log(vals)
            # Standarize scores to have mean 0.0:
            if zscore:
                zscore = (lambda x : (x-np.mean(vals)) / np.std(vals))
                vals = [zscore(x) for x in vals]                 
            locs = [i for i, d in enumerate(all_dates) if d in dates]
            plt.plot(locs, vals, linestyle='-', marker='', linewidth=4, color=colors[color_index])
            color_index += 1
    xlocs, xstrs = plt.xticks()
    plt.xticks(xlocs, ["%s-%s" % d for d in all_dates], rotation='vertical')
    ylabel = "Jensen-Shannon distance"
    if logscale:
        ylabel = "log(%s)" % ylabel
    if zscore:
        ylabel = "zscore(%s)" % ylabel        
    plt.ylabel(ylabel)
