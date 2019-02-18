# -*- coding: utf-8 -*-
####################### LIBRARIES #########################################
import pandas as pd
import math
import collections
import multiprocessing as mp
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas.plotting import table as ptable
from pylab import *
import os
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
import scipy.stats as stats
#CONSTANTS

SAMPLESIZE = 100
WIDTH_FIGURE = 11.7
HEIGHT_FIGURE =  8.27

# seaborn options
sns.set()
sns.set_palette("husl",3) 
sns.set(rc={'figure.figsize':(WIDTH_FIGURE,HEIGHT_FIGURE*2)})


folder = 'statistic_threshold'

if not os.path.exists(folder):
    os.makedirs(folder)

df =  pd.read_csv('results_out.csv', delimiter = ';', low_memory = False)

def plotHist(x,  df, n, xlabel, ylabel, title, filename):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.hist(df[x], n)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()

def saveHeatMap(df, title, filename):
    corr = df.corr()
    
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(corr, mask=mask, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    plt.title(title)
    plt.savefig(filename)
    plt.clf()
    print('saved heatmap')

def plotDataset(x_, y_, ds, xlabel, ylabel, title, filename,formatter = '%d'):
    ax = sns.lineplot(x = x_, y = y_, data = ds,estimator='mean', ci=95, n_boot=1000)
    ax.yaxis.set_major_formatter(FormatStrFormatter(formatter))
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()

def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
    """Return an axes of confidence bands using a simple approach.

    Notes
    -----
    .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
    .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}

    References
    ----------
    .. [1]: M. Duarte.  "Curve fitting," JUpyter Notebook.
       http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb

    """
    if ax is None:
        ax = plt.gca()

    ci = t*s_err*np.sqrt(1/n + (x2-np.mean(x))**2/np.sum((x-np.mean(x))**2))
    ax.fill_between(x2, y2+ci, y2-ci, color="#b9cfe7", edgecolor="")

    return ax

def plot_ci_bootstrap(n, x, y, resid, nboot=500, ax=None):
    """Return an axes of confidence bands using a bootstrap approach.

    Notes
    -----
    The bootstrap approach iteratively resampling residuals.
    It plots `nboot` number of straight lines and outlines the shape of a band.
    The density of overlapping lines indicates improved confidence.

    Returns
    -------
    ax : axes
        - Cluster of lines
        - Upper and Lower bounds (high and low) (optional)  Note: sensitive to outliers

    References
    ----------
    .. [1] J. Stults. "Visualizing Confidence Intervals", Various Consequences.
       http://www.variousconsequences.com/2010/02/visualizing-confidence-intervals.html

    """ 
    if ax is None:
        ax = plt.gca()

    bootindex = sp.random.randint
    nx = n 

    for _ in range(nboot):
        resamp_resid = resid[bootindex(0, len(resid)-1, len(resid))]
        # Make coeffs of for polys
        pc = sp.polyfit(x, y + resamp_resid, 1)                   
        # Plot bootstrap cluster
        ax.plot(x, sp.polyval(pc,x), "b-", linewidth=2, alpha=3.0/float(nboot))

    return ax

def plotInterval(df,x,y, xlabel, ylabel, title, filename):
    x = df[x].values
    y = df[y].values

    # Modeling with Numpy
    p, cov = np.polyfit(x, y, 1, cov=True)        # parameters and covariance from of the fit
    y_model = np.polyval(p, x)                    # model using the fit parameters; NOTE: parameters here are coefficients
    
    # Statistics
    n = x.size                              # number of observations
    m = p.size                                    # number of parameters
    DF = n - m                                    # degrees of freedom
    t = stats.t.ppf(0.95, n - m)                  # used for CI and PI bands
    
    # Estimates of Error in Data/Model
    resid = y - y_model                           
    chi2 = np.sum((resid/y_model)**2)             # chi-squared; estimates error in data
    chi2_red = chi2/(DF)                          # reduced chi-squared; measures goodness of fit
    s_err = np.sqrt(np.sum(resid**2)/(DF))        # standard deviation of the error

    fig, ax = plt.subplots(figsize=(8,6))

    # Data
    ax.plot(x,y,"o", color="#b9cfe7", markersize=8,
         markeredgewidth=1,markeredgecolor="b",markerfacecolor="None")
    
    # Fit
    ax.plot(x,y_model,"-", color="0.1", linewidth=1.5, alpha=0.5, label="Fit")  

    x2 = np.linspace(np.min(x), np.max(x), 100)
    y2 = np.linspace(np.min(y_model), np.max(y_model), 100)

    # Confidence Interval (select one)
    plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax)
    #plot_ci_bootstrap(n, x, y, resid, ax=ax)

    # Prediction Interval
    pi = t*s_err*np.sqrt(1+1/n+(x2-np.mean(x))**2/np.sum((x-np.mean(x))**2))   
    ax.fill_between(x2, y2+pi, y2-pi, color="None", linestyle="--")
    ax.plot(x2, y2-pi, "--", color="0.5", label="95% Prediction Limits")
    ax.plot(x2, y2+pi, "--", color="0.5")


    # Figure Modifications --------------------------------------------------------
    # Borders
    ax.spines["top"].set_color("0.5")
    ax.spines["bottom"].set_color("0.5")
    ax.spines["left"].set_color("0.5")
    ax.spines["right"].set_color("0.5")
    ax.get_xaxis().set_tick_params(direction="out")
    ax.get_yaxis().set_tick_params(direction="out")
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left() 

    # Labels
    plt.title(title, fontsize="14", fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(np.min(x)-1,np.max(x)+1)

    # Custom legend
    handles, labels = ax.get_legend_handles_labels()
    display = (0,1)
    anyArtist = plt.Line2D((0,1),(0,0), color="#b9cfe7")  # Create custom artists
    legend = plt.legend(
          [handle for i,handle in enumerate(handles) if i in display]+[anyArtist],
          [label for i,label in enumerate(labels) if i in display]+["95% Confidence Limits"],
          loc=9, bbox_to_anchor=(0, -0.21, 1., .102), ncol=3, mode="expand")  
    frame = legend.get_frame().set_edgecolor("0.5")

    # Save Figure
    plt.tight_layout()
    plt.savefig(filename, bbox_extra_artists=(legend,), bbox_inches="tight")
    plt.clf()

def saveTable(df, title, filename):
    fig, ax = plt.subplots(1, 1)
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False) 
    fig.set_size_inches(11, 14)
    ax.title.set_text(title)

    ptable(ax, df, loc='upper center')
    plt.axis('off')
    
    plt.savefig(filename)
    print('saving table ', filename)
    plt.clf()

def statiscticsThreshold(df, x, title, filename):
    ec = df[x]
    std = np.std(df[x])
    dataframe = pd.DataFrame(data = {'Name': x, 'Standar Deviation' : std})
    fig, ax = plt.subplots(1, 1)
    ptable(ax, data_Frame, loc='upper center', colWidths=[0.2, 0.2, 0.2])
    plt.savefig(filename)
    plt.clf()

    NEW_FEATURE= np.zeros_like(np.arange(len(df)))

    for i in range(len(df)):
        if df[x] > 89:
            NEW_FEATURE[i] = 2
        elif df[x] <10:
            NEW_FEATURE[i] = 1


def plotPerformance(df):
    failure_rate = 1/df['MTT']
    performance = 1 - failure_rate
    aux = pd.DataFrame(data = {'Failure': failure_rate , 'Performance': performance.values, 'Time': df['TIMESTAMP'].values})
    plotDataset('Time','Performance',aux,'Time[s]', 'Performance','Performance vs Timestamp','plot_4_perf_vs_time','%.6f')
    plotDataset('Time','Failure',aux,'Time[s]','Failure Rate','Failure Rate vs Timestamp','plot_5_failure_rate_vs_time','%.6f')


cpu_usage = 'vnf cpu usage'
mem_usage = 'vnf mem usage'
sto_usage = 'vnf sto usage'
time = 'TIMESTAMP'
sample = df.sample(1000)

#plotting hist
plotHist(cpu_usage, df, 50,'CPU USAGE','Frequency','CPU USAGE HISTOGRAM','hist_1_cpu_usage')
plotHist(mem_usage, df, 50,'MEMORY USAGE','Frequency','MEMORY USAGE HISTOGRAM','hist_2_mem_usage')
plotHist(sto_usage, df, 50,'STORAGE USAGE','Frequency','STORE USAGE HISTOGRAM','hist_3_sto_usage')

#plotting datasets
plotDataset(time, cpu_usage, sample.sort_values(time),'TIME [s]', 'CPU USAGE', 'CPU USAGE VS TIMESTAMP', 'plot_1_cpu_vs_time')
plotDataset(time, mem_usage, sample.sort_values(time),'TIME [s]', 'MEM USAGE', 'MEM USAGE VS TIMESTAMP', 'plot_2_mem_usage_vs_time')
plotDataset(time,'sla', sample.sort_values(time),'TIME [s]', 'SLA', 'SLA VS TIMESTAMP', 'plot_3_sla_vs_time')


#plotting heatmaps
saveHeatMap(sample.iloc[:,1:42], 'HEAT MAP', 'heatmap_41variables')

#save table
saveTable(sample[[cpu_usage, mem_usage]].describe(),'TABLE', 'table_1_cpu')

#saving confidence interval
plotInterval(sample.sort_values(cpu_usage), time, cpu_usage,'TIMESTAMP','CPU USAGE','CPU USAGE VS TIMESTAMP','confidence_interval_1_cpu_vs_time')

plotPerformance(sample.sort_values(time))
