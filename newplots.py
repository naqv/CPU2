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



#NO AVAILABLE ON CONDA REPOS, INSTALL WITH PIP
from SALib.analyze import sobol
from SALib.sample import saltelli
from SALib.test_functions import Ishigami
from SALib.util import read_param_file

#TO USE IN ANOVA TABLE.
import statsmodels.api as sm
from statsmodels.formula.api import ols

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

df =  pd.read_csv('nuevo_results_out.csv', delimiter = ';', low_memory = False)

def plotHist(x,  df, n, xlabel, ylabel, title, filename, labels = None):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if labels == None:
        plt.hist(df[x], n)
    else:
        plt.hist(df[x], n, label = labels)
    plt.title(title)
    plt.savefig(folder + '/' + filename)
    plt.clf()

def saveHeatMap(df, title, filename):
    corr = df.corr()
    
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(corr, mask=mask, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    plt.title(title)
    plt.savefig(folder + '/' + filename)
    plt.clf()
    print('saved heatmap')

def plotDataset(x_, y_, ds, xlabel, ylabel, title, filename,formatter = '%d'):
    ax = sns.lineplot(x = x_, y = y_, data = ds,estimator='mean', ci=95, n_boot=1000)
    ax.yaxis.set_major_formatter(FormatStrFormatter(formatter))
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.savefig(folder + '/' + filename)
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
    plt.savefig(folder + '/' + filename, bbox_extra_artists=(legend,), bbox_inches="tight")
    plt.clf()

def saveTable(df, title, filename):
    fig, ax = plt.subplots(1, 1)
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False) 
    fig.set_size_inches(11, 14)
    ax.title.set_text(title)

    ptable(ax, df, loc='upper center')
    plt.axis('off')
    
    plt.savefig(folder + '/' + filename)
    print('saving table ', filename)
    plt.clf()

def statiscticsThreshold(df, x, title, filename):
    ec = df[x]
    std = np.std(df[x])
    fig, ax = plt.subplots(1, 1)
    dataframe = pd.DataFrame(data = {'Name': [x], 'Standard Deviation' : [std]})
    saveTable(dataframe, title, filename)

    NEW_FEATURE = []
    i = 0
    
    for index, row in df.iterrows():
        if(row[x]> 89):
            NEW_FEATURE.append('OVERLOAD')
        elif(row[x] < 10):
            NEW_FEATURE.append('UNDERLOAD')
        else:
            NEW_FEATURE.append('NORMAL')
        i += 1

    df = df.assign(LOADS = NEW_FEATURE)

    sns.countplot(x = 'LOADS', data = df[df['LOADS'].isin(['OVERLOAD','UNDERLOAD'])])
    plt.xlabel('OVERLOAD, UNDERLOAD')
    plt.ylabel('Frequency')
    plt.title('OVERLOAD, UNDERLOAD FREQUENCY')
    plt.savefig(folder + '/' + 'countplot_overload_underload_cpu_usage')
    plt.clf()

def statisticThresholdWithTu(df, x, title, filename, Tu):
    ec = df[x]
    std = np.std(df[x])
    fig, ax = plt.subplots(1, 1)
    dataframe = pd.DataFrame(data = {'Name': [x], 'Standard Deviation' : [std]})
    saveTable(dataframe, title, filename)

    NEW_FEATURE = []
    i = 0

    #Tu = 1 - s*MA
    #s = (1 - Tu)/Ma
    Tu = 0.8
    
    for index, row in df.iterrows():
        if(row[x]> (Tu * 100)):
            NEW_FEATURE.append('OVERLOAD')
        else:
            NEW_FEATURE.append('NORMAL')
        i += 1

    df = df.assign(LOADS = NEW_FEATURE)

    sns.countplot(x = 'LOADS', data = df[df['LOADS'] == 'OVERLOAD'])
    plt.xlabel('OVERLOAD')
    plt.ylabel('Frequency')
    plt.title('OVERLOADFREQUENCY')
    plt.savefig(folder + '/' + 'countplot_overload_using_tu_cpu_usage')
    plt.clf()

def plotPerformance(df):
    failure_rate = 1/df['MTT']
    performance = 1 - failure_rate
    aux = pd.DataFrame(data = {'Failure': failure_rate , 'Performance': performance.values, 'Time': df['TIMESTAMP'].values})
    plotDataset('Time','Performance',aux,'Time[s]', 'Performance','Performance vs Timestamp','plot_4_perf_vs_time','%.6f')
    plotDataset('Time','Failure',aux,'Time[s]','Failure Rate','Failure Rate vs Timestamp','plot_5_failure_rate_vs_time','%.6f')


def plotNewCharts(df):
    '''
        See =>
        Equations for new graph:
        Available Bandwidth (AB) #no idea about this feature
        C: Bandwidth total !no idea about this feature
        Idle_rate: Bandwidth total - Bandwidth usage AB=C/(C+Idle_rate) #no idea about this feature
        Times==timesstamp
        Thruput = Flow traffic / times (~ 50hours), *partially done, check, assumed times = 50
        Thruput availability = avg Thruput / max Thruput, *done
        to find is to only use box plot table, *done, which will inform the upper, middle and lower limit
        failure rate : 1/MTTF performance= 1 - failure rate *done, see plotPerformance function
    '''
    #pending, i don't have AB, C or a method to calculate them

def plotBox(df, column, title, filename):
    sns.boxplot(x = column, data = df, orient = 'v')
    plt.title(title)
    plt.savefig(folder + '/' + filename)
    plt.clf()
    
def sensibilityAnalysis(df):
    #variables to use
    sdf = df[['TIMESTAMP','MTT_lower','MTT_upper']]

    #bounds
    ts_min = sdf['TIMESTAMP'].min()
    ts_max = sdf['TIMESTAMP'].max()

    mtt_lower_min = sdf['MTT_lower'].min()
    mtt_lower_max = sdf['MTT_lower'].max()

    mtt_upper_min = sdf['MTT_upper'].min()
    mtt_upper_max = sdf['MTT_upper'].max()

    problem = {
      'num_vars': 3,
      'names': ['TIMESTAMP','MTT_lower', 'MTT_upper'],
      'bounds': [[ts_min,ts_max],[mtt_lower_min,mtt_lower_max],[mtt_upper_min,mtt_upper_max]]
    }

     
    # [[mtt_lower_min,mtt_lower_max],[mtt_upper_min,mtt_upper_max]]
    # Generate samples
    param_values = saltelli.sample(problem, 1000, calc_second_order=True)

    # Run the "model" and save the output in a text file
    # This will happen offline for external models
    Y = Ishigami.evaluate(param_values)

    Si = sobol.analyze(problem, Y, calc_second_order=True, conf_level=0.95, print_to_console=True)

def getANOVA(df, target, lsFeatures):
    #replace empty spaces with underscores
    newName = target.replace(' ', '_')
    df = df.rename(columns = {target: newName})
    fts = '+'.join(lsFeatures)
    modelLine = newName + ' ~ ' + fts
    model = ols(modelLine, data = df).fit()
    print(model.summary())

cpu_usage = 'vnf cpu usage'
mem_usage = 'vnf mem usage'
sto_usage = 'vnf sto usage'
time = 'TIMESTAMP'

#adding Thruput
#check this value (50) do I have to divide by 50 or timestamp?
df = df.assign(Thruput = (df['flow traffic']/ 50).values)

sample = df.sample(1000)

thruputAvailability_Sample = sample['Thruput'].mean() / sample['Thruput'].max()
thruputAvailability_AllDF = df['Thruput'].mean() / df['Thruput'].max()

#statistics threshold
statiscticsThreshold(sample.sort_values(time), cpu_usage, 'Standard Deviation CPU USAGE', 'table_stdev_cpu_usage')

#statistics threshold with Tu
#statisticThresholdWithTu(df, x, title, filename, Tu)
statisticThresholdWithTu(sample.sort_values(time), cpu_usage, 'Standard Deviation CPU USAGE', 'table_stdev_shtu_cpu_usage',0.8)

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

#plotting boxplot
plotBox(sample, 'Thruput', 'Thruput Boxplot', 'boxplot_1_Thruput')

#printing ANOVA
getANOVA(df, cpu_usage, ['MTT_lower','MTT_upper'])
