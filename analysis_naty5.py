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
#CONSTANTS

SAMPLESIZE = 100

# seaborn options
sns.set()
sns.set_palette("husl",3) 
sns.set(rc={'figure.figsize':(11.7,8.27)})

def load_csv():
    try:
        df = pd.read_csv('results_out.csv', delimiter = ';', low_memory = False)
        return df
    except Exception as e:
        print('error reading file')
        print(e)

def getMeanConfidenceInterval(data,name):
    confidence=0.95
    a = 1.0 * data
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    inf= m-h
    sup = m+h
    print('The confidence inteval of', name, ' is [', inf, ',', sup,'].')
    return (m, m-h, m+h)

def printConfidenceInterval(df):
    #check this, the variables are not used in another place, consider avoid return a tuple in
    #getMeanConfidenceInterval function (I.A.).
    mci_mtt = getMeanConfidenceInterval(df['TF'].values,'TF')
    mci_mtt = getMeanConfidenceInterval(df['MTT'].values,'MTT')
    mci_mttfr = getMeanConfidenceInterval(df['MTTF_R'].values,'MTTF_R')
    mci_MTTF_EM = getMeanConfidenceInterval(df['MTTF_EM'].values,'MTTF_EM')
    mci_MTTF_C = getMeanConfidenceInterval(df['MTTF_C'].values,'MTTF_C')
    mci_MTTF_TDDB = getMeanConfidenceInterval(df['MTTF_TDDB'].values,'MTTF_TDDB')
    mci_MTTF_SM = getMeanConfidenceInterval(df['MTTF_SM'].values,'MTTF_SM')
    mci_MTTFF_TC = getMeanConfidenceInterval(df['MTTFF_TC'].values,'MTTFF_TC')
    mci_A = getMeanConfidenceInterval(df['A'].values,'A')
    mci_AEM = getMeanConfidenceInterval(df['AEM'].values,'AEM')
    mci_AC = getMeanConfidenceInterval(df['AC'].values,'AC')
    mci_ATDDB = getMeanConfidenceInterval(df['ATDDB'].values,'ATDDB')
    mci_ASM = getMeanConfidenceInterval(df['ASM'].values,'ASM')
    mci_ATC = getMeanConfidenceInterval(df['ATC'].values,'ATC')
    mci_TAA = getMeanConfidenceInterval(df['TAA'].values,'TAA')
    mci_QRED = getMeanConfidenceInterval(df['QRED'].values,'QRED')
    mci_QR = getMeanConfidenceInterval(df['QR'].values,'QR')
    mci_PUE = getMeanConfidenceInterval(df['PUE'].values,'PUE')
    mci_DCie = getMeanConfidenceInterval(df['DCie'].values,'DCie')
    mci_cost = getMeanConfidenceInterval(df['cost'].values,'cost')
    mci_cost = getMeanConfidenceInterval(df['MTTF_IC'].values,'MTTF_IC')
    mci_cost = getMeanConfidenceInterval(df['A_TC'].values,'A_TC')
    mci_cost = getMeanConfidenceInterval(df['Q_DIT'].values,'Q_DIT')
    mci_cost = getMeanConfidenceInterval(df['TPF'].values,'TPF')
    mci_cost = getMeanConfidenceInterval(df['AIRFLOW'].values,'AIRFLOW')
    mci_cost = getMeanConfidenceInterval(df['TAAF'].values,'TAAF')
    mci_cost = getMeanConfidenceInterval(df['DeltaT_de'].values,'DeltaT_de')
    mci_cost = getMeanConfidenceInterval(df['QD'].values,'QD')

def plotHist(x, df, n, title, filename):
    plt.hist(df[x], n)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()

def stardard_desviation(df):
    dname=["TF","MTT","MTTF_R","MTTF_EM","MTTF_C","MTTF_TDDB","MTTF_SM","MTTFF_TC","A","AEM","AC","ATDDB","ASM","ATC","TAA","QRED","QR","PUE","Dcie","Cost","MTTF_IC","A_TC","Q_DIT"]
    dvalues = [np.std(df['TF'].values),np.std(df['MTT'].values),np.std(df['MTTF_R'].values),np.std(df['MTTF_EM'].values),np.std(df['MTTF_C'].values), np.std(df['MTTF_TDDB'].values), np.std(df['MTTF_SM'].values),np.std(df['MTTFF_TC'].values),np.std(df['A'].values),np.std(df['AEM'].values),np.std(df['AC'].values),np.std(df['ATDDB'].values),np.std(df['ASM'].values),np.std(df['ATC'].values),np.std(df['TAA'].values),np.std(df['QRED'].values),np.std(df['QR'].values),np.std(df['PUE'].values),np.std(df['DCie'].values),np.std(df['cost'].values),np.std(df['MTTF_IC'].values), np.std(df['A_TC'].values),np.std(df['Q_DIT'].values)]
    dvalues = [np.std(df['TF'].values),np.std(df['MTT'].values),np.std(df['MTTF_R'].values),np.std(df['MTTF_EM'].values),np.std(df['MTTF_C'].values), np.std(df['MTTF_TDDB'].values), np.std(df['MTTF_SM'].values),np.std(df['MTTFF_TC'].values),np.std(df['A'].values),np.std(df['AEM'].values),np.std(df['AC'].values),np.std(df['ATDDB'].values),np.std(df['ASM'].values),np.std(df['ATC'].values),np.std(df['TAA'].values),np.std(df['QRED'].values),np.std(df['QR'].values),np.std(df['PUE'].values),np.std(df['DCie'].values),np.std(df['cost'].values),np.std(df['MTTF_IC'].values), np.std(df['A_TC'].values),np.std(df['Q_DIT'].values)]
    midata = {'Name': dname, 'Standard Desviation': dvalues}
    data_Frame = pd.DataFrame(data = midata)
    fig, ax = plt.subplots(1, 1)
    ptable(ax, data_Frame, loc='upper center', colWidths=[0.2, 0.2, 0.2])
    plt.savefig('stardard_desviation_table.png')
    print(data_Frame)
    plt.clf()

def plotCorrelation(df, x, y, area, colors, alpha, title, filename):
    plt.scatter(df[x],df[y], s = area, c = colors, alpha = alpha)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()

def plotDataset(x_, y_, ds, xlabel, ylabel, title, filename,formatter = '%d'):
    ax = sns.lineplot(x = x_, y = y_, data = ds)
    ax.yaxis.set_major_formatter(FormatStrFormatter(formatter))
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()

def plot3DDataset(x,y,z, ds, xlabel, ylabel, zlabel, title, filename):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(ds[x], ds[y], ds[z], cmap=plt.cm.viridis, linewidth=0.2,antialiased=True)
    ax.set_xlabel(xlabel, fontsize=15, rotation=150)
    ax.set_ylabel(ylabel, fontsize=15, rotation=150)
    ax.set_zlabel(zlabel, fontsize = 15, rotation = 0)
    ax.view_init(30,150)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()

def plot3DScatterPDataset(x,y,z, ds,xlabel, ylabel, zlabel, title, filename):
    fig = plt.figure(figsize=(11.7,8.27))
    ax = Axes3D(fig)
    ax.scatter(ds[x],ds[y],ds[z], cmap=plt.cm.viridis,marker='o')
    ax.set_xlabel(xlabel, fontsize=15, rotation=150)
    ax.set_ylabel(ylabel, fontsize=15, rotation=150)
    ax.set_zlabel(zlabel, fontsize = 15, rotation = 90)
    ax.view_init(30,150)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()

def plotDist(x,ds, title, filepath):
    try:
        ax = sns.distplot(ds[x])
        ax.set_title(title)
        plt.savefig(filepath)
        plt.clf
    except Exception as e:
        print('cannot plot distribution for variable : ', x)
        print('err: ', e)    
    
def plotGroup(lsMetrics, xMeasure, xlabel, ylabel, ds, title, filename):
    x = ds[xMeasure].values
    plt.xlabel = xlabel
    plt.ylabel = ylabel
    for e in lsMetrics:
        y = ds[e[0]].values
        plt.plot(x, y, label = e[1],marker=e[2])
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.clf()

def plotKDE(x, y, xlabel, ylabel, ds, title, filename):
    #plot a Kernel Density Estimation of "X" and "Y" variables
    sns.kdeplot(ds[x], ds[y], shade = True, legend = True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()

def plotTwoKDE(x, y1, y2, xlabel, ylabel, y1_legend, y2_legend, m1, m2, ds, title, filename):
    #m1, m2: values to multiply y1,y2. For instance: availabilities metrics should be showed as percent (x%, y%, etc..)
    #plot the Kernel Density Estimation of (x,y1) and (x,y2)
    sns.kdeplot(ds[x], ds[y1]*m1, cmap="Reds", shade=True, shade_lowest=False)
    sns.kdeplot(ds[x], ds[y2]*m2, cmap="Blues", shade=True, shade_lowest=False)

    r = sns.color_palette("Reds")[2]
    b = sns.color_palette("Blues")[2]

    red_patch = mpatches.Patch(color=r, label= y1_legend)
    blue_patch = mpatches.Patch(color=b, label= y2_legend)
    print('filename: ', filename)
    plt.legend(handles=[red_patch,blue_patch])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()

def saveTwoKDE(df):
    print('plotting two kdes')
    if not os.path.exists('kdes'):
        os.makedirs('kdes')

    sample = df.sample(SAMPLESIZE).sort_values('TIMESTAMP')

    #(x, y1, y2, xlabel, ylabel, y1_legend, y2_legend, m1=1, m2=1, ds, title, filename)
    lsV = []
    lsV.append(['ROOM_TEMP','A','AEM','Temperature(°C)', 'Availability (%)', 'Availability','A. Electromigration',100,100,'Kernel Density Estimation','kdes/1_a_aem'])
    lsV.append(['ROOM_TEMP','A','AC','Temperature(°C)', 'Availability (%)', 'Availability','A. Corrosion',100,100,'Kernel Density Estimation','kdes/2_a_ac'])
    lsV.append(['ROOM_TEMP','A','ATDDB','Temperature(°C)', 'Availability (%)', 'Availability','A. Time Depending Dielectric Breakdown',100,100,'Kernel Density Estimation','kdes/3_a_tddb'])
    lsV.append(['ROOM_TEMP','A','ASM','Temperature(°C)', 'Availability (%)', 'Availability','A. Stress Migration',100,100,'Kernel Density Estimation','kdes/4_a_asm'])
    lsV.append(['ROOM_TEMP','A','ATC','Temperature(°C)', 'Availability (%)', 'Availability','A. Thermal Cycling',100,100,'Kernel Density Estimation','kdes/5_a_atc'])
    
    for v in lsV:
        print('xlabel: ', v[4])
        plotTwoKDE(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], sample, v[9], v[10])
    print('saved two kdes \n')

def saveFigures(df):
    print('plotting figures')
    if not os.path.exists('plots'):
        os.makedirs('plots')

    if not os.path.exists('scatterplots'):
        os.makedirs('scatterplots')

    Nc = 8
    Apf = 2.3
    APs = Nc * Apf
    Cu = df['vnf cpu usage']
    ACu = (APs * Cu) / 100 #'is it 100 or 1000'
    f = APs * df['vnf cpu usage']
    
    # assign f vector to dataframe
    df = df.assign(F = f)
    #df['MTT'] = df['MTT'] + 273
    # using the same sample 
    sample = df.sample(SAMPLESIZE).sort_values('TIMESTAMP')
    
    ls = []
    #(formatter, x_,y_,ds, xlabel, ylabel, title, filename)
    #plots vs time
    
    ls.append(['TIMESTAMP', 'F', 'Time(s)', 'Frequency (MHz)','Frequency vs Time','plots/1_frecuency_time'])
    ls.append(['%.3f','TIMESTAMP', 'TF', 'Time(s)', 'Temperature (°C)','Temperature VS Time','plots/2_temperature_vs_time'])
    ls.append(['TIMESTAMP', 'MTT', 'Time(s)', 'Mean Time To Failure','Mean Time To Failure vs Time','plots/3_Mean_time_to_failure'])
    ls.append(['TIMESTAMP', 'MTTF_R', 'Time(s)', 'Mean Time To Failure due to changes on processors temperature','Mean Time To Failure due to changes on processors temperature vs Time','plots/4_mean_time_to_failure_due_to_changes_on_processors_time'])
    ls.append(['TIMESTAMP', 'MTTF_EM', 'Time(s)', 'Mean Time To Failure due to changes on the current density','Mean Time To Failure due to changes on the current density vs Time','plots/5_mean_time_to_failure_due_to_changes_on_the_current_density_time'])
    ls.append(['TIMESTAMP', 'MTTF_C', 'Time(s)', 'Mean Time To Failure due to relativite humidity','Mean Time To Failure due to relativite humidity vs Time','plots/6_Mean_time_to_failure_due_to_relativite'])
    ls.append(['TIMESTAMP', 'MTTF_TDDB', 'Time(s)', 'Mean Time To Failure on externally applied electric field','Mean Time To Failure on externally applied electric field vs Time','plots/7_mean time to failure due to changes on externally applied electric field'])
    ls.append(['%.1f','TIMESTAMP', 'MTTF_SM', 'Time(s)', 'Mean Time To Failure due to changes on thermal loads','Mean Time To Failure due to changes on thermal loads vs Time','plots/8_mean_time_to_failure_due_to_changes_on_thermal_loads'])
    ls.append(['%.1f','TIMESTAMP', 'MTTFF_TC', 'Time(s)', 'TC METRIC','TC METRIC vs Time','plots/9_tcmetric_time'])
    ls.append(['%.4f','TIMESTAMP', 'A', 'Time(s)', 'Availability','Availability','plots/10_availability_vs_time'])
    ls.append(['%.4f','TIMESTAMP', 'AEM', 'Time(s)', 'Availability Due To Electromigration','Availability Due To Electromigration vs Time','plots/11_Availability Due To Electromigration vs time'])
    ls.append(['%.4f','TIMESTAMP', 'AC', 'Time(s)', 'Availability Due To Corrosion','Availability Due To Corrosion vs Time','plots/12_Availability Due To corrosion vs time'])
    ls.append(['%.4f','TIMESTAMP', 'ATDDB', 'Time(s)', 'Availability Due Time Depending Dielectric Breakdown','Availability Due Time Depending Dielectric Breakdown vs Time','plots/13_ Availability Due Time Depending Dielectric Breakdown vs time'])
    ls.append(['%.4f','TIMESTAMP', 'ASM', 'Time(s)', 'Availability Due To Stress Migration','Availability Due To Stress Migration vs Time','plots/14 Availability Due To Stress Migration vs time'])
    ls.append(['%.4f','TIMESTAMP', 'ATC', 'Time(s)', 'Availability Due To Thermal Cycling','Availability Due To Thermal Cycling vs Time','plots/15 Availability Due To Thermal Cycling vs time'])
    ls.append(['%.2f','TIMESTAMP', 'TAA', 'Time(s)', 'Availability Due To External Temperature','Availability Due To External Temperature vs Time','plots/16 Availability Due To External Temperature vs time'])
    ls.append(['%.2f','TIMESTAMP', 'QRED', 'Time(s)', 'Thermal Load Released','Thermal Load Released vs Time','plots/17 Thermal Load Released vs time']) 
    ls.append(['%.2f','TIMESTAMP', 'QR', 'Time(s)', 'Power Required','Power Required vs Time','plots/18 Power Required vs time'])
    ls.append(['%.3f','TIMESTAMP', 'PUE', 'Time(s)', 'Power Usage Efficiency','Power Usage Efficiency vs Time','plots/19 Power Usage Efficiency vs time'])
    ls.append(['%.2f','TIMESTAMP', 'DCie', 'Time(s)', 'DCie','DCie vs Time','plots/20 DCie vs time'])
    ls.append(['%.4f','TIMESTAMP', 'cost', 'Time(s)', 'Cost','Cost vs Time','plots/21 cost vs time'])
    ls.append(['%.2f','TIMESTAMP', 'MTTF_IC', 'Time(s)', 'Unified Reliability','Unified Reliability','plots/22 Unified Reliability'])
    ls.append(['%.4f','TIMESTAMP', 'A_TC', 'Time(s)', 'Unified Availability','Unified Availability vs Time','plots/23 Unified Availability vs time'])
    ls.append(['%.3f','TIMESTAMP', 'Q_DIT', 'Time(s)', 'Amount Energy Dissipated','Amount Energy Dissipated vs Time','plots/24 Amount Energy Dissipated vs time'])
    ls.append(['%.2f','TIMESTAMP', 'TPF', 'Time(s)', 'External Temperature Impact','External Temperature Impact vs Time','plots/25 External Temperature Impact vs Time'])
    ls.append(['%.4f','TIMESTAMP', 'AIRFLOW', 'Time(s)', 'Required Volume Airflow','Required Volume Airflow vs Time','plots/26 Required Volume Airflow vs time'])
    ls.append(['%.4f','TIMESTAMP', 'TAAF', 'Time(s)', 'Thermal Accelerated Aging','Thermal Accelerated Aging vs Time','plots/27 Thermal Accelerated Aging vs time'])
    ls.append(['%.3f','TIMESTAMP', 'DeltaT_de', 'Time(s)', 'Temperature rise due to the dissipation of energy','Temperature rise due to the dissipation of energy vs Time','plots/24 Temperature rise due to the dissipation of energy vs time'])
    ls.append(['%.3f','TIMESTAMP', 'QD', 'Time(s)', 'Energy Demanded','Energy Demanded vs Time','plots/24 Energy Demanded vs time'])
    #ls.append(['TF', 'F', 'Temperature(°C)', 'Frequency (MHz)','Frequency vs Temperature','plots/25_frecuency_vs_temperature'])
    #ls.append(['TF', 'TF', 'Temperature(°C)', 'Temperature (°C)','Temperature VS Temperature','plots/26_temperature_vs_temperature'])
    ls.append(['TF', 'MTT', 'Temperature(°C)', 'Mean Temperature To Failure','Mean Temperature To Failure vs Temperature','plots/27_Mean_time_to_failure vs temperature'])
    ls.append(['TF', 'MTTF_R', 'Temperature(°C)', 'Mean Temperature To Failure due to changes on processors temperature','Mean Temperature To Failure due to changes on processors temperature vs Temperature','plots/28_mean_time_to_failure_due_to_changes_on_processors_temperature'])
    ls.append(['TF', 'MTTF_EM', 'Temperature(°C)', 'Mean Temperature To Failure due to changes on the current density','Mean Temperature To Failure due to changes on the current density vs Temperature','plots/29_mean_time_to_failure_due_to_changes_on_the_current_density_temperature'])
    ls.append(['TF', 'MTTF_C', 'Temperature(°C)', 'Mean Temperature To Failure due to relativite humidity','Mean Temperature To Failure due to relativite humidity vs Temperature','plots/30_Mean_time_to_failure_due_to_relativite vs temperature'])
    ls.append(['TF', 'MTTF_TDDB', 'Temperature(°C)', 'Mean Temperature To Failure on externally applied electric field','Mean Temperature To Failure on externally applied electric field vs Temperature','plots/31_mean temperature to failure due to changes on externally applied electric field vs temperature'])
    ls.append(['TF', 'MTTF_SM', 'Temperature(°C)', 'Mean Temperature To Failure due to changes on thermal loads','Mean Temperature To Failure due to changes on thermal loads vs Temperature','plots/32_mean_time_to_failure_due_to_changes_on_thermal_loads vs temperature'])
    ls.append(['TF', 'MTTFF_TC', 'Temperature(°C)', 'TC METRIC','TC METRIC vs Temperature','plots/33_tcmetric_vs'])
    ls.append(['TF', 'A', 'Temperature(°C)', 'Availability','Availability','plots/34_availability_vs_time vs'])
    ls.append(['TF', 'AEM', 'Temperature(°C)', 'Availability Due To Electromigration','Availability Due To Electromigration vs Temperature','plots/35_Availability Due To Electromigration vs temperature'])
    ls.append(['TF', 'AC', 'Temperature(°C)', 'Availability Due To Corrosion','Availability Due To Corrosion vs Temperature','plots/36_Availability Due To corrosion vs temperature'])
    ls.append(['TF', 'ATDDB', 'Temperature(°C)', 'Availability Due Temperature Depending Dielectric Breakdown','Availability Due Temperature Depending Dielectric Breakdown vs Temperature','plots/37_ Availability Due Temperature Depending Dielectric Breakdown vs temperature'])
    ls.append(['TF', 'ASM', 'Temperature(°C)', 'Availability Due To Stress Migration','Availability Due To Stress Migration vs Temperature','plots/38 Availability Due To Stress Migration vs temperature'])
    ls.append(['TF', 'ATC', 'Temperature(°C)', 'Availability Due To Thermal Cycling','Availability Due To Thermal Cycling vs Temperature','plots/39 Availability Due To Thermal Cycling vs temperature'])
    ls.append(['TF', 'TAA', 'Temperature(°C)', 'Availability Due To External Temperature','Availability Due To External Temperature vs Temperature','plots/40 Availability Due To External Temperature vs temperature'])
    ls.append(['TF', 'QRED', 'Temperature(°C)', 'Thermal Load Released','Thermal Load Released vs Temperature','plots/41 Thermal Load Released vs temperature'])    
    ls.append(['TF', 'QR', 'Temperature(°C)', 'Power Required','Power Required vs Temperature','plots/42 Power Required vs temperature'])
    ls.append(['TF', 'PUE', 'Temperature(°C)', 'Power Usage Efficiency','Power Usage Efficiency vs Temperature','plots/43 Power Usage Efficiency vs temperature'])
    ls.append(['TF', 'DCie', 'Temperature(°C)', 'DCie','DCie vs Temperature','plots/44 DCie vs temperature'])
    ls.append(['TF', 'cost', 'Temperature(°C)', 'Cost','Cost vs Temperature','plots/45 cost vs temperature'])
    ls.append(['TF', 'MTTF_IC', 'Temperature(°C)', 'Unified Reliability','Unified Reliability','plots/46 Unified Reliability'])
    ls.append(['TF', 'A_TC', 'Temperature(°C)', 'Unified Availability','Unified Availability vs Temperature','plots/47 Unified Availability vs temperature'])
    ls.append(['TF', 'Q_DIT', 'Temperature(°C)', 'Amount Energy Dissipated','Amount Energy Dissipated vs Temperature','plots/48 Amount Energy Dissipated vs temperature'])
    ls.append(['TF', 'TPF', 'Temperature(°C)', 'External Temperature Impact','External Temperature Impact vs Temperature','plots/25 External Temperature Impact vs Temperature'])
    ls.append(['TF', 'AIRFLOW', 'Temperature(°C)', 'Required Volume Airflow','Required Volume Airflow vs Temperature','plots/26 Required Volume Airflow vs Temperature'])
    ls.append(['TF', 'TAAF', 'Temperature(°C)', 'Thermal Accelerated Aging','Thermal Accelerated Aging vs Temperature','plots/27 Thermal Accelerated Aging vs Temperature'])
    ls.append(['TF', 'DeltaT_de', 'Temperature(°C)', 'Temperature rise due to the dissipation of energy','Temperature rise due to the dissipation of energy vs Temperature','plots/28 Temperature rise due to the dissipation of energy vs Temperature'])
    ls.append(['TF', 'QD', 'Temperature(°C)', 'Energy Demanded','Energy Demanded vs Temperature','plots/29 Energy Demanded vs Temperature'])



    for e in ls:
        
        print('plotting ',  e[5])
        if(len(e) == 6):
            #plot without formatter
            plotDataset(e[0], e[1], sample, e[2], e[3], e[4], e[5])
        else:
            plotDataset(e[1], e[2], sample, e[3], e[4], e[5], e[6], e[0])
    
    ls3d = []
    ls3d.append(['TIMESTAMP','TF','MTTFF_TC','Time(s)', 'Temperature (°C)','MTTFF_TC' ,'Time,Temperature vs MTTFF_TC','scatterplots/scatter_3d__1_MTTFF_TC'])
    ls3d.append(['TIMESTAMP','TF','A','Time(s)', 'Temperature (°C)','Availability' ,'Time,Temperature vs Availability','scatterplots/scatter_3d__2_Availability'])
    ls3d.append(['TIMESTAMP','TF','AC','Time(s)', 'Temperature (°C)','Availability due to corrosion' ,'Time,Temperature vs Availability due to corrosion','scatterplots/scatter_3d__3_Availability due to corrosion'])
    ls3d.append(['TIMESTAMP','TF','ATDDB','Time(s)', 'Temperature (°C)','Availability Due Temperature Depending Dielectric Breakdown' ,'Time,Temperature vs Availability Due Temperature Depending Dielectric Breakdown','scatterplots/scatter_3d__4_Availability Due Temperature Depending Dielectric Breakdown'])
    ls3d.append(['TIMESTAMP','TF','ASM','Time(s)', 'Temperature (°C)','Availability Due To Stress Migration' ,'Time,Temperature vs Availability Due To Stress Migration','scatterplots/scatter_3d__5_Availability Due To Stress Migration'])
    ls3d.append(['TIMESTAMP','TF','ATC','Time(s)', 'Temperature (°C)','Availability Due To Thermal Cycling' ,'Time,Temperature vs Availability Due To Thermal Cycling','scatterplots/scatter_3d__6_Availability Due To Thermal Cycling'])
    ls3d.append(['TIMESTAMP','TF','TAA','Time(s)', 'Temperature (°C)','Availability Due To External Temperature' ,'Time,Temperature vs Availability Due To Thermal Cycling','scatterplots/scatter_3d__7_Availability Due To External Temperature'])
    ls3d.append(['TIMESTAMP','TF','QRED','Time(s)', 'Temperature (°C)','Thermal Load Released' ,'Time,Temperature vs Thermal Load Released','scatterplots/scatter_3d__8_Thermal Load Released'])
    ls3d.append(['TIMESTAMP','TF','QR','Time(s)', 'Temperature (°C)','Power Required' ,'Time,Temperature vs Power Required','scatterplots/scatter_3d__9_Power Required'])
    ls3d.append(['TIMESTAMP','TF','PUE','Time(s)', 'Temperature (°C)','Power Usage Efficiency' ,'Time,Temperature vs Power Usage Efficiency','scatterplots/scatter_3d__10_Power Usage Efficiency'])
    ls3d.append(['TIMESTAMP','TF','DCie','Time(s)', 'Temperature (°C)','DCie' ,'Time,Temperature vs DCie','scatterplots/scatter_3d__11_DCie'])
    ls3d.append(['TIMESTAMP','TF','cost','Time(s)', 'Temperature (°C)','Cost' ,'Time,Temperature vs Cost','scatterplots/scatter_3d__12_Cost'])
    ls3d.append(['TIMESTAMP','TF','MTTF_IC','Time(s)', 'Temperature (°C)','Unified Reliability' ,'Time,Temperature vs DCie','scatterplots/scatter_3d__12_Unified'])
    ls3d.append(['TIMESTAMP','TF','A_TC','Time(s)', 'Temperature (°C)','Unified Availability' ,'Time,Temperature vs Unified Availability','scatterplots/scatter_3d__13_Unified Availability'])
    ls3d.append(['TIMESTAMP','TF','Q_DIT','Time(s)', 'Temperature (°C)','Amount Energy Dissipated' ,'Time,Temperature vs Amount Energy Dissipated','scatterplots/scatter_3d__14_Amount Energy Dissipated'])
    
    for e in ls3d:
        print('plotting 3D scatter plot ', e[7])
        plot3DScatterPDataset(e[0],e[1],e[2],sample,e[3],e[4],e[5],e[6],e[7])

    #TAA, QRED
    lsTAA_QRED = []
    lsTAA_QRED.append(['TAA','TAA',None])
    lsTAA_QRED.append(['QRED','QRED',None])         
    plotGroup(lsTAA_QRED,'TIMESTAMP','Time(s)', 'TAA, QRED',sample,'Thermal Load Released, Actual Ambient Temperature vs Time(s)', 'plots/qred_taa_time')
    print('saved figures \n')
    
def saveDistributionPlots(df):
    if not os.path.exists('hist_dist_plots'):
        os.makedirs('hist_dist_plots')
    n= 50
    print('plotting distribution plots (using hist)')
    plotHist('A', df, n, 'Availability Distribution', 'hist_dist_plots/1_Availability_distribution')
    plotHist('AEM', df, n, 'Availability Due to Electromigration Distribution', 'hist_dist_plots/2_Availability_electromigration_distribution')
    plotHist('AC', df, n, 'Availability Corrosion Distribution', 'hist_dist_plots/3_availability_corrosion')
    plotHist('ATDDB', df, n, 'Availability Due Time Dielectric Distribution', 'hist_dist_plots/4_availability_due_time_depende_dielectric')
    plotHist('ASM', df, n, 'Availability Stress Migration Distribution', 'hist_dist_plots/5_availability_stress_migration')
    plotHist('ATC', df, n, 'Availability Thermal Cycling Distribution', 'hist_dist_plots/6_availability_thermal_cycling')
    plotHist('TAA', df, n, 'Actual Ambient Temperature Distribution', 'hist_dist_plots/7_actual_ambient_temperature')
    plotHist('QRED', df, n, 'Thermal Load Released Distribution', 'hist_dist_plots/8_thermal_load_released')
    plotHist('QR', df, n, 'Power Required Distribution', 'hist_dist_plots/9_QR')
    plotHist('MTT', df, n, 'MTTF Distribution', 'hist_dist_plots/10_Mean Time To Failure')
    plotHist('PUE', df, n, 'Power Ussage Efficiency Distribution', 'hist_dist_plots/11_Power_Ussage_Efficiency')
    plotHist('DCie', df, n, 'DataCenter Infrastructure Efficiency Distribution', 'hist_dist_plots/12_DataCenter_Infrastructure_Efficiency')
    plotHist('cost', df, n, 'Cost Distribution', 'hist_dist_plots/13_Cost')
    plotHist('EXTERNAL_TEMP', df, n, 'External Temperature Distribution', 'hist_dist_plots/14_External_Temperature')
    plotHist('ROOM_TEMP', df, n, 'Room Temperature Distribution', 'hist_dist_plots/15_Room_Temperature')
    plotHist('AIRFLOW', df, n, 'Required Volume Airflow Distribution', 'hist_dist_plots/16_Airflow')
    plotHist('QD', df, n, 'Energy Demanded Distribution', 'hist_dist_plots/17_Energy_Demanded')
    plotHist('DeltaT_de', df, n, 'Temperature Rise Due To Dissipation Energy Distribution', 'hist_dist_plots/18_Rise_Dissipation_Energy')



    print('saved distribution plots')

def saveCorrelationPlots(dataframe):
    if not os.path.exists('correlation_plots'):
        os.makedirs('correlation_plots')
    colors = np.random.rand(SAMPLESIZE)
    area = (30 * np.random.rand(SAMPLESIZE))**2
    alpha = 0.5
    sample = df.sample(SAMPLESIZE)

    print('plotting distribution plots')
    plotCorrelation(sample, 'AC','A',area, colors, alpha,'Correlation between AC and A','correlation_plots/correlation_ac_a')
    plotCorrelation(sample, 'AEM','QRED',area, colors, alpha,'Correlation between AEM and QRED','correlation_plots/correlation_aem_qred')
    plotCorrelation(sample, 'MTTFF_TC','A',area, colors, alpha,'Correlation between MTTFF_TC and A','correlation_plots/correlation_mttff_tc_a')
    plotCorrelation(sample, 'MTT','vnf cpu usage',area, colors, alpha,'Correlation between MTT and VNF CPU Usage','correlation_plots/correlation_mtt_cpu_usage')
    plotCorrelation(sample, 'AC','A',area, colors, alpha,'Correlation between AC and A','correlation_plots/correlation_ac_a')
    #plotCorrelation(sample, 'AC','A',area, colors, alpha,'Correlation between AC and A','correlation_plots/correlation_ac_a')
    plotCorrelation(sample, 'QR','QRED',area, colors, alpha,'Correlation between QR and QRED','correlation_plots/correlation_qr_qred')
    #check the plot below
    #plotCorrelation(sample, 'A','AC',area, colors, alpha,'Correlation between AC and A','correlation_plots/correlation_ac_a')
    plotCorrelation(sample, 'TAA','AIRFLOW',area, colors, alpha,'Correlation between TAA and AIRFLOW','correlation_plots/correlation_TAA_AIRFLOW')


    #np.corrcoef(dataframe['MTT'], dataframe['A'])
    print('saved correlation plots')

def saveChartAvailability(df):
    #plotGroup(lsMetrics, xMeasure, xlabel, ylabel, ds, title, filename)
    #a metric is a list with 3-values [x-column, y-label, a marker]
    print('plotting availabilities charts')
    sample = df[(df['AEM'] > 0.95) & (df['AC'] > 0.95) & (df['ATDDB'] > 0.95) & (df['ASM'] > 0.95) & (df['ATC']> 0.95)].sample(1000).sort_values(by=['TF'])

    lsAvailabilites = []
    lsAvailabilites.append(['AEM','AEM','o'])
    lsAvailabilites.append(['AC','AC','v'])
    lsAvailabilites.append(['ATDDB','ATDDB','1'])
    lsAvailabilites.append(['ASM','ASM','2'])
    lsAvailabilites.append(['ATC','ATC','*'])
    plotGroup(lsAvailabilites,'TF','TEMPERATURE (C)', 'AVAILABILITY (%)',sample, "PROCESSOR'S TEMPERATURE VS AVAILABILITY",'plots/GROUP_availabilities_vs_temperature')
    print('saved availabilities charts')

def saveGroupPlot(df):
    #plotGroup(lsMetrics, xMeasure, xlabel, ylabel, ds, title, filename)
    #a metric is a list with 3-values [x-column, y-label, a marker]

    if not os.path.exists('group_metric_plots'):
        os.makedirs('group_metric_plots')
    
    sample = df.sample(SAMPLESIZE).sort_values('TF')

    print('plotting group metrics')

    lsMetricsAvailabilities = []
    lsMetricsAvailabilities.append(['A','Availability',None])
    lsMetricsAvailabilities.append(['AEM','A. due to electromagnetism',None])
    lsMetricsAvailabilities.append(['AC','A. due to corrosion',None])
    lsMetricsAvailabilities.append(['ATDDB','A. due to time-dependent dielectric breakdown',None])
    lsMetricsAvailabilities.append(['ASM','A. due to stress migration',None])
    lsMetricsAvailabilities.append(['ATC','A. due to thermal cycling',None])
    #lsMetricsAvailabilities.append(['MTTF_IC','Unified availability',None])
    lsMetricsAvailabilities.append(['A_TC','Unified availability',None])
    plotGroup(lsMetricsAvailabilities, 'TF','Temperature (°C)', 'AVAILABILITIES', sample,'Availability Evaluation', 'group_metric_plots/1_availability evaluation')


    lsMetricsFailures = []
    lsMetricsFailures.append(['MTTF_R','MTTF based in temperature',None])
    lsMetricsFailures.append(['MTTF_C','Corrosion',None])
    lsMetricsFailures.append(['MTTF_TDDB','Time-Dependent Dielectric Breakdown',None])
    lsMetricsFailures.append(['MTTF_SM','Stress Migration',None])
    lsMetricsFailures.append(['MTTFF_TC','Thermal Cycling',None])
    plotGroup(lsMetricsFailures, 'TF','Temperature (°C)', 'Evaluation Failures', sample,'Evaluation Failures', 'group_metric_plots/2_failure evaluation')

    #plt.plot(x, sample['TAA'], label='Thermal Accelearated Aging')
    #plt.plot(x, sample['TAA'], label='Actual Ambient Temperature')
    #pending 2 plots: Delta_T_DE, A
    #plt.title('Thermal evaluation')
    #plt.legend()
    #plt.savefig('plots/thermal evaluation')
    #plt.clf()

    lsMetricsCooling = []
    lsMetricsCooling.append(['QRED','Thermal Load Released',None])
    lsMetricsCooling.append(['QR','Energy Required',None])
    plotGroup(lsMetricsCooling, 'TF','Temperature (°C)', 'Cooling Evaluation', sample,'Cooling Evaluation', 'group_metric_plots/3_cooling evaluation')

    lsMetricsEnergy = []
    lsMetricsEnergy.append(['PUE','Power Ussage Effectiveness',None])
    lsMetricsEnergy.append(['DCie','DataCenter Infraestructure Efficiency',None])
    lsMetricsEnergy.append(['cost','Cost',None])
    lsMetricsEnergy.append(['Q_DIT','Amount of Energy Dissipated',None])
    plotGroup(lsMetricsEnergy, 'TF','Temperature (°C)', 'Energy Evaluation', sample,'Energy Evaluation', 'group_metric_plots/4_energy evaluation')
    print('saved group metrics')
    
def saveDistPlots(df):
    if not os.path.exists('distribution_plots'):
        os.makedirs('distribution_plots')
    sample = df.sample(SAMPLESIZE)

    print('plotting dist plots')
    plotDist('MTT',sample,'MTTF Distribution','distribution_plots/1_Distribution MTTF')
    plotDist('A',sample,'Availability Distribution','distribution_plots/2_Distribution Availability')
    plotDist('AC',sample,'Availability Corrosion Distribution','distribution_plots/3_Distribution Availability Corrosion')
    plotDist('AEM',sample,'Availability Electromigration Distribution','distribution_plots/4_Distribution Availability Electromigration')
    plotDist('ATDDB',sample,'Availability Time-Dependent Dielectric Breakdown Distribution','distribution_plots/5_Distribution Availability Time-Dependent Dielectric Breakdown')
    plotDist('ASM',sample,'Availability Stress Migration Distribution','distribution_plots/6_Distribution Availability Stress Migration')
    plotDist('ATC',sample,'Availability Thermal Cycling Distribution','distribution_plots/7_Distribution Availability Thermal Cycling')
    plotDist('TAA',sample,'Actual Ambient Temperature Distribution','distribution_plots/8_Distribution Actual Ambient Temperature')
    plotDist('QRED',sample,'Thermal Load Released Distribution','distribution_plots/9_Distribution Thermal Load Released')
    plotDist('PUE',sample,'Power Ussage Effectiveness Distribution','distribution_plots/10_Distribution Power Ussage Effectiveness')
    plotDist('DCie',sample,'DataCenter Infraestructure Efficiency Distribution','distribution_plots/11_Distribution DataCenter Infraestructure Efficiency')
    plotDist('cost',sample,'Cost Distribution','distribution_plots/12_Distribution Cost')
    plotDist('EXTERNAL_TEMP','External Temperature Distribution',sample,'distribution_plots/13_Distribution External Temperature')
    plotDist('ROOM_TEMP',sample,'Room Temperature Distribution','distribution_plots/14_Distribution Room Temperature')
    plotDist('MTTF_IC',sample,'Unified Reliability Distribution','distribution_plots/15_Distribution Unified Reliability')
    plotDist('A_TC',sample,'Availability Thermal Cycling Distribution', 'distribution_plots/16_Distribution Availability Thermal Cycling')
    plotDist('Q_DIT',sample,'Amount Energy Dissipated Distribution','Distribution Amount Energy Dissipated')
    print('saved distribution plots')


df = load_csv()
#print (df.iloc[:,41:63].describe())
#plt.table(df.iloc[:,41:63].describe())
saveTwoKDE(df)
saveFigures(df)
saveGroupPlot(df)
#printConfidenceInterval(df)
#stardard_desviation(df)
saveDistributionPlots(df)
#saveDistPlots(df)
saveChartAvailability(df)
saveCorrelationPlots(df)

