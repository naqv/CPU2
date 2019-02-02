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


def mean_confidence_interval(data,name):
	confidence=0.95
	a = 1.0 * data
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
	inf= m-h
	sup = m+h
	print ("The confidence interval of "+name+"  ["+str(inf)+","+str(sup)+" ]")
	return (m, m-h, m+h)

def print_confidence_interval(df):
	mci_mtt= mean_confidence_interval(df['MTT'].values,"MTT")
	mci_mttfr = mean_confidence_interval(df['MTTF_R'].values,"MTTF_R")
	mci_MTTF_EM= mean_confidence_interval(df['MTTF_EM'].values,"MTTF_EM")
	mci_MTTF_C= mean_confidence_interval(df['MTTF_C'].values,"MTTF_C")
	mci_MTTF_TDDB= mean_confidence_interval(df['MTTF_TDDB'].values,"MTTF_TDDB")
	mci_MTTF_SM= mean_confidence_interval(df['MTTF_SM'].values,"MTTF_SM")
	mci_MTTFF_TC = mean_confidence_interval(df['MTTFF_TC'].values,"MTTFF_TC")
	mci_A= mean_confidence_interval(df['A'].values,"A")
	mci_AEM =mean_confidence_interval(df['AEM'].values,"AEM")
	mci_AC= mean_confidence_interval(df['AC'].values,"AC")
	mci_ATDDB = mean_confidence_interval(df['ATDDB'].values,"ATDDB")
	mci_ASM= mean_confidence_interval(df['ASM'].values,"ASM")
	mci_ATC = mean_confidence_interval(df['ATC'].values,"ATC")
	mci_TAA= mean_confidence_interval(df['TAA'].values,"TAA")
	mci_QRED= mean_confidence_interval(df['QRED'].values,"QRED")
	mci_QR= mean_confidence_interval(df['QR'].values,"QR")
	mci_PUE=mean_confidence_interval(df['PUE'].values,"PUE")
	mci_DCie= mean_confidence_interval(df['DCie'].values,'DCie')
	mci_cost= mean_confidence_interval(df['cost'].values,"cost")
	mci_cost= mean_confidence_interval(df['cost'].values,"cost")

def some_distribution_plots(df):
	if not os.path.exists('distribution_plots'):
		os.makedirs('distribution_plots')
	plt.clf()
	plt.hist(df['A'].values, 50)
	plt.title("Availability Distribution")
	plt.savefig('distribution_plots/Availability_distribution.png')
	plt.clf()
	plt.hist(df['AEM'].values, 50)
	plt.title("Availability Due to Electromigration Distribution")
	plt.savefig('distribution_plots/availability_due_to_Electromigration.png')
	plt.clf()
	plt.hist(df['AEM'].values, 50)
	plt.title("Availability Due to Electromigration Distribution")
	plt.savefig('distribution_plots/availability_due_to_Electromigration.png')
	plt.clf()
	plt.hist(df['AC'].values, 50)
	plt.title("Availability Corrosion Distribution")
	plt.savefig('distribution_plots/availability_corrosion.png')
	plt.clf()
	plt.hist(df['ATDDB'].values, 50)
	plt.title("Availability Due Time Dielectric Distribution")
	plt.savefig('distribution_plots/availability_due_time_depende_dielectric.png')
	plt.clf()
	plt.hist(df['ASM'].values, 50)
	plt.title("Availability Stress Migration Distribution")
	plt.savefig('distribution_plots/availability_stress_migration.png')
	plt.clf()
	plt.hist(df['ATC'].values, 50)
	plt.title("Availability Thermal Cycling Distribution")
	plt.savefig('distribution_plots/availability_thermal_cycling.png')
	plt.clf()
	plt.hist(df['TAA'].values, 50)
	plt.title("Actual Ambient Temperature Distribution")
	plt.savefig('distribution_plots/actual_ambient_temperature.png')
	plt.clf()
	plt.hist(df['QRED'].values, 50)
	plt.title("Thermal Load Released Distribution")
	plt.savefig('distribution_plots/thermal_load_released.png')
	plt.clf()
	plt.hist(df['QR'].values, 50)
	plt.title("Power Required Distribution")
	plt.savefig('distribution_plots/QR.png')
	plt.clf()



def stardard_desviation():
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

def correlation_plots(dataframe):
	if not os.path.exists('correlation_plots'):
		os.makedirs('correlation_plots')
	plt.clf()
	#number one
	colors = np.random.rand(1000)
	area = (30 * np.random.rand(1000))**2
	plt.title("Correlation between AC and A ")
	plt.scatter(dataframe['AC'][:1000], dataframe['A'][:1000],s=area, c=colors, alpha=0.5)
	plt.savefig('correlation_plots/correlation_ac_a.png')
	print('saved correlation plot')
	#np.corrcoef(dataframe['MTT'], dataframe['A'])
	plt.clf()
	#number two 
	plt.title("Correlation between AEM and QRED ")
	plt.scatter(dataframe['AEM'][:1000], dataframe['QRED'][:1000],s=area, c=colors, alpha=0.5)
	plt.savefig('correlation_plots/correlation_aem_qred.png')
	print('saved correlation plot')
	#number three
	plt.clf()
	plt.title("Correlation between MTTFF_TC and A ")
	plt.scatter(dataframe['MTTFF_TC'][:1000], dataframe['A'][:1000],s=area, c=colors, alpha=0.5)
	plt.savefig('correlation_plots/correlation_mttff_tc_a.png')
	print('saved correlation plot')
	plt.clf()
	#number four
	plt.title("Correlation between MTT and Vnf cpu usage ")
	plt.scatter(dataframe['MTT'][:1000], dataframe['vnf cpu usage'][:1000],s=area, c=colors, alpha=0.1)
	plt.savefig('correlation_plots/correlation_mtt_cpu_usage.png')
	print('saved correlation plot')

def plotDataset(x_,y_,ds, xlabel, ylabel, title, filename):
		ax = sns.lineplot(x = x_, y = y_, data = ds)
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
	
def plotFigures(df):
	print('plotting figures')
	if not os.path.exists('plots'):
		os.makedirs('plots')

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
	#(x_,y_,ds, xlabel, ylabel, title, filename)
	#plots vs time
	
	ls.append(['TIMESTAMP', 'F', 'Time(seconds)', 'Frequency (MHz)','Frequency vs Time','plots/1_frecuency_time'])
	ls.append(['TIMESTAMP', 'TF', 'Time(seconds)', 'Temperature (°C)','Temperature VS Time','plots/2_temperature_vs_time'])
	ls.append(['TIMESTAMP', 'MTT', 'Time(seconds)', 'Mean Time To Failure','Mean Time To Failure vs Time','plots/3_Mean_time_to_failure'])
	ls.append(['TIMESTAMP', 'MTTF_R', 'Time(seconds)', 'Mean Time To Failure due to changes on processors temperature','Mean Time To Failure due to changes on processors temperature vs Time','plots/4_mean_time_to_failure_due_to_changes_on_processors_time'])
	ls.append(['TIMESTAMP', 'MTTF_EM', 'Time(seconds)', 'Mean Time To Failure due to changes on the current density','Mean Time To Failure due to changes on the current density vs Time','plots/5_mean_time_to_failure_due_to_changes_on_the_current_density_time'])
	ls.append(['TIMESTAMP', 'MTTF_C', 'Time(seconds)', 'Mean Time To Failure due to relativite humidity','Mean Time To Failure due to relativite humidity vs Time','plots/6_Mean_time_to_failure_due_to_relativite'])
	ls.append(['TIMESTAMP', 'MTTF_TDDB', 'Time(seconds)', 'Mean Time To Failure on externally applied electric field','Mean Time To Failure on externally applied electric field vs Time','plots/7_mean time to failure due to changes on externally applied electric field'])
	ls.append(['TIMESTAMP', 'MTTF_SM', 'Time(seconds)', 'Mean Time To Failure due to changes on thermal loads','Mean Time To Failure due to changes on thermal loads vs Time','plots/8_mean_time_to_failure_due_to_changes_on_thermal_loads'])
	ls.append(['TIMESTAMP', 'MTTFF_TC', 'Time(seconds)', 'TC METRIC','TC METRIC vs Time','plots/9_tcmetric_time'])
	ls.append(['TIMESTAMP', 'A', 'Time(seconds)', 'Availability','Availability','plots/10_availability_vs_time'])
	ls.append(['TIMESTAMP', 'AEM', 'Time(seconds)', 'Availability Due To Electromigration','Availability Due To Electromigration vs Time','plots/11_Availability Due To Electromigration vs time'])
	ls.append(['TIMESTAMP', 'AC', 'Time(seconds)', 'Availability Due To Corrosion','Availability Due To Corrosion vs Time','plots/12_Availability Due To corrosion vs time'])
	ls.append(['TIMESTAMP', 'ATDDB', 'Time(seconds)', 'Availability Due Time Depending Dielectric Breakdown','Availability Due Time Depending Dielectric Breakdown vs Time','plots/13_ Availability Due Time Depending Dielectric Breakdown vs time'])
	ls.append(['TIMESTAMP', 'ASM', 'Time(seconds)', 'Availability Due To Stress Migration','Availability Due To Stress Migration vs Time','plots/14 Availability Due To Stress Migration vs time'])
	ls.append(['TIMESTAMP', 'ATC', 'Time(seconds)', 'Availability Due To Thermal Cycling','Availability Due To Thermal Cycling vs Time','plots/15 Availability Due To Thermal Cycling vs time'])
	ls.append(['TIMESTAMP', 'TAA', 'Time(seconds)', 'Availability Due To External Temperature','Availability Due To External Temperature vs Time','plots/16 Availability Due To External Temperature vs time'])
	ls.append(['TIMESTAMP', 'QRED', 'Time(seconds)', 'Thermal Load Released','Thermal Load Released vs Time','plots/17 Thermal Load Released vs time'])	
	ls.append(['TIMESTAMP', 'QR', 'Time(seconds)', 'Power Required','Power Required vs Time','plots/18 Power Required vs time'])
	ls.append(['TIMESTAMP', 'PUE', 'Time(seconds)', 'Power Usage Efficiency','Power Usage Efficiency vs Time','plots/19 Power Usage Efficiency vs time'])
	ls.append(['TIMESTAMP', 'DCie', 'Time(seconds)', 'DCie','DCie vs Time','plots/20 DCie vs time'])
	ls.append(['TIMESTAMP', 'cost', 'Time(seconds)', 'Cost','Cost vs Time','plots/21 cost vs time'])
	ls.append(['TIMESTAMP', 'MTTF_IC', 'Time(seconds)', 'Unified Reliability','Unified Reliability','plots/22 Unified Reliability'])
	ls.append(['TIMESTAMP', 'A_TC', 'Time(seconds)', 'Unified Availability','Unified Availability vs Time','plots/23 Unified Availability vs time'])
	ls.append(['TIMESTAMP', 'Q_DIT', 'Time(seconds)', 'Amount Energy Dissipated','Amount Energy Dissipated vs Time','plots/24 Amount Energy Dissipated vs time'])

	ls.append(['TF', 'F', 'Temperature(°C)', 'Frequency (MHz)','Frequency vs Temperature','plots/25_frecuency_vs_temperature'])
	ls.append(['TF', 'TF', 'Temperature(°C)', 'Temperature (°C)','Temperature VS Temperature','plots/26_temperature_vs_temperature'])
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

	
	for e in ls:
		#(x_,y_,ds, xlabel, ylabel, title, filename)
		print('plotting ',  e[5])
		plotDataset(e[0], e[1], sample, e[2], e[3], e[4], e[5])
	
	ls3d = []
	ls3d.append(['TIMESTAMP','TF','MTTFF_TC','Time (s)', 'Temperature (°C)','MTTFF_TC' ,'Time,Temperature vs MTTFF_TC','plots/scatter_3d__1_MTTFF_TC'])
	ls3d.append(['TIMESTAMP','TF','A','Time (s)', 'Temperature (°C)','Availability' ,'Time,Temperature vs Availability','plots/scatter_3d__2_Availability'])
	ls3d.append(['TIMESTAMP','TF','AC','Time (s)', 'Temperature (°C)','Availability due to corrosion' ,'Time,Temperature vs Availability due to corrosion','plots/scatter_3d__3_Availability due to corrosion'])
	ls3d.append(['TIMESTAMP','TF','ATDDB','Time (s)', 'Temperature (°C)','Availability Due Temperature Depending Dielectric Breakdown' ,'Time,Temperature vs Availability Due Temperature Depending Dielectric Breakdown','plots/scatter_3d__4_Availability Due Temperature Depending Dielectric Breakdown'])
	ls3d.append(['TIMESTAMP','TF','ASM','Time (s)', 'Temperature (°C)','Availability Due To Stress Migration' ,'Time,Temperature vs Availability Due To Stress Migration','plots/scatter_3d__5_Availability Due To Stress Migration'])
	ls3d.append(['TIMESTAMP','TF','ATC','Time (s)', 'Temperature (°C)','Availability Due To Thermal Cycling' ,'Time,Temperature vs Availability Due To Thermal Cycling','plots/scatter_3d__6_Availability Due To Thermal Cycling'])
	ls3d.append(['TIMESTAMP','TF','TAA','Time (s)', 'Temperature (°C)','Availability Due To External Temperature' ,'Time,Temperature vs Availability Due To Thermal Cycling','plots/scatter_3d__7_Availability Due To External Temperature'])
	ls3d.append(['TIMESTAMP','TF','QRED','Time (s)', 'Temperature (°C)','Thermal Load Released' ,'Time,Temperature vs Thermal Load Released','plots/scatter_3d__8_Thermal Load Released'])
	ls3d.append(['TIMESTAMP','TF','QR','Time (s)', 'Temperature (°C)','Power Required' ,'Time,Temperature vs Power Required','plots/scatter_3d__9_Power Required'])
	ls3d.append(['TIMESTAMP','TF','PUE','Time (s)', 'Temperature (°C)','Power Usage Efficiency' ,'Time,Temperature vs Power Usage Efficiency','plots/scatter_3d__10_Power Usage Efficiency'])
	ls3d.append(['TIMESTAMP','TF','DCie','Time (s)', 'Temperature (°C)','DCie' ,'Time,Temperature vs DCie','plots/scatter_3d__11_DCie'])
	ls3d.append(['TIMESTAMP','TF','cost','Time (s)', 'Temperature (°C)','Cost' ,'Time,Temperature vs Cost','plots/scatter_3d__12_Cost'])
	ls3d.append(['TIMESTAMP','TF','MTTF_IC','Time (s)', 'Temperature (°C)','Unified Reliability' ,'Time,Temperature vs DCie','plots/scatter_3d__12_Unified'])
	ls3d.append(['TIMESTAMP','TF','A_TC','Time (s)', 'Temperature (°C)','Unified Availability' ,'Time,Temperature vs Unified Availability','plots/scatter_3d__13_Unified Availability'])
	ls3d.append(['TIMESTAMP','TF','Q_DIT','Time (s)', 'Temperature (°C)','Amount Energy Dissipated' ,'Time,Temperature vs Amount Energy Dissipated','plots/scatter_3d__14_Amount Energy Dissipated'])
	
	for e in ls3d:
		#plot3DScatterPDataset('TIMESTAMP', 'TF', zlabel, sample, 'Timestamp (s)', 'Temperature (°C)', zlabel, title, filename)
		print('plotting 3D scatter plot ', e[7])
		plot3DScatterPDataset(e[0],e[1],e[2],sample,e[3],e[4],e[5],e[6],e[7])
	
	
def saveChartAvailability(df):
	sample = df[(df['AEM'] > 0.95) & (df['AC'] > 0.95) & (df['ATDDB'] > 0.95) & (df['ASM'] > 0.95) & (df['ATC']> 0.95)].sample(1000).sort_values(by=['TF'])
	
	x = sample['TF'].values
	y1 = sample['AEM'].values
	y2 = sample['AC'].values
	y3 = sample['ATDDB'].values
	y4 = sample['ASM'].values
	y5 = sample['ATC'].values
	plt.xlabel('TEMPERATURE (C)')
	plt.ylabel('AVAILABILITY (%)')
	plt.title("PROCESSOR'S TEMPERATURE VS AVAILABILITY ")
	plt.plot(x,y1,label='AEM',marker='o')
	plt.plot(x,y2,label='AC', marker='v')
	plt.plot(x,y3,label='ATDDB', marker='1')
	plt.plot(x,y4,label='ASM', marker='2')
	plt.plot(x,y5,label='ATC', marker='*')
	plt.legend()
	plt.savefig('plots/availability_vs_temperature')
	print('finished')

def saveGroupPlot(df):
	SAMPLESIZE=1000
	sample = df.sample(SAMPLESIZE)
	x = sample['TF']
	plt.xlabel = 'TEMPERATURE (°C)'
	plt.ylabel = 'AVAILABILITIES'
	plt.plot(x, sample['A'], label='Availability')
	plt.plot(x, sample['AEM'], label='A. due to electromagnetism')
	plt.plot(x, sample['AC'], label='A. due to corrosion')
	plt.plot(x, sample['ATDDB'], label='A. due to time-dependent dielectric breakdown')
	plt.plot(x, sample['ASM'], label='A. due to stress migration')
	plt.plot(x, sample['ATC'], label='A. due to thermal cycling')
	plt.plot(x, sample['MTTF_IC'], label='Unified reliability')
	plt.plot(x, sample['A_TC'], label='Unified availability')
	plt.title('Availability Evaluation')
	plt.legend()
	plt.savefig('plots/availability evaluation group')
	plt.clf()


	plt.plot(x, sample['MTTF_R'], label='MTTF based in temperature')
	plt.plot(x, sample['MTTF_C'], label='Corrosion')
	plt.plot(x, sample['MTTF_TDDB'], label='Time-Dependent Dielectric Breakdown')
	plt.plot(x, sample['MTTF_SM'], label='Stress Migration')
	plt.plot(x, sample['MTTFF_TC'], label='Thermal Cycling')
	plt.title('Evaluation Failures')
	plt.xlabel = 'TEMPERATURE (°C)'
	plt.ylabel = 'Evaluation Failures'
	plt.legend()
	plt.savefig('plots/evaluation failure group')
	plt.clf()

	#plt.plot(x, sample['TAA'], label='Thermal Accelearated Aging')
	#plt.plot(x, sample['TAA'], label='Actual Ambient Temperature')
	#pending 2 plots: Delta_T_DE, A
	#plt.title('Thermal evaluation')
	#plt.legend()
	#plt.savefig('plots/thermal evaluation')
	#plt.clf()

	plt.plot(x, sample['QRED'], label='Thermal Load Released')
	plt.plot(x, sample['QR'], label='Energy Required')
	plt.title('Cooling evaluation')
	plt.xlabel = 'TEMPERATURE (°C)'
	plt.ylabel = 'Cooling Evaluation'
	plt.legend()
	plt.savefig('plots/cooling evaluation')
	plt.clf()

	plt.plot(x, sample['PUE'], label='Power Ussage Effectiveness')
	plt.plot(x, sample['DCie'], label='DataCenter Infraestructure Efficiency')
	plt.plot(x, sample['cost'], label='Cost')
	plt.plot(x, sample['Q_DIT'], label='Amount of Energy Dissipated')
	plt.xlabel = 'TEMPERATURE (°C)'
	plt.ylabel = 'Energy Evaluation'
	#pending 1 plots: Q_D
	plt.title('Energy evaluation')
	plt.legend()
	plt.savefig('plots/energy evaluation')
	plt.clf()

	
def saveDistPlots(df):
	ax=sns.distplot(df['MTT'])
	ax.set_title("Distribution MTT")
	plt.savefig('distribution_MTT.png')

	ax=sns.distplot(df['A'])
	ax.set_title("Distribution Availability")
	plt.savefig('distribution_A.png')

	ax= sns.distplot(df['AC'])
	ax.set_title("Distribution Availability")
	plt.savefig('distribution_AEM.png')

	ax=sns.distplot(df['AEM'])
	ax.set_title("Distribution Availability")
	plt.savefig('distribution_AEM.png')

	ax=sns.distplot(df['ATDDB'])
	ax.set_title("Distribution Availability")
	plt.savefig('distribution_AEM.png')

	ax=sns.distplot(df['ASM'])
	ax.set_title("Distribution Availability")
	plt.savefig('distribution_AS.png')

	ax=sns.distplot(df['ASM'])
	ax.set_title("Distribution Availability")
	plt.savefig('distribution_ASM.png')

	ax=sns.distplot(df['ATC'])
	ax.set_title("Distribution Availability")
	plt.savefig('distribution_ATC.png')

	ax=sns.distplot(df['TAA'])
	ax.set_title("Distribution Availability")
	plt.savefig('distribution_TAA.png')

	ax=sns.distplot(df['QRED'])
	ax.set_title("Distribution Availability")
	plt.savefig('distribution_QRED.png')


	ax=sns.distplot(df['PUE'])
	ax.set_title("Distribution Availability")
	plt.savefig('distribution_PUE.png')

	ax=sns.distplot(df['DCie'])
	ax.set_title("Distribution Availability")
	plt.savefig('distribution_DCie.png')

	ax=sns.distplot(df['cost'])
	ax.set_title("Distribution Availability")
	plt.savefig('distribution_cost.png')


	ax=sns.distplot(df['EXTERNAL_TEMP'])
	ax.set_title("Distribution Availability")
	plt.savefig('distribution_EXTERNAL_TEMP.png')


	ax=sns.distplot(df['ROOM_TEMP'])
	ax.set_title("Distribution Availability")
	plt.savefig('distribution_ROOM_TEMP.png')

	ax=sns.distplot(df['MTTF_IC'])
	ax.set_title("Distribution Availability")
	plt.savefig('distribution_MTTF_IC.png')

	ax=sns.distplot(df['A_TC'])
	ax.set_title("Distribution Availability")
	plt.savefig('distribution_A_TC.png')

	ax=sns.distplot(df['Q_DIT'])
	ax.set_title("Distribution Availability")
	plt.savefig('distribution_Q_DIT.png')


df = load_csv()
#print (df.iloc[:,41:63].describe())
#plt.table(df.iloc[:,41:63].describe())
plotFigures(df)
#saveGroupPlot(df)
#print_confidence_interval(df)
#stardard_desviation()
#some_distribution_plots(df)

#saveDistPlots(df)
#saveChartAvailability(df)
#correlation_plots(df)