
####################### LIBRARIES #########################################
import pandas as pd
import math
import collections
import multiprocessing as mp
import random
import numpy as np
###########################################################################
# THERE ARE THE CONSTANTS TO CALCULATE EVERY EQUATIONS. GENERAL VARIABLES #
###########################################################################

NCORES = 8
T_inf =22
I= 6
V=1
activity_factor= 0.1
cp=0.1
h=50
as_motherboard = 60*(10**-4)
mass= 50*(10**-3)
C=900
t=41
tp_0= 293.15
e0=0.642
k=0.000863
mttftpo= 27800
E_a = 0.642
K = 8.623*(10**-5)
MTTR = 8
MTTF_t = 27800
J_inf_adv = 2.5 * (10**5)
J_inf = 2*(10**5)
N = 2
RH_inf_adv = 85
RH_inf = 60
gamma = 1
E_inf_adv = 4
E_inf = 3.25
T_inf_adv = 90
q = 4
nt=0.7
te= 1 
td=15
SEER= 13
ne=0.8
ATI=400
nti=0.95


def load_csv():
	try:
		df = pd.read_csv('results.csv', delimiter = ',', low_memory = False)
		return df
	except Exception as e:
		print('error reading file')
		print(e)

########## THIS DEFINITIONS ADD EACH METRIC TO THE NEW CSV ###################################
def add_metrics_to_new_csv(df,tf,mtt,mtt_upper,mtt_lower, mttfr, mttfem,mttfc,mttftddb,mttfsm,mttftc,a,aem,ac,
                           atddb,asm,atc,taa,qred,qr,PUE,DCie,cost, timestamp, externaltemp, roomtemp,
                           mttfic, a_tc, qdit,TPF):
	try:
		#df is loaded form main
		df = df.assign(TF = tf.values)
		df = df.assign(MTT = mtt.values)
		df = df.assign(MTT_upper = mtt_upper.values)
		df = df.assign(MTT_lower = mtt_lower.values)
		df = df.assign(MTTF_R = mttfr.values)
		df = df.assign(MTTF_EM = mttfem.values)
		df = df.assign(MTTF_C = mttfc.values)
		df = df.assign(MTTF_TDDB = mttftddb.values)
		df = df.assign(MTTF_SM = mttfsm.values)
		df = df.assign(MTTFF_TC = mttftc.values)
		df = df.assign(A = a.values)
		df = df.assign(AEM = aem.values)
		df = df.assign(AC = ac.values)
		df = df.assign(ATDDB = atddb.values)
		df = df.assign(ASM = asm.values)
		df = df.assign(ATC = atc.values)
		df = df.assign(TAA = taa.values)
		df = df.assign(QRED = qred.values)
		df = df.assign(QR = qr.values)
		df = df.assign(PUE = PUE.values)
		df = df.assign(DCie = DCie.values)
		df = df.assign(cost = cost.values)
		df = df.assign(TIMESTAMP = timestamp)
		df = df.assign(EXTERNAL_TEMP = externaltemp)
		df = df.assign(ROOM_TEMP = roomtemp)
		df = df.assign(MTTF_IC = mttfic.values)
		df = df.assign(A_TC = a_tc.values)
		df = df.assign(Q_DIT = qdit.values)
		df = df.assign(TPF = TPF.values)
		df.to_csv('results_out.csv', sep = ';')
	except Exception as e:
		print('error reading file')
		print(e)



# this get the dataframe from the CSV using pandas.
def getDataframeFromCsv(filePath, delimiter):
	try:
		return pd.read_csv(filePath, delimiter)
	except Exception as e:
		print('Error reading CSV')
		print(e)

#this describeDataFrame.
def describeDataFrame(df):
	try:
		print(df.describe())
	except Exception as e:
		print('Error describing dataframe')
		print(e)

#null features.
def describeNullFeature(dataframe, feature):
    try:
        records = len(dataframe)
        feature_nulls = dataframe[feature].isnull().sum()
    
        print('# records: ', records)
        print('flow traffic null: ', feature_nulls)
        print('% null flow traffic: ', (100*feature_nulls/(1.0*records)))
    except Exception as e:
        print('Error describing csv, dataframe and/or feature invalid')
        print(e)

#function that get the cpu_frecuency from the vnf cpu usage
#equation 1
def get_average_cpu_freceuncy(dataframe):
	ec_1 = dataframe['vnf cpu usage'] * NCORES * 2.3
	return ec_1

#equation 2
def output_frequency(vectorFrequency,room_temperature):
  factor_a = ((I * V) + (activity_factor * cp * V**2 ) ) / ( h * as_motherboard )
  factor_b = (1 - math.e ** (-((h * as_motherboard * 3600) / (mass * C))* t))
  ec_2 = room_temperature + factor_a * vectorFrequency * factor_b 
  return (ec_2)

#equation 3
def mttf(temperature):
	mtt= mttftpo*(math.e**((e0/k)*(1/temperature - 1/tp_0)))
	return mtt

#upper
def mttf_upper(mtt):
	mtt_upper = mtt + mtt*0.5
	return mtt_upper

#lower
def mttf_lower(mtt):
	mtt_lower = mtt - mtt*0.5 
	return mtt_lower

def mttfr(tf,room_temperature):
	#E_a div k
	E_a_k = E_a / k
	mttr = MTTF_t * math.e **( E_a_k * (1/(tf +273)  - 1/(np.array(room_temperature) +273)))
	return mttr

def mttfem(tf,room_temperature):
	#J_inf_adv div J_inf
	J_inf_adv_J_inf = J_inf_adv / J_inf
	
	#E_a div k
	E_a_k = E_a / k
	
	mttfem = MTTF_t * (J_inf_adv_J_inf ** (-N)) * math.e**(E_a_k * (1/(tf+273) - 1/(np.array(room_temperature)+273)))
	return mttfem
	
def mttfc(tf,room_temperature):
	#RH_inf_adv div RH_inf
	RH_inf_adv_RH_inf = RH_inf_adv / RH_inf
	
	#E_a div k
	E_a_k = E_a / k
	
	mttfc = MTTF_t * (RH_inf_adv_RH_inf ** (-2.7)) * math.e**(E_a_k * (1/(tf+273) - 1/(np.array(room_temperature)+273)))
	return mttfc

def mttftddb(tf,room_temperature):
	#E_a div k
	E_a_k = E_a / k
	mttftddb = MTTF_t * (math.e**(- gamma * (E_inf_adv - E_inf))) * (math.e ** (E_a_k * (1/(tf+273) - 1/(np.array(room_temperature) +273))))
	return mttftddb

def mttfsm(tf,room_temperature):
	#E_a div k
	E_a_k = E_a / k
	
	mttfsm = MTTF_t * (abs((tf - T_inf_adv )/(tf - room_temperature))) ** (-2.5) * (math.e ** (E_a_k * (1/(tf+273) - 1/(np.array(room_temperature)+273))))
	return mttfsm

def mttftc(tf,room_temperature):
	mttftc = MTTF_t * (abs((tf - T_inf_adv )/(tf - room_temperature))) ** (-q)
	return mttftc

def availability(temperature,room_temperature):
	up= mttftpo*math.e**((E_a/k)*(1/(temperature + 273) - 1/(273 + np.array(room_temperature)) ))
	down = mttftpo*math.e**((E_a/k)*(1/(temperature + 273) - 1/(np.array(room_temperature) + 273))) + MTTR
	a=up/down
	return a

# this is availability equation
def availability_due_to_Electromigration(temperature,room_temperature):
	up= mttftpo*(((2.5*(10**5))/(2*(10**5)))**(-2))*math.e**((E_a/k)*(1/(temperature + 273) - 1/(273+np.array(room_temperature))))
	down = mttftpo*(((2.5*(10**5))/(2*(10**5)))**(-2))*math.e**((E_a/k)*(1/(temperature + 273) - 1/(273 + np.array(room_temperature)))) + MTTR
	aem= up/down
	return aem

# this is availability equation
def availability_corrosion(temperature,room_temperature):
	up= mttftpo*((85/60)**(-2.7))*math.e**((E_a/k)*(1/(temperature + 273) - 1/ (273 + np.array(room_temperature))  ))
	down = mttftpo*((85/60)**(-2.7))*math.e**((E_a/k)*(1/(temperature + 273) - 1/(273 + np.array(room_temperature)))) + MTTR
	ac= up/down
	return ac

# this is availability equation
def availability_due_time_depende_dielectric(temperature,room_temperature):
	up= mttftpo*(math.e**(-1*(E_inf_adv -E_inf)))*math.e**((E_a/k)*(1/(temperature + 273) - 1/(273 + np.array(room_temperature))  ))
	down = mttftpo*(math.e**(-1*(E_inf_adv -E_inf)))*math.e**((E_a/k)*(1/(temperature + 273) - 1/   (273+np.array(room_temperature))   )) + MTTR
	atddb= up/down
	return atddb

# this is availability equation

def availability_stress_migration(temperature,room_temperature):
	up= mttftpo*(abs((temperature - T_inf_adv)/(temperature - room_temperature))**(-2.5))*math.e**((E_a/k)*(1/(temperature + 273) - 1/  (273 + np.array(room_temperature))))
	down = mttftpo*(abs((temperature - T_inf_adv)/(temperature - room_temperature))**(-2.5))*math.e**((E_a/k)*(1/(temperature + 273) - 1/  (273 + np.array(room_temperature)))) + MTTR
	asm= up/down
	return asm

def availability_thermal_cycling(temperature,room_temperature):
	up= mttftpo*(abs((temperature - T_inf_adv)/(temperature - room_temperature))**(-4))
	down = mttftpo*(abs((temperature - T_inf_adv)/(temperature - room_temperature))**(-4)) + MTTR
	atc = up/down
	return atc

def UnifiedReliability(MTTF_TC, MTTF_SM):
	return (MTTF_TC * MTTF_SM) /(MTTF_TC + MTTF_SM)

def external_temperature_impact(room_temperature,external_temperature,timestamp,qr):
	Pti= qr/3.412141633
	up= (ATI*h)*(((3.413*Pti*(1-nti)*timestamp)/(ATI*h)) + external_temperature - room_temperature) - qr
	down=   (ATI*h)
	TPF = up/down + np.array(room_temperature) 
	return TPF                   

def UnifiedAvailability(MTTF_IC):
	return MTTF_IC / (MTTF_IC + MTTR)

#this receive frecuency ec_1
def thermal_load_released(frecuency,temperature_room):
	qred= 3.413*((I*V + activity_factor*cp*V*V*frecuency)*(1 - nt) + h*as_motherboard*(np.array(temperature_room) -td))*te
	return qred


def power_required(frecuency,temperature_room):
	temp= 3.413
	upper= (3.792 + SEER)*((I*V+activity_factor*cp*V*V*frecuency)*(1- nt)) +3.792*(h*as_motherboard*(np.array(temperature_room)-td)) 
	low= ne*SEER
	qr= temp*(upper/low)*te
	return qr

























#this receive frecuency ec_1
def actual_ambient_temperature(frecuency,room_temperature):
	taa= room_temperature + ((I*V + activity_factor*cp*V*V*frecuency)*(1- nt))/(h*as_motherboard)
	return taa



#this receive frecuency ec_1

#need to check
def power_usage_efficiency(frecuency):
	PUE=3.413*((I*V + cp*activity_factor*V*te*frecuency)/(12.5325 + 0.01653*frecuency)) 
	return PUE

#need to check
def DCie(frecuency):
	DCie= (1253.25 + 1.653*frecuency)/(20.478 +0.03413*frecuency)
	return DCie

#need to check
def cost(frecuency):
	cost = 0.00048 + 0.00020478*frecuency
	return cost

def addTimeStamp(sizePop):
        return [i for i in range(1, sizePop + 1)]

def addRangeExternalTemp(sizePop):
	return [random.uniform(10,60) for i in range(0,sizePop)]

def addRangeRoomTemp(sizePop):
	return [random.uniform(20,30) for i in range(0,sizePop)]

def UnifiedReliability(MTTF_TC, MTTF_SM):
	return (MTTF_TC * MTTF_SM) /(MTTF_TC + MTTF_SM)

def UnifiedAvailability(MTTF_IC):
	return MTTF_IC / (MTTF_IC + MTTR)

def AmountEnergyDissipated(Qr, PUE):
	Ptotal = (Qr * PUE ) / 3.413
	return (Ptotal * te) * (1 - nt)

def main():
    try:
        print('starting')
        df = getDataframeFromCsv('results.csv',',')
        temperature_room= addRangeRoomTemp(len(df))
        external_temperature= addRangeRoomTemp(len(df))
        freq= get_average_cpu_freceuncy(df)
        tf = output_frequency(freq*1000,temperature_room)
        

        
        with mp.Pool(processes = 27) as pool:
            r1 = pool.apply_async(mttf,([tf]))
            r1_ = pool.apply_async(mttf_upper,([r1.get()]))
            r1__ = pool.apply_async(mttf_lower,([r1.get()]))
            r2 = pool.apply_async(mttfr,([tf,temperature_room]))
            r3 = pool.apply_async(mttfem,([tf,temperature_room]))
            r4 = pool.apply_async(mttfc,([tf,temperature_room]))
            r5 = pool.apply_async(mttftddb,([tf,temperature_room]))
            r6 = pool.apply_async(mttfsm,([tf,temperature_room]))
            r7 = pool.apply_async(mttftc,([tf,temperature_room]))
            r8 = pool.apply_async(availability,([tf,temperature_room]))
            r9 = pool.apply_async(availability_due_to_Electromigration,([tf,temperature_room]))
            r10 = pool.apply_async(availability_corrosion,([tf,temperature_room]))
            r11 = pool.apply_async(availability_due_time_depende_dielectric,([tf,temperature_room]))
            r12 = pool.apply_async(availability_stress_migration,([tf,temperature_room]))
            r13 = pool.apply_async(availability_thermal_cycling,([tf,temperature_room]))
            r14 = pool.apply_async(actual_ambient_temperature,([freq,temperature_room]))
            r15 = pool.apply_async(thermal_load_released,([freq,temperature_room]))
            r16 = pool.apply_async(power_required,([freq,temperature_room]))
            r17 = pool.apply_async(power_usage_efficiency,([freq]))
            r18 = pool.apply_async(DCie,([freq]))
            r19 = pool.apply_async(cost,([freq]))
            r20 = pool.apply_async(addTimeStamp,([len(df)]))
            #r21 = pool.apply_async(addRangeExternalTemp, ([len(df)]))
            #r22 = pool.apply_async(addRangeRoomTemp, ([len(df)]))
            r23 = pool.apply_async(UnifiedReliability, ([r7.get(), r6.get()]))           
            r24 = pool.apply_async(UnifiedAvailability, ([r23.get()]))
            r25 = pool.apply_async(AmountEnergyDissipated, ([r16.get(), r17.get()]))
            r26= pool.apply_async(external_temperature_impact,([temperature_room, external_temperature, r20.get(), r16.get()]))
			
            add_metrics_to_new_csv(df,tf, r1.get(), r1_.get(), r1__.get(), r2.get(), r3.get(), r4.get(), r5.get(), r6.get(),
                                   r7.get(), r8.get(), r9.get(), r10.get(), r11.get(), r12.get(), r13.get(), r14.get(), r15.get(), r16.get(),
                                   r17.get(), r18.get(), r19.get(), r20.get(), external_temperature, temperature_room, r23.get(), r24.get(), r25.get(),r26.get())
        #add_metrics_to_new_csv(c,atddb,asm,atc,taa,qred,qr,PUE,DCie,cost)        
        print('finished')
    except Exception as e:
        print('Error generatig new csv')
        print(e)


if __name__ == '__main__':
	main()

