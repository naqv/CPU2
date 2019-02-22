import pandas as pd
import numpy as np

def fillColumn(df, column, dsfill):
    try:
        print('starting {}'.format(column))
        i = 0
        j = 0
        while(i < len(df)):
            if j >= len(dsfill):
                break
            else:
                if(np.isnan(df[column][i])):
                    df.loc[i,column] = dsfill.loc[j][0]
                    j += 1
            i += 1
        print('finished')
    except Exception as e:
        print('Err {}, i: {}, ds: {}'.format(e,i,column))

try:
    df =  pd.read_csv('results.csv', delimiter = ',', low_memory = False)

    print('it is assumed that the separator of the folloing files is a semicolon (;)')
    flow_traffic = pd.read_csv('flow_traffic.csv', delimiter = ';', header = None)
    flow_bdusage = pd.read_csv('flow_bdusage.csv', delimiter = ';', header = None)
    flow_packet_loss = pd.read_csv('flow.packet.loss.csv', delimiter = ';', header = None)
    flow_latency = pd.read_csv('nuevos_datos.csv', delimiter = ';', header = None)

    fillColumn(df,'flow traffic',flow_traffic)
    fillColumn(df,'flow bnd usage',flow_bdusage)
    fillColumn(df,'flow packet loss',flow_packet_loss)
    fillColumn(df,'flow latency',flow_latency)

    df.to_csv('results_out_with_new_values.csv', sep=';', index = False)
    print('saved')
    
except Exception as e:
    print('Err: {}'.format(e))
