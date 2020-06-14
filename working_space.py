import numpy as np
from Model import Market_Env,FundData
from Model import profilios_to_csv
import logging

logging.basicConfig(level=logging.DEBUG)

logging.info('test')

fund_return_src = r'.\data\Monthly_Fund_Return.csv'
feature_src = r'.\data\Feature.csv'
fund_map_src =r'.\data\FUND_MAP_SELECTED.csv'
env = Market_Env(feature_src,fund_map_src,fund_return_src)
profilios=[]
for i in range(6):
    inputs = np.random.rand(env.action_dim)
    profilo = env.create_profilio(inputs,6)
    #print(profilo)
    profilios.append(profilo)

cagr,mdd,transfer_count = env.fund_data.profilios_return(profilios,2015,1)
#print (f'cagr:{cagr},mdd:{mdd},transfer_count:{transfer_count}')
#profilios_to_csv(profilios,2012,10,r'.\Temp\test3.csv')

data = env.feature_data
dfMax = data.max(axis=0)
dfMin = data.min(axis=0)
#data = data.divide(dfMax, axis=1)
print(env.feature_data)
print(data[1])
