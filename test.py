import numpy as np
from Model import Market,Market_Env,Market
from Model import profilios_to_csv
import logging


a = range(10)
print(list(a))
a = np.array(a[:-1]) + 100
print(list(a))

logging.basicConfig(level=logging.DEBUG)

logging.info('test')

fund_return_src = r'.\data\Monthly_Fund_Return.csv'
feature_src = r'.\data\Feature.csv'
fund_map_src =r'.\data\FUND_MAP_SELECTED.csv'
market =Market(data_src =fund_return_src)

env = Market_Env(feature_src,fund_map_src)
profilios=[]
for i in range(6):
    inputs = np.random.rand(env.action_dim)
    profilo = env.create_profilio(inputs,6)
    #print(profilo)
    profilios.append(profilo)

cagr,mdd,transfer_count = market.get_result(profilios,2015,1)
#print (f'cagr:{cagr},mdd:{mdd},transfer_count:{transfer_count}')
profilios_to_csv(profilios,2012,10,r'.\Temp\test3.csv')
