import pandas as pd
import os
import numpy as np
import math
import datetime
import gym
import csv
import matplotlib.pyplot as plt
import logging

def next_year(year,month):
    year,month = (year+1,1) if (month ==12) else (year,month+1)
    return (year,month)
  
def acc_return(profits):
    return (1+profits).cumprod()-1
def maxdrawdown(x,display=False):
    e = np.argmax(np.maximum.accumulate(x) - x) # end of the period
    s = e if (e==0) else  np.argmax(x[:e])
    if (display):
        plt.plot(x)
        plt.plot([e, s], [x[e], x[s]], 'o', color='Red', markersize=10)
        plt.show()
    return (x[e]-x[s])/(1+x[s])
def get_cagr(profits):
    return math.pow(1+acc_return(profits)[-1],12.0/len(profits))-1

def get_score(cagr,mdd):
    if (mdd > 0.2):
        b=3
    elif (mdd>0.1):
        b=2
    else:
        b=1
    return cagr/b

def profilios_to_csv(profilios,start_year,start_month,file):
    ids,data = np.unique([item[0] for sublist in profilios for item in sublist]),{}
    data['DAYM']=[]
    year,month = start_year,start_month
    for id in ids:
        data[id]=[]
        for profilio in profilios:
            data[id].append(next(iter([i[1] for i in profilio if i[0]==id]),'NA'))
    for profilio in profilios:
        data['DAYM'].append(datetime.datetime(year,month,1))
        year,month = next_year(year,month)
    pd.DataFrame(data=data).to_csv(file,index=False)


class FundData:
    def __init__(self,data_src: str):
        self.data = None
        parse_dates = ['DATAYM']
        self.data=pd.read_csv(data_src,parse_dates=parse_dates)
    def get_risk_types(self):
        return self.data['RISKTYPENAME'].unique()
    def get_perofrmence_id(self,risk_type=None):
        src = self.data[self.data['RISKTYPENAME']==risk_type] if (risk_type!=None) else self.data
        return src['ISINCODE'].unique()
    def get_return(self,isin_code,date):
        ret = self.data[(self.data.ISINCODE==isin_code)& (self.data.DATAYM ==date)]['RET1M'].values
        #return 0 for non-existing data
        return ret[0] if(len(ret)==1 and not np.isnan(ret[0])) else 0

class Market:
    def __init__(self,data_src):
        self.fund_data = FundData(data_src = data_src)
    def get_result(self,portfolios,start_year,start_month):
        profits = []
        year,month = start_year,start_month
        for portfolio in portfolios:
            profit =0
            for id, weight in portfolio:
                ret = self.fund_data.get_return(id,datetime.datetime(year,month,1))
                profit += weight*ret
            year,month = next_year(year,month)
            profits.append(profit)
        profits=np.array(profits)
        transfer_count=0
        for i in range(1,len(portfolios)):
            for id,_ in portfolios[i]:
                if (not id in [item[0] for item in portfolios[i-1]]):
                    transfer_count+=1

        cagr = get_cagr(profits)
        mdd = maxdrawdown(acc_return(profits))        
        return cagr,mdd,transfer_count

class Market_Env():
    def __init__ (self,feature_src,fund_map_src,equity_limit=0.75,episode_limit=30):
        self.feature_data = pd.read_csv(feature_src)
        fund_map_src
        self.fund_map = pd.read_csv(fund_map_src)
        self.funds = self.fund_map['ISINCODE'].values
        self.state_dim =self.feature_data.shape[1]
        self.max_action = 0.5
        self.action_dim = len(self.funds)+1 #one more action for risk off
        self.equity_limit=equity_limit
        self.episode_limit=episode_limit
        logging.info(f'model par: state:{self.state_dim} action:{self.action_dim}')
    def create_profilio(self,inputs,max_number):
        funds = self.funds
        if(len(inputs)!=len(funds)+1):
            logging.warning(f"size of inputs and funds does not match should be {len(funds)+1}")
            return None
        threshold = inputs[np.argsort(inputs[:-1])[-max_number]]
        weights = [i if i >= threshold else 0 for i in inputs[:-1]]
        weights = weights/sum(weights)
        weights = self.adjust_weight(weights)
       
        profilio =[]
        for batch_id, weight in enumerate(weights):
            if(not np.isclose(weight,0)):
                profilio.append((self.funds[batch_id],weight))
        return profilio

    def adjust_weight(self,weights):
        #adjust equity to be under equity_limit
        equity_sum = 0
        for batch_id, weight in enumerate(weights):
            fund = self.funds[batch_id]
            risk_type = self.get_fund_risk_type(fund)
            equity_sum += (weight if risk_type=='Equity' else 0)
        logging.debug(f'equity_weight_sum:{equity_sum}')
        if(equity_sum>self.equity_limit):
            ratio = (self.equity_limit-equity_sum*self.equity_limit) / (equity_sum-equity_sum*self.equity_limit)
            for batch_id, _ in enumerate(weights):
                fund = self.funds[batch_id]
                risk_type = self.get_fund_risk_type(fund)
                if risk_type=='Equity':
                    weights[batch_id] = weights[batch_id]*ratio
            weights = weights/sum(weights)
            equity_sum=0
            for batch_id, weight in enumerate(weights):
                fund = self.funds[batch_id]
                risk_type = self.get_fund_risk_type(fund)
                equity_sum += (weight if risk_type=='Equity' else 0)
            logging.debug(f'equity_weight_sum after adjust:{equity_sum}') 
        return weights
    def get_fund_risk_type(self,fund):
        return self.fund_map[self.fund_map['ISINCODE']==fund]['RISKTYPENAME'].values[0]
    
    def seed(self,seed):
        pass
    def reset(self,validation=True):
        state=0
        self.episode=0
        state = np.zeros(self.state_dim)
        return state
    def step(self,action):   
        reward, done=0,False
        state = np.zeros(self.state_dim)
        self.episode+=1
        done = True if (self.episode>=self.episode_limit) else False
            
        return state, reward, done






