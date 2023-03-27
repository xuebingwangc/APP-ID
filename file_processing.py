# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 14:56:27 2021

@author: wxb3
"""
import pandas as pd
import numpy as dp
import  spectrum_featurization as spefea
# blank substraction 
def blank_substraction(sample,ck,MS_error=0.01,rt_error=0.5,sn=3):
    index=[]
    for i in range(len(sample)):        
        if(pd.isnull(sample["MSMS spectrum"][i])):
            continue
        flag=1
        for j in range(len(ck)):
            if(abs(sample["Precursor m/z"][i]-ck["Precursor m/z"][j])<=MS_error and abs(sample["RT (min)"][i]-ck["RT (min)"][j])<=rt_error and (sample["Area"][i]/ck["Area"][j])>3):
                flag=0
                break
        if(flag==1):
            index.append(i)
    return sample.iloc[index].reset_index(drop=True)
#judge aduuct ion from previously identidied peaks, M-CO2-H and 2M-H were flitered, others with similar MS2 were marked
def adduct_(data,MS_error=0.002,rt_error=0.5,sim=0.7):
    data1=data[data["Round"]==data["Round"][len(data)-1]]
    data1=data1.reset_index(drop=True)
    for i in range(len(data1)):
        data2=data[(data["RT(min)"]>data1["RT(min)"][i]-rt_error)&(data["RT(min)"]<data1["RT(min)"][i]+rt_error)].reset_index(drop=True)
        data3=data2[(data2["Precursor m/z"]>data1["Precursor m/z"][i]+43.98983-MS_error)&(data2["Precursor m/z"]<data1["Precursor m/z"][i]+43.98983+MS_error)].reset_index(drop=True)        
        if(len(data3)>0):
            data["Adduct"][i+len(data)-len(data1)]="[M-CO2-H]-"+";"+str(data3["PeakID"][0])
            data["Adduct1"][i+len(data)-len(data1)]=0
            continue
        data4=data2[(data2["Precursor m/z"]>2*data1["Precursor m/z"][i]+1.00728-MS_error)&(data2["Precursor m/z"]<2*data1["Precursor m/z"][i]+1.00728+MS_error)].reset_index(drop=True)        
        if(len(data4)>0):
            data["Adduct"][i+len(data)-len(data1)]="[2M-H]-"+";"+str(data4["PeakID"][0])
            data["Adduct1"][i+len(data)-len(data1)]=0
            continue      
        for j in range(len(data2)):
            if(data2["PeakID"][j]!=data1["PeakID"][i] and spefea.DP(data1["MSMS spectrum"][i],data2["MSMS spectrum"][j])>sim):
                data["Adduct"][i+len(data)-len(data1)]="similar MS2 with"+str(data2["PeakID"][j])
                data["Adduct1"][i+len(data)-len(data1)]=1
                continue
    return data
