# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 14:56:27 2021

@author: wxb3
"""
import spectrum_featurization as spefea
import maching_learning as machlea
import caculator as ccl
import main_search 
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import time
import re
import joblib
import os 
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
def process(sample0,train_test,index,pubchem_dic,df_svm,in_house_database,database0,output):
    start=time.time()
    result0=open(output,"a")
    Round=0
    print("Round,PeakID,RT(min),Precursor m/z,Adduct,Adduct1,Height,Area,SMILES,Formula,Comment,mzerror (ppm),fragratio,S/N,MS1 isotopes,MSMS spectrum,Reference SMILES,Reference MSMS spectrum,all possible SMILES and scores",file=result0)
    for i in range(len(sample0)):
        if (~pd.isnull(sample0["MSMS spectrum"][i])):
            if(i%100==0):
                print("Round",Round,"Tentative PFAS peaks progress Bars：",round(100*i/len(sample0),1),"%",sep=" ")
            mzerror=0.000005*(sample0["Precursor m/z"][i])
            database1=database0[(database0['m/z']>sample0["Precursor m/z"][i]-mzerror) & (database0['m/z']<sample0["Precursor m/z"][i]+mzerror)].reset_index(drop=True)
            for j in range(len(database1)):
                error1=sample0["Precursor m/z"][i]-database1["m/z"][j]             
                fragratio=spefea.fragratio(database1["MSMS spectrum"][j],sample0["MSMS spectrum"][i],database1["m/z"][j],sample0["Precursor m/z"][i]) 
                if(fragratio>=0.5):
                    error2=round(error1*1000,2)
                    print(Round,sample0["PeakID"][i],sample0["RT (min)"][i],sample0["Precursor m/z"][i],"[M-H]-",1,sample0["Height"][i],sample0["Area"][i],database1["SMILES"][j],database1["formula"][j],database1["Name"][j],error2,round(fragratio,2),sample0['S/N'][i],sample0['MS1 isotopes'][i],sample0['MSMS spectrum'][i],"NA",database1["MSMS spectrum"][j],"NA","NA","NA","NA",sep=",",file=result0)
                    break
    result0.close()
    #begin to iterate from round 1
    flag=1
    while(flag==1):
        flag=0
        #database is identified PFASs with MS/MS spectra in the previous round
        dataall=pd.read_csv(output,index_col=False)
        if(len(dataall)==0):
            break
        dataall=adduct_(dataall)
        dataall.to_csv(output,index=False)
        database=dataall.loc[dataall['Round']==Round]
        database=database.loc[database['Adduct1']==1]
        database=database.dropna(subset=["SMILES"])
        database=database.reset_index(drop=True)
        list1=list(dataall["PeakID"])
        result0=open(output,"a")
        Round=Round+1
        for i in range(len(sample0)):
            if ((not pd.isnull(sample0["MSMS spectrum"][i])) and (sample0["PeakID"][i] not in list1)):
                if(i%100==0):
                    print("Round",Round,"Tentative PFAS peaks progress Bars：",round(100*i/len(sample0),1),"%",sep=" ")
                list4=[]
                list5=[]
                list6=[]
                for j in range(len(database)):             
                    fragratio=spefea.FNR(database["MSMS spectrum"][j],sample0["MSMS spectrum"][i],float(database["Precursor m/z"][j]),sample0["Precursor m/z"][i]) 
                    if(fragratio>=0.5):
                        list4.append(fragratio)
                        list5.append(database["MSMS spectrum"][j])
                        list6.append(database["SMILES"][j])
                if(list4==[]):
                    continue
                maxlist4=max(list4)
                maxlist5=list5[list4.index(maxlist4)]
                maxlist6=list6[list4.index(maxlist4)]
                df_formula=ccl.formula_caculator(spefea.MSMSspe(sample0["MS1 isotopes"][i]),sample0["Precursor m/z"][i],error_ppm=5,iso_thre=0.85)
                if(len(df_formula)==0):
                    continue
                df_pub,formula_0=main_search.rank_candidate_2(train_test,index,df_svm,sample0["Precursor m/z"][i],df_formula,spefea.MSMSspe(sample0["MSMS spectrum"][i]),pubchem_dic,in_house_database)
                if(len(df_pub)==0):
                    continue
                else:
                    list7=list(df_pub["SMILES"])
                    list8=list(df_pub["score"])
                    list9=list(df_pub["trans"])
                    list10=list(df_pub["error"])             
                    allsmiles=spefea.listtostr2(list7,list8,list9,list10)
                    flag=1                
                    if(df_pub["trans"][0]==""):                                                  
                        print(Round,sample0["PeakID"][i],sample0["RT (min)"][i],sample0["Precursor m/z"][i],"[M-H]-",1,sample0["Height"][i],sample0["Area"][i],df_pub["SMILES"][0],formula_0,"",df_pub["error"][0],round(maxlist4,2),sample0['S/N'][i],sample0['MS1 isotopes'][i],sample0['MSMS spectrum'][i],maxlist6,maxlist5,allsmiles,sep=",",file=result0)
                    else:
                        print(Round,sample0["PeakID"][i],sample0["RT (min)"][i],sample0["Precursor m/z"][i],"[M-H]-",1,sample0["Height"][i],sample0["Area"][i],"",formula_0,"",df_pub["error"][0],round(maxlist4,2),sample0['S/N'][i],sample0['MS1 isotopes'][i],sample0['MSMS spectrum'][i],maxlist6,maxlist5,allsmiles,sep=",",file=result0)
        result0.close()
    dataall=pd.read_csv(output)
    if(len(dataall)!=0):
        dataall=adduct_(dataall)
    del dataall["Adduct1"]
    dataall.to_csv(output,index=False)
    end=time.time()
    print("Round",Round,"The running time is %s hours."%(round((end-start)/(3600),2)),sep=" ")
    print("The results has been saved in ",output,".",sep="")