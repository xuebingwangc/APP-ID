# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 22:14:00 2020

@author: Bing
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
import file_processing as filepro
import os 
# input data
filename=input("Please input the filename: ")

output=filename.split(".")[0]+"output.csv"
sample0=pd.read_csv(filename)
f1=open("train_test.joblib","rb")
f2=open("indexlist.joblib","rb")
f3=open("pubchem_dic.joblib","rb")
train_test=joblib.load(f1)
index=joblib.load(f2)
pubchem_dic=joblib.load(f3)
df_svm=pd.read_csv("fnr_svm.csv")  
in_house_database=machlea.read_inhouse_database("PFAS_database.csv")  
database0=pd.read_csv("PFASMSMSdatabase.csv")
f1.close()
f2.close()
f3.close()
# collecting seeds based on MS/MS database
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
    dataall=filepro.adduct_(dataall)
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
dataall=filepro.adduct_(dataall)
del dataall["Adduct1"]
dataall.to_csv(output,index=False)
end=time.time()
print("Round",Round,"The running time is %s hours."%(round((end-start)/(3600),2)),sep=" ")
print("The results has been saved in ",output,".",sep="")


        
