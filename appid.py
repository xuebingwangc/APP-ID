# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:52:31 2023

@author: wxb4
"""

import pandas as pd
import joblib
import file_processing as filepro
import maching_learning as machlea

# input data
filename="test.csv"
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

#process data
filepro.process(sample0,train_test,index,pubchem_dic,df_svm,in_house_database,database0,output)