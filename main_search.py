# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 09:09:35 2020

@author: Bing
"""
import maching_learning as machlea
import  spectrum_featurization as spefea
import pandas as pd
import caculator as ccl
import joblib
import numpy as np
import warnings 
import math
import re

warnings.filterwarnings("ignore")
def formulatolist(formula):
    formula=formula.replace("Cl","A")
    ptn=re.compile('([A-Z])([\d]*)')
    frag=ptn.findall(formula)
    list1=[]
    list2=[]
    list3=[]
    for i in range(len(frag)):
        list1.append(frag[i][0])
        if(frag[i][1]==""):
            list2.append(1)
        else:
            list2.append(int(frag[i][1]))
    if("C" in list1):
        list3.append(list2[list1.index("C")])
    else:
        list3.append(0)
    if("H" in list1):
        list3.append(list2[list1.index("H")])
    else:
        list3.append(0)
    if("A" in list1):
        list3.append(list2[list1.index("A")])
    else:
        list3.append(0)
    if("F" in list1):
        list3.append(list2[list1.index("F")])
    else:
        list3.append(0)
    if("N" in list1):
        list3.append(list2[list1.index("N")])
    else:
        list3.append(0)
    if("O" in list1):
        list3.append(list2[list1.index("O")])
    else:
        list3.append(0)
    if("P" in list1):
        list3.append(list2[list1.index("P")])
    else:
        list3.append(0)
    if("S" in list1):
        list3.append(list2[list1.index("S")])
    else:
        list3.append(0)
    return list3
def listtoformula(list_):
    formula=""
    list2=["C","H","Cl","F","N","O","P","S"]
    for i in range(len(list_)):
        if(list_[i]!=0):
            formula=formula+list2[i]+str(list_[i])
    return formula
def trans_pfas(cac_formula,database_formula,num_thre):
    list_cac=formulatolist(cac_formula)
    list_data=formulatolist(database_formula)
    num=0
    kind=0
    kind_hf=0
    list3=[]
    for i in range(len(list_cac)):
        list3.append(list_cac[i]-list_data[i])
        if(list3[i]!=0):
            if(i!=1 and i!=3):
                num=num+abs(list3[i])
                kind=kind+1
            else:
                num=num+0.5*abs(list3[i])
                kind_hf+=1
            if(num>num_thre):
                return -1,-1,-1,""
    return num,kind,kind_hf,listtoformula(list3)
def transform_pfas1(list_database,df_formula,num_thre=10):
    list_for=[]
    list_error=[]
    list_t=[]
    list_da=[]
    list_n=[]
    list_k=[]
    list_f=[]
    for i in range(len(df_formula)):
        list_data=[]
        list_trans=[]
        list_num=[]
        list_kind=[]
        f=[]       
        for j in range(len(list_database)):
            num_,kind_,f_,trans_=trans_pfas(df_formula["Formula_mole"][i],list_database[j],num_thre)
            if(num_==-1):
                continue
            list_data.append(list_database[j])
            list_trans.append(trans_)
            list_num.append(num_)
            list_kind.append(kind_)
            f.append(f_)
        if(list_data==[]):
            continue
        df_i=pd.DataFrame({"Formula_":list_data,"trans":list_trans,"num":list_num,"kind":list_kind,"f":f,})
        df_i=df_i.sort_values(by=["num","kind","f"],ascending=["True","True","False"])
        df_i=df_i.reset_index(drop=True)
        list_for.append(df_formula["Formula_mole"][i])
        list_error.append(df_formula["Error(ppm)"][i])
        list_t.append(df_i["trans"][0])
        list_da.append(df_i["Formula_"][0])
        list_n.append(df_i["num"][0])
        list_k.append(df_i["kind"][0])
        list_f.append(df_i["f"][0])
    if(list_for==[]):
        return pd.DataFrame()
    df_=pd.DataFrame({"Formula":list_for,"Error(ppm)":list_error,"Database_formula":list_da,"Trans":list_t,"num":list_n,"kind":list_k,"f":list_f})
    df_=df_.sort_values(by=["num","kind","f"],ascending=["True","True","False"])
    df_=df_.reset_index(drop=True)
    return df_
# rank candidate from offline pubchem database
def rank_candidate(train_test,index,df_svm,mz,df_formula,MS2spe,database_pubchem,MS2_error_Da=0.01,inhouse_database=None):
    list_smi=[]
    if(type(database_pubchem)==dict and type(inhouse_database)==dict):
        for i in range(len(df_formula)):
            if(df_formula["Formula_mole"][i] in database_pubchem.keys()):
                list_smi=list_smi+database_pubchem[df_formula["Formula_mole"][i]]
            if(df_formula["Formula_mole"][i] in inhouse_database.keys()):
                list_smi=list_smi+inhouse_database[df_formula["Formula_mole"][i]]
        list_smi=list(set(list_smi))
    elif(type(database_pubchem)==dict and type(inhouse_database)!=dict):
        for i in range(len(df_formula)):
            if(df_formula["Formula_mole"][i] in database_pubchem.keys()):
                list_smi=list_smi+database_pubchem[df_formula["Formula_mole"][i]]
    elif(type(database_pubchem)!=dict and type(inhouse_database)==dict):
        for i in range(len(df_formula)):
            if(df_formula["Formula_mole"][i] in inhouse_database.keys()):
                list_smi=list_smi+inhouse_database[df_formula["Formula_mole"][i]]
    else:
        return pd.DataFrame(),[]                
    
    sim_=np.zeros(shape=(1,len(train_test)))
    MS2=spefea.spetostr(MS2spe)
    for j in range(len(train_test)):
        sim_[0][j]=spefea.FNR(train_test["MSMS spectrum"][j],MS2,train_test["m/z"][j],mz,MS2_error_Da,minmz=50,maxmz=1250)
    fp_pre=np.zeros(shape=(1,len(index)))    
    for k in range(len(index)):
        model_name="model\svm"+str(k)+".model"
        f1=open(model_name,"rb")
        model_svm=joblib.load(f1)
        f1.close()
        fp_pre[0,k]=model_svm.predict(sim_)
    if(list_smi==[]):
        return pd.DataFrame(),fp_pre
    list_score=[]
    for m in range(len(list_smi)):        
        fp_cac=np.array([machlea.get_cdk_fingerprints(list_smi[m])])
        fp_cac=fp_cac[:,index]
        score_=machlea.score(fp_pre,fp_cac,df_svm)
        list_score.append(score_)
    df_fp=pd.DataFrame({"SMILES":list_smi,"score":list_score})
    df_fp=df_fp.sort_values(by="score",ascending=False)    
    df_fp=df_fp.reset_index(drop=True)
    return df_fp,fp_pre
# rank along with transformation with offline database
def rank_candidate_1(train_test,index,df_svm,mz,df_formula,MS2spe,database_pubchem,inhouse_database,MS2_error_Da=0.01,thresh=0.9):
    list_smi=[]
    if(type(database_pubchem)==dict and type(inhouse_database)==dict):
        for i in range(len(df_formula)):
            if(df_formula["Formula_mole"][i] in database_pubchem.keys()):
                list_smi=list_smi+database_pubchem[df_formula["Formula_mole"][i]]
            if(df_formula["Formula_mole"][i] in inhouse_database.keys()):
                list_smi=list_smi+inhouse_database[df_formula["Formula_mole"][i]]
        list_smi=list(set(list_smi))
    else:
        return pd.DataFrame()                   
    sim_=np.zeros(shape=(1,len(train_test)))
    MS2=spefea.spetostr(MS2spe)
    for j in range(len(train_test)):
        sim_[0][j]=spefea.FNR(train_test["MSMS spectrum"][j],MS2,train_test["m/z"][j],mz,MS2_error_Da,minmz=50,maxmz=1250)
    fp_pre=np.zeros(shape=(1,len(index)))    
    for k in range(len(index)):
        model_name="model\svm"+str(k)+".model"
        f1=open(model_name,"rb")
        model_svm=joblib.load(f1)
        f1.close()
        fp_pre[0,k]=model_svm.predict(sim_)
    smi_t1=[]
    sco_t1=[]
    trans_t1=[]
    error_ppm=[]
    formula_0=""
# caculate transformation score based on inhouse database with threshhold 
    list_database=list(inhouse_database.keys())
    df1=transform_pfas1(list_database,df_formula,num_thre=10)
    if(len(df1)==0):
        df2=pd.DataFrame()
    elif(len(df1)>3):
        df2=df1[0:3].copy()
    else:
        df2=df1.copy()
    if(len(df2)!=0): 
        formula_0=df2["Formula"][0]
        list_smi1=inhouse_database[df2["Database_formula"][0]]
        list_score1=[]  
        for m in range(len(list_smi1)):        
            fp_cac=np.array([machlea.get_cdk_fingerprints(list_smi1[m])])
            fp_cac=fp_cac[:,index]
            score_=machlea.score(fp_pre,fp_cac,df_svm)
            list_score1.append(score_)
        sco1=max(list_score1)
        smi1=list_smi1[list_score1.index(sco1)]
        smi_t1.append(smi1)
        sco_t1.append(sco1*thresh)
        trans_t1.append(df2["Trans"][0])  
        error_ppm.append(str(round(df2["Error(ppm)"][0],2)))
        if(len(df2)>=2):
            list_smi2=inhouse_database[df2["Database_formula"][1]]
            list_score2=[] 
            for m in range(len(list_smi2)):        
                fp_cac=np.array([machlea.get_cdk_fingerprints(list_smi2[m])])
                fp_cac=fp_cac[:,index]
                score_=machlea.score(fp_pre,fp_cac,df_svm)
                list_score2.append(score_)
            sco2=max(list_score2)
            smi2=list_smi2[list_score2.index(sco2)]
            smi_t1.append(smi2)
            sco_t1.append(sco2*thresh)
            trans_t1.append(df2["Trans"][1]) 
            error_ppm.append(str(round(df2["Error(ppm)"][1],2)))
        if(len(df2)>=3):
            list_smi3=inhouse_database[df2["Database_formula"][2]]
            list_score3=[]   
            for m in range(len(list_smi3)):        
                fp_cac=np.array([machlea.get_cdk_fingerprints(list_smi3[m])])
                fp_cac=fp_cac[:,index]
                score_=machlea.score(fp_pre,fp_cac,df_svm)
                list_score3.append(score_)
            sco3=max(list_score3)
            smi3=list_smi3[list_score3.index(sco3)]
            smi_t1.append(smi3)
            sco_t1.append(sco3*thresh)
            trans_t1.append(df2["Trans"][2]) 
            error_ppm.append(str(round(df2["Error(ppm)"][2],2)))
# caculate pubchem smiles score
    sco_t0=[]
    trans_t0=[]
    error_ppm0=[]
    for m in range(len(list_smi)):        
        fp_cac=np.array([machlea.get_cdk_fingerprints(list_smi[m])])
        fp_cac=fp_cac[:,index]
        score_=machlea.score(fp_pre,fp_cac,df_svm)
        sco_t0.append(score_)
        trans_t0.append("")
        error_ppm0.append(str(round((mz-machlea.SmilestoMW(list_smi[m]))*1000000/mz,2)))
    if(sco_t1==[] and sco_t0==[]):
        return pd.DataFrame(),""
    else:
        df_fp=pd.DataFrame({"SMILES":(list_smi+smi_t1),"score":(sco_t0+sco_t1),"trans":(trans_t0+trans_t1),"error":(error_ppm0+error_ppm)})
        df_fp=df_fp.sort_values(by="score",ascending=False)    
        df_fp=df_fp.reset_index(drop=True)
        if(df_fp["trans"][0]==""):
            return df_fp,machlea.SmilestoFormula(df_fp["SMILES"][0])
        else:
            return df_fp,formula_0
        
# rank along with transformation with offline database
def rank_candidate_2(train_test,index,df_svm,mz,df_formula,MS2spe,database_pubchem,inhouse_database,MS2_error_Da=0.01,num_thresh=10):
    list_smi=[]
    if(type(database_pubchem)==dict and type(inhouse_database)==dict):
        for i in range(len(df_formula)):
            if(df_formula["Formula_mole"][i] in database_pubchem.keys()):
                list_smi=list_smi+database_pubchem[df_formula["Formula_mole"][i]]
            if(df_formula["Formula_mole"][i] in inhouse_database.keys()):
                list_smi=list_smi+inhouse_database[df_formula["Formula_mole"][i]]
        list_smi=list(set(list_smi))
    else:
        return pd.DataFrame()                   
    sim_=np.zeros(shape=(1,len(train_test)))
    MS2=spefea.spetostr(MS2spe)
    for j in range(len(train_test)):
        sim_[0][j]=spefea.FNR(train_test["MSMS spectrum"][j],MS2,train_test["m/z"][j],mz,MS2_error_Da,minmz=50,maxmz=1250)
    fp_pre=np.zeros(shape=(1,len(index)))    
    for k in range(len(index)):
        model_name="model\svm"+str(k)+".model"
        f1=open(model_name,"rb")
        model_svm=joblib.load(f1)
        f1.close()
        fp_pre[0,k]=model_svm.predict(sim_)
    smi_t1=[]
    sco_t1=[]
    trans_t1=[]
    error_ppm=[]
    formula_0=""
# caculate transformation score based on inhouse database with threshhold 
    list_database=list(inhouse_database.keys())
    df1=transform_pfas1(list_database,df_formula,num_thre=num_thresh)
    if(len(df1)==0):
        df2=pd.DataFrame()
    else:
        df2=df1.copy()
    for i in range(len(df2)):
        list_smi1=inhouse_database[df2["Database_formula"][i]]
        list_score1=[]  
        for m in range(len(list_smi1)):        
            fp_cac=np.array([machlea.get_cdk_fingerprints(list_smi1[m])])
            fp_cac=fp_cac[:,index]
            score_=machlea.score(fp_pre,fp_cac,df_svm)
            list_score1.append((score_*(1-0.02*(math.pow(df2["kind"][i],2)+df2["num"][0]))))
        sco1=max(list_score1)
        smi1=list_smi1[list_score1.index(sco1)]
        smi_t1.append(smi1)
        sco_t1.append(sco1)
        trans_t1.append(df2["Trans"][i])  
        error_ppm.append(str(round(df2["Error(ppm)"][i],2)))
        if(i==0):
            formula_0=df2["Formula"][i]
            score_for1=sco_t1[i]
        else:
            if(sco_t1[i]>score_for1):
                formula_0=df2["Formula"][i]
                score_for1=sco_t1[i]
# caculate pubchem smiles score
    sco_t0=[]
    trans_t0=[]
    error_ppm0=[]
    for m in range(len(list_smi)):        
        fp_cac=np.array([machlea.get_cdk_fingerprints(list_smi[m])])
        fp_cac=fp_cac[:,index]
        score_=machlea.score(fp_pre,fp_cac,df_svm)
        sco_t0.append(score_)
        trans_t0.append("")
        error_ppm0.append(str(round((mz-machlea.SmilestoMW(list_smi[m]))*1000000/mz,2)))
    if(sco_t1==[] and sco_t0==[]):
        return pd.DataFrame(),""
    else:
        df_fp=pd.DataFrame({"SMILES":(list_smi+smi_t1),"score":(sco_t0+sco_t1),"trans":(trans_t0+trans_t1),"error":(error_ppm0+error_ppm)})
        df_fp=df_fp.sort_values(by="score",ascending=False)    
        df_fp=df_fp.reset_index(drop=True)
        if(df_fp["trans"][0]==""):
            return df_fp,machlea.SmilestoFormula(df_fp["SMILES"][0])
        else:
            return df_fp,formula_0


# rank candidate from pubchem online database
def rank_candidate_3(train_test,index,df_svm,mz,MS1spe,MS2spe,MS1_error_ppm=5,MS2_error_Da=0.01,iso_thre=0.85,database_pubchem=1,df_inhouse_database=None):
    df_formula=ccl.formula_caculator(MS1spe,mz,MS1_error_ppm,iso_thre)
    list_smi=[]
    for i in range(len(df_formula)):
        list_smi1=[]
        if(database_pubchem==1):
            list_smi1=machlea.searchpubchem(df_formula["Formula_mole"][i])
        if isinstance(df_inhouse_database,pd.DataFrame): 
            list_smi1=list_smi1+machlea.search_inhouse(df_formula["Formula_mole"][i],df_inhouse_database)
        if(list_smi1==[]):
            continue
        list_smi=list_smi+list_smi1
    if(list_smi==[]):
        return pd.DataFrame()
    list_smi1=list_smi.copy()
    list_smi=[]
    for i in range(len(list_smi1)):
        if("." not in list_smi1[i]):
            list_smi.append(machlea.iso(list_smi1[i]))
    list_smi=list(set(list_smi))
    if(list_smi==None):
        return pd.DataFrame()
    sim_=np.zeros(shape=(1,len(train_test)))
    MS2=spefea.spetostr(MS2spe)
    for j in range(len(train_test)):
        sim_[0][j]=spefea.FNR(train_test["MSMS spectrum"][j],MS2,train_test["m/z"][j],mz,MS2_error_Da,minmz=50,maxmz=1250)
    fp_pre=np.zeros(shape=(1,len(index)))    
    for k in range(len(index)):
        model_name="model\svm"+str(k)+".model"
        f1=open(model_name,"rb")
        model_svm=joblib.load(f1)
        f1.close()
        fp_pre[0,k]=model_svm.predict(sim_)    
    list_score=[]
    for m in range(len(list_smi)):        
        fp_cac=np.array([machlea.get_cdk_fingerprints(list_smi[m])])
        fp_cac=fp_cac[:,index]
        score_=machlea.score(fp_pre,fp_cac,df_svm)
        list_score.append(score_)
    df_fp=pd.DataFrame({"SMILES":list_smi,"score":list_score})
    df_fp=df_fp.sort_values(by="score",ascending=False)    
    df_fp=df_fp.reset_index(drop=True)
    return df_fp  
  




#testdata
"""
f1=open("train_test.joblib","rb")
f2=open("indexlist.joblib","rb")
f3=open("pubchem_dic.joblib","rb")
train_test=joblib.load(f1)
index=joblib.load(f2)
pubchem_dic=joblib.load(f3)
f1.close()
f2.close()
f3.close()
df_svm=pd.read_csv("fnr_svm.csv")    
mz1=412.9655
MS1=spefea.MSMSspe("412.74103:3127450 412.96548:1523914368 413.1864:2288183 413.97018:141754464 414.97256:9901278")
MS2=spefea.MSMSspe("51.42956:1915798 51.49224:1804712 52.9576:2716837 54.07448:1813418 60.1084:2088622 63.00193:2126126 63.95168:2230482 91.4044:2479800 104.21696:2502640 105.41184:2266984 118.99163:12679638 125.08211:2412372 132.90079:2263684 133.22545:2655879 154.87143:2430069 168.98938:60777184 174.95171:2404028 218.98671:16229503 236.58173:2453787 368.97702:20838016")   
in_house_database=machlea.read_inhouse_database("in_house_PFAS_database.csv")
df_formula=ccl.formula_caculator(MS1,mz1,error_ppm=5,iso_thre=0.85)
df=rank_candidate(train_test,index,df_svm,mz1,df_formula,MS2,database_pubchem=pubchem_dic,inhouse_database=in_house_database)
df2=transform_pfas(list(in_house_database.keys()),df_formula)
"""
