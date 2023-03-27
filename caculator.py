# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:48:53 2020

@author: Bing
"""

import re
import numpy as np   
import time
import pandas as pd
from IsoSpecPy import IsoSpecPy 
def fragtomz(str1):
    str1=str1.replace("Cl","A")
    dd={"C":12,"H":1.007825032,"O":15.99491462,"N":14.003074,"S":31.972071,"P":30.97376163,"A":34.96885268,"F":18.99840322}
    ptn=re.compile('([A-Z])([\d]*)')
    frag=ptn.findall(str1)
    mz=0
    for code, nums in frag:
        if code in dd:
            if(nums==""):
                num=1
            else:
                num=int(nums)
            mz=mz+dd[code]*num
    return mz+0.00055
 

def formula_cac(mz,error):
    formulas=[]
    formulas2=[]
    formulas3=[]
    mz_act=[]
    error_ab=[]
    error_re=[]
    error_re2=[]
    rdb=[]
    for c in range(2,min(int(mz/12)+2,51)):
        for f in range(3,min(int((mz-c*12)/18.9984)+3,51)):
            for o in range(min(int((mz-12*c-18.9984*f)/15.9949)+2,21)):
                for s in range(min(int((mz-12*c-18.9984*f-15.9949*o)/31.9721)+2,3)):
                    for n in range(min(int((mz-12*c-18.9984*f-15.9949*o-s*31.9721)/14.0031)+2,6)):
                        for p in range(min(int((mz-12*c-18.99840322*f-15.99491462*o-s*31.972072071-n*14.003074)/30.97376)+2,3)):
                            for h in range(min(int((mz-12*c-18.99840322*f-15.99491462*o-s*31.972072071-n*14.003074-p*30.97376163)/1.007825032)+2,101)):
                                rdb_2=h+f-p-n+1
                                if(rdb_2%2==1):
                                    continue
                                rdb_=c+1-0.5*rdb_2
                                if(-0.5<rdb_<20):
                                    mz_=c*12+f*18.99840322+o*15.99491462+s*31.972071+n*14.003074+p*30.97376163+h*1.007825032+0.00055
                                    ab_error=mz-mz_
                                    re_error=(ab_error)/mz*1000000
                                    re_error2=abs(re_error)
                                    if(re_error2<=error):
                                        mz_act.append(mz_)
                                        error_ab.append(ab_error*1000)
                                        error_re.append(re_error)
                                        rdb.append(rdb_)
                                        formula="C"+str(c)
                                        formula2="C"+str(c)
                                        if(h!=0):
                                            formula+="H"
                                            formula2+="H"+str(h)
                                            formula3=formula+str(h+1)
                                            if(h!=1):
                                                formula+=str(h)
                                        else:
                                            formula3=formula+"H"
                                        if(f!=0):
                                            formula+="F"
                                            formula3+="F"
                                            formula2+="F"+str(f)
                                            if(f!=1):
                                                formula+=str(f)
                                                formula3+=str(f)
                                        if(n!=0):
                                            formula+="N"
                                            formula3+="N"
                                            formula2+="N"+str(n)
                                            if(n!=1):
                                                formula+=str(n)
                                                formula3+=str(n)
                                        if(o!=0):
                                            formula+="O"
                                            formula3+="O"
                                            formula2+="O"+str(o)
                                            if(o!=1):
                                                formula+=str(o)
                                                formula3+=str(o)                                        
                                        if(p!=0):
                                            formula+="P"
                                            formula3+="P"
                                            formula2+="P"+str(p)
                                            if(p!=1):
                                                formula+=str(p)
                                                formula3+=str(p)
                                        if(s!=0):
                                            formula+="S"
                                            formula3+="S"
                                            formula2+="S"+str(s)
                                            if(s!=1):
                                                formula+=str(s)
                                                formula3+=str(s)
                                        formulas.append(formula)
                                        formulas2.append(formula2)
                                        formulas3.append(formula3)
                                        error_re2.append(re_error2)
    df=pd.DataFrame({"Formula":formulas,"m/z":mz_act,"Error(ppm)":error_re,"Error(mDa)":error_ab,"RDB":rdb,"abs_re_error":error_re2,"Formula_1":formulas2,"Formula_mole":formulas3})
    df=df.sort_values(by="abs_re_error",ascending=True)    
    df=df.reset_index(drop=True)
    df=df[["Formula","m/z","Error(ppm)","Error(mDa)","RDB","Formula_1","Formula_mole"]]
    return df
def formula_cac_cl(mz,error=5):
    formulas=[]
    formulas2=[]
    formulas3=[]
    mz_act=[]
    error_ab=[]
    error_re=[]
    error_re2=[]
    rdb=[]
    for cl in range(1,5):
        for c in range(2,min(int((mz-cl*34.96885268)/12)+2,51)):
            for f in range(3,min(int((mz-c*12-cl*34.96885268)/18.9984)+3,51)):
                for o in range(min(int((mz-12*c-cl*34.96885268-18.9984*f)/15.9949)+2,21)):
                    for s in range(min(int((mz-12*c-cl*34.96885268-18.9984*f-15.9949*o)/31.9721)+2,3)):
                        for n in range(min(int((mz-12*c-cl*34.96885268-18.9984*f-15.9949*o-s*31.9721)/14.0031)+2,6)):
                            for p in range(min(int((mz-12*c-cl*34.96885268-18.99840322*f-15.99491462*o-s*31.972072071-n*14.003074)/30.97376)+2,3)):
                                for h in range(min(int((mz-12*c-cl*34.96885268-18.99840322*f-15.99491462*o-s*31.972072071-n*14.003074-p*30.97376163)/1.007825032)+2,101)):
                                    rdb_2=h+cl+f-p-n+1
                                    if(rdb_2%2==1):
                                        continue
                                    rdb_=c+1-0.5*rdb_2
                                    if(-0.5<rdb_<20):
                                        mz_=c*12+f*18.99840322+o*15.99491462+s*31.972071+n*14.003074+h*1.007825032+cl*34.96885268+p*30.97376163+0.00055
                                        ab_error=mz-mz_
                                        re_error=(ab_error)/mz*1000000
                                        re_error2=abs(re_error)
                                        if(re_error2<=error):
                                            mz_act.append(mz_)
                                            error_ab.append(ab_error)
                                            error_re.append(re_error)
                                            rdb.append(rdb_)
                                            formula="C"+str(c)
                                            formula2="C"+str(c)
                                            if(h!=0):
                                                formula+="H"
                                                formula2+="H"+str(h)
                                                formula3=formula+str(h+1)
                                                if(h!=1):
                                                    formula+=str(h)
                                            else:
                                                formula3=formula+"H"
                                            if(cl!=0):
                                                formula+="Cl"
                                                formula3+="Cl"
                                                formula2+="Cl"+str(cl)
                                                if(cl!=1):
                                                    formula+=str(cl)
                                                    formula3+=str(cl)
                                            if(f!=0):
                                                formula+="F"
                                                formula3+="F"
                                                formula2+="F"+str(f)
                                                if(f!=1):
                                                    formula+=str(f)
                                                    formula3+=str(f)
                                            if(n!=0):
                                                formula+="N"
                                                formula3+="N"
                                                formula2+="N"+str(n)
                                                if(n!=1):
                                                    formula+=str(n)
                                                    formula3+=str(n)
                                            if(o!=0):
                                                formula+="O"
                                                formula3+="O"
                                                formula2+="O"+str(o)
                                                if(o!=1):
                                                    formula+=str(o)
                                                    formula3+=str(o)
                                            if(p!=0):
                                                formula+="P"
                                                formula3+="P"
                                                formula2+="P"+str(p)
                                                if(p!=1):
                                                    formula+=str(p)
                                                    formula3+=str(p)
                                            if(s!=0):
                                                formula+="S"
                                                formula3+="S"
                                                formula2+="S"+str(s)
                                                if(s!=1):
                                                    formula+=str(s)
                                                    formula3+=str(s)
                                            formulas.append(formula)
                                            formulas2.append(formula2)
                                            formulas3.append(formula3)
                                            error_re2.append(re_error2)
    df=pd.DataFrame({"Formula":formulas,"m/z":mz_act,"Error(ppm)":error_re,"Error(mDa)":error_ab,"RDB":rdb,"abs_re_error":error_re2,"Formula_1":formulas2,"Formula_mole":formulas3})
    df=df.sort_values(by="abs_re_error",ascending=True)    
    df=df.reset_index(drop=True)
    df=df[["Formula","m/z","Error(ppm)","Error(mDa)","RDB","Formula_1","Formula_mole"]]
    return df
def formula_caculator(MS1spe,mz,error_ppm=5,iso_thre=0.8):
    MS1=MS1clean(MS1spe,mz,0.002)
    if(len(MS1)==0):
        return pd.DataFrame()
    cl_=is_Cl(MS1)
    if(cl_):
        df_formula=formula_cac_cl(mz,error_ppm) 
    else:
        df_formula=formula_cac(mz,error_ppm) 
    i=0
    iso_list=[]
    MS1=pd.DataFrame(MS1)
    MS1.columns=["mass","intensity"]
    while(i<len(df_formula)):
        MS1_exp=isotope_pattern(df_formula["Formula_1"][i])    
        iso_score=compare_isotope(MS1,MS1_exp)
        if(iso_score>iso_thre):
            i=i+1
            iso_list.append(iso_score)
        else:
            df_formula.drop([i],inplace=True)
            df_formula=df_formula.reset_index(drop=True)
    df_formula["iso_score"]=iso_list
    return df_formula        

def isotope_pattern(formula, thres=0.99):
    isotope = IsoSpecPy.IsoSpec.IsoFromFormula(formula, thres)
    isotope = isotope.getConfsNumpy()
    output = pd.DataFrame({'mass': isotope[0]+0.00055, 'intensity': np.exp(isotope[1])})
    return output

def compare_isotope(measured, expected, error=0.002):
    measured['intensity'] = measured['intensity']/measured['intensity'][0]
    expected['intensity'] = expected['intensity']/expected['intensity'][0]
    expected_m1 = sum(expected['intensity'][(expected['mass'] > expected['mass'][0] - error + 0.997) & (expected['mass'] < expected['mass'][0] + error + 1.006)])
    measured_m1 = sum(measured['intensity'][(measured['mass'] > measured['mass'][0] - error + 0.997) & (measured['mass'] < measured['mass'][0] + error + 1.006)])
    expected_m2 = sum(expected['intensity'][(expected['mass'] > expected['mass'][0] - error + 1.994) & (expected['mass'] < expected['mass'][0] + error + 2.013)])
    measured_m2 = sum(measured['intensity'][(measured['mass'] > measured['mass'][0] - error + 1.994) & (measured['mass'] < measured['mass'][0] + error + 2.013)])
    expected_m3 = sum(expected['intensity'][(expected['mass'] > expected['mass'][0] - error + 2.991) & (expected['mass'] < expected['mass'][0] + error + 3.018)])
    measured_m3 = sum(measured['intensity'][(measured['mass'] > measured['mass'][0] - error + 2.991) & (measured['mass'] < measured['mass'][0] + error + 3.018)])
    score = (1 - abs(expected_m1 - measured_m1)) * (1 - abs(expected_m2 - measured_m2)) * (1 - abs(expected_m3 - measured_m3))
    return score
def MS1clean(MS1spe,mz,error=0.002):
    while(MS1spe[0,0]<(mz-error)):
        MS1spe=np.delete(MS1spe,0,0)
        if(len(MS1spe)==0):
            break
    return MS1spe         
def is_Cl(MS1spe,error=0.002,tol=0.25):
    MS1=pd.DataFrame(MS1spe)
    MS1.columns=["mass","intensity"]
    MS1['intensity'] =MS1['intensity']/MS1['intensity'][0]
    MS1_m2 = sum(MS1['intensity'][(MS1['mass'] > MS1['mass'][0] - error + 1.994) & (MS1['mass'] < MS1['mass'][0] + error + 2.013)])
    if(MS1_m2<0.25):
        return 0
    else:
        return 1

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
# caculate the number and kind of atoms  changed after the transformtion, the transformed formula was also returned               
def trans_pfas(cac_formula,database_formula,num_thre):
    list_cac=formulatolist(cac_formula)
    list_data=formulatolist(database_formula)
    count1=0
    count2=0
    list3=[]
    for i in range(len(list_cac)):
        list3.append(list_cac[i]-list_data[i])
        count1=count1+abs(list3[i])
        if(count1>num_thre):
            return -1,-1,-1,""
        if(list3[i]!=0):
            count2=count2+1
    return count1,count2,abs(list3[3]),listtoformula(list3)

# judge the change of atom number is higher than num_thre or not
def trans_pfas2(cac_formula,database_formula,num_thre):
    list_cac=formulatolist(cac_formula)
    list_data=formulatolist(database_formula)
    count1=0
    list3=[]
    for i in range(len(list_cac)):
        list3.append(list_cac[i]-list_data[i])
        count1=count1+abs(list3[i])
        if(count1>num_thre):
            return 0
    return 1

# return formula change
def trans_pfas3(cac_formula,database_formula):
    list_cac=formulatolist(cac_formula)
    list_data=formulatolist(database_formula)
    list1=[]
    for i in range(len(list_cac)):
        list1.append(list_cac[i]-list_data[i])
    return listtoformula(list1)
    
        
        
    
    
           
                
