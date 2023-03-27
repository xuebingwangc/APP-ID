# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 11:58:07 2021

@author: wxb1
"""

import pandas as pd
import numpy as np

def MSMSpre(str1,mz1,error=0.01,minmz=0,maxmz=2000):
    msmsspe1=MSMSclean(str1,mz1,error,minmz,maxmz)
    if(len(msmsspe1)==0):
        return np.zeros(shape=(0,0))
    msmsspe1=MSMSnoise(msmsspe1)         
    list1=[]
    list2=[]
    for i in range(len(msmsspe1)):
        if(msmsspe1[i][0]>100 or 82.96085-error<msmsspe1[i][0]<82.96085+error or 98.95577-error<msmsspe1[i][0]<98.95577+error or 68.99576-error<msmsspe1[i][0]<68.99576+error or 84.99067-error<msmsspe1[i][0]<84.99067+error):
            list1.append(msmsspe1[i][0])
            list2.append(msmsspe1[i][1])
    if(len(list1)==0):
        return np.zeros(shape=(0,0))
    else:
        msmsspe=np.zeros(shape=(len(list1),2))
        for j in range(len(msmsspe)):
            msmsspe[j][0]=list1[j]
            msmsspe[j][1]=list2[j]
        msmsspe[:,1]=1 
        return msmsspe
def MSMSpre1(str1,mz1,error=0.01,minmz=0,maxmz=2000):
    msmsspe1=MSMSclean(str1,mz1,error,minmz,maxmz)
    if(len(msmsspe1)==0):
        return np.zeros(shape=(0,0))
    msmsspe1=MSMSnoise(msmsspe1)             
    return msmsspe1
def MSMSpre2(str1,mz1,error=0.01,minmz=0,maxmz=2000):
    msmsspe1=MSMSclean(str1,mz1,error,minmz,maxmz)
    if(len(msmsspe1)==0):
        return np.zeros(shape=(0,0))
    msmsspe1=MSMSnoise(msmsspe1)         
    list1=[]
    list2=[]
    for i in range(len(msmsspe1)):
        if(msmsspe1[i][0]>100 or 82.96085-error<msmsspe1[i][0]<82.96085+error or 98.95577-error<msmsspe1[i][0]<98.95577+error or 68.99576-error<msmsspe1[i][0]<68.99576+error or 84.99067-error<msmsspe1[i][0]<84.99067+error or 79.95736-error<msmsspe1[i][0]<79.95736+error):
            list1.append(msmsspe1[i][0])
            list2.append(msmsspe1[i][1])
    if(len(list1)==0):
        return np.zeros(shape=(0,0))
    else:
        msmsspe=np.zeros(shape=(len(list1),2))
        for j in range(len(msmsspe)):
            msmsspe[j][0]=list1[j]
            msmsspe[j][1]=list2[j]
        msmsspe[:,1]=1 
        return msmsspe
 
def MSMSclean(str1,mz,error=0.01,minmz=0,maxmz=2000):
    msmsspe1=MSMSspe(str1)
    list1=[]
    list2=[]
    list3=[]
    list4=[]
    list5=[]
    list6=[]
    for i in range(len(msmsspe1)):
        if((msmsspe1[i][0]<(mz-10) or ((mz-error)<msmsspe1[i][0]<(mz+error))) and (minmz<msmsspe1[i][0]<maxmz)):
            list1.append(msmsspe1[i][0])
            list2.append(msmsspe1[i][1])
        else:
            list3.append(msmsspe1[i][0])
            list4.append(msmsspe1[i][1])
    if(len(list1)==0):
        return np.zeros(shape=(0,0))
    for i in range(len(list1)):
        flag=0
        for j in range(len(list3)):
            if(43.98983-error<(list3[j]-list1[i])<43.98983+error or 79.95682-error<(list3[j]-list1[i])<79.95682+error):
                flag=1
                break
        if(flag==0):
            list5.append(list1[i])
            list6.append(list2[i])    
    msmsspe=np.zeros(shape=(len(list5),2))
    for j in range(len(msmsspe)):
        msmsspe[j][0]=list5[j]
        msmsspe[j][1]=list6[j]
    return msmsspe

def MSMSnoise(spe,abu=0.05):
    if(len(spe)==1):
        return spe
    else:
        max1=max(spe[:,1])
        for i in range(len(spe)):
            spe[i,1]=spe[i,1]/max1
        j=0
        while(j<len(spe)):
            if(spe[j,1]>=abu):               
                j=j+1
            else:
                spe=np.delete(spe,j,0)
    return spe
def MSMSspe(str1):
    count=str1.count(":")
    msmsspe=np.zeros(shape=(count,2))
    strlist=str1.split(" ")
    for i in range(count):
        s=strlist[i].split(":")
        msmsspe[i][0]=eval(s[0])
        msmsspe[i][1]=eval(s[1])
    return msmsspe

def DP(str1,str2,error=0.01,minmz=0,maxmz=2000):
    spec1=MSMSspe(str1)
    spec2=MSMSspe(str2)
    if(len(spec1)==0 or len(spec2)==0):
        return 0 
    spec=spec1.copy()
    for i in range(len(spec2)):
        flag=1
        for j in range(len(spec1)):
            if (abs(spec2[i,0]-spec1[j,0])<=error):
                flag=0
                break
        if(flag==1):
            spec=np.row_stack((spec,spec2[i]))
    add=len(spec)-len(spec1)
    if(add==0):
        spec1_1=spec1.copy()
    else:
        spec1_1=np.row_stack(((spec1),np.zeros(shape=(add,2))))
    spec2_1=spec.copy()
    spec2_1[:,1]=0
    for i in range(len(spec2_1)):
        for j in range(len(spec2)):
            if (abs(spec2[j,0]-spec2_1[i,0])<=error and spec2[j,1]>spec2_1[i,1]):
                spec2_1[i,1]=spec2[j,1] 
    if(len(spec1_1)==1):
        max1=spec1_1[0,1]
    else:
        max1=max(spec1_1[:,1])
    if(len(spec2_1)==1):
        max2=spec2_1[0,1]
    else:
        max2=max(spec2_1[:,1])
    if(max1==0 or max2==0):
        return 0
    else:
        for i in range(len(spec1_1)):
            spec1_1[i,1]=spec1_1[i,1]/max1
        for j in range(len(spec2_1)):
            spec2_1[j,1]=spec2_1[j,1]/max2
    return (np.dot(spec1_1[:,1],spec2_1[:,1]))/np.sqrt(((np.dot(spec1_1[:,1],spec1_1[:,1]))*(np.dot(spec2_1[:,1],spec2_1[:,1])))) 
def HSS(str1,str2,mz1,mz2,error=0.01,minmz=0,maxmz=2000):
    spec1_0=MSMSspe(str1)
    spec2_0=MSMSspe(str2)
    if(len(spec1_0)==0 or len(spec2_0)==0):
        return 0 
    diff=mz2-mz1
    if(diff>=0):
        for i in range(len(spec2_0)):
            for j in range(len(spec1_0)):
                if (abs(spec2_0[i,0]-spec1_0[j,0]-diff)<=error):
                    spec2_0[i,0]=spec2_0[i,0]-diff
    else:
        for i in range(len(spec1_0)):
            for j in range(len(spec2_0)):
                if (abs(spec1_0[i,0]-spec2_0[j,0]+diff)<=error):
                    spec1_0[i,0]=spec1_0[i,0]+diff
    spec2=spec2_0.copy()
    spec1=spec1_0.copy()
    spec=spec1.copy()
    for i in range(len(spec2)):
        flag=1
        for j in range(len(spec1)):
            if (abs(spec2[i,0]-spec1[j,0])<=error):
                flag=0
                break
        if(flag==1):
            spec=np.row_stack((spec,spec2[i]))
    add=len(spec)-len(spec1)
    if(add==0):
        spec1_1=spec1.copy()
    else:
        spec1_1=np.row_stack(((spec1),np.zeros(shape=(add,2))))
    spec2_1=spec.copy()
    spec2_1[:,1]=0
    for i in range(len(spec2_1)):
        for j in range(len(spec2)):
            if (abs(spec2[j,0]-spec2_1[i,0])<=error and spec2[j,1]>spec2_1[i,1]):
                spec2_1[i,1]=spec2[j,1] 
    if(len(spec1_1)==1):
        max1=spec1_1[0,1]
    else:
        max1=max(spec1_1[:,1])
    if(len(spec2_1)==1):
        max2=spec2_1[0,1]
    else:
        max2=max(spec2_1[:,1])
    if(max1==0 or max2==0):
        return 0
    else:
        for i in range(len(spec1_1)):
            spec1_1[i,1]=spec1_1[i,1]/max1
        for j in range(len(spec2_1)):
            spec2_1[j,1]=spec2_1[j,1]/max2
    return (np.dot(spec1_1[:,1],spec2_1[:,1]))/np.sqrt(((np.dot(spec1_1[:,1],spec1_1[:,1]))*(np.dot(spec2_1[:,1],spec2_1[:,1]))))

def FR(str1,str2,mz1,mz2,error=0.01,minmz=0,maxmz=2000):
    spec1=MSMSpre(str1,mz1,error,minmz,maxmz)
    spec2=MSMSpre(str2,mz2,error,minmz,maxmz)
    if(len(spec1)==0 or len(spec2)==0):
        return 0
    count1=0
    for i in range(len(spec1)):
        for j in range(len(spec2)):
            if (abs(spec2[j,0]-spec1[i,0])<=error):
                count1=count1+1
                break
    count2=0
    for i in range(len(spec2)):
        for j in range(len(spec1)):
            if (abs(spec1[j,0]-spec2[i,0])<=error):
                count2=count2+1
                break
    return np.sqrt(count1/len(spec1))*np.sqrt(count2/ len(spec2))
def fragratio(str1,str2,mz1,mz2,error=0.01,minmz=0,maxmz=2000):
    spec1=MSMSpre1(str1,mz1,error,minmz,maxmz)
    spec2=MSMSpre1(str2,mz2,error,minmz,maxmz)
    if(len(spec1)==0 or len(spec2)==0):
        return 0
    count1=0
    for i in range(len(spec1)):
        for j in range(len(spec2)):
            if (abs(spec2[j,0]-spec1[i,0])<=error):
                count1=count1+1
                break
    count2=0
    for i in range(len(spec2)):
        for j in range(len(spec1)):
            if (abs(spec1[j,0]-spec2[i,0])<=error):
                count2=count2+1
                break
    return np.sqrt(count1/len(spec1))*np.sqrt(count2/ len(spec2))

def FNR(str1,str2,mz1,mz2,error=0.01,minmz=0,maxmz=2000):
    spec1=MSMSpre(str1,mz1,error,minmz,maxmz)
    spec2=MSMSpre(str2,mz2,error,minmz,maxmz)
    if(len(spec1)==0 or len(spec2)==0):
        return 0
    count1=0
    diff=mz2-mz1
    if(abs(diff)<error):
        for i in range(len(spec1)):
            for j in range(len(spec2)):
                if (abs(spec2[j,0]-spec1[i,0])<=error and abs(spec2[j,0]-mz2)>error):
                    count1=count1+1
                    break
        return count1/np.sqrt(len(spec1)*len(spec2))
    for i in range(len(spec1)):
        for j in range(len(spec2)):
            if (abs(spec2[j,0]-spec1[i,0])<=error or (abs(spec2[j,0]-spec1[i,0]-diff)<=error and abs(spec2[j,0]-mz2)>error and (mz1-spec1[i,0])>10 and not (43.98983-error<mz1-spec1[i,0]<43.98983+error or 79.95682-error<mz1-spec1[i,0]<79.95682+error))):
                count1=count1+1
                break
    count2=0
    diff=mz2-mz1
    for i in range(len(spec2)):
        for j in range(len(spec1)):
            if (abs(spec1[j,0]-spec2[i,0])<=error or (abs(spec2[i,0]-spec1[j,0]-diff)<=error and abs(spec2[i,0]-mz2)>error and (mz1-spec1[j,0])>10 and not (43.98983-error<mz1-spec1[j,0]<43.98983+error or 79.95682-error<mz1-spec1[j,0]<79.95682+error))):
                count2=count2+1
                break
    return np.sqrt(count1/len(spec1))*np.sqrt(count2/ len(spec2))

def spetostr(spe):
    str1=""
    for i in range(len(spe)):        
        if(i==0):
            str1=str1+str(spe[i][0])+":"+str(spe[i][1])
        else:
            str1=str1+" "+str(spe[i][0])+":"+str(spe[i][1])
    return str1
def spe_from_csv(file):
    df1=pd.read_csv(file)
    df1=df1.values
    return spetostr(df1)

def listtostr(list2,list3):
    str1=""
    for i in range(len(list2)):
        if(i==0):
            str1=str1+str(list2[i])+":"+str(round(list3[i],2))
        else:
            str1=str1+" "+str(list2[i])+":"+str(round(list3[i],2))
    return str1
def listtostr2(list1,list2,list3,list4):
    str1=""
    for i in range(len(list1)):
        if(i==0):
            str1=str1+str(list1[i])+":"+str(round(list2[i],1))+":"+str(list3[i])+":"+str(list4[i])
        else:
            str1=str1+" "+str(list1[i])+":"+str(round(list2[i],1))+":"+str(list3[i])+":"+str(list4[i])
    return str1