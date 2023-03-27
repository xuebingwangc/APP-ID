# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 15:18:44 2020

@author: Bing
"""

import  spectrum_featurization as spefea
import maching_learning as machlea
import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import numpy as np
import math
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.AtomPairs import Pairs
import matplotlib.pyplot as plt
import time

# import data
train_test=pd.read_csv("train_test.csv")
#get fingerprints of molecules,and remove unbalanced fingerprints <10% and >90%
mols=list((train_test["SMILES"]))
fp_train_test=np.zeros(shape=(len(train_test),8034))
for i in range(len(train_test)):
    fp_train_test[i]=machlea.get_cdk_fingerprints(mols[i])
fp_0=fp_train_test.copy()
fp_sum=[]
for i in range(8034):
    fp_num=0
    for j in range(len(train_test)):
        fp_num=fp_num+fp_train_test[j][i]
    fp_sum.append(fp_num)
index=[]
for i in range(8034):
    if(int(len(train_test)*0.1)<fp_sum[i]<int(len(train_test)*0.9)):
        index.append(i)
fp_train_test=fp_train_test[:,index] 
#try del same dimension
index2=[0]
for i in range(len(index)):
    flag=1
    for j in range(len(index2)):
        if(list(fp_train_test[:,i])==list(fp_train_test[:,index2[j]])):
            flag=0
            break
    if(flag==1):
        index2.append(i)
index3=[]
for i in index2:
    index3.append(index[i])
fp_train_test=fp_0[:,index3] 

#creat fragment kernel matrix on train_test data
fr_train_test=np.zeros(shape=(len(train_test),len(train_test)))
fnr_train_test=np.zeros(shape=(len(train_test),len(train_test)))
frag_train_test=np.zeros(shape=(len(train_test),120000))
for i in range(len(train_test)):
    for j in range(len(train_test)):
        fr_train_test[i][j]=spefea.FR(train_test["MSMS spectrum"][i],train_test["MSMS spectrum"][j],train_test["m/z"][i],train_test["m/z"][j])
        fnr_train_test[i][j]=spefea.FNR(train_test["MSMS spectrum"][i],train_test["MSMS spectrum"][j],train_test["m/z"][i],train_test["m/z"][j])       
for i in range(len(train_test)):
    spe_i=spefea.MSMSpre1(train_test["MSMS spectrum"][i],train_test["m/z"][i],minmz=50,maxmz=1250)
    if(len(spe_i)==0):
        continue
    for j in range(len(spe_i)):
        mz_j=int(spe_i[j,0]*100-5000+0.5)
        frag_train_test[i][mz_j]=spe_i[j,1] 
        
# train the dataset using different spectrum featurization method using svm
df_fr_svm_linear=machlea.mach_lea(fr_train_test,fp_train_test,machlea.svm_linear)
df_fnr_svm_linear=machlea.mach_lea(fnr_train_test,fp_train_test,machlea.svm_linear)
df_frag_svm_linear=machlea.mach_lea(frag_train_test,fp_train_test,machlea.svm_linear)
df_shuffle_fnr_svm_linear=machlea.shuffle(fnr_train_test,fp_train_test,machlea.svm_linear)

#visualization results using violinplot
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24, 16))
plt.violinplot
axes[0, 0].violinplot([ df_fr_svm_linear["acc"],df_fnr_svm_linear["acc"],df_frag_svm_linear["acc"], df_shuffle_fnr_svm_linear["acc"]], showmeans=False, showmedians=True)
axes[0, 1].violinplot([df_fr_svm_linear["pre"],df_fnr_svm_linear["pre"], df_frag_svm_linear["pre"],df_shuffle_fnr_svm_linear["pre"]], showmeans=False, showmedians=True)
axes[1, 0].violinplot([df_fr_svm_linear["rec"], df_fnr_svm_linear["rec"],df_frag_svm_linear["rec"], df_shuffle_fnr_svm_linear["rec"]], showmeans=False, showmedians=True)
axes[1, 1].violinplot([df_fr_svm_linear["f1"],df_fnr_svm_linear["f1"],df_frag_svm_linear["f1"], df_shuffle_fnr_svm_linear["f1"]], showmeans=False, showmedians=True)
font={"family":"Arial","weight":"normal","size":30}   
axes[0, 0].set_ylabel('Accuracy',font)
axes[0, 1].set_ylabel('Precision',font)
axes[1, 0].set_ylabel('Recall',font)
axes[1, 1].set_ylabel('F1 Score',font)
# plt.setp(axes, xticklabels=["","FR","",'FNR',"","Frag","","Random"])
xticklabels=["","FR","",'FNR',"","Frag","","Shu"]
yticklabel1=["","0.4","0.5","0.6","0.7","0.8","0.9","1.0"]
yticklabel2=["","0.0","0.2","0.4","0.6","0.8","1.0"]
axes[0, 0].set_xticklabels(xticklabels,fontproperties="Arial",size=30)
axes[0, 1].set_xticklabels(xticklabels,fontproperties="Arial",size=30)
axes[1, 0].set_xticklabels(xticklabels,fontproperties="Arial",size=30)
axes[1, 1].set_xticklabels(xticklabels,fontproperties="Arial",size=30)
axes[0, 0].set_yticklabels(yticklabel1,fontproperties="Arial",size=30)
axes[0, 1].set_yticklabels(yticklabel2,fontproperties="Arial",size=30)
axes[1, 0].set_yticklabels(yticklabel2,fontproperties="Arial",size=30)
axes[1, 1].set_yticklabels(yticklabel2,fontproperties="Arial",size=30)
plt.show()
 
# compare different maching learning algorithm 

df_svm=machlea.mach_lea(fnr_train_test,fp_train_test,machlea.svm_linear)
df_logi=machlea.mach_lea(fnr_train_test,fp_train_test,machlea.logi)
df_bay=machlea.mach_lea(fnr_train_test,fp_train_test,machlea.bay)
df_dec=machlea.mach_lea(fnr_train_test,fp_train_test,machlea.dec)
df_ran=machlea.mach_lea(fnr_train_test,fp_train_test,machlea.ran)
df_knn=machlea.mach_lea(fnr_train_test,fp_train_test,machlea.knn5)
df_ann=machlea.mach_lea(fnr_train_test,fp_train_test,machlea.ann)
df_shuffle_fnr_svm_linear=machlea.shuffle(fnr_train_test,fp_train_test,machlea.svm_linear)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24, 16))
plt.violinplot
axes[0, 0].violinplot([df_svm["acc"], df_logi["acc"], df_bay["acc"], df_dec["acc"], df_ran["acc"], df_knn["acc"], df_ann["acc"], df_shuffle_fnr_svm_linear["acc"]], showmeans=False, showmedians=True)
axes[0, 1].violinplot([df_svm["pre"], df_logi["pre"],df_bay["pre"],df_dec["pre"],df_ran["pre"],df_knn["pre"],df_ann["pre"],df_shuffle_fnr_svm_linear["pre"]], showmeans=False, showmedians=True)
axes[1, 0].violinplot([df_svm["rec"], df_logi["rec"], df_bay["rec"], df_dec["rec"], df_ran["rec"], df_knn["rec"], df_ann["rec"], df_shuffle_fnr_svm_linear["rec"]], showmeans=False, showmedians=True)
axes[1, 1].violinplot([df_svm["f1"], df_logi["f1"], df_bay["f1"], df_dec["f1"], df_ran["f1"], df_knn["f1"], df_ann["f1"], df_shuffle_fnr_svm_linear["f1"]], showmeans=False, showmedians=True)
font={"family":"Arial","weight":"normal","size":30}   
axes[0, 0].set_ylabel('Accuracy',font)
axes[0, 1].set_ylabel('Precision',font)
axes[1, 0].set_ylabel('Recall',font)
axes[1, 1].set_ylabel('F1 Score',font)
xticklabels=["",'SVM',"LOG","BAY","DEC","RAN","KNN","ANN","Shu"]
yticklabel1=["","0.4","0.5","0.6","0.7","0.8","0.9","1.0"]
yticklabel2=["","0.0","0.2","0.4","0.6","0.8","1.0"]
axes[0, 0].set_xticklabels(xticklabels,fontproperties="Arial",size=30)
axes[0, 1].set_xticklabels(xticklabels,fontproperties="Arial",size=30)
axes[1, 0].set_xticklabels(xticklabels,fontproperties="Arial",size=30)
axes[1, 1].set_xticklabels(xticklabels,fontproperties="Arial",size=30)
axes[0, 0].set_yticklabels(yticklabel1,fontproperties="Arial",size=30)
axes[0, 1].set_yticklabels(yticklabel2,fontproperties="Arial",size=30)
axes[1, 0].set_yticklabels(yticklabel2,fontproperties="Arial",size=30)
axes[1, 1].set_yticklabels(yticklabel2,fontproperties="Arial",size=30)
plt.show()

# import validation data
validation=pd.read_csv("validation.csv")
# return a list of top n and a list of formula/SMILES which is not found in pubchem
listtop=[]
listNA=[]
for i in range(len(validation)):
    print("validation progress",i,"of",len(validation))
    list_smiles=machlea.searchpubchem(validation["Formula"][i])
    if(list_smiles==None):
        listNA.append(validation["Formula"][i])
        continue
    for l in range(len(list_smiles)):
        list_smiles[l]=machlea.iso(list_smiles[l])
    list_smiles=list(set(list_smiles)) 
    if(validation["SMILES"][i] not in list_smiles):
        listNA.append(validation["SMILES"][i])
        continue
    fnr_=np.zeros(shape=(1,len(train_test)))
    for j in range(len(train_test)):
        fnr_[0][j]=spefea.FNR(train_test["MSMS spectrum"][j],validation["MSMS spectrum"][i],train_test["m/z"][j],validation["m/z"][i])
    fp_pre=np.zeros(shape=(1,fp_train_test.shape[1]))         
    for k in range(fp_train_test.shape[1]):
        fp_pre[0,k]=machlea.svm_(fnr_train_test,fp_train_test[:,k],fnr_)
    list_score=[]
    for m in range(len(list_smiles)):        
        fp_cac=np.array([machlea.get_cdk_fingerprints(list_smiles[m])])
        fp_cac=fp_cac[:,index3]
        score_=machlea.score(fp_pre,fp_cac,df_fnr_svm_linear)
        list_score.append(score_)
    df_fp=pd.DataFrame({"SMILES":list_smiles,"score":list_score})
    df_fp=df_fp.sort_values(by="score",ascending=False)    
    df_fp=df_fp.reset_index(drop=True)
    list_smiles2=list(df_fp["SMILES"])
    listtop.append(list_smiles2.index(validation["SMILES"][i])+1)
    
df_topn=pd.DataFrame({"SMILES":validation["SMILES"],"top":listtop})
top1=0
top3=0
top5=0
for i in listtop:
    if(i<6):
        top5+=1
        if(i<4):
            top3+=1
            if(i==1):
                top1+=1
print("The top1 rate is %.1f%%"%(top1/len(validation)*100),".",sep="")
print("The top3 rate is %.1f%%"%(top3/len(validation)*100),".",sep="")
print("The top5 rate is %.1f%%"%(top5/len(validation)*100),".",sep="")

                
    
    
        
        
        
        
        
        