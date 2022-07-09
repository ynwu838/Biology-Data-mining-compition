import pandas as pd
import numpy as np
import woe as WOE
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV

antibody_a=pd.read_csv("features/antibody_seq_a.csv",sep="\t")
antibody_a.drop(columns=["#"],inplace=True)
num=len(antibody_a.columns)
names=[]
for i in np.linspace(1,num,num):
    i=int(i)
    names.append("antibody_a-"+str(i))
antibody_a.columns=names
antibody_a=antibody_a[0:1884]
print(antibody_a)
#--------------------------------------antibody_a-------------------------------------------------
antibody_b=pd.read_csv("features/antibody_seq_b.csv",sep="\t")
antibody_b.drop(columns=["#"],inplace=True)
num=len(antibody_b.columns)
names=[]
for i in np.linspace(1,num,num):
    i=int(i)
    names.append("antibody_b-"+str(i))
antibody_b.columns=names
antibody_b=antibody_b[0:1884]
print(antibody_b)
#--------------------------------------antibody_b----------------------------------------------
antigen_seq=pd.read_csv("features/antigen_seq.csv",sep="\t")
antigen_seq.drop(columns=["#"],inplace=True)
num=len(antigen_seq.columns)
names=[]
for i in np.linspace(1,num,num):
    i=int(i)
    names.append("antigen-"+str(i))
antigen_seq.columns=names
antigen_seq=antigen_seq[0:1884]
print(antigen_seq)
Data=pd.concat([antibody_a,antibody_b,antigen_seq],axis=1)
print(Data)