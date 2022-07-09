import pandas as pd
import numpy as np
import math
import lightgbm
from sklearn.decomposition import PCA
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE



def cal_pccs(x, y):
    x = np.array(x)
    y = np.array(y)
    length = len(x)
    meanX = x.mean()
    meanY = y.mean()
    StdX = x.std()
    StdY = y.std()
    sum = 0
    for i in np.linspace(0, length - 1, length):
        i = int(i)
        sum = sum + (x[i] - meanX) * (y[i] - meanY)
    sum = sum / length
    result = sum / (StdX * StdY)
    return result


def kinds_of_letter(Str):
    letters = []
    for letter in Str:
        if letter not in letters:
            letters.append(letter)
    return letters


def jiaoji(lst1, lst2):
    rets = list(set(lst1).union(set(lst2)))
    return rets


def normalization(Data):
    Data = np.array(Data)
    Data = (Data - Data.mean()) / Data.std()
    return Data


def correlated_factor(Seq, numbers):
    global Index1
    global Index2
    Amino_acid = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
                  'X']
    Hydrophobicity = [0.62, -2.53, -0.78, -0.9, 0.29, -0.74, -0.85, 0.48, -0.4, 1.38, 1.06, -1.5, 0.64, 1.19, 0.12,
                      -0.18, -0.05, 0.81, 0.26, 1.08, 0]
    Hydrophilicity = [-0.5, 3, 0.2, 3, -1, 3, 0.2, 0, -0.5, -1.8, -1.8, 3, -1.3, -2.5, 0, 0.3, -0.4, -3.4, -2.3, -1.5,
                      0]
    SideChainMass = [15, 101, 58, 59, 47, 73, 72, 1, 82, 57, 57, 73, 75, 91, 42, 31, 45, 130, 107, 43, 50]
    # 数据标准化
    Hydrophilicity = normalization(Hydrophilicity)
    Hydrophobicity = normalization(Hydrophobicity)
    SideChainMass = normalization(SideChainMass)
    results = []
    results = []
    for i in np.linspace(1, numbers, numbers):
        i = int(i)
        result = []
        for j in np.linspace(0, len(Seq) - i - 1, len(Seq) - i):
            j = int(j)
            for index in np.linspace(0, len(Amino_acid) - 1, len(Amino_acid)):
                index = int(index)
                if Amino_acid[index] == Seq[j]:
                    Index1 = index
                    break;
            for index in np.linspace(0, len(Amino_acid) - 1, len(Amino_acid)):
                index = int(index)
                if Amino_acid[index] == Seq[j + i]:
                    Index2 = index
                    break;
            Value = (Hydrophobicity[Index1] - Hydrophobicity[Index2]) ** 2 + (
                    Hydrophilicity[Index1] - Hydrophilicity[Index2]) ** 2 + (
                            SideChainMass[Index1] - SideChainMass[Index2]) ** 2
            result.append(Value / 3)
        result = np.array(result)
        Theta = result.mean()
        results.append(Theta)
    return results


train_data = pd.read_csv('F:/data/final_dataset_train.tsv', sep='\t')
test_data = pd.read_csv('F:/data/mix_data_test.tsv', sep='\t')
labels = train_data["delta_g"]
train_data.drop(columns=["pdb","delta_g"],inplace=True)
test_data.drop(columns=["id"],inplace=True)
Data = pd.concat([train_data, test_data])
#-----------------------------------------antibody_seq_a-----------------------------------------------------------------
letters = []
Matrix = []
for item in Data["antibody_seq_a"]:
    temp = kinds_of_letter(item)
    letters = jiaoji(letters, temp)
letters.sort()
for item in Data["antibody_seq_a"]:
    Vector = []
    for letter in letters:
        Count=item.count(letter)
        Vector.append(Count)
    Matrix.append(Vector)

A = pd.DataFrame(Matrix)
Columns = []
for letter in letters:
    Columns.append("antibody_a_" + "_" + letter)
A.columns = Columns
print(A)
#-----------------------------------------antibody_seq_b-----------------------------------------------------------------
letters = []
Matrix = []
for item in Data["antibody_seq_b"]:
    temp = kinds_of_letter(item)
    letters = jiaoji(letters, temp)
letters.sort()
for item in Data["antibody_seq_b"]:
    Vector = []
    for letter in letters:
        Count=item.count(letter)
        Vector.append(Count)
    Matrix.append(Vector)

B= pd.DataFrame(Matrix)
Columns = []
for letter in letters:
    Columns.append("antibody_b_" + "_" + letter)
B.columns = Columns
print(B)
#-----------------------------------------antibody_seq_c-----------------------------------------------------------------
letters = []
Matrix = []
for item in Data["antigen_seq"]:
    temp = kinds_of_letter(item)
    letters = jiaoji(letters, temp)
letters.sort()
for item in Data["antigen_seq"]:
    Vector = []
    for letter in letters:
        Count=item.count(letter)
        Vector.append(Count)
    Matrix.append(Vector)

C= pd.DataFrame(Matrix)
Columns = []
for letter in letters:
    Columns.append("antigen_seq" + "_" + letter)
C.columns = Columns
print(C)
# ----------------------------------------------------------antigen_seq-----------------------------------------
Data = pd.concat([A, B, C], axis=1)
Data.to_csv("Data.csv")
border = 1706
traindata = Data[0:border]
testdata = Data[border:len(Data)]

Selection = LGBMRegressor()
Selection.fit(traindata, labels)
print(Selection.feature_importances_)
for i in np.linspace(0, len(Selection.feature_importances_) - 1, len(Selection.feature_importances_)):
    i = int(i)
    if Selection.feature_importances_[i] < 5:
        featurename = Data.columns[i]
        print(featurename)
        traindata.drop(featurename, axis=1, inplace=True)
        testdata.drop(featurename, axis=1, inplace=True)
print(traindata.columns)
print(testdata.columns)

# _----------------------------------------递归特征消除法-----------------------------------------------------------------------
'''selectmodel=LGBMRegressor(Random_state=2222)
Sel=RFE(selectmodel,58).fit(traindata,labels)
newData=Sel.transform(traindata)
print(Sel.support_)
for i in np.linspace(0,len(Sel.support_)-1,len(Sel.support_)):
    i=int(i)
    if Sel.support_[i]==False:
        print(Data.columns[i])
        traindata.drop(Data.columns[i], axis=1, inplace=True)
        testdata.drop(Data.columns[i], axis=1, inplace=True)'''

# --------------------------------------------feature_selection---------------------------------------------------------

model1 = LGBMRegressor(random_state=1, seed=2222,n_estimators=100)
model2 = XGBRegressor(random_state=1, seed=2222,n_estimators=100)

model2.fit(traindata, labels)
result = model2.predict(testdata)
result = pd.DataFrame(result)
result.columns = ["label"]
print(result)
result.to_csv("result2.csv", encoding='utf-8')