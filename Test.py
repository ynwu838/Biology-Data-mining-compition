import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import missingno as msno
from matplotlib import pyplot as plt


def freture_counts(data, feature):
    data = data[feature].value_counts()
    data.to_csv(feature + ".csv")
    print(data)


def freture_describe(data, feature):
    print(data[feature].describe())

def jiaoji(lst1, lst2):
    rets = list(set(lst1).union(set(lst2)))
    return rets

def kinds_of_letter(Str):
    letters = []
    for letter in Str:
        if letter not in letters:
            letters.append(letter)
    return letters
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