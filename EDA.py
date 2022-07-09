import pandas as pd
import numpy as np
import pandas as pd
import  numpy as np
import missingno as msno
from matplotlib import pyplot as plt
def freture_counts (data,feature):
    data=data[feature].value_counts()
    data.to_csv(feature+".csv")
    print(data)
def freture_describe (data,feature):

    print(data[feature].describe())
train_data=pd.read_csv('F:/data/final_dataset_train.tsv', sep='\t')
test_data=pd.read_csv('F:/data/final_dataset_train.tsv', sep='\t')
Data=pd.concat([train_data,test_data])
print(train_data)
print("-----------------------数据的类型，名字和缺失值情况----------------------------------")
print(train_data.info())
picture=msno.bar(train_data, labels=True)
plt.savefig("missing")
print("--------------------Every feature-------------------------------------------------")
features=['pdb','antibody_seq_a','antibody_seq_b','antigen_seq','delta_g']
for feature in features:

    print( "feature name:"+feature)
    freture_counts(train_data,feature)
    freture_describe(train_data,feature)
    print('-------------------------------------------------------------------------------')

