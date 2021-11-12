import pandas as pd
import os
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

train = pd.read_csv('./data/train.csv')
drop_columns = ['PassengerId', 'Name', 'Cabin', 'Ticket']
# drop_columns = ['PassengerId','Name','Ticket']
train.drop(columns=drop_columns, inplace=True)
columns = train.columns
list_nan = list(train.isna().sum())


X = train[columns]
X.drop(columns=['Survived'], inplace=True)
Y = train['Survived']
c_factorize = columns[1:]
for c in c_factorize:
    val = train[c]
    X[c] = val.factorize()[0]

X = X.to_numpy()
Y = Y.to_numpy()


def imputDataFrame(pd):
    imputer = KNNImputer(n_neighbors=2, weights="uniform", missing_values=-1)
    return imputer.fit_transform(pd)


X = imputDataFrame(X)
df_temp = pd.DataFrame(X)
df_temp = list(df_temp[df_temp == -1].sum())
del df_temp
X = MinMaxScaler().fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(
    X_test, Y_test, test_size=1/3, random_state=42)
