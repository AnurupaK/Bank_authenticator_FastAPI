import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import warnings
warnings.filterwarnings("ignore")


bank = pd.read_csv('BankNote_Authentication.csv')
df = bank.copy()

X = df.drop(columns='class',axis=1)
Y = df['class']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.3, stratify=Y, random_state=3)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier()

model.fit(X_train,Y_train)

# pred_train = model.predict(X_train)
# print(accuracy_score(pred_train,Y_train))
#
# pred_test = model.predict(X_test)
# print(accuracy_score(pred_test,Y_test))


pickle.dump(model,open('classifier.pkl','wb'))
