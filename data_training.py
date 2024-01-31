import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import json


# Load data
data = pd.read_csv('preprocess.csv')

X = data.drop('Survived',axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=data['Survived'],test_size=0.2,random_state=248)

encode_geneder = LabelEncoder()
encode_embarked = LabelEncoder()

X_train['Sex'] = encode_geneder.fit_transform(X_train['Sex'])
X_test['Sex'] = encode_geneder.transform(X_test['Sex'])

X_train['Embarked'] = encode_embarked.fit_transform(X_train['Embarked'])
X_test['Embarked'] = encode_embarked.transform(X_test['Embarked'])

scale = MinMaxScaler()
scaled_train_df = pd.DataFrame(scale.fit_transform(X_train[['Age','Fare']]),columns=['Age','Fare'])
scaled_test_df = pd.DataFrame(scale.transform(X_test[['Age','Fare']]),columns=['Age','Fare'])
X_train = X_train.drop(['Age','Fare'],axis=1)
X_test = X_test.drop(['Age','Fare'],axis=1)

X_train['Age'] = scaled_train_df['Age'].values
X_train['Fare'] = scaled_train_df['Fare'].values

X_test['Age'] = scaled_test_df['Age'].values
X_test['Fare'] = scaled_test_df['Fare'].values

modelKNN = KNeighborsClassifier(n_neighbors = 9, weights='distance')
modelKNN.fit(X_train, y_train)
predictionsKNN = modelKNN.predict(X_test)
accuracyKNN = metrics.accuracy_score(y_test, predictionsKNN)

modelLogReg = LogisticRegression()
modelLogReg.fit(X_train, y_train)
predictionsLogReg = modelLogReg.predict(X_test)
accuracyLogReg = modelLogReg.score(X_test, y_test)

probsKNN = modelKNN.predict_proba(X_test)[:, 1]
probsLogReg = modelLogReg.predict_proba(X_test)[:, 1]

fprLR, tprLR, thresholdsLR = metrics.roc_curve(y_test, probsLogReg)
fprKNN, tprKNN, thresholdsKNN = metrics.roc_curve(y_test, probsKNN)

fig = plt.figure()
axes = fig.add_axes([0,0,1,1])
axes.plot(fprLR, tprLR, label = "LogReg")
axes.plot(fprKNN, tprKNN, label = "KNN")
axes.set_xlabel("False positive rate")
axes.set_ylabel("True positive rate")
axes.set_title("ROC Curve for KNN, Logistic regression, Dummy")
axes.grid(which = 'major', c='#cccccc', linestyle='--', alpha=0.5)
axes.legend(shadow=True)
plt.savefig('ROC.png', dpi=120)

auc_logistic_regression = metrics.auc(fprLR, tprLR)
auc_knn                 = metrics.auc(fprKNN, tprKNN)


with open("metrics.json", 'w') as outfile:
        json.dump(
        	{ 
        	  "accuracy_KNN"                   : accuracyKNN,
        	  "accuracy_logistic-regression"   : accuracyLogReg,
        	  "AUC_logistic-regression"        : auc_logistic_regression,
        	  "AUC_KNN"                        : auc_knn}, 
        	  outfile
        	)