import numpy as np
import scipy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score

import matplotlib.pyplot as plt


df = pd.read_csv('/tmp/adult.csv', encoding="utf-8")
print(df.head())
print(df.o.value_counts())

df.o = df.o.astype(str).map({' <=50K': 0, ' >50K': 1 })
print(df.o.value_counts())

le = preprocessing.LabelEncoder()
for col in df.columns:
    if df[col].dtype == object:
        df[col] = le.fit_transform(df[col])
    else:
        pass

y = df.o
print(y.head())

x = df.drop('o', axis=1)
print(x.head())

seed = 5
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=seed)

rfc = RandomForestClassifier(n_estimators=475, min_samples_split=60, min_samples_leaf=10,
                             max_depth=17, criterion='entropy', class_weight=None)

rfc = rfc.fit(xtrain, ytrain)
# 二分类问题，用roc_auc_score进行评估
result = rfc.score(xtest, ytest)
print(result)

#print('所有树：%s' % rfc.estimators_)

print(rfc.classes_)
print(rfc.n_classes_)

print('判定结果：%s' % rfc.predict(xtest))
print('判定结果：%s' % rfc.predict_proba(xtest)[:,:])
print('判定结果：%s' % rfc.predict_proba(xtest)[:,1])

d1 = np.array(pd.Series(rfc.predict_proba(xtest)[:,1]>0.5).map({False: 0, True: 1 }))
d2 = rfc.predict(xtest)
np.array_equal(d1, d2)

print('正确率：%s' % roc_auc_score(ytest, rfc.predict_proba(xtest)[:,1]))

print('各feature的重要性：%s' % rfc.feature_importances_)

importances = rfc.feature_importances_
std = np.std([importances for tree in rfc.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
print('Feature ranking:')
for f in range(min(20, xtrain.shape[1])):
    print('%2d) %-*s %f' % (f +1, 30, xtrain.columns[indices[f]], importances[indices[f]]))
plt.figure()
plt.title('Feature importances')
plt.bar(range(xtrain.shape[1]), importances[indices], color='r', yerr=std[indices], align='center')
plt.xticks(range(xtrain.shape[1]), indices)
plt.xlim([-1, xtrain.shape[1]])
#plt.show()

#cross_val_score
#x 数据特征
#y 数据标签
#sorting: 调用方法
#cv：几折交叉验证
#n_jobs:
#
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, xtrain, ytrain)
print(scores.mean())

clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, xtrain, ytrain)
print(scores.mean())

#调优
print(rfc.get_params)

param_test1 = { 'n_estimators': range(25, 500, 25)}
gsearch1 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=100,
                                                         min_samples_leaf=20,
                                                         max_depth=8,
                                                         random_state=10),
                        param_grid=param_test1, scoring='roc_auc', cv=5)
gsearch1.fit(xtrain, ytrain)
print(gsearch1.best_params_, gsearch1.best_score_)

param_test2 = {'min_samples_split' : range(60,200,20), 'min_samples_leaf' : range(10,110,10)}
gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=475,
                                                         max_depth=8,
                                                         random_state=10),
                        param_grid=param_test2, scoring='roc_auc', cv=5)
gsearch2.fit(xtrain, ytrain)
print(gsearch2.best_params_, gsearch2.best_score_)

param_test3 = {'max_depth' : range(3,30,2)}
gsearch3 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=60,
                                                         min_samples_leaf=10,
                                                         n_estimators=475,
                                                         random_state=10),
                        param_grid=param_test3, scoring='roc_auc', cv=5)
gsearch3.fit(xtrain, ytrain)
print(gsearch3.best_params_, gsearch3.best_score_)

param_test4 = {'criterion':['gini','entropy'],'class_weight':[None, 'balanced']}
gsearch4 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=60,
                                                         min_samples_leaf=10,
                                                         n_estimators=475,
                                                         max_depth=17,
                                                         random_state=10),
                        param_grid=param_test4, scoring='roc_auc', cv=5)
gsearch4.fit(xtrain, ytrain)
print(gsearch4.best_params_, gsearch4.best_score_)

print('优化后的正确率：%s' % roc_auc_score(ytest, gsearch1.best_estimator_.predict_proba(xtest)[:,1]))

