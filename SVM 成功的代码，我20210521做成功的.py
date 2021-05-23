# -*- coding: utf-8 -*-
"""
Created on Sun May 23 23:55:31 2021

@author: kz
"""

#SVM 成功的代码，我20210521做成功的
#导入包
import matplotlib.pyplot as plt
import numpy as pd
from sklearn import datasets, linear_model,model_selection,svm
import sklearn

#加载数据集的函数
def load_data_regression():
    diabetes = datasets.load_diabetes()
    return model_selection.train_test_split(datasets.data,diabetes.target,
                                             test_size=0.25,random_state=0)
                                             
#加载数据的函数
def load_data_classfication():
    iris=datasets.load_iris()
    X_train=iris.data
    y_train=iris.target
    return model_selection.train_test_split(X_train, y_train,test_size=0.25,
        random_state=0,stratify=y_train)

#线性分类SVM原型：
sklearn.svm.LinearSVC(penalty='l2',loss='squared_hinge',dual=True,tol=0.0001,C=1.0,
                      multi_class='ovr',fit_intercept=True,intercept_scaling=1,
                      class_weight=None,verbose=0,random_state=None,max_iter=1000)

#使用LinearSVC类考察线性分类支持向量机的预测能力，给出函数
def test_LinearSVC(*data):
    X_train,X_test,y_train,y_test=data
    cls=svm.LinearSVC()
    cls.fit(X_train,y_train)
    print('Coefficients:%s, intercept %s'%(cls.coef_,cls.intercept_))
    print('Score: %.2f' % cls.score(X_test, y_test))
    
#调用test_LinearSVC函数
X_train,X_test,y_train,y_test=load_data_classfication()
test_LinearSVC(X_train,X_test,y_train,y_test)