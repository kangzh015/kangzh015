# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Lasso 成功的代码20210523，树志帮助做成功的

#导入包
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model,discriminant_analysis,model_selection

#加载数据集的函数
def load_data():
    # iris = datasets.load_iris()
    diabetes = datasets.load_iris()
    # return model_selection.train_test_split(iris.data,iris.target,
                                            #  test_size=0.25,random_state=0)
    return model_selection.train_test_split(diabetes.data,diabetes.target,
                                             test_size=0.25,random_state=0)

#使用Lasso的函数
def test_Lasso(*data):
    X_train,X_test,y_train,y_test=data
    regr = linear_model.Lasso()
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept %.2f'%(regr.coef_,regr.intercept_))
    print("Residual sum of squares: %.2f"% np.mean((regr.predict(X_test) - y_test) **2))
    print('Score: %.2f' % regr.score(X_test, y_test))
X_train,X_test,y_train,y_test=load_data()
test_Lasso(X_train,X_test,y_train,y_test)

#检验不同α值对于预测性能的影响，给出测试函数
def test_Lasso_alpha(*data):
    X_train, X_test, y_train, y_test = data
    alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
    scores = []
    for i,alpha in enumerate(alphas):
        regr = linear_model.Lasso(alpha=alpha)
        regr.fit(X_train, y_train)
        scores.append(regr.score(X_test, y_test))
    ## 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alphas, scores)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"score")
    ax.set_xscale('log')
    ax.set_title("Lasso")
    plt.show()

#调用该函数
X_train, X_test, y_train, y_test = load_data()
test_Lasso_alpha(X_train, X_test, y_train, y_test)

#结果是当α超过1之后，随着α的增长，预测性能急剧下降。
