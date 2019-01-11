# -*- coding: utf-8 -*-
"""
优化算法
如何去求模型当中的W，使得损失最小？（目的是找到最小损失对应的W值）

线性回归经常使用的两种优化算法1.
正规方程

理解：X为特征值矩阵，y为目标值矩阵。直接求到最好的结果
缺点：当特征过多过复杂时，求解速度太慢并且得不到结果
"""

def linear1():
    """
    用正规方程的优化方法对波士顿房价进行预测
    :return:
    """
    # 1、获取数据
    from sklearn.datasets import load_boston
    boston = load_boston()
    
    # 2.划分数据集
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,random_state=22)
    
    # 3、标准化
    from sklearn.preprocessing import StandardScaler
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    
    # 4、线性回归预估器
    from sklearn.linear_model import LinearRegression
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)  # 训练
    
    # 5、得出预测结果和模型
    y_predict = estimator.predict(x_test)
    print("查看正规方程优化方法的预测结果：\n", y_predict)
    print("查看正规方程优化方法的权重系数：\n", estimator.coef_)
    print("查看正规方程优化方法的偏置：\n", estimator.intercept_)
    
    # 6、模型评估
    from sklearn.metrics import mean_squared_error
    error = mean_squared_error(y_test, y_predict)
    print("查看正规方程优化方法的均方误差为：\n", error)

def linear2():
    """
    用梯度下降的优化方法对波士顿房价进行预测
    :return:
    """
    # 1、获取数据
    from sklearn.datasets import load_boston
    boston = load_boston()
    
    # 2.划分数据集
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,random_state=22)
    
    # 3、标准化
    from sklearn.preprocessing import StandardScaler
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    
    # 4、梯度下降预估器流程
    from sklearn.linear_model import SGDRegressor
    estimator = SGDRegressor(tol=1e-3)
    estimator.fit(x_train, y_train)
    
    # 5、得出预测结果和模型
    y_predict = estimator.predict(x_test)
    print("查看正规方程优化方法的预测结果：\n", y_predict)
    print("查看正规方程优化方法的权重系数：\n", estimator.coef_)
    print("查看正规方程优化方法的偏置：\n", estimator.intercept_)
    
    # 6、模型评估
    from sklearn.metrics import mean_squared_error
    error = mean_squared_error(y_test, y_predict)
    print("查看正规方程优化方法的均方误差为：\n", error)

def linear3():
    """
    用岭回归对波士顿房价进行预测
    :return:
    """
    # 1、获取数据
    from sklearn.datasets import load_boston
    boston = load_boston()
    
    # 2.划分数据集
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,random_state=22)
    
    # 3、标准化
    from sklearn.preprocessing import StandardScaler
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    
    # 4、线性回归预估器流程
    #from sklearn.linear_model import Ridge
    #estimator = Ridge()
    #estimator.fit(x_train, y_train)
    
   
    from sklearn.externals import joblib
    # 模型保存,只有保存了模型,读取就行了
    #joblib.dump(estimator, "ridge67.pkl")
    # 加载和保存图形
    estimator = joblib.load("ridge67.pkl")
    
    
    # 5、得出预测结果和模型
    y_predict = estimator.predict(x_test)
    print("查看正规方程优化方法的预测结果：\n", y_predict)
    print("查看正规方程优化方法的权重系数：\n", estimator.coef_)
    print("查看正规方程优化方法的偏置：\n", estimator.intercept_)
    
    # 6、模型评估
    from sklearn.metrics import mean_squared_error
    error = mean_squared_error(y_test, y_predict)
    print("查看正规方程优化方法的均方误差为：\n", error)

    

if __name__ == "__main__":
    # 代码1：用正规方程的优化方法对波士顿房价进行预测
    # linear1()
    
    # 代码2：用梯度下降的优化方法对波士顿房价进行预测
    # linear2()
    
    # 代码3：用岭回归对波士顿房价进行预测
    linear3()
