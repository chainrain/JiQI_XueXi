# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 19:31:44 2019

@author: chain
"""


# estimator:[统计] 估计量；评价者

def knn_iris():
    """
    KNN算法对鸢尾花进行分类
    :return:
    """
    # 1.获取数据
    from sklearn.datasets import load_iris
    iris = load_iris()

    # 2.划分数据集:train:训练集,test:测试集
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)
    # print(x_train,x_test,y_train,y_test)

    # 3.标准化
    """
    fit(): Method calculates the parameters μ and σ and saves them as internal objects.
    解释：简单来说，就是求得训练集X的均值，方差，最大值，最小值,这些训练集X固有的属性。

    transform(): Method using these calculated parameters apply the transformation to a particular dataset.
    解释：在fit的基础上，进行标准化，降维，归一化等操作（看具体用的是哪个工具，如PCA，StandardScaler等）。

    fit_transform(): joins the fit() and transform() method for transformation of dataset.
    解释：fit_transform是fit和transform的组合，既包括了训练又包含了转换。
    transform()和fit_transform()二者的功能都是对数据进行某种统一处理（比如标准化~N(0,1)，将数据缩放(映射)到某个固定区间，归一化，正则化等）
    """
    from sklearn.preprocessing import StandardScaler
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.KNN预估器流程
    from sklearn.neighbors import KNeighborsClassifier
    # 4.1实例化预估器类
    estimator = KNeighborsClassifier(n_neighbors=3)
    # 4.2调用fit进行训练
    estimator.fit(x_train, y_train)
    # 4.3得出预测结果
    y_predict = estimator.predict(x_test)
    print("KNN算法对鸢尾花进行分类,预测结果：\n", y_predict)
    # 4.4模型评估
    print("KNN算法对鸢尾花进行分类,对比真实值和预测值:\n", y_predict == y_test)  # 1）对比真实值和预测值
    accuracy = estimator.score(x_test, y_test)  # 2）直接计算准确率
    print("KNN算法对鸢尾花进行分类,准确率为：\n", accuracy)


def knn_iris_gscv():
    """
    KNN算法对鸢尾花进行分类，添加模型选择与调优
    :return:
    """
    # 1.获取数据
    from sklearn.datasets import load_iris
    iris = load_iris()

    # 2.划分数据集:train:训练集,test:测试集
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    # 3.标准化
    """
    fit(): Method calculates the parameters μ and σ and saves them as internal objects.
    解释：简单来说，就是求得训练集X的均值，方差，最大值，最小值,这些训练集X固有的属性。

    transform(): Method using these calculated parameters apply the transformation to a particular dataset.
    解释：在fit的基础上，进行标准化，降维，归一化等操作（看具体用的是哪个工具，如PCA，StandardScaler等）。

    fit_transform(): joins the fit() and transform() method for transformation of dataset.
    解释：fit_transform是fit和transform的组合，既包括了训练又包含了转换。
    transform()和fit_transform()二者的功能都是对数据进行某种统一处理（比如标准化~N(0,1)，将数据缩放(映射)到某个固定区间，归一化，正则化等）
    """
    from sklearn.preprocessing import StandardScaler
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.KNN预估器流程
    from sklearn.neighbors import KNeighborsClassifier
    # 4.1实例化预估器类
    estimator = KNeighborsClassifier(n_neighbors=5)
    # 添加模型选择与调优
    # 准备要调节的参数
    param_dict = {"n_neighbors": [1, 3, 5]}
    from sklearn.model_selection import GridSearchCV
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)
    # 4.2调用fit进行训练
    estimator.fit(x_train, y_train)
    # 4.3得出预测结果
    y_predict = estimator.predict(x_test)
    print("KNN算法对鸢尾花进行分类，添加模型选择与调优,预测结果：\n", y_predict)
    # 4.4模型评估
    print("KNN算法对鸢尾花进行分类，添加模型选择与调优,对比真实值和预测值:\n", y_predict == y_test)  # 1）对比真实值和预测值
    accuracy = estimator.score(x_test, y_test)  # 2）直接计算准确率
    print("KNN算法对鸢尾花进行分类，添加模型选择与调优,准确率为：\n", accuracy)

    # 查看调优结果
    print("KNN算法对鸢尾花进行分类，添加模型选择与调优,最佳参数:\n", estimator.best_params_)
    print("KNN算法对鸢尾花进行分类，添加模型选择与调优,最佳结果:\n", estimator.best_score_)
    print("KNN算法对鸢尾花进行分类，添加模型选择与调优,最佳估计器:\n", estimator.best_estimator_)
    print("KNN算法对鸢尾花进行分类，添加模型选择与调优,交叉验证结果:\n", estimator.cv_results_)


def nb_news():
    """
    用朴素贝叶斯算法对新闻分类
    :return:
    """
    # 1.获取数据
    from sklearn.datasets import fetch_20newsgroups
    news = fetch_20newsgroups(subset="all")

    # 2.划分数据集
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)

    # 3.文本特征抽取
    from sklearn.feature_extraction.text import TfidfVectorizer
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4、朴素贝叶斯的预估器流程
    from sklearn.naive_bayes import MultinomialNB
    estimator = MultinomialNB()
    estimator.fit(x_train, y_train)

    # 5.得出预测结果
    y_predict = estimator.predict(x_test)
    print("预测结果: \n", y_predict)

    # 模型评估
    # 5.1对比真实值和预测值
    print("用朴素贝叶斯算法对新闻分类,对比真实值和预测值:\n", y_predict == y_test)  # 1）对比真实值和预测值
    accuracy = estimator.score(x_test, y_test)  # 2）直接计算准确率
    print("用朴素贝叶斯算法对新闻分类,准确率为：\n", accuracy)


def tree_iris():
    """
    决策树算法对鸢尾花进行分类
    :return:
    """
    # 1.获取数据
    from sklearn.datasets import load_iris
    iris = load_iris()

    # 2.划分数据集:train:训练集,test:测试集
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    # 3、决策树预估器流程
    # 3.1实例化预估器类
    from sklearn.tree import DecisionTreeClassifier
    estimator = DecisionTreeClassifier(criterion="gini")  # entropy:熵（热力学函数）
    # 3.2调用fit进行训练
    estimator.fit(x_train, y_train)
    # 3.3得出预测结果
    y_predict = estimator.predict(x_test)
    print("决策树算法对鸢尾花进行分类,预测结果：\n", y_predict)

    # 4）模型评估
    # 4.1 对比真实值和预测值
    print("决策树算法对鸢尾花进行分类,对比真实值和预测值:\n", y_predict == y_test)
    # 4.2 直接计算准确率
    accuracy = estimator.score(x_test, y_test)
    print("决策树算法对鸢尾花进行分类,准确率为：\n", accuracy)

    # 保存树到本地
    from sklearn.tree import export_graphviz
    export_graphviz(estimator, out_file="iris_tree67.dot", feature_names=iris.feature_names)


if __name__ == "__main__":
    # 代码1：KNN算法对鸢尾花进行分类
    knn_iris()

    # 代码2：KNN算法对鸢尾花进行分类，添加模型选择与调优
    # knn_iris_gscv()

    # 代码3：用朴素贝叶斯算法对新闻分类
    # nb_news()

    # 代码4：决策树算法对鸢尾花进行分类
    tree_iris()