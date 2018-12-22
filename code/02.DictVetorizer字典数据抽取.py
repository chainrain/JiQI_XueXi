from sklearn.feature_extraction import DictVectorizer

def dictvec():
    """
    字典数据抽取:把字典中的一些类别数据,分别进行转换成特征
    :return:
    """
    # 实例化
    dict1 = DictVectorizer(sparse=True)  # 为false转numpy格式
    # 分析特征,转换为特征
    data = dict1.fit_transform([{'city': '北京','temperature': 100}, {'city': '上海','temperature':60}, {'city': '深圳','temperature': 30}])

    print(data)
    print(type(data))  # <class 'scipy.sparse.csr.csr_matrix'> 这个是sparse矩阵格式,节约内存,方便数据处理
    print(dict1.get_feature_names())  # 特征值

if __name__ == '__main__':
    dictvec()


"""
如果没有添加sparse的参数:dict1 = DictVectorizer()
那么它就是个sparse的矩阵格式:
对应的numpy数组为
 数值在数组   数值
   的位置
  (0, 1)	1.0
  (0, 3)	100.0
  (1, 0)	1.0
  (1, 3)	60.0
  (2, 2)	1.0
  (2, 3)	30.0
<class 'scipy.sparse.csr.csr_matrix'>


如果添加了sparse=False的参数:dict1 = DictVectorizer(sparse=False)
那么它就是个numpy数组格式
[[  0.   1.   0. 100.]
 [  1.   0.   0.  60.]
 [  0.   0.   1.  30.]]
<class 'numpy.ndarray'>

两个格式对应的是
"""