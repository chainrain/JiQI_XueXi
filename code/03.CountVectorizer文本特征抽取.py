import jieba
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler  # 归一化
from sklearn.preprocessing import StandardScaler  # 标准化


def countvec():
    """
    对文本进行特征值化
    :return:
    """
    cv = CountVectorizer()  # countvectorizer没有sparse的参数

    # data = cv.fit_transform(['life is short,i like python',
    #                             'life is long,i dislike python'])

    data2 = cv.fit_transform(["人生 苦短，我喜欢 python", "人生漫长，不用 python"])  # 对于中文,默认不支持特征抽取,需要空格隔开单词,这里要用到jieba的包,pip install jieba

    print(data2)

    print(cv.get_feature_names())  # 统计所有文章当中不重复的词

    print(data2.toarray())  # 按照get_feature_names()的顺序,统计该文章出现这个词的顺序,返回0的表示没出现,1表示出现了

def cutword():

    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")

    con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")

    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")
    # 生成器:<generator object Tokenizer.cut at 0x7f6a13d2be08>
    # print(con1)
    # print(con2)
    # print(con3)

    # 转换成列表
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)
    # 一个生成器转为list列表,
    # print(content1)
    # print(content2)
    # print(content3)

    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)
    # 列表转字符串
    # print(c1)
    # print(c2)
    # print(c3)
    return c1, c2, c3


def hanzi():
    """
    中文特征值化
    :return:
    """
    c1,c2,c3 = cutword()

    cv = CountVectorizer()

    data = cv.fit_transform([c1,c2,c3])

    print(cv.get_feature_names())

    print(data.toarray())

    """
    特征结果:
    ['一种', '不会', '不要', '之前', '了解', '事物', '今天', '光是在', '几百万年', '发出', '取决于', '只用', '后天', '含义', '大部分', '如何', '如果', '宇宙', '我们', '所以', '放弃', '方式', '明天', '星系', '晚上', '某样', '残酷', '每个', '看到', '真正', '秘密', '绝对', '美好', '联系', '过去', '这样']
    [[0 0 1 0 0 0 2 0 0 0 0 0 1 0 1 0 0 0 0 1 1 0 2 0 1 0 2 1 0 0 0 1 1 0 0 0]
    [0 0 0 1 0 0 0 1 1 1 0 0 0 0 0 0 0 1 3 0 0 0 0 1 0 0 0 0 2 0 0 0 0 0 1 1]
    [1 1 0 0 4 3 0 0 0 0 1 1 0 1 0 1 1 0 1 0 0 1 0 0 0 1 0 0 0 2 1 0 0 1 0 0]]
    """

def tfidfvec():
    """
    中文特征值化,频率版,重要性
    :return: None
    """
    c1, c2, c3 = cutword()

    print(c1, c2, c3)

    tf = TfidfVectorizer()

    data = tf.fit_transform([c1, c2, c3])

    print(tf.get_feature_names())

    print(data.toarray())
    """
    特征结果
    ['一种', '不会', '不要', '之前', '了解', '事物', '今天', '光是在', '几百万年', '发出', '取决于', '只用', '后天', '含义', '大部分', '如何', '如果', '宇宙', '我们', '所以', '放弃', '方式', '明天', '星系', '晚上', '某样', '残酷', '每个', '看到', '真正', '秘密', '绝对', '美好', '联系', '过去', '这样']
    [[0.         0.         0.21821789 0.         0.         0.
    0.43643578 0.         0.         0.         0.         0.
    0.21821789 0.         0.21821789 0.         0.         0.
    0.         0.21821789 0.21821789 0.         0.43643578 0.
    0.21821789 0.         0.43643578 0.21821789 0.         0.
    0.         0.21821789 0.21821789 0.         0.         0.        ]
    [0.         0.         0.         0.2410822  0.         0.
    0.         0.2410822  0.2410822  0.2410822  0.         0.
    0.         0.         0.         0.         0.         0.2410822
    0.55004769 0.         0.         0.         0.         0.2410822
    0.         0.         0.         0.         0.48216441 0.
    0.         0.         0.         0.         0.2410822  0.2410822 ]
    [0.15698297 0.15698297 0.         0.         0.62793188 0.47094891
    0.         0.         0.         0.         0.15698297 0.15698297
    0.         0.15698297 0.         0.15698297 0.15698297 0.
    0.1193896  0.         0.         0.15698297 0.         0.
    0.         0.15698297 0.         0.         0.         0.31396594
    0.15698297 0.         0.         0.15698297 0.         0.        ]]
    """


def mm():
    """
    归一化处理
    比如说,几个特征同等重要的时候,进行归一化
    目的,让某一个特征对最终结果不会造成更大的影响

    数据标准化（归一化）处理是数据挖掘的一项基础工作，
    不同评价指标往往具有不同的量纲和量纲单位，这样的情况会影响到数据分析的结果，
    为了消除指标之间的量纲影响，需要进行数据标准化处理，以解决数据指标之间的可比性.
    原始数据经过数据标准化处理后，各指标处于同一数量级，适合进行综合对比评价.
    :return:
    """
    mm = MinMaxScaler()

    data = mm.fit_transform([[90,2,10,40],[60,4,15,45],[75,3,13,46]])

    print(data)

    """
    结果:
    [[1.         0.         0.         0.        ]
    [0.         1.         1.         0.83333333]
    [0.5        0.5        0.6        1.        ]]
    """


def stand():
    """
    标准化缩放,通过对原始数据进行变换把数据变换到均值为0,方差为1范围内

    标准化：将特征数据的分布调整成标准正太分布，也叫高斯分布，
    过程为两步：去均值的中心化（均值变为0）；方差的规模化（方差变为1）。
    :return:
    """
    std = StandardScaler()

    data = std.fit_transform([[1., -1., 3.], [2., 4., 2.], [4., 6., -1.]])

    print(data)

    """
    结果:
    [[-1.06904497 -1.35873244  0.98058068]
    [-0.26726124  0.33968311  0.39223227]
    [ 1.33630621  1.01904933 -1.37281295]]
    """


def im():
    """
    缺失值处理
    :return:NOne
    """
    # missing_values: 填补值   strategy: 按什么计算mean平均值   axis: 按行还是列
    # im = Imputer(missing_values='NaN', strategy='mean', axis=0)  # 这是老版本方法,已废除,不过还能用,就是有警告而已
    im = SimpleImputer(missing_values=np.nan, strategy='mean',fill_value=0)  # missing_values参数可不带,默认为NaN

    data = im.fit_transform([[1, 2], [np.nan, 3], [7, 6]])

    print(data)

    """
    结果:把NaN填补成4
    [[1. 2.]
    [4. 3.]
    [7. 6.]]
    """

if __name__ == '__main__':
    # countvec()
    # cutword()
    # hanzi()
    # tfidfvec()
    # mm()
    # stand()
    im()