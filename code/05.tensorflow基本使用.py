import tensorflow as tf
"""
Tensorflow的意思是: Tensor数据张量,flow流动
tensorflow属于计算密集型

理解:
IO密集型:Django/scrapy,请求网页http比较多,磁盘操作比较多
计算密集型:Tensorflow,用到CPU/GPU比较多

使用tensorflow模块时候,如果是直接pip安装的,可能会报编译警告,
解决方法一:
import os  
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息  
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error   
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error

解决方法二:
在: https://github.com/lakshayg/tensorflow-build 里,找到对应的包,下载拉到虚拟机中
然后源码安装: pip install xxxxxx.whl,就行了
"""

# 创建新的一张图包含了一组op和tensor,上下文环境,内存空间会再次分配一份给它
# 什么是op: 只要使用tensorflow定义的函数,都是OP
# tensor: 就指代的是数据,计算的化都是tensor计算


g = tf.Graph()
print(g)
with g.as_default():
    c = tf.constant(11.0)
    print(c.graph)

# 实现一个加法运算
a = tf.constant(5.0)
b = tf.constant(6.0)

sum1 = tf.add(a,b)
# print(sum1)

# 默认分配内存空间
graph = tf.get_default_graph()
print(graph)

# 会话
with tf.Session() as sess:
    print(sess.run(sum1))
    # 查看属性的内存空间,结果都在默认的内存空间里
    print(a.graph)
    print(sum1.graph)
    print(sess.graph)

