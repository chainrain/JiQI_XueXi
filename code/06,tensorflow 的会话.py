"""
tensorflow 分为:前段系统和后端系统
前段系统用来定义程序的图的结构
后端系统用来运算图结构

会话的功能:
1.运行图的结构
2.分配资源计算
3.掌握资源(变量的资源,队列,线程)
"""
import tensorflow as tf

g = tf.Graph()
# print(g)
with g.as_default():
    c = tf.constant(11.0)
    # print(c.graph)


a = tf.constant(5.0)
b = tf.constant(6.0)
sum1 = tf.add(a,b)
# 会话,只能运行一个图,但可以在会话中指定运行其他图
graph = tf.get_default_graph()
with tf.Session(graph=g) as sess:  # 使用上下文管理器比较好,因为不需要close关闭会话
    print(sess.run(c))
    # 查看属性的内存空间,结果都在默认的内存空间里
    print(a.graph)
    print(sum1.graph)
    print(sess.graph)

# config=tf.ConfigProto(log_device_placement=True 看到程序在哪里,那个cpu运算
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:  # 使用上下文管理器比较好,因为不需要close关闭会话
    print(sess.run(sum1))
    # 查看属性的内存空间,结果都在默认的内存空间里
    print(a.graph)
    print(sum1.graph)
    print(sess.graph)

# ipython 操作数据
# import tensorflow
# a = tensorflow.constant(3.0)
# tensorflow.InteractiveSession
# tensorflow.InteractiveSession()
# a.eval()
# Out[6]: 3.0
