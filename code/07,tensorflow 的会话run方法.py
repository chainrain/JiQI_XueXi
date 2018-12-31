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

a = tf.constant(5.0)
b = tf.constant(6.0)
sum1 = tf.add(a,b)
# 会话,只能运行一个图,但可以在会话中指定运行其他图
graph = tf.get_default_graph()

# 不是OP不能运行
var1 = 2
# var2 = 3
# sum2 = var1 + var2

# 在这里有重载机制,只要有一部分是OP,那么也可以运行
sum2 = a + var1
print(sum2)

# 训练模型
# plt = tf.placeholder(tf.float32,[2,3])  # 指定数组模型,行,列
plt = tf.placeholder(tf.float32,[None,3])  # 任意行数

# config=tf.ConfigProto(log_device_placement=True 看到程序在哪里,那个cpu运算
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:  # 使用上下文管理器比较好,因为不需要close关闭会话
    # print(sess.run([a,b,sum1]))
    # print(sess.run([sum2]))
    print(sess.run(plt,feed_dict={plt:[[1,2,3],[4,5,6],[2,3,4]]}))
    # 查看属性的内存空间,结果都在默认的内存空间里
    # print(a.graph)
    # print(sum1.graph)
    # print(sess.graph)


