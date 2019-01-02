import tensorflow as tf

"""
变量OP:
    1.变量OP能够持久化保存,普通张量OP是不行的
    2.当定义了一个变量OP的时候,一定要在会话当中运行初始化设置
"""
# a = tf.constant([1,2,3,4,5])
a = tf.constant(3.0,name='a')
b = tf.constant(4.0,name='b')
c = tf.add(a,b,name='add')

var = tf.Variable(tf.random_normal([2,3],mean=0.0,stddev=1.0),name='variable')
# print(type(a))
# print(type(var))

#  在开启会话前,要使用这个方法进行初始化,不然会报错:Attempting to use uninitialized value Variable
init_op = tf.global_variables_initializer()

# 开启会话
with tf.Session() as sess:
    # 运行初始化设置
    sess.run(init_op)

    """
    tensorboard,网页可视化工具
    把程序的图结构写入事件文件,graph:把指定图写入事件文件当中,
    这个文件可以用终端命令tensorboard --logdir="目录地址(注意,只是文件的目录,不是文件)"打开,然后终端回给个入口你进入后台查看
    重新生成的话,终端会按时间读取最近的一个图结构
    """
    filewriter = tf.summary.FileWriter('./test/',graph=sess.graph)

    print(sess.run([c,var]))