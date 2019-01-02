import tensorflow as tf

def myregression():
    """
    自实现一个线性回归预测
    :return:
    """
    # 作用域.目的,让后台图标清晰明了
    with tf.variable_scope('data'):
        # 准备数据,x特征值 [100,10]  y目标值[100]
        x = tf.random_normal([100,1],mean=1.75,stddev=0.5,name='x_data')  # mean平均值,stddev方差
        # 矩阵相乘必须是二维的
        y_true = tf.matmul(x,[[0.7]]) + 0.8

    # 作用域.目的,让后台图标清晰明了
    with tf.variable_scope('model'):
        # 建立线性回归模型 1个特征,1个权重,一个偏置 y = x w + b
        weight = tf.Variable(tf.random_normal([1,1],mean=0.0 , stddev=1.0),name='w',trainable=True)  # trainable,指定之歌变量能跟着梯度下降一起优化
        # 优化
        bias = tf.Variable(0.0,name='b')

        y_predict = tf.matmul(x,weight) + bias

    # 作用域.目的,让后台图标清晰明了
    with tf.variable_scope('loss'):
        # 建立损失函数,均方误差
        loss = tf.reduce_mean(tf.square(y_true - y_predict))

    # 作用域.目的,让后台图标清晰明了
    with tf.variable_scope('optimizer'):
        # 梯度下降优化损失,如果GradientDescentOptimizer(0.1)过大,就会造成梯度爆炸
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss=loss)  # GradientDescentOptimizer(0.1)学习率不能拉太高0.1,0.001...(不要超过0.2)

    # 通过会话运行程序
    init_op = tf.global_variables_initializer()  # 定义一个初始化变量的OP
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)
        # 打印随机最先初始化的权重和偏置
        print('随机初始化的参数权重为: %f, 偏置为: %f' %(weight.eval(),bias.eval()))
        # 建立事件
        tf.summary.FileWriter('./test/',graph=sess.graph)
        # 循环运行优化,目标为权重0.7 偏置为0.8
        for i in range(2000):
            sess.run(train_op)

            print('第%d次优化的参数权重为: %f, 偏置为: %f' %(i,weight.eval(),bias.eval()))


if __name__ == '__main__':
    myregression()