import tensorflow as tf

# 开启交互式会话
tf.InteractiveSession()

# 固定值张量
zeros = tf.zeros([3,4],tf.float32)
# print(zeros.eval())
# print(zeros.dtype)


a = tf.cast([[1,2,3],[4,5,6]],tf.float32)
# print(a.eval())
# print(a.dtype)

# 合并
one = [[1,2,3],[4,5,6]]
two = [[7,8,9],[10,11,12]]
three = tf.concat([one,two],axis=1)
print(three.eval())
print(three.dtype)

"""
tensorflow的其他api
https://tensorflow.google.cn/api_docs/python/
"""