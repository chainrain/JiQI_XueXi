from sklearn.feature_extraction.text import CountVectorizer

# 实例化CountVectorizer
vector = CountVectorizer()

response = vector.fit_transform(['life is short,i like python',
                                'life is long,i dislike python'])

print(response)
"""
结果:未知意思
  (0, 5)	1
  (0, 3)	1
  (0, 6)	1
  (0, 1)	1
  (0, 2)	1
  (1, 0)	1
  (1, 4)	1
  (1, 5)	1
  (1, 1)	1
  (1, 2)	1
  """

print(vector.get_feature_names())
"""
结果:特征名词吧,就是不重复的名词
['dislike', 'is', 'life', 'like', 'long', 'python', 'short']
"""

print(response.toarray())
"""
结果:一个二维数组
[[0 1 1 1 0 1 1]
 [1 1 1 0 1 1 0]]
 """

