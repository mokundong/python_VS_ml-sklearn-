# -*- coding: utf-8 -*-
#独热码
from sklearn.preprocessing import OneHotEncoder
x=[[1,2,3,4,5],
   [5,4,3,2,1],
   [3,3,3,3,3],
   [1,1,1,1,1]]
print("before transform:" ,x)
encoder = OneHotEncoder(sparse=False)
encoder.fit(x)
print("active_features_:",encoder.active_features_)
print("feature_indices_:",encoder.feature_indices_)
print("n_values_:",encoder.n_values_)
print("after transform:",encoder.transform( [[1,2,3,4,5]]))
