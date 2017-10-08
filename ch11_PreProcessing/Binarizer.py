# -*- coding: utf-8 -*-
#二元化
from sklearn.preprocessing import Binarizer
x=[[1,2,3,4,5],
   [5,4,3,2,1],
   [3,3,3,3,3],
   [1,1,1,1,1]]
print("before transform:" ,x)
binarizer = Binarizer(threshold=2.5)
print("after transform:",binarizer.transform(x))
