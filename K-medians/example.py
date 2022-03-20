# https://github.com/UBC-MDS/KMediansPy
import numpy as np
from KMediansPy.distance import distance
from KMediansPy.KMedians import KMedians
from KMediansPy.summary import summary

X = np.array([[1, 2], [5, 4], [6, 7]])
medians = np.array([[1, 2], [5, 4], [6, 7]])
dist = distance(X, medians)
print(dist)


X = np.array([[1, 2], [5, 4], [6, 7], [-1, 2], [9, 8]])
k = 2
a = KMedians(X, 2)

X = np.array([[1, 2], [5, 4]])
med = np.array([[1, 2]])
labels = np.array([0, 0])
y = summary(X, med, labels)
print(y)
