import kNN as k

dataSet, labels = k.createDataSet()
label_r = k.classify0([0, 0], dataSet, labels, 3)
print(label_r)
