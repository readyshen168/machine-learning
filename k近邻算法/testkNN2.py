import kNN2 as knn

dataSet, labels = knn.createDataSet()
k = 3
inX = [0, 0]
label, classCount = knn.classify0(inX, dataSet, labels, k)
print(label)
print(classCount)  # {'B':2, 'A': 1}
