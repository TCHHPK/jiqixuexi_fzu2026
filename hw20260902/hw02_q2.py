import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

x_train,y_train=make_circles(100)
x_train.shape
y_train.shape

x_test=np.random.uniform(-1, 1, (100, 2))
x_test.shape

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)
y_test=knn.predict(x_test)


from sklearn import datasets    
iris =datasets.load_iris()

data =iris.data
label=iris.target

ans_cnt=0

for i in range(len(data)):
#i=0
    x_test=data[i:i+1]
#label[i]
    x_train=np.delete(data,i,axis=0)
    y_train=np.delete(label,i)
#print(x_train.shape)
#print(y_train.shape)
#print(x_test.shape)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(data[i:i+1])
# print(label[i])
# print(y_pred)
    if(label[i] == y_pred[0]):
        ans_cnt+=1

accuracy=ans_cnt/150
print("correct: ",ans_cnt)
print(f"accuracy: {accuracy:.2%}")