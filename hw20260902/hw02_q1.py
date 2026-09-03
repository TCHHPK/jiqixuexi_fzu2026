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

plt.scatter(x_train[:,0],x_train[:,1],c=y_train,marker="o")
plt.scatter(x_test[:,0],x_test[:,1],c=y_test,marker="x")
plt.show()


