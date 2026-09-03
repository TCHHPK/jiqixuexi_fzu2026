from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
n_samples = 400
X,Y= make_circles(400)
X.shape
Y.shape
plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()
from sklearn.datasets import make_moons
X,Y=make_moons()
X.shape
Y.shape
plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()