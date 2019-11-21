import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


N = 15
K = 2
X = np.array([[-3,0] , [-2,-2] , [-1,-1] , [-1,-4] , [0,-3] , [1,-2], [1,-1] , [1,1], [1,3] , [3,0],[3,3], [4,0], [4,2], [5,3], [5,-2]])

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print("Number of iterations till convergence: " + str(kmeans.n_iter_ ))
print(kmeans.cluster_centers_)

Clusters = kmeans.cluster_centers_
midpoint = np.array([np.mean(Clusters[:,0] , axis=0) , np.mean(Clusters[:,1] , axis=0) ])
slope = np.divide(np.subtract(Clusters[0][0] , Clusters[1][0]) , np.subtract(Clusters[1][1] , Clusters[0][1]))
xLine = np.linspace(-4,5,20)
yLine = slope*(xLine-midpoint[0]) + midpoint[1];


y = np.array([-1,1])
plt.scatter(X[:,0] , X[:,1] , c = kmeans.labels_, cmap=plt.cm.Paired)
plt.plot(Clusters[:,0],Clusters[:,1] , 'k*' , label = "cluster centroids")
plt.plot(xLine,yLine)
plt.xlabel('x')
plt.ylabel('y')
#plt.title('Assign cluster groups to data points')
plt.legend()
plt.title('Finalized K-means clustering result')
plt.grid()

plt.show()

