import numpy as np
import matplotlib.pyplot as plt
N = 15
K = 2
X = np.array([[-3,0] , [-2,-2] , [-1,-1] , [-1,-4] , [0,-3] , [1,-2], [1,-1] , [1,1], [1,3] , [3,0],[3,3], [4,0], [4,2], [5,3], [5,-2] ])
Clusters  = np.array([[-2,-3] , [0,2]])
Clusters_initial = Clusters
#plt.plot(Clusters_initial[:,0],Clusters_initial[:,1] , 'r*')
midpoint = np.array([np.mean(Clusters[:,0] , axis=0) , np.mean(Clusters[:,1] , axis=0) ])
slope = np.divide(np.subtract(Clusters[0][0] , Clusters[1][0]) , np.subtract(Clusters[1][1] , Clusters[0][1]))
xLine = np.linspace(-7,7,20)
yLine = slope*(xLine-midpoint[0]) + midpoint[1];
#Begin step2 of K-means
for l in range(5):
    class_points = []
    for i in range(N):
        distance = []
        for j in range(K):
            distance.append(np.power(np.linalg.norm(np.subtract(X[i,:] , Clusters[j,:])) ,2))
        print(distance)
        class_points.append(distance.index(min(distance)))

    class0Labels = [index for index, value in enumerate(class_points) if value == 0]
    class1Labels = [index for index, value in enumerate(class_points) if value == 1]

    Clusters[0,:] = np.mean(X[class0Labels,:], axis=0)
    Clusters[1,:] = np.mean(X[class1Labels,:] , axis=0)


    print(Clusters)

#Draw perpendicular bisector:

y = np.array([-1,1])
plt.scatter(X[:,0] , X[:,1] , c = class_points, cmap=plt.cm.Paired)
plt.plot(Clusters[:,0],Clusters[:,1] , 'k*')
#plt.plot(xLine,yLine)
plt.xlabel('x')
plt.ylabel('y')
#plt.title('Assign cluster groups to data points')
plt.title('Finalized K-means clustering result')
plt.grid()

plt.show()


