import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

# Plot the maximum margin separating hyperplane within a two-class
# separable dataset using a Support Vector Machine classifier
# with linear kernel

np.random.seed(0)
X = np.array([[-3,0] , [0,0] , [-2,-2] , [2,2] , [4,1] , [0,2] ,[-2,-4] , [0,-5] , [1,-3] , [2,-2] , [4,-2] , [8,1] , [-3,-5]])
y = np.array([1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1])

clf = svm.SVC(kernel= 'linear',C=1)
clf.fit(X,y)

#get the separating hyperplace
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

#Plot the parallels and the separating hyperplace that pass through the support vectors:
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])


plt.scatter(X[:,0] , X[:,1] , c=y, cmap=plt.cm.Paired)
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('SVM plot, Q4a')

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,linewidth=1, facecolors='none', edgecolors='k')

print("Support vectors are: " + str(clf.support_vectors_))
print("Indices of support vectors are: " + str(clf.support_) )
print("Number of support vectors for each class: " + str(clf.n_support_))
print("w = " + str(clf.coef_))
print("b = " + str(clf.intercept_))
print()
plt.show()