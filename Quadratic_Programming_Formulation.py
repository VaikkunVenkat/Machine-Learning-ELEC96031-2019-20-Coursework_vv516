from cvxopt import matrix
from cvxopt import solvers
import numpy as np
x_neg = np.array([[-2,-4] , [0,-5] , [1,-3] , [2,-2] , [4,-2],[8,1],[-3,-5] ])
y_neg = np.array([-1,-1,-1,-1,-1,-1,-1])
x_pos = np.array([[-3,0] , [0,0] , [-2,-2] , [2,2] , [4,1] , [0,2]])
y_pos = np.array([1,1,1,1,1,1])

X = np.vstack((x_pos, x_neg))
print(X)
y = np.concatenate((y_pos,y_neg))
N = len(y)

y = y.reshape(-1,1) * 1.
X_dash = y * X
P = np.dot(X_dash , X_dash.T) * 1.

print(P)

P = matrix(P, tc='d')
q = matrix(np.dot(-1,np.ones((N,1))) , tc='d')
G = matrix(np.dot(-1,np.eye((N))) , tc='d')
h = matrix(np.zeros((N,1)) , tc='d')
A = matrix(y.reshape(1,N) , tc = 'd')
b= matrix(0, tc = 'd')

sol = solvers.qp(P,q,G,h,A,b)
alphas = np.array(sol['x'])
print(sol['x'])
print(sol['primal objective'])

#w parameter in vectorized form
w = ((y * alphas).T @ X).reshape(-1,1)

#Selecting the set of indices S corresponding to non zero parameters
S = (alphas > 1e-4).flatten()

#Computing b
b = y[S] - np.dot(X[S], w)

#Display results
print('Alphas = ',alphas[alphas > 1e-4])
print('w = ', w.flatten())
print('b = ', b[0])



