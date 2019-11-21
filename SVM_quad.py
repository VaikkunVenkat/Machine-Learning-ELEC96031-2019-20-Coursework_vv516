import numpy
from cvxopt import matrix

P = matrix(numpy.diag([1,0]), tc= 'd')
q = matrix(numpy.array([3,4]), tc='d')
G = matrix(numpy.array([[-1,0],[0,-1],[-1,-3],[2,5],[3,4]]), tc='d')
h = matrix(numpy.array([0,0,-15,100,80]), tc='d')
from cvxopt import solvers
sol = solvers.qp(P,q,G,h)

print(sol['x'])