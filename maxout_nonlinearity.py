# 2x+1 , -x + 3

import numpy as np
import matplotlib.pyplot as plt

alpha1 , beta1 , alpha2 , beta2 = 3,1,-1,3
intersection = (beta2 - beta1) / (alpha1 - alpha2)
print(intersection)

x_1 = np.linspace(intersection,5,20)
x_2 = np.linspace(-2,intersection,20)
y_1 = alpha1*x_1 + beta1
y_2 = alpha2*x_2 + beta2

y_1_derivative = alpha1*np.ones((20,1))
y_2_derivative = alpha2*np.ones((20,1))

plt.figure()
plt.title('Example maxout non-linear activation function')
plt.plot(x_1,y_1,'r' , label=r"$ \theta(s) = \alpha_1s + \beta_1$" + r"$ , \alpha_1=3 , \beta_1 = 1$")
plt.plot(x_2 , y_2, 'g' , label=r"$ \theta(s) = \alpha_2s + \beta_2$" + r"$ , \alpha_2=-1 , \beta_2 = 3$")
plt.xlabel('s')
plt.ylabel(r"$\theta(s)$")
plt.grid()
plt.legend()

plt.figure()
plt.title('Example maxout non-linear activation derivative')
plt.plot(x_1,y_1_derivative,'r' , label=r"$ \theta^\prime(s) = \alpha_1$" + r"$ , \alpha_1=3$")
plt.plot(x_2 , y_2_derivative, 'g' , label=r"$ \theta^\prime(s) = \alpha_2$" + r"$ , \alpha_2=-1$")
plt.xlabel('s')
plt.ylabel(r"$\theta^\prime(s)$")
plt.grid()
plt.legend()





plt.show()