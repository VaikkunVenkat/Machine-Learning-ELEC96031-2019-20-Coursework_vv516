import numpy as np
import matplotlib.pyplot as plt

transition_probabilities_matrix_a1 = np.array([[0.8 , 0.1 , 0.1] , [0.6 , 0.3 , 0.1] , [0.1 , 0.5 , 0.4]])
transition_probabilities_matrix_a2 = np.array([[0.4 , 0.4 , 0.2] , [0.7 , 0.2 , 0.1] , [0.1 , 0.1 , 0.8]])
transition_probs = [transition_probabilities_matrix_a1 , transition_probabilities_matrix_a2]
N = 3
R1,R2,R3 = 1,2,-1
Rewards = np.array([R1,R2,R3])
gamma = 0.1
theta = 0.0000000005
delta = 0
num_actions = 2
V_s = np.zeros((N,1))   #Initialise the state value function
count = 0
error_gamma1 = []
while True :
    count = count + 1
    v = V_s
    V_stemp = np.zeros((N,1))
    for l in range(N):  #stick into the elements of V(s)
        V_max_comList = []
        for i in range(num_actions):  #2 different actions with associated prob table
            V=0
            for j in range(N):  # iteratively add for each s
                V = np.add(V, transition_probs[i][l,j]*(Rewards[l] + (gamma * V_s[j])))
            V_max_comList.append(V)
        V_stemp[l] = max(V_max_comList)
    V_s = V_stemp
    print("V(s) [i = " + str(count) + "] = ")
    print(V_s)
    norm_diff = (np.linalg.norm(np.subtract(V_s , v)))
    error_gamma1.append(norm_diff)
    print("Norm differance = " + str(norm_diff))
    if norm_diff < theta:
        break


plt.figure()
plt.plot(error_gamma1 , 'r' , label = 'error, ' + r"$\gamma = 0.9$")
#plt.plot(error_gamma2 , 'b' , label = 'error, ' + r"$\gamma = 0.1$")
plt.xlabel('iteration number')
plt.ylabel('error')
plt.title(r"$|v_{t+1}(s) - v_{t}(s)|$" + " error plot with iteration")
plt.grid()
plt.legend()
plt.show()