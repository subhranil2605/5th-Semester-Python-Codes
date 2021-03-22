# Programmed by Subhranil Sarkar
# 4th Problem

import numpy as np
import matplotlib.pyplot as plt

#constants
h = 1973
e = 3.795
m = 940e6
D = 0.755501
alpha = 1.44
r_0 = 0.131349

# Setting range of "r"
r_min = -5
r_max = 5
n = 100

# Creating grids
r = np.linspace(r_min, r_max, n)
d = r[1] - r[0]


# Creating potential matrix
def v_pot(x):
    r_prime = (x - r_0) / x
    q = 2 * alpha * r_prime
    result = D * (np.power(e, -q) - np.power(e, q))
    return result


V = np.zeros((n, n))
for i in range(n):
    V[i, i] = v_pot(r[i])

# Creating Kinteic matrix
K = np.eye(n)*(-2)
for i in range(n-1):
    K[i, i+1] = 1
    K[i+1, i] = 1

# Creating hamiltonian matrix
H = (-(h**2)/(2*m*d**2))*K + V

# Getting eigen values and eigen vectors
val, vec = np.linalg.eig(H)

ground_state_energy = val[1]
first_excited_state_energy = val[2]


print(f'Lowest Vibrational energy is : {ground_state_energy}')

plt.plot(r, vec[:, 1], linewidth=3)

plt.xlabel('r')
plt.ylabel('wavefunction $\psi(r)$')
plt.grid()
plt.title('S-wave Schrodinger equation for the vibration of the Hydrogen Atom')
plt.show()


