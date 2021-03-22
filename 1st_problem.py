# This code is programmed by Subhranil Sarkar
# Finding the energy of the ground state of an atom and plot the corresponding wave function
# 18-March-2021
# 1st problem of the practical of CC-H-T-11 of the CBCS syllabus

import numpy as np
import matplotlib.pyplot as plt

#constants
h = 1973
e = 3.795
m = 0.511e6

# Setting range of "r"
r_min = 1e-10
r_max = 10
n = 110

# Creating grids
r = np.linspace(r_min, r_max, n)
d = r[1] - r[0]

# Creating potential matrix
v_pot = lambda x: - np.power(e, 2) / x
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
print(val)

ground_state_energy = val[1]
first_excited_state_energy = val[2]

print(f'Ground State energy is : {ground_state_energy} and 1st excited state energy is : {first_excited_state_energy}')

plt.plot(r, np.abs(vec[:, 1]), label="Ground State", linewidth=3)
plt.plot(r, np.abs(vec[:, 2]), label="1st Excited State", linewidth=3)

plt.xlabel('r')
plt.ylabel('wavefunction $\psi(r)$')
plt.grid()
plt.legend()
plt.title('S-wave Schrodinger equation for the ground state and the first excited state of the Hydrogen Atom')
plt.show()
