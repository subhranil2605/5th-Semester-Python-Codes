# Solving Schroedinger equation for the Anharmonic oscillator
# 3rd Problem

import numpy as np
import matplotlib.pyplot as plt

# constants
k = 100
h_cut = 197.3
m = 940

rmin = 1e-10
rmax = 10
N = 150

# creating grids
r = np.linspace(rmin, rmax, N)
d = r[1] - r[0]

# different values of 'b'
b_vals = [0, 10, 30]

# anharmonic potential function
v_func = lambda x, b: (1/2) * k * x**2 + (1/3) * b * x**3


# creating potential matrix
def potential_matrix(b):
    v = np.zeros((N, N))
    for i in range(N):
        v[i, i] = v_func(r[i], b)
    return v


pot_matrices_map = map(potential_matrix, b_vals)
pot_matrices_list = list(pot_matrices_map)


#Constructing Kinetic energy matrix
T = np.zeros((N, N), float)
for i in range(N):
    for j in range(N):
        if i == j:
            T[i,j] = -2.0
        elif np.abs(i-j) == 1.0:
            T[i,j] = 1.0
        else:
            T[i,j] = 0.0



hamiltonian = lambda potenital_matrix: (-(h_cut ** 2) / (2 * m * d ** 2)) * T + potenital_matrix

hamiltonians_map = map(hamiltonian, pot_matrices_list)
hamiltonians_list = list(hamiltonians_map)

# Storing eigen values and eigen vectors
eigen_values = []
eigen_vectors = []

for i in hamiltonians_list:
    val, vec = np.linalg.eig(i)
    eigen_values.append(val)
    eigen_vectors.append(vec)

# Showing the values of ground state energies for different values of "a"
for i in range(len(eigen_values)):
    print('For b = %d MeV, Ground state energy is : %0.2f MeV' % (b_vals[i], eigen_values[i][0]))



#Plotting the eigen functions
plt.plot(r, np.abs(eigen_vectors[0][:, 0]), '--', label=f"Ground State with $b={b_vals[0]}$")
plt.plot(r, np.abs(eigen_vectors[1][:, 0]), '.-', label=f"Ground State with $b={b_vals[1]}$")
plt.plot(r, np.abs(eigen_vectors[2][:, 0]), '.', label=f"Ground State with $b={b_vals[2]}$")

plt.title(r'Solution of : $ \frac{d^2y(r)}{dr^2}+\frac{2m}{\hbar^2}(E-V(r))y(r)=0$ with anharmonic oscillator potential : $V(r)=\frac{1}{2}kr^2+\frac{1}{3}br^3$')
plt.xlabel('r', size=14)
plt.ylabel('wavefunction $\psi(r)$', size=14)
plt.grid()
plt.legend()

plt.show()
