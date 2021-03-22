# This code is programmed by Subhranil Sarkar
# Finding the energy of the ground state of an atom and plot the corresponding wave function
# 18-March-2021
# 2nd problem of the practical of CC-H-T-11 of the CBCS syllabus

# importing required libraries
import numpy as np
import matplotlib.pyplot as plt


# constants
h = 1973
e = 3.795
m = 0.511e6

# Setting range of r
r_min = 1e-10
r_max = 10
n = 110

# values of a
a_values = [3, 5, 7]

# creating grid of points
r = np.linspace(r_min, r_max, n)

# setting the value of distance
d = r[1] - r[0]

# potential function using lambda
v_pot_func = lambda x, a: (- np.power(e, 2) / x) * np.exp(- x / a)


# Creating Potential matrix
def potential_matrix(a):
    v = np.zeros((n, n))
    for i in range(n):
        v[i, i] = v_pot_func(r[i], a)
    return v


pot_matrices_map = map(potential_matrix, a_values)
pot_matrices_list = list(pot_matrices_map)

# Creating Kinetic Matrix
K = np.eye(n) * (-2)
for i in range(n - 1):
    K[i, i + 1] = 1
    K[i + 1, i] = 1

hamiltonian = lambda potenital_matrix: (-(h ** 2) / (2 * m * d ** 2)) * K + potenital_matrix

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
    print('For a = %d A, Ground state energy is : %0.2f eV' % (a_values[i], eigen_values[i][1]))

plt.plot(r, np.abs(eigen_vectors[0][:, 1]), '--', label="Ground State, $a=3A$")
plt.plot(r, np.abs(eigen_vectors[1][:, 1]), '.-', label="Ground State, $a=5A$")
plt.plot(r, np.abs(eigen_vectors[2][:, 1]), '.', label="Ground State, $a=7A$")

plt.xlabel('r', size=14)
plt.ylabel('wavefunction $|\psi(r)|$', size=14)
plt.grid()
plt.legend()

plt.show()
