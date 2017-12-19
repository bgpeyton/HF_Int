#import math
#import numpy as np
#import scipy.special as sps
#import scipy.linalg as spl
from ints import *

def P_mat(molecule, basis, c_norm):
    """Returns density matrix P"""
    P = np.zeros([molecule.shape[0]*basis.shape[1],molecule.shape[0]*basis.shape[1]])
    for i in range (0,molecule.shape[0]*basis.shape[1]):
        for j in range (0,molecule.shape[0]*basis.shape[1]):
            for m in range (0,1): # M is number of occuppied orbitals, should automate
                P[i,j] += c_norm[i,m]*c_norm[j,m]
    return P

molecule = np.array([[0,0,0],[1.4,0,0]])
basis = np.array([[5.447178, 0.824547, 0.183192000],[5.447178, 0.824547, 0.183192000]])

# Core Hamiltonian
H = gen_H_core(molecule, basis)

# Orthogonalization matrix
S = S_mat(molecule, basis)
orth = spl.sqrtm(spl.inv(S))

# MO coefficients
eps, c = np.linalg.eigh(np.dot(np.dot(orth.T,H),orth))
c_norm = np.dot(orth,c)

# Density matrix
P = P_mat(molecule,basis,c_norm)

# J+K matrix
G = eri_mat(molecule,basis,P)

# Fock matrix
F = H + G

# Energy
E = np.sum((2*H+G)*P)
print("E = {}".format(E))        
