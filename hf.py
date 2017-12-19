import scipy.linalg as spl
from ints import *
# math, numpy as np, and scipy.scpecial as sps are pulled from ints

def P_mat(molecule, basis, c_norm):
    """Returns density matrix P"""
    P = np.zeros([molecule.shape[0]*basis.shape[1],molecule.shape[0]*basis.shape[1]])
    for i in range (0,molecule.shape[0]*basis.shape[1]):
        for j in range (0,molecule.shape[0]*basis.shape[1]):
            for m in range (0,1): # M is number of occuppied orbitals, should automate
                P[i,j] += c_norm[i,m]*c_norm[j,m]
    return P


def loop_hf(molecule, basis, F, H):
    """Solves the Hartree Fock equations once given a molecule, a basis, an initial Fock matrix, and a core Hamiltonian"""

    # Orthogonalization matrix
    S = S_mat(molecule, basis)
    orth = spl.sqrtm(spl.inv(S))

    # MO coefficients
    eps, c = np.linalg.eigh(np.dot(np.dot(orth.T,F),orth))
    c_norm = np.dot(orth,c)
    
    # Density matrix
    P = P_mat(molecule,basis,c_norm)
    
    # J+K matrix
    G = eri_mat(molecule,basis,P)
    
    # Fock matrix
    F = H + G
    
    # Energy
    E = np.sum((2*H+G)*P)

    return E, F
    

def full_hf(molecule, basis, convergence=10E-12):
    """Solves the HF equations self-consistently given a molecule, basis, and optional convergence""" 
    # Initial core Hamiltonian
    H = gen_H_core(molecule, basis)
    
    # Initial orthogonalization matrix
    S = S_mat(molecule, basis)
    orth = spl.sqrtm(spl.inv(S))
    
    # Initial MO coefficients
    eps, c = np.linalg.eigh(np.dot(np.dot(orth.T,H),orth))
    c_norm = np.dot(orth,c)
    
    # Initial density matrix
    P = P_mat(molecule,basis,c_norm)
    
    # Initial J+K matrix
    G = eri_mat(molecule,basis,P)
    
    # Initial Fock matrix
    F = H + G
    
    # Initial energy
    E = 0
    E_new = np.sum((2*H+G)*P)
    
    # Repeat until convergence 
    while (math.fabs(E_new - E)) > convergence:
        E = E_new    
        E_new, F = loop_hf(molecule, basis, F, H)

    # Nuclear repulsion energy
    R = 1 / np.linalg.norm(molecule[0]-molecule[1])
    
    # Final total HF energy
    E_HF = E_new + R

    return E_HF

#molecule = np.array([[0,0,0],[1.4,0,0]])
#basis = np.array([[5.447178, 0.824547, 0.183192000],[5.447178, 0.824547, 0.183192000]])

#full_hf(molecule, basis)
