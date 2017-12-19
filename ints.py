import math
import numpy as np # for norm
import scipy.special as sps # for binom

def Pn(a, An, b, Bn):
    """Returns equation 9, takes individual coords of A and B"""
    if (a + b == 0):
        return 0 
    else:
        return (a*An + b*Bn)/(a+b)


def In(nn, a, An, mn, b, Bn):
    """Returns equation 8"""
    val = 0
    for i in range(0, nn+1):
        for j in range(0, mn+1):
            val += (Pn(a, An, b, Bn) - An) ** (nn - i) * sps.binom(nn,i) * (Pn(a, An, b, Bn) - Bn) ** (mn - j) * sps.binom(mn,j) * 0.5 * (1 + (-1) ** (i+j)) * (a + b) ** ((-i - j - 1)/2.) * math.gamma((1 + i + j)/2.) 
    return val


def gen_S(n, a, A, m, b, B):
    """Returns equation 7: overlap integral"""
    return math.exp(-a*b*np.linalg.norm(A-B)**2/(a+b))*In(n[0],a,A[0],m[0],b,B[0])*In(n[1],a,A[1],m[1],b,B[1])*In(n[2],a,A[2],m[2],b,B[2]) 


def normalize(n, a, A):
    """Returns a normalization constant from vector A and basis a"""
    val = gen_S(n, a, A, n, a, A)
    return (1/gen_S(n, a, A, n, a, A)**0.5)


def gen_K(a, A, b, B):
    """Returns equation 12: kinetic energy integral (specifically for S gaussians!)"""
    val = 0
    n = np.array([0,0,0])
    m = np.array([0,0,0])
    mx = np.array([2,0,0])
    my = np.array([0,2,0])
    mz = np.array([0,0,2])
    val = -2*b*gen_S(n, a, A, m, b, B) + 4*b**2*gen_S(n, a, A, mx, b, B)
    val += -2*b*gen_S(n, a, A, m, b, B) + 4*b**2*gen_S(n, a, A, my, b, B)
    val += -2*b*gen_S(n, a, A, m, b, B) + 4*b**2*gen_S(n, a, A, mz, b, B)
    val = -0.5*val
    return val


def gen_V(a, A, b, B, molecule):
    """Returns equation 14: nuclear attraction integral (specifically for S gaussians!)"""
    val1 = 0
    val2 = 0
    val1 = -2*math.pi*math.exp(-a*b*np.linalg.norm(A-B)**2/(a+b))/(a+b)

    Px = Pn(a, A[0], b, B[0])
    Py = Pn(a, A[1], b, B[1])
    Pz = Pn(a, A[2], b, B[2])
    P = np.array([Px, Py, Pz])

    C = molecule[0] # Hard-coding two atoms for now
    if np.array_equal(C, P):
        val2 += 1
    else:
        val2 += math.pi**0.5*math.erf((a+b)**0.5*np.linalg.norm(P-C))/(2*(a+b)**0.5*np.linalg.norm(P-C))

    C = molecule[1] # Hard-coding two atoms for now
    if np.array_equal(C, P):
        val2 += 1
    else:
        val2 += math.pi**0.5*math.erf((a+b)**0.5*np.linalg.norm(P-C))/(2*(a+b)**0.5*np.linalg.norm(P-C)) 

    return val1*val2


def gen_eri(a1, A1, b1, B1, a2, A2, b2, B2):
    """Returns equation 15: electron repulsion integral (specifically for S gaussians!)""" 

    Px1 = Pn(a1, A1[0], b1, B1[0])
    Py1 = Pn(a1, A1[1], b1, B1[1])
    Pz1 = Pn(a1, A1[2], b1, B1[2])
    P1 = np.array([Px1, Py1, Pz1])

    Px2 = Pn(a2, A2[0], b2, B2[0])
    Py2 = Pn(a2, A2[1], b2, B2[1])
    Pz2 = Pn(a2, A2[2], b2, B2[2])
    P2 = np.array([Px2, Py2, Pz2])

    K1 = math.exp(-a1*b1*np.linalg.norm(A1-B1)**2/(a1+b1))
    K2 = math.exp(-a2*b2*np.linalg.norm(A2-B2)**2/(a2+b2))

    T = np.linalg.norm(P1-P2)**2*(a1 + b1)*(a2+b2)/((a1+b1)+(a2+b2))

    if T == 0:
        F = 1
    else:
        F = math.pi**0.5/2./(T**0.5)*math.erf(T**0.5)  

    val = 2*math.pi**2.5*K1*K2/((a1+b1)*(a2+b2)*((a1+b1)+(a2+b2))**0.5)*F
    return val


def S_mat(molecule, basis):
    """Returns overlap matrix S"""
    S = np.zeros([molecule.shape[0]*basis.shape[1],molecule.shape[0]*basis.shape[1]])
    n = np.array([0,0,0])
    m = np.array([0,0,0])
    natom = molecule.shape[0]
    nbasis = basis.shape[1]

    for i in range(0,natom): # Atom 1 loop
        for j in range(0,natom): # Atom 2 loop
            for p in range(0,nbasis): # Basis function 1 loop
                for q in range(0,nbasis): # Basis function 2 loop
                    S[p+3*i,q+3*j] = gen_S(n, basis[i,p], molecule[i], m, basis[j,q], molecule[j]) * normalize(n, basis[i,p], molecule[i]) * normalize(m, basis[j,q], molecule[j])

    return S


def K_mat(molecule, basis):
    """Returns kinetic energy matrix K"""
    K = np.zeros([molecule.shape[0]*basis.shape[1],molecule.shape[0]*basis.shape[1]])
    n = np.array([0,0,0])
    m = np.array([0,0,0])
    natom = molecule.shape[0]
    nbasis = basis.shape[1]

    for i in range(0,natom): # Atom 1 loop
        for j in range(0,natom): # Atom 2 loop
            for p in range(0,nbasis): # Basis function 1 loop
                for q in range(0,nbasis): # Basis function 2 loop
                    K[p+3*i,q+3*j] = gen_K(basis[i,p], molecule[i], basis[j,q], molecule[j]) * normalize(n, basis[i,p], molecule[i]) * normalize(m, basis[j,q], molecule[j])

    return K


def V_mat(molecule, basis):
    """Returns kinetic energy matrix V"""
    V = np.zeros([molecule.shape[0]*basis.shape[1],molecule.shape[0]*basis.shape[1]])
    n = np.array([0,0,0])
    m = np.array([0,0,0])
    natom = molecule.shape[0]
    nbasis = basis.shape[1]

    for i in range(0,natom): # Atom 1 loop
        for j in range(0,natom): # Atom 2 loop
            for p in range(0,nbasis): # Basis function 1 loop
                for q in range(0,nbasis): # Basis function 2 loop
                    V[p+3*i,q+3*j] = gen_V(basis[i,p], molecule[i], basis[j,q], molecule[j], molecule) * normalize(n, basis[i,p], molecule[i]) * normalize(m, basis[j,q], molecule[j])

    return V


def gen_H_core(molecule, basis):
    """Returns Hamiltonian matrix H"""
    H_core = K_mat(molecule, basis) + V_mat(molecule, basis)
    return H_core


def eri_mat(molecule, basis, P):
    """Returns eri matrix G"""
    na = np.array([0,0,0])
    ma = np.array([0,0,0])
    natom = molecule.shape[0] # this is 2 for us
    nbasis = basis.shape[1] # this is 3 for us
    G = np.zeros([molecule.shape[0]*basis.shape[1],molecule.shape[0]*basis.shape[1]])
    J = np.zeros([molecule.shape[0]*basis.shape[1],molecule.shape[0]*basis.shape[1]])
    K = np.zeros([molecule.shape[0]*basis.shape[1],molecule.shape[0]*basis.shape[1]])

    for i in range(0,natom): 
        for j in range(0,natom):
            for ii in range(0,nbasis):
                for jj in range(0,nbasis):
                    for m in range(0,natom):
                        for n in range(0,natom):
                            for mm in range(0,nbasis):
                                for nn in range(0,nbasis):
                                    G[ii+3*i,jj+3*j] += P[mm+3*m,nn+3*n]*(2*gen_eri(basis[i,ii],molecule[i],basis[j,jj],molecule[j],basis[m,mm],molecule[m],basis[n,nn],molecule[n]) - gen_eri(basis[i,ii],molecule[i],basis[n,nn],molecule[n],basis[m,mm],molecule[m],basis[j,jj],molecule[j])) * normalize(na, basis[i,ii], molecule[i]) * normalize(na, basis[j,jj],molecule[j]) * normalize(na, basis[m,mm],molecule[m]) * normalize(na, basis[n,nn],molecule[n])
   
    return G
