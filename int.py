import math
import numpy as np # for norm
import scipy.special as sps # for binom

def Pn(a, A, b, B):
    """Returns equation 9"""
    if (a + b == 0):
        return 0 
    else:
        return (a*A + b*B)/(a+b)


def In(nn, a, An, mn, b, Bn):
    """Returns equation 8"""
    val = 0
    for i in range(0, nn+1):
        for j in range(0, mn+1):
            val += (Pn(a, An, b, Bn) - An) ** (nn - i) * sps.binom(nn,i) * (Pn(a, An, b, Bn) - Bn) ** (mn - j) * sps.binom(mn,j) * 0.5 * (1 + (-1) ** (i+j)) * (a + b) ** ((-i - j - 1)/2.) * math.gamma((1 + i + j)/2.) 
    return val


def S(n, a, A, m, b, B):
    """Returns equation 7"""
    return math.exp(-a*b*np.linalg.norm(A-B)**2/(a+b))*In(n[0],a,A[0],m[0],b,B[0])*In(n[1],a,A[1],m[1],b,B[1])*In(n[2],a,A[2],m[2],b,B[2]) 


def K(a, A, b, B):
    """Returns equation 12: specifically for S gaussians!"""
    val = 0
    n = np.array([0,0,0])
    m = np.array([0,0,0])
    mx = np.array([2,0,0])
    my = np.array([0,2,0])
    mz = np.array([0,0,2])
    val = -2*b*S(n, a, A, m, b, B) + 4*b**2*S(n, a, A, mx, b, B)
    print("Val = {}".format(val))
    print("Overlap1 = {} and Overlap2 = {}".format(S(n, a, A, m, b, B),S(n, a, A, mx, b, B)))
    val += -2*b*S(n, a, A, m, b, B) + 4*b**2*S(n, a, A, my, b, B)
    val += -2*b*S(n, a, A, m, b, B) + 4*b**2*S(n, a, A, mz, b, B)
    val = -0.5*val
    return val




n = np.array([2,0,0])
m = np.array([0,0,0])
a = 5.447178
A = np.array([0,0,0])
B = np.array([1,0,0])
nc1 = 2.5411995

#print(S(n, a, A, m, a, B))
print(K(a, A, a, A)*nc1**2)
