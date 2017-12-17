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
            print("Pn = {}".format(Pn(a,An,b,Bn)))
    print("In = {}".format(val))
    return val

def S(n, a, A, m, b, B):
    """Returns equation 7"""
    return math.exp(-a*b*np.linalg.norm(A-B)**2/(a+b))*In(n[0],a,A[0],m[0],b,B[0])*In(n[1],a,A[1],m[1],b,B[1])*In(n[2],a,A[2],m[2],b,B[2]) 





n = np.array([2,0,0])
m = np.array([0,0,0])
a = 5.447178
A = np.array([0,0,0])
B = np.array([1,0,0])

print(S(n, a, A, m, a, B))
