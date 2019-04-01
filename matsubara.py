import numpy as np
from scipy.linalg import block_diag
from functions import *

class matsubara:
    def __init__(self, beta, ntau, norb, sig):
        self.ntau = ntau
        self.beta = beta
        self.norb = norb
        self.sig = sig

        self.dtau = 1.0*self.beta/(self.ntau-1)
        self.M = np.zeros([ntau*norb, norb], dtype=np.complex128)
    #---------------------------------------------------
    def scale(self, c):
        self.M *= c
    #---------------------------------------------------
    def mycopy(self, b):
        pass
    #---------------------------------------------------
    def mysave(self, myfile):
        pass
    #---------------------------------------------------
    def myload(self, myfile):
        pass
        
#---------------------------------------------------
def compute_G0M(ik1, ik2, UksR, UksI, eks, fks, Rs, myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump):
    G0 = matsubara(beta, ntau, norb, -1)
    
    kx, ky = get_kx_ky(ik1, ik2, Nkx, Nky, ARPES)

    print('myrank', myrank)
    print('k2p', k2p[ik1,ik2])
    
    if myrank==k2p[ik1,ik2]:
        index = k2i[ik1,ik2]
            
        # R * e^((beta-tau)*ek) * [-f(ek)] * R^dagger
        G = 1j*np.einsum('ab,mb,b,cb->mac', Rs[index], UksI[1,index], -fks[index], np.conj(Rs[index]))
        G0.M = np.reshape(G, [ntau*norb, norb])
        
    return G0
#---------------------------------------------------
def compute_D0M(omega, beta, ntau, norb):

    D0 = matsubara(beta, ntau, norb, +1)

    nB   = 1./(np.exp(beta*omega)-1.0)
    #theta = np.tril(np.ones([Ntau,Ntau]), -1) + np.diag(0.5*np.ones(Ntau)) 
    
    taus = np.linspace(0, beta, ntau)

    # check this carefully
    x = -1j*(nB+0.0)*np.exp(omega*taus) - 1j*(nB+1.0)*np.exp(-omega*(taus))
    D0.M = np.reshape(np.einsum('t,ab->tab', x, np.diag(np.ones(norb))), [ntau*norb, norb]) 
        
    return D
