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
        self.M = np.zeros([ntau, norb, norb], dtype=np.complex128)
        self.deltaM = np.zeros([norb, norb], dtype=np.complex128)    
    #---------------------------------------------------
    def add(self, b):
        self.M += b.M
        self.deltaM += b.deltaM
    #---------------------------------------------------
    def scale(self, c):
        self.M *= c
        self.deltaM *= c
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
def compute_G00M(ik1, ik2, myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump):
    G0 = matsubara(beta, ntau, norb, -1)
    
    kx, ky = get_kx_ky(ik1, ik2, Nkx, Nky, ARPES)

    if myrank==k2p[ik1,ik2]:
        index = k2i[ik1,ik2]

        f = np.diag(-0.5*np.ones(norb))
        G0.M = 1j*np.einsum('ab,mab->mab', f, np.ones([ntau, norb, norb], dtype=np.complex128))
        
    return G0
        
#---------------------------------------------------
def compute_G0M(ik1, ik2, UksR, UksI, eks, fks, Rs, myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump):
    G0 = matsubara(beta, ntau, norb, -1)
    
    kx, ky = get_kx_ky(ik1, ik2, Nkx, Nky, ARPES)

    if myrank==k2p[ik1,ik2]:
        index = k2i[ik1,ik2]
            
        # R * e^((beta-tau)*ek) * [-f(ek)] * R^dagger
        G0.M = 1j*np.einsum('ab,mb,b,cb->mac', Rs[index], UksI[1,index], -fks[index], np.conj(Rs[index]))
        
    return G0
#---------------------------------------------------
def compute_D0M(omega, beta, ntau, norb):

    D0 = matsubara(beta, ntau, norb, +1)

    nB   = 1./(np.exp(beta*omega)-1.0)
    
    taus = np.linspace(0, beta, ntau)

    # check this carefully
    x = -1j*(nB+0.0)*np.exp(omega*taus) - 1j*(nB+1.0)*np.exp(-omega*(taus))
    D0.M = np.einsum('t,ab->tab', x, np.diag(np.ones(norb)))
        
    return D0
