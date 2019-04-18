import numpy as np
from scipy.linalg import expm
from functions import *

class langreth:
    def __init__(self, nt, tmax, M):
        '''
        M is the known Matsubara piece
        '''
        self.nt   = nt
        self.tmax = tmax
        self.ntau = M.ntau
        self.beta = M.beta
        self.sig  = M.sig
        self.norb  = M.norb
        
        self.dtau = M.dtau
        self.dt   = 1.0*self.tmax/(self.nt-1)

        nt = self.nt
        ntau = self.ntau
        norb = self.norb
        
        self.L  = np.zeros([nt, norb, nt, norb], dtype=np.complex128)
        self.R  = np.zeros([nt, norb, nt, norb], dtype=np.complex128)
        self.IR = np.zeros([ntau, norb, nt, norb], dtype=np.complex128)
        self.M  = M.M
        self.deltaR = np.zeros([nt, norb, norb], dtype=np.complex128)
        self.deltaM = M.deltaM
    #---------------------------------------------------     
    def add(self, b):
        self.L  += b.L
        self.R  += b.R
        self.IR += b.IR
        self.deltaR += b.deltaR

        # NOTE : langreth operations should never change the matsubara piece
        #self.M  += b.M
        #self.deltaM += b.deltaM
    #---------------------------------------------------
    def scale(self, c):
        self.L  *= c
        self.R  *= c
        self.IR *= c
        self.deltaR *= c

        # NOTE : langreth operations should never change the matsubara piece
        #self.M  *= c
        #self.deltaM *= c
    #---------------------------------------------------     
    def zero(self):
        self.L  = np.zeros_like(self.L)
        self.R  = np.zeros_like(self.R)
        self.IR = np.zeros_like(self.IR)
        self.M  = np.zeros_like(self.M)
        self.deltaR = np.zeros_like(self.deltaR)
        self.deltaM = np.zeros_like(self.deltaM)
    #---------------------------------------------------     
    def mysave(self, myfile):
        np.save(myfile+'L', self.L)
        np.save(myfile+'R', self.R)
        np.save(myfile+'IR', self.IR)
        np.save(myfile+'M', self.M)
        np.save(myfile+'deltaR', self.deltaR)
        np.save(myfile+'deltaM', self.deltaM)
    #---------------------------------------------------
    def myload(self, myfile):
        self.L  = np.load(myfile+'L.npy')
        self.R  = np.load(myfile+'R.npy')
        self.IR = np.load(myfile+'IR.npy')
        self.M  = np.load(myfile+'M.npy')
        self.deltaR = np.load(myfile+'deltaR.npy')
        self.deltaM = np.load(myfile+'deltaM.npy')
    #---------------------------------------------------
    def __str__(self):
        return 'L  max %1.3e mean %1.3e'%(np.amax(np.abs(self.L)), np.mean(np.abs(self.L))) +'\n' \
              +'R  max %1.3e mean %1.3e'%(np.amax(np.abs(self.R)), np.mean(np.abs(self.R))) +'\n' \
              +'IR max %1.3e mean %1.3e'%(np.amax(np.abs(self.IR)),np.mean(np.abs(self.IR)))+'\n' \
              +'M  max %1.3e mean %1.3e'%(np.amax(np.abs(self.M)), np.mean(np.abs(self.M))) +'\n' \
              +'dR  max %1.3e mean %1.3e'%(np.amax(np.abs(self.deltaR)), np.mean(np.abs(self.deltaR)))+'\n' \
              +'dM  max %1.3e mean %1.3e'%(np.amax(np.abs(self.deltaM)), np.mean(np.abs(self.deltaM)))
#---------------------------------------------------
def init_Ht(H, myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump):
    
    # the Hamiltonian in general is a function of kx,ky,t,orbital
    # H is a function which accepts a kx and ky value and returns an norb x norb Hamiltonian

    Ht = np.zeros([kpp, nt, norb, norb], dtype=np.complex128)

    dt = 1.0*tmax/(nt-1)
    ts = np.arange(0, nt*dt-dt/2.0, dt)
    assert len(ts)==nt
    A  = np.zeros([2, nt])
    for it in range(nt):
        A[0,it], A[1,it], _ = compute_A(ts[it], nt, dt, pump)
    
    for ik1 in range(Nkx):
        for ik2 in range(Nky):
            if myrank==k2p[ik1,ik2]:                
                index = k2i[ik1,ik2]

                kx, ky = get_kx_ky(ik1, ik2, Nkx, Nky, ARPES)

                for it in range(nt):
                    #Ht[index,it] = H(kx-A[0,it], ky-A[1,it]) * np.cos(0.02*ts[it])
                    Ht[index,it] = H(kx-A[0,it], ky-A[1,it])

    return Ht
#---------------------------------------------------
def init_Uks(Ht, myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump):
    '''
    for ARPES, use Nky = 1 
    '''

    dt = 1.0*tmax/(nt-1)
    
    taus = np.linspace(0, beta, ntau)    

    UksR = np.zeros([kpp, nt, norb, norb], dtype=np.complex128)
    UksI = np.zeros([2, kpp, ntau, norb], dtype=np.complex128)
    fks  = np.zeros([kpp, norb], dtype=np.complex128)
    eks  = np.zeros([kpp, norb], dtype=np.complex128)
    Rs   = np.zeros([kpp, norb, norb], dtype=np.complex128)
    
    for ik1 in range(Nkx):
        for ik2 in range(Nky):
            if myrank==k2p[ik1,ik2]:                
                index = k2i[ik1,ik2]

                kx, ky = get_kx_ky(ik1, ik2, Nkx, Nky, ARPES)

                prod = np.diag(np.ones(norb))
                UksR[index,0] = prod.copy()
                for it in range(1, nt):
                    tt = it*dt # - dt/2.0
                    Ax, Ay, _ = compute_A(tt, nt, dt, pump)
                    #prod = np.dot(expm(-1j*Ht(kx-Ax, ky-Ay)*dt), prod)
                    prod = np.dot(expm(-1j*Ht[index,it]*dt), prod)
                    UksR[index,it] = prod.copy()

                #eks[index], Rs[index] = np.linalg.eig(Ht(kx,ky))
                eks[index], Rs[index] = np.linalg.eig(Ht[index,0])
                
                fks[index] = 1.0/(np.exp(beta*eks)+1.0)

                # better way since all H commute at t=0
                # pull R across the U(tau,0) when computing bare G so that we work with diagonal things

                # Uk(tau)
                UksI[0,index] = np.exp(-eks[index][None,:]*taus[:,None])
                # Uk(beta-tau)
                UksI[1,index] = np.exp(+eks[index][None,:]*(beta-taus[:,None]))

    return UksR, UksI, eks, fks, Rs
#---------------------------------------------------
def compute_G0R(ik1, ik2, GM, UksR, UksI, eks, fks, Rs, myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump):
    
    G0 = langreth(nt, tmax, GM)
    
    kx, ky = get_kx_ky(ik1, ik2, Nkx, Nky, ARPES)

    if myrank==k2p[ik1,ik2]:
        index = k2i[ik1,ik2]
            
        G0.L = 1j*np.einsum('mab,bc,c,dc,ned->mane', UksR[index], Rs[index], fks[index]-0.0, np.conj(Rs[index]), np.conj(UksR[index]))

        GG = 1j*np.einsum('mab,bc,c,dc,ned->mane', UksR[index], Rs[index], fks[index]-1.0, np.conj(Rs[index]), np.conj(UksR[index]))
        G0.R = GG - G0.L
                
        G0.IR  = 1j*np.einsum('ab,mb,b,cb,ndc->mand', Rs[index], UksI[1,index], -fks[index], np.conj(Rs[index]), np.conj(UksR[index]))
        
    return G0
#---------------------------------------------------
def compute_D0R(DM, omega, nt, tmax):

    D0 = langreth(nt, tmax, DM)

    beta = D0.beta
    tmax = D0.tmax
    norb = D0.norb
    ntau = D0.ntau
    
    nB = 1./(np.exp(beta*omega)-1.0)
    
    ts = np.linspace(0, tmax, nt)
    taus = np.linspace(0, beta, ntau)

    x = -1j*(nB+1.0-0.0)*np.exp(1j*omega*(ts[:,None]-ts[None,:])) - 1j*(nB+0.0)*np.exp(-1j*omega*(ts[:,None]-ts[None,:])) 
    D0.L = block_diag(x, norb)

    x = -1j*(nB+1.0-1.0)*np.exp(1j*omega*(ts[:,None]-ts[None,:])) - 1j*(nB+1.0)*np.exp(-1j*omega*(ts[:,None]-ts[None,:]))
    D0.R = block_diag(x, norb) - D0.L
    
    #x = -1j*(nB+1.0-0.0)*np.exp(1j*omega*(ts[:,None]+1j*taus[None,:])) - 1j*(nB+0.0)*np.exp(-1j*omega*(ts[:,None]+1j*taus[None,:]))
    #D.RI = block_diag(*[x]*Norbs)

    x = -1j*(nB+1.0-1.0)*np.exp(1j*omega*(-1j*taus[:,None]-ts[None,:])) - 1j*(nB+1.0)*np.exp(-1j*omega*(-1j*taus[:,None]-ts[None,:]))
    D0.IR = block_diag(x, norb)
    
    return D0
#---------------------------------------------------

