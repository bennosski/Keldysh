import numpy as np
from scipy.linalg import expm
from functions import *
from plotting import *
from itertools import product
import h5py
import os
from profiler import timed

class langreth:
    def __init__(self, norb, nt, tmax, ntau, beta, sig):
        '''
        M is the known Matsubara piece
        '''
        self.nt   = nt
        self.tmax = tmax
        self.ntau = ntau
        self.beta = beta
        self.sig  = sig
        self.norb = norb
        #self.ntau = M.ntau
        #self.beta = M.beta
        #self.sig  = M.sig
        #self.norb  = norb        
        #self.dtau = dtau
        self.dtau = 1.0*beta/(ntau-1)
        self.dt   = 1.0*self.tmax/(self.nt-1)

        nt = self.nt
        ntau = self.ntau
        norb = self.norb
        
        self.L  = np.zeros([nt, norb, nt, norb], dtype=np.complex128)
        self.R  = np.zeros([nt, norb, nt, norb], dtype=np.complex128)
        self.RI = np.zeros([nt, norb, ntau, norb], dtype=np.complex128)
        self.deltaR = np.zeros([nt, norb, norb], dtype=np.complex128)
    #---------------------------------------------------     
    def add(self, b):
        self.L  += b.L
        self.R  += b.R
        self.RI += b.RI
        self.deltaR += b.deltaR
    #---------------------------------------------------
    def scale(self, c):
        self.L  *= c
        self.R  *= c
        self.RI *= c
        self.deltaR *= c
    #---------------------------------------------------
    def multiply(self, B):
        #self.L  *= B.L
        #self.R  *= B.R
        #self.RI *= B.RI
        #self.deltaR *= B.deltaR
        #self.sig *= B.sig
        self.L = np.einsum('manb,mn->manb', self.L, B.L)
        self.R = np.einsum('manb,mn->manb', self.R, B.R)
        self.RI = np.einsum('manb,mn->manb', self.RI, B.RI)
        self.sig = self.sig * B.sig
    #---------------------------------------------------
    def copy(self, B):
        self.L = B.L
        self.R = B.R
        self.RI = B.RI
        self.deltaR = B.deltaR
        self.sig = B.sig
    #---------------------------------------------------     
    def zero(self):
        self.L  = np.zeros_like(self.L)
        self.R  = np.zeros_like(self.R)
        self.RI = np.zeros_like(self.RI)
        self.deltaR = np.zeros_like(self.deltaR)
    #---------------------------------------------------     
    def save(self, folder, myfile):
        assert not os.path.exists(folder+myfile), 'Cannot Overwrite Existing Data'

        f = h5py.File(folder+myfile, 'w')
        params = f.create_dataset('/params', dtype='f')
        params.attrs['ntau'] = self.ntau
        params.attrs['norb'] = self.norb
        params.attrs['dtau'] = self.dtau
        params.attrs['beta'] = self.beta
        params.attrs['tmax'] = self.tmax        
        params.attrs['nt']   = self.nt
        params.attrs['dt']   = self.dt

        f.create_dataset('/L', data=self.L)
        f.create_dataset('/RI', data=self.RI)
        f.create_dataset('/R', data=self.R)
        f.create_dataset('/deltaR', data=self.deltaR)
        #f.create_dataset('/M', data=self.M)
        #f.create_dataset('/deltaM', data=self.deltaM)
        f.close()
    #---------------------------------------------------
    def load(self, folder, myfile):        
        f = h5py.File(folder+myfile, 'r')
        #self.M      = f['/M'][...]a
        #self.deltaM = f['/deltaM'][...]
        self.R      = f['/R'][...]
        self.deltaR = f['/deltaR'][...]
        self.RI     = f['/RI'][...]
        self.L      = f['/L'][...]
        self.ntau   = f['/params'].attrs['ntau']
        self.norb   = f['/params'].attrs['norb']
        self.dtau   = f['/params'].attrs['dtau']
        self.beta   = f['/params'].attrs['beta']
        self.tmax   = f['/params'].attrs['tmax']
        self.nt     = f['/params'].attrs['nt']
        self.dt     = f['/params'].attrs['dt']
        f.close()
    #---------------------------------------------------
    def __str__(self):
        return 'L  max %1.3e mean %1.3e'%(np.amax(np.abs(self.L)), np.mean(np.abs(self.L))) +'\n' \
              +'R  max %1.3e mean %1.3e'%(np.amax(np.abs(self.R)), np.mean(np.abs(self.R))) +'\n' \
              +'RI max %1.3e mean %1.3e'%(np.amax(np.abs(self.RI)),np.mean(np.abs(self.RI)))+'\n' \
              +'dR  max %1.3e mean %1.3e'%(np.amax(np.abs(self.deltaR)), np.mean(np.abs(self.deltaR)))
#---------------------------------------------------
def compute_U(H, kx, ky, t0, dt, pump, dt_fine, norb):
    # should give 4th order convergence
    #
    # c1 = 1/2 - sq3/6
    # c2 = 1/2 + sq3/6
    #
    # t1 = t + c1*dt
    # t2 = t + c2*dt
    #
    # H1 = H(t1)
    # H2 = H(t2)
    #
    # a1 = (3-2*sq3)/12
    # a2 = (3+2*sq3)/12
    #
    # U(t+dt, t) = exp(-1j*dt*(a1*H1+a2*H2)) * exp(-1j*dt*(a2*H1+a1*H2))
    
    steps = int(np.ceil(dt/dt_fine))
    dt_fine = dt/steps

    #assert steps==1, 'steps = %d'%steps
    
    sq3 = np.sqrt(3.0)
    c1  = 0.5 - sq3/6.0
    c2  = 0.5 + sq3/6.0
    a1  = (3.0 - 2.0*sq3)/12.0
    a2  = (3.0 + 2.0*sq3)/12.0
    
    U = np.diag(np.ones(norb, dtype=np.complex128))
    
    for step in range(steps):
        t  = t0 + dt_fine*step
        t1 = t + c1*dt_fine
        t2 = t + c2*dt_fine
        Ax, Ay = compute_A(t1, pump)
        H1 = H(kx-Ax, ky-Ay, t1)
        Ax, Ay = compute_A(t2, pump)
        H2 = H(kx-Ax, ky-Ay, t2)
        U  = expm(-1j*dt_fine*(a1*H1+a2*H2)) @ expm(-1j*dt_fine*(a2*H1+a1*H2)) @ U
        
    return U
#---------------------------------------------------
@timed
def init_Uks(H, dt_fine, myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump, version='regular'):
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
    Ht   = np.zeros([kpp, nt, norb, norb], dtype=np.complex128)
    
    for ik1 in range(Nkx):
        for ik2 in range(Nky):
            if myrank==k2p[ik1,ik2]:                
                index = k2i[ik1,ik2]

                kx, ky = get_kx_ky(ik1, ik2, Nkx, Nky, ARPES)

                UksR[index,0] = np.diag(np.ones(norb))
                                    
                Ax, Ay = compute_A(0, pump)
                Ht[index,0] = H(kx-Ax, ky-Ay, 0)
                
                for it in range(1, nt):
                    t = (it-1)*dt
                    Ax, Ay = compute_A(it*dt, pump)
                    Ht[index,it] = H(kx-Ax, ky-Ay, it*dt)
                    
                    if version=='regular':
                        UksR[index,it] = expm(-1j*H(kx-Ax, ky-Ay, t+dt/2.0)*dt) @ UksR[index,it-1]
                    elif version=='higher order':
                        UksR[index,it] = compute_U(H, kx, ky, t, dt, pump, dt_fine, norb) @ UksR[index,it-1]
                    else:
                        raise ValueError
                        
                eks[index], Rs[index] = np.linalg.eig(H(kx,ky,0))
                #eks[index], Rs[index] = np.linalg.eig(Ht[index,0])
                
                fks[index] = 1.0/(np.exp(beta*eks[index])+1.0)

                # better way since all H commute at t=0
                # pull R across the U(tau,0) when computing bare G so that we work with diagonal things

                # Uk(tau)
                UksI[0,index] = np.exp(-eks[index][None,:]*taus[:,None])
                # Uk(beta-tau)
                UksI[1,index] = np.exp(+eks[index][None,:]*(beta-taus[:,None]))

    return UksR, UksI, eks, fks, Rs, Ht
#---------------------------------------------------
def compute_G00R(ik1, ik2, myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump):
    
    G0 = langreth(norb, nt, tmax, ntau, beta, -1)
    
    kx, ky = get_kx_ky(ik1, ik2, Nkx, Nky, ARPES)

    if myrank==k2p[ik1,ik2]:
        index = k2i[ik1,ik2]

        f = np.diag(+0.5*np.ones(norb))
        G0.L = 1j*np.einsum('ab,manb->manb', f, np.ones([nt, norb, nt, norb], dtype=np.complex128))

        f = np.diag(-1.0*np.ones(norb))
        G0.R = 1j*np.einsum('ab,manb->manb', f, np.ones([nt, norb, nt, norb], dtype=np.complex128))

        #f = np.diag(-0.5*np.ones(norb))
        #G0.IR = 1j*np.einsum('ab,manb->manb', f, np.ones([ntau, norb, nt, norb], dtype=np.complex128))

        f = np.diag(+0.5*np.ones(norb))        
        G0.RI = 1j*np.einsum('ab,manb->manb', f, np.ones([nt, norb, ntau, norb], dtype=np.complex128))

    return G0
#---------------------------------------------------
@timed
def compute_G0R(ik1, ik2, UksR, UksI, eks, fks, Rs, myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump):

    G0 = langreth(norb, nt, tmax, ntau, beta, -1)
        
    kx, ky = get_kx_ky(ik1, ik2, Nkx, Nky, ARPES)

    if myrank==k2p[ik1,ik2]:
        index = k2i[ik1,ik2]
            
        G0.L = 1j*np.einsum('mab,bc,c,dc,ned->mane', UksR[index], Rs[index], fks[index]-0.0, np.conj(Rs[index]), np.conj(UksR[index]))

        # in accuracy with the minus 1?
        #GG = 1j*np.einsum('mab,bc,c,dc,ned->mane', UksR[index], Rs[index], fks[index]-1.0, np.conj(Rs[index]), np.conj(UksR[index]))
        #G0.R = GG - G0.L
        G0.R = -1j*np.einsum('mab,ncb->manc', UksR[index], np.conj(UksR[index]))
        
        #G0.IR  = 1j*np.einsum('ab,mb,b,cb,ndc->mand', Rs[index], UksI[1,index], -fks[index], np.conj(Rs[index]), np.conj(UksR[index]))

        G0.RI = 1j*np.einsum('mab,bc,c,nc,dc->mand', UksR[index], Rs[index], fks[index], 1.0/UksI[0,index], np.conj(Rs[index]))
        
    return G0
#---------------------------------------------------
@timed
def compute_D0R(norb, omega, nt, tmax, ntau, beta, sig):

    #D0 = langreth(norb, nt, tmax, ntau, beta, sig)

    #beta = D0.beta
    #tmax = D0.tmax
    #norb = D0.norb
    #ntau = D0.ntau
    
    class D:
        pass

    D0 = D()

    nB = 1./(np.exp(beta*omega)-1.0)
    
    ts = np.linspace(0, tmax, nt)
    taus = np.linspace(0, beta, ntau)

    D0.L = -1j*(nB+1.0-0.0)*np.exp(1j*omega*(ts[:,None]-ts[None,:])) - 1j*(nB+0.0)*np.exp(-1j*omega*(ts[:,None]-ts[None,:])) 
    #D0.L = block_diag(x, norb)
    

    x = -1j*(nB+1.0-1.0)*np.exp(1j*omega*(ts[:,None]-ts[None,:])) - 1j*(nB+1.0)*np.exp(-1j*omega*(ts[:,None]-ts[None,:]))
    D0.R = x - D0.L
    
    #D0.R = -1j*np.sin(omega*(ts[:,None]-ts[None,:]))
    #D0.R = block_diag(x, norb)

    x = -1j*(nB+1.0-0.0)*np.exp(1j*omega*(ts[:,None]+1j*taus[None,:])) - 1j*(nB+0.0)*np.exp(-1j*omega*(ts[:,None]+1j*taus[None,:]))
    #D0.RI = block_diag(x, norb)
    D0.RI = x

    D0.sig = +1

    #x = -1j*(nB+1.0-1.0)*np.exp(1j*omega*(-1j*taus[:,None]-ts[None,:])) - 1j*(nB+1.0)*np.exp(-1j*omega*(-1j*taus[:,None]-ts[None,:]))
    #D0.IR = block_diag(x, norb)

    return D0
#---------------------------------------------------

