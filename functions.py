import numpy as np
from scipy.linalg import expm
from scipy.linalg import block_diag
import subprocess, os

class langreth:
    # would be easier to have Nt, Ntau, Norbs as member variables

    def __init__(self, Nt, Ntau, Norbs):
        self.G  = np.zeros([Norbs*Nt, Norbs*Nt], dtype=complex)
        self.L  = np.zeros([Norbs*Nt, Norbs*Nt], dtype=complex)
        self.IR = np.zeros([Norbs*Ntau, Norbs*Nt], dtype=complex)
        self.RI = np.zeros([Norbs*Nt, Norbs*Ntau], dtype=complex)
        self.M  = np.zeros([Norbs*Ntau, Norbs*Ntau], dtype=complex)
        self.DR = np.zeros(Norbs*Nt, dtype=complex)
        self.DM = np.zeros(Norbs*Ntau, dtype=complex)

    def add(self, b):
        self.G  += b.G
        self.L  += b.L
        self.IR += b.IR
        self.RI += b.RI
        self.M  += b.M

    def directMultiply(self, b):
        self.G  *= b.G
        self.L  *= b.L
        self.IR *= b.IR
        self.RI *= b.RI
        self.M  *= b.M
        
    def scale(self, c):
        self.G  *= c
        self.L  *= c
        self.IR *= c
        self.RI *= c
        self.M  *= c
        
    def mycopy(self, b):
        self.G  = b.G.copy()
        self.L  = b.L.copy()
        self.IR = b.IR.copy()
        self.RI = b.RI.copy()
        self.M  = b.M.copy()
        self.DR = b.DR.copy()
        self.DM = b.DM.copy()        
        
    def transpose(self):
        Gt = np.transpose(self.G)
        Lt = np.transpose(self.L)
        IRt = np.transpose(self.IR)
        RIt = np.transpose(self.RI)
        # note the proper switching between components
        self.G = Lt     
        self.L = Gt
        self.IR = RIt
        self.RI = IRt
        self.M = np.transpose(self.M)
        
    def zero(self, Nt, Ntau, Norbs):
        self.G  = np.zeros([Norbs*Nt, Norbs*Nt], dtype=complex)
        self.L  = np.zeros([Norbs*Nt, Norbs*Nt], dtype=complex)
        self.IR = np.zeros([Norbs*Ntau, Norbs*Nt], dtype=complex)
        self.RI = np.zeros([Norbs*Nt, Norbs*Ntau], dtype=complex)
        self.M  = np.zeros([Norbs*Ntau, Norbs*Ntau], dtype=complex)
        self.DR = np.zeros(Norbs*Nt, dtype=complex)
        self.DM = np.zeros(Norbs*Ntau, dtype=complex)
        
    def mysave(self, myfile):
        np.save(myfile+'G', self.G)
        np.save(myfile+'L', self.L)
        np.save(myfile+'RI', self.RI)
        np.save(myfile+'IR', self.IR)
        np.save(myfile+'M', self.M)
        np.save(myfile+'DR', self.DR)
        np.save(myfile+'DM', self.DM)

    def myload(self, myfile):
        self.G  = np.load(myfile+'G.npy')
        self.L  = np.load(myfile+'L.npy')
        self.RI = np.load(myfile+'RI.npy')
        self.IR = np.load(myfile+'IR.npy')
        self.M  = np.load(myfile+'M.npy')
        
    def __str__(self):
        return 'L  max %1.3e mean %1.3e'%(np.amax(np.abs(self.L)), np.mean(np.abs(self.L))) +'\n' \
              +'G  max %1.3e mean %1.3e'%(np.amax(np.abs(self.G)), np.mean(np.abs(self.G))) +'\n' \
              +'IR max %1.3e mean %1.3e'%(np.amax(np.abs(self.IR)),np.mean(np.abs(self.IR)))+'\n' \
              +'RI max %1.3e mean %1.3e'%(np.amax(np.abs(self.RI)),np.mean(np.abs(self.RI)))+'\n' \
              +'M  max %1.3e mean %1.3e'%(np.amax(np.abs(self.M)), np.mean(np.abs(self.M))) 

def parseline(mystr):
    ind = mystr.index('#')
    return mystr[ind+1:]

def bash_command(cmd):
    subprocess.Popen(['/bin/bash', '-c', cmd])

def mymkdir(mydir):
    if not os.path.exists(mydir):
        print 'making ',mydir
        os.mkdir(mydir)

def setup_cuts(Nk):
    if False:
        # gamma X M gamma
        cut_ikxs = []
        cut_ikys = []
        for i in range(Nk//3):
            cut_ikxs.append(i)
            cut_ikys.append(i)
        for i in range(Nk//6):
            cut_ikxs.append(Nk//3+i)
            cut_ikys.append(Nk//3-2*i)
        for i in range(Nk//2+1):
            cut_ikxs.append(Nk//2-i)
            cut_ikys.append(0)

    if True:
        # gamma X
        cut_ikxs = []
        cut_ikys = []
        for i in range(Nk//2):
            cut_ikxs.append(i)
            cut_ikys.append(i)

    return cut_ikxs, cut_ikys

# kpoint on the y axis
def Hk(kx, ky):
    # graphene
    ''' 
    mat = np.zeros([2,2], dtype=complex)
    gammak = 1 + np.exp(1j*kx*np.sqrt(3.)) + np.exp(1j*np.sqrt(3)/2*(kx + np.sqrt(3)*ky))
    mat[0,1] = gammak*2.8
    mat[1,0] = np.conj(gammak)*2.8
    return mat
    '''
    
    #x = np.array([[-0.1, 0.2], [0.2, 0.1]])
    
    x = np.zeros([1,1])
    x[0] = -0.1
    
    return x

# k point on the y axis
def band(kx, ky):
    #return 2.8 * sqrt(1. + 4*cos(sqrt(3.)/2*kx)*cos(ky/2) + 4*cos(ky/2)**2)
    #return 2.8 * np.sqrt(1. + 4*np.cos(3.0/2*kx)*np.cos(np.sqrt(3)*ky/2) + 4*np.cos(np.sqrt(3.)*ky/2)**2)
    
    #return 2.8 * np.sqrt(1. + 4*np.cos(3.0/2*ky)*np.cos(np.sqrt(3)*kx/2) + 4*np.cos(np.sqrt(3.)*kx/2)**2)

    #return sorted(np.linalg.eigvals(Hk(kx,ky)), reverse=True)

    # what about eigenvalue sorting? Does it matter
    return np.linalg.eigvals(Hk(kx,ky))

def get_kx_ky(ik1, ik2, Nkx, Nky, ARPES=False):
    if not ARPES:
        ky = 4*np.pi/3*ik1/Nkx + 2*np.pi/3*ik2/Nky
        kx = 2*np.pi/np.sqrt(3.)*ik2/Nky
    else:

        '''
        f = (1./4+1./24) + (1./12) * ik1/(Nkx-1)
        # cut along gamma - X
        # ik runs from 0 to Nk/2
        ky = 4*np.pi/3*f + 2*np.pi/3*f
        kx = 2*np.pi/np.sqrt(3.)*f
        '''
        kx = 0.0
        ky = 0.0

    return kx, ky

def compute_A(mytime, Nt, dt, pump):
    if pump==0:
        return 0.0, 0.0, 0.0

    if pump==11:
        Amax = 0.5

        fieldAngle = np.pi*150./180.
        cosA    = np.cos(fieldAngle)
        sinA    = np.sin(fieldAngle)

        A = 0.
        if mytime>=18.0 and mytime<=20.0:            
            A =  Amax*np.sin(np.pi/2.*(mytime-18.0))**2
        elif mytime>20.0 and mytime<22.0:
            A = -Amax*np.sin(np.pi/2.*(mytime-20.0))**2

        return A*cosA, A*sinA, fieldAngle
    
    return None

def init_k2p_k2i_i2k(Nkx, Nky, nprocs, myrank):
    k2p = np.zeros([Nkx, Nky], dtype=int)
    k2i = np.zeros([Nkx, Nky], dtype=int)
    i2k = []
    for ik1 in range(Nkx):
        for ik2 in range(Nky):
            k2p[ik1,ik2] = (ik1*Nky + ik2)%nprocs
            k2i[ik1,ik2] = (ik1*Nky + ik2)//nprocs
            if k2p[ik1,ik2]==myrank:
                i2k.append([ik1,ik2])
    return k2p, k2i, i2k

def init_block_theta(Nt, Norbs):
    # could try using scipy.block_diag for this
    theta = np.zeros([Norbs*Nt, Norbs*Nt])
    for a in range(Norbs):
        for b in range(Norbs):
            for i in range(Nt):
                theta[a*Nt+i,b*Nt+i] = 0.5
                for j in range(i):
                    theta[a*Nt+i,b*Nt+j] = 1.0
    return theta

def init_Uks(myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, Nt, Ntau, dt, dtau, pump, Norbs):
    '''
    for ARPES, use Nky = 1 
    '''

    beta = Ntau*dtau    
    taus = np.linspace(0, (Ntau-1)*dtau, Ntau)
    

    UksR = np.zeros([kpp, Nt, Norbs, Norbs], dtype=complex)
    UksI = np.zeros([2, kpp, Ntau, Norbs], dtype=complex)
    fks  = np.zeros([kpp, Norbs], dtype=complex)
    eks  = np.zeros([kpp, Norbs], dtype=complex)
    
    for ik1 in range(Nkx):
        for ik2 in range(Nky):
            if myrank==k2p[ik1,ik2]:                
                index = k2i[ik1,ik2]

                kx, ky = get_kx_ky(ik1, ik2, Nkx, Nky, ARPES)

                prod = np.diag(np.ones(Norbs))
                UksR[index,0] = prod.copy()
                for it in range(1,Nt):
                    tt = it*dt # - dt/2.0
                    Ax, Ay, _ = compute_A(tt, Nt, dt, pump)
                    prod = np.dot(expm(-1j*Hk(kx-Ax, ky-Ay)*dt), prod)
                    UksR[index,it] = prod.copy()

                eks[index] = band(kx, ky)
                fks[index] = 1.0/(np.exp(beta*eks)+1.0)

                # better way since all Hk commute at t=0
                # pull R across the U(tau,0) when computing bare G so that we work with diagonal things

                # Uk(tau)
                UksI[0,index] = np.exp(-eks[index][None,:]*taus[:,None])
                # Uk(beta-tau)
                UksI[1,index] = np.exp(+eks[index][None,:]*(beta-taus[:,None]))

    return UksR, UksI, eks, fks

def compute_G0(ik1, ik2, fks, UksR, UksI, eks, myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, Nt, Ntau, dt, dtau, pump, Norbs):

    G0 = langreth(Nt, Ntau, Norbs)
    
    beta  = dtau*Ntau
        
    kx, ky = get_kx_ky(ik1, ik2, Nkx, Nky, ARPES)

    if myrank==k2p[ik1,ik2]:
        index = k2i[ik1,ik2]
    
        # test this for 1x1 and 2x2 cases
        evals, R = np.linalg.eig(Hk(kx, ky))

        #assert np.amax(abs(Hk(kx,ky)-np.einsum('ij,j,kj', R, evals, np.conj(R))))<1e-14
        #print('evals', evals)
        #print('eks[index]', eks[index])
        #assert np.amax(abs(evals-eks[index]))<1e-14

        G  = 1j*np.einsum('mab,bc,c,dc,ned->amen', UksR[index], R, fks[index]-0.0, np.conj(R), np.conj(UksR[index]))
        G0.L = np.reshape(G, [Nt*Norbs, Nt*Norbs])

        G  = 1j*np.einsum('mab,bc,c,dc,ned->amen', UksR[index], R, fks[index]-1.0, np.conj(R), np.conj(UksR[index]))
        G0.G = np.reshape(G, [Nt*Norbs, Nt*Norbs])

        G  = 1j*np.einsum('mab,bc,c,nc,dc->amdn', UksR[index], R, fks[index], 1.0/UksI[0,index], np.conj(R))
        G0.RI = np.reshape(G, [Nt*Norbs, Ntau*Norbs])
        
        G  = 1j*np.einsum('ab,mb,b,cb,ndc->amdn', R, UksI[1,index], -fks[index], np.conj(R), np.conj(UksR[index]))
        G0.IR = np.reshape(G, [Ntau*Norbs, Nt*Norbs])
        
        theta = np.tril(np.ones([Ntau,Ntau]), -1) + np.diag(0.5*np.ones(Ntau)) 
        G  = 1j*np.einsum('ab,mb,mnb,nb,cb->amcn', R, UksI[0,index], fks[index][None,None,:]-theta[:,:,None], 1.0/UksI[0,index], np.conj(R))
        G0.M = np.reshape(G, [Ntau*Norbs, Ntau*Norbs])

    return G0


def init_D(omega, Nt, Ntau, dt, dtau, Norbs):
    D = langreth(Nt, Ntau, Norbs)

    beta = dtau*Ntau
    nB   = 1./(np.exp(beta*omega)-1.0)
    theta = np.tril(np.ones([Ntau,Ntau]), -1) + np.diag(0.5*np.ones(Ntau)) 
    
    #ts = np.arange(0, Nt*dt, dt)
    #taus = np.arange(0, Ntau*dtau, dtau)
    ts = np.linspace(0, (Nt-1)*dt, Nt)
    taus = np.linspace(0, (Ntau-1)*dtau, Ntau)

    x = -1j*(nB+1.0-0.0)*np.exp(1j*omega*(ts[:,None]-ts[None,:])) - 1j*(nB+0.0)*np.exp(-1j*omega*(ts[:,None]-ts[None,:]))
    D.L = block_diag(*[x]*Norbs)

    x = -1j*(nB+1.0-1.0)*np.exp(1j*omega*(ts[:,None]-ts[None,:])) - 1j*(nB+1.0)*np.exp(-1j*omega*(ts[:,None]-ts[None,:]))
    D.G = block_diag(*[x]*Norbs)

    x = -1j*(nB+1.0-0.0)*np.exp(1j*omega*(ts[:,None]+1j*taus[None,:])) - 1j*(nB+0.0)*np.exp(-1j*omega*(ts[:,None]+1j*taus[None,:]))
    D.RI = block_diag(*[x]*Norbs)

    x = -1j*(nB+1.0-1.0)*np.exp(1j*omega*(-1j*taus[:,None]-ts[None,:])) - 1j*(nB+1.0)*np.exp(-1j*omega*(-1j*taus[:,None]-ts[None,:]))
    D.IR = block_diag(*[x]*Norbs)
    
    x = -1j*(nB+np.transpose(theta))*np.exp(omega*(taus[:,None]-taus[None,:])) - 1j*(nB+theta)*np.exp(-omega*(taus[:,None]-taus[None,:]))
    D.M = block_diag(*[x]*Norbs)

    return D
    
def initRA(L, Nt, Norbs):
    # theta for band case
    theta = init_block_theta(Nt, Norbs)
    
    R = np.zeros([Norbs*Nt, Norbs*Nt], dtype=complex)
    A = np.zeros([Norbs*Nt, Norbs*Nt], dtype=complex)
    
    R =  theta * (L.G - L.L)
    A = -np.transpose(theta) * (L.G - L.L)

    return R, A
      
def computeRelativeDifference(a, b):    
    change = [np.sum(abs(a.L - b.L))/np.sum(abs(a.L)),
              np.sum(abs(a.G - b.G))/np.sum(abs(a.G)),
              np.sum(abs(a.R - b.R))/np.sum(abs(a.R)),
              np.sum(abs(a.A - b.A))/np.sum(abs(a.A)),
              np.sum(abs(a.RI - b.RI))/np.sum(abs(a.RI)),
              np.sum(abs(a.IR - b.IR))/np.sum(abs(a.IR)),
              np.sum(abs(a.M - b.M))/np.sum(abs(a.M))]
    return change
        

def multiply(a, b, Nt, Ntau, dt, dtau, Norbs):
    '''
    computes the langreth product of a and b and stores result in c
    '''

    aR, aA = initRA(a, Nt, Norbs)
    bR, bA = initRA(b, Nt, Norbs)

    aR  += np.diag(a.DR)
    aA  += np.diag(a.DR)
    a.M += np.diag(a.DM)

    bR  += np.diag(b.DR)
    bA  += np.diag(b.DR)
    b.M += np.diag(b.DM)

    c = langreth(Nt, Ntau, Norbs)

    c.zero(Nt, Ntau, Norbs)
    
    c.M = -1j*dtau*np.dot(a.M, b.M)

    mixed_product = -1j*dtau*np.dot(a.RI, b.IR)

    c.G = dt*(np.dot(a.G, bA) + np.dot(aR, b.G)) + mixed_product
    c.L = dt*(np.dot(a.L, bA) + np.dot(aR, b.L)) + mixed_product

    c.RI = dt*np.dot(aR, b.RI) - 1j*dtau*np.dot(a.RI, b.M)
    c.IR = dt*np.dot(a.IR, bA) - 1j*dtau*np.dot(a.M, b.IR)

    return c
    
    
#invert a * b = c to solve for b
def solve(a, c, Nt, Ntau, dt, dtau, Norbs):

    aR, aA = initRA(a, Nt, Norbs)
    cR, cA = initRA(c, Nt, Norbs)

    aR  += np.diag(a.DR)
    aA  += np.diag(a.DR)
    a.M += np.diag(a.DM)

    cR  += np.diag(c.DR)
    cA  += np.diag(c.DR)
    c.M += np.diag(c.DM)
    
    b = langreth(Nt, Ntau, Norbs)

    aMinv = np.linalg.inv(a.M)
    aRinv = np.linalg.inv(aR)
    aAinv = np.linalg.inv(aA)
    
    b.M = np.dot(aMinv, c.M) / (-1j*dtau)

    bR = np.dot(aRinv, cR) / (dt)
    bA = np.dot(aAinv, cA) / (dt)

    b.RI = np.dot(aRinv, c.RI - np.dot(a.RI, b.M)*(-1j*dtau) ) / (dt)
    b.IR = np.dot(aMinv, c.IR - np.dot(a.IR, bA)*(dt) ) / (-1j*dtau)

    mixed_product = np.dot(a.RI, b.IR)*(-1j*dtau)

    b.G  = np.dot(aRinv, c.G  - np.dot(a.G, bA)*(dt) - mixed_product ) / (dt)
    b.L  = np.dot(aRinv, c.L  - np.dot(a.L, bA)*(dt) - mixed_product ) / (dt)
    
    return b

