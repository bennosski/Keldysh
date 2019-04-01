import numpy as np

def block_diag(x, norb):
    return np.reshape(np.einsum('xy,ab->xayb', x, np.diag(np.ones(norb))), [np.shape(x)[0]*norb, np.shape(y)[0]*norb])

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
def H(kx, ky):
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



