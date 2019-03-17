from functions import *

def run_tests_vectorized_version(omega, i2k, constants):

    myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, Nt, Ntau, dt, dtau, pump, Norbs = constants

    if myrank!=0: return

    import time

    time0 = time.time()
    UksR, UksI, eks, fks = init_Uks(*constants)
    print 'new init Uks took ',time.time()-time0

    time0 = time.time()
    UksR_old, UksI_old, eks_old, fks_old = init_Uks_old(*constants)
    print 'old init Uks took ',time.time()-time0
    
    print 'diffs Uks'
    print np.amax(abs(UksR-UksR_old))
    print np.amax(abs(UksI-UksI_old))
    print np.amax(abs(eks-eks_old))
    print np.amax(abs(fks-fks_old))

    time0 = time.time()
    D = init_D(omega, Nt, Ntau, dt, dtau, Norbs)
    print 'new D time',time.time()-time0
    time0 = time.time()
    D_old = init_D_old(omega, Nt, Ntau, dt, dtau, Norbs)
    print 'old D time', time.time()-time0
    if myrank==0:
        print 'D\n', D
        print 'D_test\n',D_old
        D_old.scale(-1.0)
        D_old.add(D)
        print 'D-D_old\n',D_old

    # compute local Greens function for each processor
    for ik in range(kpp):
        ik1,ik2 = i2k[ik]

        G0k = compute_G0(ik1, ik2, fks, UksR, UksI, eks, *constants)
        G0k_old = compute_G0_old(ik1, ik2, fks, UksR, UksI, eks, *constants)

        print 'G0k\n',G0k
        print 'G0k2\n',G0k_old
        G0k_old.scale(-1.0)
        G0k_old.add(G0k)
        print 'G0k-G0k_old\n',G0k_old


def init_Uks_old(myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, Nt, Ntau, dt, dtau, pump, Norbs):
    '''
    for ARPES, use Nky = 1 
    '''
    
    beta = Ntau*dtau
    
    UksR = np.zeros([kpp, Nt, Norbs, Norbs], dtype=complex)
    UksI = np.zeros([kpp, Ntau, Norbs], dtype=complex)
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
                # also pull R across the U(tau,0) so that we work with diagonal things
                for it in range(Ntau):
                    UksI[index,it] = np.exp(-eks[index]*dtau*it)
                
    return UksR, UksI, eks, fks

def compute_G0_old(ik1, ik2, fks, UksR, UksI, eks, myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, Nt, Ntau, dt, dtau, pump, Norbs):
    # for ARPES use ik2 = 0

    G0 = langreth(Nt, Ntau, Norbs)
    
    beta  = dtau*Ntau
    theta = init_theta(Ntau)
        
    kx, ky = get_kx_ky(ik1, ik2, Nkx, Nky, ARPES)

    # check if this is the right k point for this proc
    # this should have been checked before calling this function
    if myrank==k2p[ik1,ik2]:
        index = k2i[ik1,ik2]

        _, R  = np.linalg.eig(Hk(kx, ky))
        G0L = 1j*np.einsum('ij,j,jk->ik', R, fks[index]-0.0, np.conj(R).T) # - 0.0 for lesser Green's function    
        G0G = 1j*np.einsum('ij,j,jk->ik', R, -fks[index]*np.exp(beta*eks[index]), np.conj(R).T) # - 1.0 for greater Green's function    
        for it1 in range(Nt):
            for it2 in range(Nt):
                G = np.einsum('ij,jk,kl->il', UksR[index,it1], G0L, np.conj(UksR[index,it2]).T)
                for a in range(Norbs):
                    for b in range(Norbs):
                        G0.L[a*Nt+it1,b*Nt+it2] = G[a,b]

                G = np.einsum('ij,jk,kl->il', UksR[index,it1], G0G, np.conj(UksR[index,it2]).T)
                for a in range(Norbs):
                    for b in range(Norbs):
                        G0.G[a*Nt+it1,b*Nt+it2] = G[a,b]


        for it1 in range(Nt):
            t1 = it1 * dt
            for it2 in range(Ntau):
                t2 = -1j * it2 * dtau

                #UksI_inv = [UksI[index,it2,1], UksI[index,it2,0]]
                #G = fks[index] * UksI_inv
                G = fks[index] / UksI[index,it2]
                G = 1j*np.einsum('ij,jk,kl,lm->im', UksR[index,it1], R, np.diag(G), np.conj(R).T)
                #G = np.einsum('ij,jk,kl->il', UksR[index,it1], G0L,  )
                for a in range(Norbs):
                    for b in range(Norbs):
                        G0.RI[a*Nt+it1,b*Ntau+it2] = G[a,b]

        for it1 in range(Ntau):
            t1 = -1j * it1 * dtau
            for it2 in range(Nt):
                t2 = it2 * dt

                G = -UksI[index,it1]*fks[index]*np.exp(beta*eks[index])
                G = 1j*np.einsum('ij,jk,kl,lm->im', R, np.diag(G), np.conj(R).T, np.conj(UksR[index,it2]).T)  
                #G = np.einsum('ij,jk,kl->il',  , G0G, np.conj(UksR[index,it2]).T)
                for a in range(Norbs):
                    for b in range(Norbs):
                        G0.IR[a*Ntau+it1,b*Nt+it2] = G[a,b]


        #G = 1j*np.einsum('ij,mnj,kj->mnik', R,  , np.conj(R))

        for it1 in range(Ntau):
            t1 = -1j * it1 * dtau
            for it2 in range(Ntau):
                t2 = -1j * it2 * dtau

                if it1==it2:
                    e1 = (fks[index,0]-0.5) * np.exp(-1j*(+eks[index,0])*(-1j*dtau*(it1-it2)))
                    e2 = (fks[index,1]-0.5) * np.exp(-1j*(+eks[index,1])*(-1j*dtau*(it1-it2)))
                elif it1>it2:
                    e1 = (-fks[index,0]*np.exp(beta*eks[index,0])) * np.exp(-1j*(+eks[index,0])*(-1j*dtau*(it1-it2)))
                    e2 = (-fks[index,1]*np.exp(beta*eks[index,1])) * np.exp(-1j*(+eks[index,1])*(-1j*dtau*(it1-it2)))
                else:
                    e1 = (fks[index,0]) * np.exp(-1j*(+eks[index,0])*(-1j*dtau*(it1-it2)))
                    e2 = (fks[index,1]) * np.exp(-1j*(+eks[index,1])*(-1j*dtau*(it1-it2)))

                G = 1j*np.einsum('ij,j,jk->ik', R, [e1, e2], np.conj(R).T)

                for a in range(Norbs):
                    for b in range(Norbs):
                        G0.M[a*Ntau+it1,b*Ntau+it2] = G[a,b]
                
    return G0
                

# do this in the orbital basis
# so D has zeros in off-diagonal blocks
# no U transformations needed
def init_D_old(omega, Nt, Ntau, dt, dtau, Norbs):

    D = langreth(Nt, Ntau, Norbs)

    beta = dtau*Ntau
    nB   = 1./(np.exp(beta*omega)-1.0)
    theta = init_theta(Ntau)
    theta_transpose = np.transpose(theta)

    for it1 in range(Nt):
        t1 = it1 * dt
        for it2 in range(Nt):
            t2 = it2 * dt
            for ib in range(Norbs):
                D.L[ib*Nt+it1, ib*Nt+it2] = -1j*(nB + 1.0 - 0.0)*np.exp(1j*omega*(t1-t2)) - 1j*(nB + 0.0)*np.exp(-1j*omega*(t1-t2))
                D.G[ib*Nt+it1, ib*Nt+it2] = -1j*(nB + 1.0 - 1.0)*np.exp(1j*omega*(t1-t2)) - 1j*(nB + 1.0)*np.exp(-1j*omega*(t1-t2))


    for it1 in range(Ntau):
        t1 = -1j * it1 * dtau
        for it2 in range(Ntau):
            t2 = -1j * it2 * dtau
            for ib in range(Norbs):
                D.M[ib*Ntau+it1, ib*Ntau+it2] = -1j*(nB + theta_transpose[it1,it2])*np.exp(1j*omega*(t1-t2)) - 1j*(nB +  theta[it1,it2])*np.exp(-1j*omega*(t1-t2))


    for it1 in range(Nt):
        t1 = it1 * dt
        for it2 in range(Ntau):
            t2 = -1j * it2 * dtau
            for ib in range(Norbs):
                D.RI[ib*Nt+it1,ib*Ntau+it2] = -1j*(nB + 1.0 - 0.0)*np.exp(1j*omega*(t1-t2)) - 1j*(nB + 0.0)*np.exp(-1j*omega*(t1-t2))


    for it1 in range(Ntau):
        t1 = -1j * it1 * dtau
        for it2 in range(Nt):
            t2 = it2 * dt
            for ib in range(Norbs):
                D.IR[ib*Ntau+it1,ib*Nt+it2] = -1j*(nB + 1.0 - 1.0)*np.exp(1j*omega*(t1-t2)) - 1j*(nB + 1.0)*np.exp(-1j*omega*(t1-t2))

    return D


def old_multiply(a, b, c, Nt, Ntau, dt, dtau, Norbs):

    aR, aA = initRA(a, Nt, Norbs)
    bR, bA = initRA(b, Nt, Norbs)

    #will this change a and b?

    aR  += np.diag(a.DR)
    aA  += np.diag(a.DR)
    a.M += np.diag(a.DM)

    bR  += np.diag(b.DR)
    bA  += np.diag(b.DR)
    b.M += np.diag(b.DM)

    #c = langreth(Nt, Ntau, Norbs)
    c.zero(Nt, Ntau, Norbs)
    
    c.M = np.dot(a.M, b.M) * (-1j*dtau)

    # what are these lines for??
    cR = np.dot(aR, bR) * (dt)
    cA = np.dot(aA, bA) * (dt)

    mixed_product = np.dot(a.RI, b.IR) * (-1j*dtau)
    
    c.G = (np.dot(a.G, bA) + np.dot(aR, b.G)) * (dt) + mixed_product
    c.L = (np.dot(a.L, bA) + np.dot(aR, b.L)) * (dt) + mixed_product
    
    c.RI = np.dot(aR, b.RI) * (dt) + np.dot(a.RI, b.M) * (-1j*dtau)
    c.IR = np.dot(a.IR, bA) * (dt) + np.dot(a.M, b.IR) * (-1j*dtau)
    
    #return c
