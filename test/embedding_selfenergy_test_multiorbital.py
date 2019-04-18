# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 01:44:04 2016

@author: Ben
"""

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    nprocs = comm.size
    myrank = comm.rank
except ImportError:
    print('MPI not found')
    myrank = 0
    nprocs = 1

import src    
import pdb
import numpy as np
import time
import sys, os
from langreth import *
import shutil
from util import *
from functions import *
import integration
from matsubara import *
from plotting import *
import time

if myrank==0:
    time0 = time.time()    
    print(' ')
    print('nprocs = ',nprocs)
    
Nkx = 1
Nky = 1
k2p, k2i, i2k = init_k2p_k2i_i2k(Nkx, Nky, nprocs, myrank)
kpp = np.count_nonzero(k2p==myrank)

def main():
    
    beta = 10.0
    ARPES = False
    pump = 0
    g2 = None
    omega = None
    tmax = 10.0
    
    order = 6
    ntau = 800
    
    #nts = [400,800,1000]
    #nts = [10, 50, 100, 500]

    #nts = [50, 100, 500]
    nts = [50, 100, 500]
    
    diffs = {}
    diffs['nts'] = nts
    diffs['M']  = []
    diffs['IR'] = []
    diffs['R']  = []
    diffs['L']  = []

    '''
    np.random.seed(1)
    norb = 3
    Hmat = np.random.randn(norb, norb)
    Hmat += np.conj(Hmat).T
    '''
    
    norb = 3
    e0   = -0.2
    e1   = -0.1
    e2   =  0.2
    lamb1 = 1.0
    lamb2 = 1.2
    Hmat = np.array([[e0, lamb1, lamb2],
                     [np.conj(lamb1), e1, 0],
                     [np.conj(lamb2), 0, e2]],
                     dtype=complex)
    

    print('H : ')
    print(Hmat)
    print('')
    
    for nt in nts:
        
        #---------------------------------------------------------
        # compute non-interacting G for the 3x3 problem
        norb = np.shape(Hmat)[0]
        def H(kx, ky): return Hmat
            #return np.array([[e0, lamb1, lamb2],
            #                 [np.conj(lamb1), e1, 0],
            #                 [np.conj(lamb2), 0, e2]],
            #                 dtype=complex)
            
        constants = (myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump)
        Ht = init_Ht(H, *constants)
        UksR, UksI, eks, fks, Rs = init_Uks(Ht, *constants)
        GexactM = compute_G0M(0, 0, UksR, UksI, eks, fks, Rs, *constants)
        Gexact  = compute_G0R(0, 0, GexactM, UksR, UksI, eks, fks, Rs, *constants)
        
        #------------------------------------------------------
        # compute Sigma_embedding
        # Sigma = sum_{i,j} H0i(t) Gij(t,t') Hj0(t')

        norb = np.shape(Hmat)[0]-1
        SigmaM = matsubara(beta, ntau, norb, -1)     
        #def H(kx, ky): return Hmat[1:, 1:]
        constants = (myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump)
        #Ht = init_Ht(H, *constants)
        UksR, UksI, eks, fks, Rs = init_Uks(Ht[:,:,1:,1:], *constants)

        SM = compute_G0M(0, 0, UksR, UksI, eks, fks, Rs, *constants)
        SM.M = np.einsum('i,mij,j->m', Ht[0,0,0,1:], SM.M, Ht[0,0,1:,0])[:,None,None]

        taus = np.linspace(0, beta, ntau)
        plt(taus, [SM.M[:,0,0].real, SM.M[:,0,0].imag], 'SM')
        #exit()
        
        S = compute_G0R(0, 0, SM, UksR, UksI, eks, fks, Rs, *constants)
        S.R  = np.einsum('mi,minj,nj->mn', Ht[0,:,0,1:], S.R, Ht[0,:,1:,0])[:,None,:,None]
        S.L  = np.einsum('mi,minj,nj->mn', Ht[0,:,0,1:], S.L, Ht[0,:,1:,0])[:,None,:,None]
        S.IR = np.einsum('i,minj,nj->mn', Ht[0,0,0,1:], S.IR, Ht[0,:,1:,0])[:,None,:,None]

        dt = 1.0*tmax/(nt-1)
        ts = np.arange(0, nt*dt-dt/2.0, dt)
        assert len(ts)==nt
        plt(ts, [S.R[:,0,0,0].real, S.R[:,0,0,0].imag], 'SR')
            
        SigmaM = matsubara(beta, ntau, 1, -1)
        SigmaM.M = SM.M
        Sigma = langreth(nt, tmax, SigmaM)
        Sigma.L = S.L
        Sigma.R = S.R
        Sigma.IR = S.IR
        
        #------------------------------------------------------
        # solve the embedding problem
        
        norb = 1
        def H(kx, ky): return Hmat[0,0]*np.ones([1,1])
        constants = (myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump)
        Ht = init_Ht(H, *constants)
        UksR, UksI, eks, fks, Rs = init_Uks(Ht, *constants)
        G0M = compute_G0M(0, 0, UksR, UksI, eks, fks, Rs, *constants)
        G0  = compute_G0R(0, 0, G0M, UksR, UksI, eks, fks, Rs, *constants)
                
        integrator = integration.integrator(6, nt, beta, ntau, norb)

        GM = matsubara(beta, ntau, norb, -1)
        integrator.dyson_matsubara(G0M, SigmaM, GM)
        
        G  = langreth(nt, tmax, GM)
        integrator.dyson_langreth(G0, Sigma, G)
        
        #------------------------------------------------------
        
        # compute differences        
        diff = np.mean(abs(GM.M[:,0,0]-GexactM.M[:,0,0]))
        print('diff = %1.3e'%diff)
        diffs['M'].append(diff)
        
        diff = np.mean(abs(G.R[:,0,:,0]-Gexact.R[:,0,:,0]))
        print('diff langreth R = %1.3e'%diff)
        diffs['R'].append(diff)
                
        diff = np.mean(abs(G.IR[:,0,:,0]-Gexact.IR[:,0,:,0]))
        print('diff langreth IR = %1.3e'%diff)
        diffs['IR'].append(diff)

        diff = np.mean(abs(G.L[:,0,:,0]-Gexact.L[:,0,:,0]))
        print('diff langreth L = %1.3e'%diff)
        diffs['L'].append(diff)
                
        #------------------------------------------------------

    plt_diffs(diffs)
        
    if 'MPI' in sys.modules:
        MPI.Finalize()

    
if __name__=='__main__':
    main()
        


