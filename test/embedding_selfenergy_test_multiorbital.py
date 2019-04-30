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

def main():

    if myrank==0:
        time0 = time.time()    
        print(' ')
        print('nprocs = ',nprocs)
    
    Nkx = 1 
    Nky = 1
    k2p, k2i, i2k = init_k2p_k2i_i2k(Nkx, Nky, nprocs, myrank)
    kpp = np.count_nonzero(k2p==myrank)
    
    beta = 2.0
    ARPES = False
    pump = 0
    g2 = None
    omega = None
    tmax = 5.0

    dim_embedding = 2
    
    order = 6
    ntau = 200
    
    #nts = [400,800,1000]
    #nts = [10, 50, 100, 500]

    #nts = [50, 100, 500]
    #nts = [50, 100, 500]
    nts = [10, 50, 100, 500]
    
    diffs = {}
    diffs['nts'] = nts
    diffs['M']  = []
    diffs['RI'] = []
    diffs['R']  = []
    diffs['L']  = []

    # random H
    np.random.seed(1)
    norb = 4
    Hmat = 0.1*np.random.randn(norb, norb) + 0.1*1j*np.random.randn(norb, norb)
    Hmat += np.conj(Hmat).T
    Tmat = 0.1*np.random.randn(norb, norb) + 0.1*1j*np.random.randn(norb, norb)
    Tmat += np.conj(Tmat).T
    

    # example H
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
    '''
    
    print('\nH : ')
    print(Hmat)
    print('')
    
    #def f(t): return np.cos(0.01*t)
    def f(t): return 1.0
    
    for nt in nts:

        dt_fine = 0.1*tmax/(nt-1)
        
        #---------------------------------------------------------
        # compute non-interacting G for the norb x norb problem
        norb = np.shape(Hmat)[0]
        def H(kx, ky, t): return Hmat + Tmat * np.cos(0.2*t)
            #return np.array([[e0, lamb1, lamb2],
            #                 [np.conj(lamb1), e1, 0],
            #                 [np.conj(lamb2), 0, e2]],
            #                 dtype=complex)
            
        constants = (myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump)
        UksR, UksI, eks, fks, Rs, Ht = init_Uks(H, dt_fine, *constants, version='higher order')
        
        GexactM = compute_G0M(0, 0, UksR, UksI, eks, fks, Rs, *constants)
        Gexact  = compute_G0R(0, 0, UksR, UksI, eks, fks, Rs, *constants)
        
        #------------------------------------------------------
        # compute Sigma_embedding
        # Sigma = sum_{i,j} H0i(t) Gij(t,t') Hj0(t')

        norb = np.shape(Hmat)[0]-dim_embedding
        SigmaM = matsubara(beta, ntau, norb, -1)     
        def H(kx, ky, t): return Hmat[dim_embedding:, dim_embedding:] + Tmat[dim_embedding:, dim_embedding:] * np.cos(0.2*t)
        constants = (myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump)
        UksR, UksI, eks, fks, Rs, _ = init_Uks(H, dt_fine, *constants, version='higher order')

        SM = compute_G0M(0, 0, UksR, UksI, eks, fks, Rs, *constants)
        SM.M = np.einsum('hi,mij,jk->mhk', Ht[0,0,:dim_embedding,dim_embedding:], SM.M, Ht[0,0,dim_embedding:,:dim_embedding])

        #taus = np.linspace(0, beta, ntau)
        #plt(taus, [SM.M[:,0,0].real, SM.M[:,0,0].imag], 'SM')
        #exit()
        
        S = compute_G0R(0, 0, UksR, UksI, eks, fks, Rs, *constants)
        S.R  = np.einsum('mhi,minj,njk->mhnk', Ht[0,:,:dim_embedding,dim_embedding:], S.R, Ht[0,:,dim_embedding:,:dim_embedding])
        S.L  = np.einsum('mhi,minj,njk->mhnk', Ht[0,:,:dim_embedding,dim_embedding:], S.L, Ht[0,:,dim_embedding:,:dim_embedding])
        S.RI = np.einsum('mhi,minj,jk->mhnk', Ht[0,:,:dim_embedding,dim_embedding:], S.RI, Ht[0,0,dim_embedding:,:dim_embedding])
        
        #dt = 1.0*tmax/(nt-1)
        #ts = np.linspace(0, tmax, nt)
            
        SigmaM = matsubara(beta, ntau, dim_embedding, -1)
        SigmaM.M = SM.M
        Sigma = langreth(norb, nt, tmax, ntau, beta, -1)
        Sigma.L = S.L
        Sigma.R = S.R
        Sigma.RI = S.RI
        
        #------------------------------------------------------
        # solve the embedding problem
        
        norb = dim_embedding
        def H(kx, ky, t): return Hmat[:dim_embedding,:dim_embedding] + Tmat[:dim_embedding, :dim_embedding] * np.cos(0.2*t)
        constants = (myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump)
        #Ht = init_Ht(H, *constants)
        UksR, UksI, eks, fks, Rs, _ = init_Uks(H, dt_fine, *constants, version='higher order')
        G0M = compute_G0M(0, 0, UksR, UksI, eks, fks, Rs, *constants)
        G0  = compute_G0R(0, 0, UksR, UksI, eks, fks, Rs, *constants)
                
        integrator = integration.integrator(6, nt, beta, ntau)

        GM = matsubara(beta, ntau, norb, -1)
        integrator.dyson_matsubara(G0M, SigmaM, GM)

        print('differences Matsubara')
        diff = np.mean(abs(GM.M-GexactM.M[:,:dim_embedding,:dim_embedding]))
        print('diff = %1.3e'%diff)
                
        G  = langreth(norb, nt, tmax, ntau, beta, -1)
        integrator.dyson_langreth(G0M, SigmaM, GM, G0, Sigma, G)
        
        #------------------------------------------------------
        
        # compute differences        
        diff = np.mean(abs(GM.M-GexactM.M[:,:dim_embedding,:dim_embedding]))
        print('diff = %1.3e'%diff)
        diffs['M'].append(diff)
        
        diff = np.mean(abs(G.R-Gexact.R[:,:dim_embedding,:,:dim_embedding]))
        print('diff langreth R = %1.3e'%diff)
        diffs['R'].append(diff)

        diff = np.mean(abs(G.RI-Gexact.RI[:,:dim_embedding,:,:dim_embedding]))
        print('diff langreth RI = %1.3e'%diff)
        diffs['RI'].append(diff)

        diff = np.mean(abs(G.L-Gexact.L[:,:dim_embedding,:,:dim_embedding]))
        print('diff langreth L = %1.3e'%diff)
        diffs['L'].append(diff)
        
        #------------------------------------------------------

    plt_diffs(diffs)
        
    if 'MPI' in sys.modules:
        MPI.Finalize()

    
if __name__=='__main__':
    main()
        


