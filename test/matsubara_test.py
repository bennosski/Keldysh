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

if myrank==0:
    time0 = time.time()
    
if myrank==0:
    print(' ')
    print('nprocs = ',nprocs)
    
Nkx = 1
Nky = 1
k2p, k2i, i2k = init_k2p_k2i_i2k(Nkx, Nky, nprocs, myrank)
kpp = np.count_nonzero(k2p==myrank)

def main():
    
    beta = 2.0
    ARPES = False
    pump = 0
    g2 = None
    omega = None
    tmax = 1.0
    nt = 10

    e1 = -0.1
    e2 =  0.1
    lamb = 1.0
    
    diffs_vs_ntau = []
    ntaus = [200]
    
    for ntau in ntaus:
        
        #---------------------------------------------------------
        # compute non-interacting G for the 2x2 problem (exact solution)

        norb = 2
        def H(kx, ky):
            return np.array([[e1, lamb], [np.conj(lamb), e2]], dtype=complex)

        constants = (myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump)
        UksR, UksI, eks, fks, Rs = init_Uks(H, *constants)
        G2x2 = compute_G0M(0, 0, UksR, UksI, eks, fks, Rs, *constants)
        
        #------------------------------------------------------
        # compute Sigma_embedding
        # Sigma = |lambda|^2 * g22(t,t')

        norb = 1
        def H(kx, ky): return e2*np.ones([1,1])
        constants = (myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump)
        UksR, UksI, eks, fks, Rs = init_Uks(H, *constants)
        Sigma = compute_G0M(0, 0, UksR, UksI, eks, fks, Rs, *constants)
        Sigma.scale(lamb*np.conj(lamb))
        
        # solve the embedding problem
        
        norb = 1
        def H(kx, ky): return e1*np.ones([1,1])
        constants = (myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump)
        UksR, UksI, eks, fks, Rs = init_Uks(H, *constants)
        G0 = compute_G0M(0, 0, UksR, UksI, eks, fks, Rs, *constants)
        
        G = matsubara(beta, ntau, norb, -1)

        integrator = integration.integrator(5, nt, beta, ntau, norb)

        integrator.dyson_matsubara(G0, Sigma, G)

        #plt(linspace(0,beta,ntau), [G.M[:,0].imag, G2x2M[:,0,0].imag], 'Gsol and G2x2')
        
        print('diff = %1.3e'%np.amax(abs(G.M[:,0,0]-G2x2.M[:,0,0])))

        #------------------------------------------------------
        
    #np.save(savedir+'diffs', diffs_vs_deltat)
    #np.save(savedir+'dts', dts)

    if 'MPI' in sys.modules:
        MPI.Finalize()

if __name__=='__main__':
    main()
        


