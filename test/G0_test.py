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
from itertools import product

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
    tmax = 1.0
    dt_fine = 0.005
    
    order = 6
    ntau = 800
    
    #nts = [400,800,1000]
    #nts = [10, 50, 100, 500]

    #nts = [50, 100, 500]
    #nts = [50, 100, 500]
    #nts = [50, 100, 500]
    #nts = [123]

    nts = [200]
    
    diffs = {}
    diffs['nts'] = nts
    diffs['M']  = []
    diffs['IR'] = []
    diffs['R']  = []
    diffs['L']  = []

    # random H
    '''
    np.random.seed(1)
    norb = 3
    Hmat = np.random.randn(norb, norb)
    Hmat += np.conj(Hmat).T
    '''

    # 1x1
    '''
    norb = 1
    e0   = -0.2
    Hmat = np.array([[e0]], dtype=np.complex128)
    '''
    
    # 2x2 H
    norb = 2
    e0   = -0.2
    e1   =  0.2
    lamb = 1.0
    Hmat = np.array([[e0, lamb],
                     [np.conj(lamb), e1]], dtype=np.complex128)
        
    # 3x3 H
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

    def f(t): return np.array([[1.0, np.cos(0.01*t)], [np.cos(0.01*t), 1.0]])
    #def f(t): return 1.0

    def compute_time_dependent_G0(H, myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump):
        # check how much slower this is than computing G0 using U(t,t')
        
        '''
        norb = np.shape(H(0,0,0))[0]
        def H0(kx, ky, t): return np.zeros([norb,norb])
        constants = (myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump)
        UksR, UksI, eks, fks, Rs, Ht = init_Uks(H0, dt_fine, *constants)
        G0M = compute_G0M(0, 0, UksR, UksI, eks, fks, Rs, *constants)
        G0  = compute_G0R(0, 0, G0M, UksR, UksI, eks, fks, Rs, *constants)
        '''

        dt = 1.0*tmax/(nt-1)

        G0M = compute_G00M(0, 0, *constants)
        G0  = compute_G00R(0, 0, G0M, *constants)        

        '''
        G00M = compute_G00M(0, 0, *constants)
        G00  = compute_G00R(0, 0, G0M, *constants)

        print('shpae G0M', np.shape(G0M.M))
        print('shape G00M', np.shape(G00M.M))
        
        #for (i,j) in product(range(norb), repeat=2):
        #    plt(np.linspace(0,beta,ntau), [G0M.M[:,i,j].imag, G00M.M[:,i,j].imag], 'G00M %d %d'%(i,j))
        #    plt(np.linspace(0,beta,ntau), [G0M.M[:,i,j].real, G00M.M[:,i,j].real], 'G00M %d %d'%(i,j))

        print('diffs')
        print(dist(G0M.M, G00M.M))
        print(dist(G0.R, G00.R))
        print(dist(G0.L, G00.L))
        print(dist(G0.IR, G00.IR))
        exit()
        '''
        
        GM = matsubara(beta, ntau, norb, -1)
        SigmaM  = matsubara(beta, ntau, norb, -1)
        SigmaM.deltaM = H(0, 0, 0)
        integrator.dyson_matsubara(G0M, SigmaM, GM)

        # check if SigmaM is the same as before
        
        G = langreth(nt, tmax, GM)
        Sigma = langreth(nt, tmax, SigmaM)
        for it in range(nt):
            Sigma.deltaR[it] = H(0, 0, it*dt)
        integrator.dyson_langreth(G0, Sigma, G)

        return GM, G
    

    
    for nt in nts:

        integrator = integration.integrator(6, nt, beta, ntau)
        
        '''
        #---------------------------------------------------------
        # compute Ht used later for computing the embedding selfenergy
        def H(kx, ky, t): return Hmat * f(t)
        norb = np.shape(H(0,0,0))[0]        
        constants = (myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump)
        _, _, _, _, _, Ht = init_Uks(H, dt_fine, *constants)
        '''

        #---------------------------------------------------------
        # compute non-interacting G for the norb x norb problem
        # we compute this by solving Dyson's equation with the time-dependent hamiltonian as the selfenergy

        norb = np.shape(Hmat)[0]
        def H(kx, ky, t): return Hmat * f(t)
        constants = (myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump)
        GexactM, Gexact = compute_time_dependent_G0(H, *constants)
        print('done computing exact solution')

        print('diff GexactM.M and Gexact.M', dist(GexactM.M, Gexact.M))
        
        #---------------------------------------------------------
        # compare to G0 computed with U(t,t')

        def H(kx, ky, t): return Hmat * f(t)
        print('H(0,0,0)')
        print(H(0,0,0))
        norb = np.shape(Hmat)[0]
        constants = (myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump)
        UksR, UksI, eks, fks, Rs, _ = init_Uks(H, dt_fine, *constants)
        G0M = compute_G0M(0, 0, UksR, UksI, eks, fks, Rs, *constants)
        G0  = compute_G0R(0, 0, G0M, UksR, UksI, eks, fks, Rs, *constants)

        # test for U(t,t')
        ts = np.linspace(0,tmax,nt)
        Uexact = np.array([expm(-1j*H(0,0,0)*t) for t in ts])
        print('diff Uexact UksR', dist(Uexact, UksR[0]))
        '''
        ts = np.linspace(0,tmax,nt)
        Uexact = np.array([expm(-1j*H(0,0,0)*t) for t in ts])
        print('diff Uexact UksR', dist(Uexact, UksR[0]))
        for (i,j) in product(range(2), repeat=2):
            plt(ts, [UksR[0,:,i,j].real, Uexact[:,i,j].real], 'real part %d %d'%(i,j))
            plt(ts, [UksR[0,:,i,j].imag, Uexact[:,i,j].imag], 'imag part %d %d'%(i,j))            
        exit()
        '''
        
        print("done computing G0 using U(t,t')")

        for (i,j) in product(range(norb), repeat=2):
            plt(np.linspace(0, beta, ntau), [G0.M[:,i,j].imag, GexactM.M[:,i,j].imag], 'G0M')
        
        print('G0M diff')
        print(np.mean(abs(G0M.M-GexactM.M)))
        print('G0 diff (R,L,IR)')
        print(np.mean(abs(G0.R-Gexact.R)))
        print(np.mean(abs(G0.L-Gexact.L)))
        print(np.mean(abs(G0.IR-Gexact.IR)))

        '''
        for (i,j) in product(range(norb), repeat=2):
            print('i j %d %d'%(i,j))
            im([G0.R[:,i,:,j].imag, Gexact.R[:,i,:,j].imag], [0,tmax,0,tmax], 'R imag')
            im([G0.R[:,i,:,j].real, Gexact.R[:,i,:,j].real], [0,tmax,0,tmax], 'R real')
        '''

    #plt_diffs(diffs)
        
    if 'MPI' in sys.modules:
        MPI.Finalize()

    
if __name__=='__main__':
    main()
        


