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
    
if myrank==0:
    print(' ')
    print('nprocs = ',nprocs)
    
Nkx = 1
Nky = 1
k2p, k2i, i2k = init_k2p_k2i_i2k(Nkx, Nky, nprocs, myrank)
kpp = np.count_nonzero(k2p==myrank)

def im_plot(x, y):
    y1 = x[:,0,:,0]
    y2 = y[:,0,:,0]
    im([y1.imag, y2.imag, y1.imag-y2.imag], [0,tmax,0,tmax], 'imag')
    im([y1.real, y2.real, y1.real-y2.real], [0,tmax,0,tmax], 'real')

def main():
    
    beta = 10.0
    ARPES = False
    pump = 0
    g2 = None
    omega = None
    tmax = 10.0
    e1   = -0.1
    e2   =  0.1
    lamb = 1.0
    order = 6
    ntau = 800
    dt_fine = 0.01

    nts = [10, 50, 100, 500]
    
    diffs = {}
    diffs['nts'] = nts
    diffs['M']  = []
    diffs['IR'] = []
    diffs['R']  = []
    diffs['L']  = []
    
    for nt in nts:
        
        #---------------------------------------------------------
        # compute non-interacting G for the 2x2 problem
        norb = 2
        def H(kx, ky, t): return np.array([[e1, lamb], [np.conj(lamb), e2]], dtype=complex)
        constants = (myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump)
        UksR, UksI, eks, fks, Rs, _ = init_Uks(H, dt_fine, *constants, version='higher order')
        G2x2M = compute_G0M(0, 0, UksR, UksI, eks, fks, Rs, *constants)
        G2x2  = compute_G0R(0, 0, G2x2M, UksR, UksI, eks, fks, Rs, *constants)
        
        #------------------------------------------------------
        # compute Sigma_embedding
        # Sigma = |lambda|^2 * g22(t,t')

        norb = 1
        def H(kx, ky, t): return e2*np.ones([1,1])
        constants = (myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump)
        UksR, UksI, eks, fks, Rs, _ = init_Uks(H, dt_fine, *constants, version='higher order')
        SigmaM = compute_G0M(0, 0, UksR, UksI, eks, fks, Rs, *constants)
        SigmaM.scale(lamb*np.conj(lamb))
        Sigma = compute_G0R(0, 0, SigmaM, UksR, UksI, eks, fks, Rs, *constants)
        Sigma.scale(lamb*np.conj(lamb))

        #------------------------------------------------------
        # solve the embedding problem
        
        norb = 1
        def H(kx, ky, t): return e1*np.ones([1,1])
        constants = (myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump)
        UksR, UksI, eks, fks, Rs, _ = init_Uks(H, dt_fine, *constants, version='higher order')
        G0M = compute_G0M(0, 0, UksR, UksI, eks, fks, Rs, *constants)
        G0  = compute_G0R(0, 0, G0M, UksR, UksI, eks, fks, Rs, *constants)
                
        integrator = integration.integrator(6, nt, beta, ntau)

        GM = matsubara(beta, ntau, norb, -1)
        integrator.dyson_matsubara(G0M, SigmaM, GM)
        
        G  = langreth(nt, tmax, GM)
        integrator.dyson_langreth(G0, Sigma, G)
        
        #------------------------------------------------------
        # compute differences
        
        diff = np.mean(abs(GM.M[:,0,0]-G2x2M.M[:,0,0]))
        print('diff = %1.3e'%diff)
        diffs['M'].append(diff)
        
        diff = np.mean(abs(G.R[:,0,:,0]-G2x2.R[:,0,:,0]))
        print('diff langreth R = %1.3e'%diff)
        diffs['R'].append(diff)
        
        #im_plot(G.R, G2x2.R)
        
        diff = np.mean(abs(G.IR[:,0,:,0]-G2x2.IR[:,0,:,0]))
        print('diff langreth IR = %1.3e'%diff)
        diffs['IR'].append(diff)
        
        #im_plot(G.IR, G2x2.IR)

        diff = np.mean(abs(G.L[:,0,:,0]-G2x2.L[:,0,:,0]))
        print('diff langreth L = %1.3e'%diff)
        diffs['L'].append(diff)
        
        #im_plot(G.L, G2x2.L)
        
        #------------------------------------------------------

    plt_diffs(diffs)
        
    if 'MPI' in sys.modules:
        MPI.Finalize()

    
if __name__=='__main__':
    main()
        


