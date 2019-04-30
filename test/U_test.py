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
    tmax = 10.0
    dt_fine = 0.01
    #dt_fine = 10.0
    order = 6
    ntau = 800    

    nts = [10, 50, 100, 500]
    
    diffs = {}
    diffs['nts'] = nts
    diffs['U']  = []
    diffs['U_higher_order'] = []
    
    delta = 0.3
    omega = 0.2
    V = 0.5
    norb = 2

    def H(kx, ky, t):
        #Bx = 2.0*V*np.cos(2.0*omega*t)
        #By = 2.0*V*np.sin(2.0*omega*t)
        #Bz = 2.0*delta
        #return 0.5*np.array([[Bz, Bx-1j*By], [Bx+1j*By, -Bz]], dtype=complex)
        return np.array([[delta, V*np.exp(-2.0*1j*omega*t)], [V*np.exp(+2.0*1j*omega*t), -delta]], dtype=complex)
    
    def compute_time_dependent_G0(H, myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump):
        # check how much slower this is than computing G0 using U(t,t')
        
        dt = 1.0*tmax/(nt-1)

        G0M = compute_G00M(0, 0, *constants)
        
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

        constants = (myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump)

        integrator = integration.integrator(6, nt, beta, ntau)

        #---------------------------------------------------------
        # compute U(t,t') exactly and the corresponding G0

        ts = np.linspace(0, tmax, nt)

        UksR, UksI, eks, fks, Rs, _ = init_Uks(H, dt_fine, *constants)

        Omega = sqrt((delta-omega)**2 + V**2)
        Uexact = np.zeros([1, nt, norb, norb], dtype=np.complex128)
        Uexact[0, :, 0, 0] = np.exp(-1j*omega*ts)*(np.cos(Omega*ts) - 1j*(delta-omega)/Omega*np.sin(Omega*ts))
        Uexact[0, :, 0, 1] = -1j*V/Omega*np.exp(-1j*omega*ts)*np.sin(Omega*ts)
        Uexact[0, :, 1, 0] = -1j*V/Omega*np.exp(1j*omega*ts)*np.sin(Omega*ts)
        Uexact[0, :, 1, 1] = np.exp(1j*omega*ts)*(np.cos(Omega*ts) + 1j*(delta-omega)/Omega*np.sin(Omega*ts))

        print('check unitary ')
        p = np.einsum('tba,tbc->tac', np.conj(Uexact[0,:]), Uexact[0,:])
        print(dist(p, np.einsum('t,ab->tab', np.ones(nt), np.diag(np.ones(norb)))))
        
        #---------------------------------------------------------
        # compute G0 computed with U(t,t') via integration        
        UksR, UksI, eks, fks, Rs, _ = init_Uks(H, dt_fine, *constants, version='regular')

        # test for U(t,t')
        d = dist(Uexact, UksR)
        print('diff Uexact UksR', d)
        print("done computing U regular")
        diffs['U'].append(d)

        #---------------------------------------------------------
        # compute G0 computed with U(t,t') via higher-order integration        
        UksR, UksI, eks, fks, Rs, _ = init_Uks(H, dt_fine, *constants, version='higher order')

        # test for U(t,t')
        d = dist(Uexact, UksR)
        print('diff Uexact UksR', d)
        print("done computing U higher order")
        diffs['U_higher_order'].append(d)


    plt_diffs(diffs)
        
    if 'MPI' in sys.modules:
        MPI.Finalize()

    
if __name__=='__main__':
    main()
        


