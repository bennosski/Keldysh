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

def main():

    if myrank==0:
        time0 = time.time()    
        print(' ')
        print('nprocs = ',nprocs)
    
    Nkx = 1
    Nky = 1
    k2p, k2i, i2k = init_k2p_k2i_i2k(Nkx, Nky, nprocs, myrank)
    kpp = np.count_nonzero(k2p==myrank)
    
    beta = 10.0
    ARPES = False
    pump = 0
    g2 = None
    omega = None
    tmax = 1.0
    #dt_fine = 0.001
    dt_fine = 0.001
    order = 6
    ntau = 800    

    #nts = [10, 50, 100, 500]

    nts = [100]
    
    diffs = {}
    diffs['nts'] = nts
    diffs['U']  = []
    diffs['M']  = []
    diffs['IR'] = []
    diffs['R']  = []
    diffs['L']  = []

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

        norb = np.shape(H(0,0,0))[0]
        def H0(kx, ky, t): return np.zeros([norb,norb])
        constants = (myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump)
        UksR, UksI, eks, fks, Rs, Ht = init_Uks(H0, dt_fine, *constants, version='higher order')
        G0M_ref = compute_G0M(0, 0, UksR, UksI, eks, fks, Rs, *constants)
        G0_ref  = compute_G0R(0, 0, G0M_ref, UksR, UksI, eks, fks, Rs, *constants)

        dt = 1.0*tmax/(nt-1)

        G0M = compute_G00M(0, 0, *constants)
        G0  = compute_G00R(0, 0, G0M, *constants)        

        print('test G00')
        differences(G0_ref, G0)
        
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

        constants = (myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump)

        integrator = integration.integrator(6, nt, beta, ntau)

        #---------------------------------------------------------
        # compute U(t,t') exactly and the corresponding G0

        #Uexact = np.array([expm(-1j*H(0,0,0)*t) for t in ts])

        ts = np.linspace(0, tmax, nt)
        _, UksI, eks, fks, Rs, _ = init_Uks(H, dt_fine, *constants)

        Omega = sqrt((delta-omega)**2 + V**2)
        Uexact = np.zeros([1, nt, norb, norb], dtype=np.complex128)
        Uexact[0, :, 0, 0] = np.exp(-1j*omega*ts)*(np.cos(Omega*ts) - 1j*(delta-omega)/Omega*np.sin(Omega*ts))
        Uexact[0, :, 0, 1] = -1j*V/Omega*np.exp(-1j*omega*ts)*np.sin(Omega*ts)
        Uexact[0, :, 1, 0] = -1j*V/Omega*np.exp(1j*omega*ts)*np.sin(Omega*ts)
        Uexact[0, :, 1, 1] = np.exp(1j*omega*ts)*(np.cos(Omega*ts) + 1j*(delta-omega)/Omega*np.sin(Omega*ts))

        print('check unitary ')
        p = np.einsum('tba,tbc->tac', np.conj(Uexact[0,:]), Uexact[0,:])
        print(dist(p, np.einsum('t,ab->tab', np.ones(nt), np.diag(np.ones(norb)))))

        GMexact = compute_G0M(0, 0, Uexact, UksI, eks, fks, Rs, *constants)
        Gexact = compute_G0R(0, 0, GMexact, Uexact, UksI, eks, fks, Rs, *constants)
        
        #---------------------------------------------------------
        # compute G0 computed with U(t,t') via integration        
        UksR, UksI, eks, fks, Rs, _ = init_Uks(H, dt_fine, *constants, version='higher order')
        G0M = compute_G0M(0, 0, UksR, UksI, eks, fks, Rs, *constants)
        G0  = compute_G0R(0, 0, G0M, UksR, UksI, eks, fks, Rs, *constants)

        # test for U(t,t')
        d = dist(Uexact, UksR)
        print('diff Uexact UksR', d)
        print("done computing G0 using U(t,t')")
        diffs['U'].append(d)
        
        #---------------------------------------------------------
        # compute non-interacting G for the norb x norb problem
        # we compute this by solving Dyson's equation with the time-dependent hamiltonian as the selfenergy

        GdysonM, Gdyson = compute_time_dependent_G0(H, *constants)
        print('done computing G0 via Dyson equation')
        
        '''
        ts = np.linspace(0,tmax,nt)
        Uexact = np.array([expm(-1j*H(0,0,0)*t) for t in ts])
        print('diff Uexact UksR', dist(Uexact, UksR[0]))
        for (i,j) in product(range(2), repeat=2):
            plt(ts, [UksR[0,:,i,j].real, Uexact[:,i,j].real], 'real part %d %d'%(i,j))
            plt(ts, [UksR[0,:,i,j].imag, Uexact[:,i,j].imag], 'imag part %d %d'%(i,j))            
        exit()
        '''

        
        #for (i,j) in product(range(norb), repeat=2):
        #    plt(np.linspace(0, beta, ntau), [G0.M[:,i,j].imag, GdysonM.M[:,i,j].imag], 'G0M')

        '''
        for (i,j) in product(range(norb), repeat=2):
            im([Gexact.L[:,i,:,j].real, G0.L[:,i,:,j].real, Gdyson.L[:,i,:,j].real], [0,tmax,0,tmax], 'G0 real L %d %d'%(i,j))
            im([Gexact.L[:,i,:,j].imag, G0.L[:,i,:,j].imag, Gdyson.L[:,i,:,j].imag], [0,tmax,0,tmax], 'G0 imag L %d %d'%(i,j))
        '''
        
        print('differences between G0 and Gexact')
        differences(G0, Gexact)

        print('differences between Gdyson and Gexact')
        differences(Gdyson, Gexact)
        
        '''
        for (i,j) in product(range(norb), repeat=2):
            print('i j %d %d'%(i,j))
            im([G0.R[:,i,:,j].imag, Gexact.R[:,i,:,j].imag], [0,tmax,0,tmax], 'R imag')
            im([G0.R[:,i,:,j].real, Gexact.R[:,i,:,j].real], [0,tmax,0,tmax], 'R real')
        '''

    plt_diffs(diffs)
        
    if 'MPI' in sys.modules:
        MPI.Finalize()

    
if __name__=='__main__':
    main()
    print('\nPASSED TEST')
        


