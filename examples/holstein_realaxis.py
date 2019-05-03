# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 01:44:04 2016

@author: Ben
"""

#
# Add Hartree term
# Run with MPI
# timing/profiling
#
#


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
from params import *

def main():

    if myrank==0:
        time0 = time.time()
    
    if myrank==0:
        print(' ')
        print('nprocs = ',nprocs)

    assert not os.path.exists(savedir+'Sigma') and not os.path.exists(savedir+'Gloc'), 'Cannot overwrite existing data'

    volume = Nkx*Nky

    k2p, k2i, i2k = init_k2p_k2i_i2k(Nkx, Nky, nprocs, myrank)
    kpp = np.count_nonzero(k2p==myrank)

    integrator = integration.integrator(6, nt, beta, ntau)

    def H(kx, ky, t):
        return -2.0*np.cos(kx)*np.ones([norb, norb])
    constants = (myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump)
    UksR, UksI, eks, fks, Rs, Ht = init_Uks(H, dt_fine, *constants, version='higher order')

    print('Done initializing Us')

    SigmaM = matsubara(0,0,0,0)
    SigmaM.load(savedir, 'SigmaM')

    # Solve real axis part xo
    #---------------------------------------------------------

    D = compute_D0R(norb, omega, nt, tmax, ntau, beta, +1)

    print('Done initializing D')

    Sigma0 = langreth(norb, nt, tmax, ntau, beta, -1)
    Sigma  = langreth(norb, nt, tmax, ntau, beta, -1)
    iter_selfconsistency = 4
    change = 0.0
    for i in range(iter_selfconsistency):
        print('iteration : %d'%i)

        Sigma0.copy(Sigma)

        Gloc = langreth(norb, nt, tmax, ntau, beta, -1)
        for ik in range(kpp):
            ik1,ik2 = i2k[ik]

            G0M = compute_G0M(ik1, ik2, UksI, eks, fks, Rs, *constants)
            G0 = compute_G0R(ik1, ik2, UksR, UksI, eks, fks, Rs, *constants)

            if i==0:
                G = G0
            else:
                G  = langreth(norb, nt, tmax, ntau, beta, -1)
                integrator.dyson_langreth(G0M, SigmaM, G0, Sigma, G)
            
            Gloc.add(G)

        Sigma = langreth(norb, nt, tmax, ntau, beta, -1)
        if nprocs==1:
            Sigma.copy(Gloc)
        else:
            comm.Allreduce(Gloc.L,  Sigma.L,  op=MPI.SUM)
            comm.Allreduce(Gloc.R,  Sigma.R,  op=MPI.SUM)
            comm.Allreduce(Gloc.RI,  Sigma.RI,  op=MPI.SUM)
            comm.Allreduce(Gloc.deltaR,  Sigma.deltaR,  op=MPI.SUM)
        Sigma.multiply(D)
        Sigma.scale(1j * g2 / volume)

        print('Done computing Sigma')
        print('sigma size')
        print(np.mean(np.abs(Sigma.R)))
        print(np.mean(np.abs(Sigma.RI)))
        print(np.mean(np.abs(Sigma.L)))


        change = max([np.mean(abs(Sigma0.R-Sigma.R)), \
                      np.mean(abs(Sigma0.L-Sigma.L)), \
                      np.mean(abs(Sigma0.RI-Sigma.RI)), \
                      np.mean(abs(Sigma0.deltaR-Sigma.deltaR))])                      
        print('change = %1.3e'%change)

        
    Sigma.save(savedir, 'Sigma')
    Gloc.save(savedir, 'Gloc')    
    saveparams(savedir)
        
    if 'MPI' in sys.modules:
        MPI.Finalize()

    
if __name__=='__main__':
    main()
        


