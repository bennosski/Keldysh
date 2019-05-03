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

from params import *

def main():

    if myrank==0:
        time0 = time.time()
    
    if myrank==0:
        print(' ')
        print('nprocs = ',nprocs)
    
    if not os.path.exists(savedir): os.mkdir(savedir)
    assert not os.path.exists(savedir+'SigmaM') and not os.path.exists(savedir+'GlocM'), 'Cannot overwrite existing data'

    volume = Nkx*Nky

    k2p, k2i, i2k = init_k2p_k2i_i2k(Nkx, Nky, nprocs, myrank)
    kpp = np.count_nonzero(k2p==myrank)

    integrator = integration.integrator(6, nt, beta, ntau)

    def H(kx, ky):
        return -2.0*np.cos(kx)*np.ones([norb, norb])
    constants = (myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump)
    UksI, eks, fks, Rs = init_UksM(H, dt_fine, *constants, version='higher order')

    print('Done initializing Us')
    
    # Solve Matsubara problem
    #---------------------------------------------------------

    DM = compute_D0M(omega, beta, ntau, norb)

    SigmaM = matsubara(beta, ntau, norb, -1)
    iter_selfconsistency = 8
    change = 0.0
    for i in range(iter_selfconsistency):
        print('iteration : %d'%i)

        SigmaM0 = SigmaM.M.copy()

        GlocM = matsubara(beta, ntau, norb, -1)
        for ik in range(kpp):
            ik1,ik2 = i2k[ik]
            G0M = compute_G0M(ik1, ik2, UksI, eks, fks, Rs, *constants)

            if i==0:
                GM = G0M
            else:
                GM = matsubara(beta, ntau, norb, -1)
                integrator.dyson_matsubara(G0M, SigmaM, GM)
            
            GlocM.add(GM)

        SigmaM = matsubara(beta, ntau, norb, -1)
        if nprocs==1:
            SigmaM.M = GlocM.M
        else:
            comm.Allreduce(GlocM.M,  SigmaM.M,  op=MPI.SUM)
        SigmaM.multiply(DM)
        SigmaM.scale(1j * g2 / volume)

        print('Done computing SigmaM')

        change = np.mean(abs(SigmaM0-SigmaM.M))
        print('change = %1.3e'%change)

        print('Done computing GlocM')


    SigmaM.save(savedir, 'SigmaM')
    GlocM.save(savedir, 'GlocM')
    saveparams(savedir)

    if 'MPI' in sys.modules:
        MPI.Finalize()

    
if __name__=='__main__':
    main()
        


