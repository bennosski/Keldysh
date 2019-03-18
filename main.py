# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 01:44:04 2016

@author: Ben
"""

# stability of bare G and D on imaginary axis -- rewrite fermi functions and exponentials

import pdb
import numpy as np
import time
import sys, os
from functions import *
from mpi4py import MPI
import shutil
from testing import *

savedir   = sys.argv[1]

comm = MPI.COMM_WORLD
nprocs = comm.size
myrank = comm.rank

if myrank==0:
    time0 = time.time()
    
if myrank==0:
    print ' '
    print 'nprocs = ',nprocs
    
mymkdir(savedir)
mymkdir(savedir+'Gdir/')
mymkdir(savedir+'G2x2dir/')
comm.barrier()

def main(Sigma, Nt, Ntau, dt, dtau, Nkx, Nky, g2, omega, pump):
        
    ARPES = False
    Norbs = np.shape(Hk(0,0))[0]
    volume = Nkx*Nky

    if myrank==0:
        print '\n','---------------\nParams:'
        print 'Nt    = ',Nt
        print 'Ntau  = ',Ntau
        print 'dt    = ',dt
        print 'dtau  = ',dtau
        print 'Nkx   = ',Nkx
        print 'Nky   = ',Nky
        print 'g2    = ',g2
        print 'omega = ',omega
        print 'pump  = ',pump
        print 'Norbs = ',Norbs
        print '----------------\n'    

    if myrank==0:
        startTime = time.time()

    #######------------ for embedding test ---------------#########


    ## k2p is k indices to processor number
    k2p, k2i, i2k = init_k2p_k2i_i2k(Nkx, Nky, nprocs, myrank)

    # kpp is the number of k points on this process
    kpp = np.count_nonzero(k2p==myrank)

    constants = (myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, Nt, Ntau, dt, dtau, pump, Norbs)

    UksR, UksI, eks, fks = init_Uks(*constants)

    if myrank==0:
        print "done with Uks initialization"

    comm.barrier()        
    if myrank==0:
        print "Initialization time ", time.time()-startTime,'\n'

    if myrank==0:
        timeStart = time.time()

    G0 = langreth(Nt, Ntau, Norbs)
    temp = langreth(Nt, Ntau, Norbs)

    for ik in range(kpp):
        ik1,ik2 = i2k[ik]
        G0 = compute_G0(ik1, ik2, fks, UksR, UksI, eks, *constants)

    if myrank==0:
        print "Total initialization of D and G0k time ", time.time()-timeStart,'\n'

    temp.zero(Nt, Ntau, Norbs)

    multiply(G0, Sigma, temp, Nt, Ntau, dt, dtau, Norbs)
    temp.scale(-1.0)
    temp.DR = np.ones(Norbs*Nt) / dt
    temp.DM = np.ones(Norbs*Ntau) / (-1j*dtau)
    G0 = solve(temp, G0, Nt, Ntau, dt, dtau, Norbs)

    comm.barrier()        
    if myrank==0:
        print 'finished program'
        print 'total time ',time.time()-time0

    return G0


if __name__=='__main__':

    Nt = 100
    dt = 0.01
    Ntau = 100
    dtau = 0.01
    Nkx = 1
    Nky = 1
    pump = 0
    g2 = None
    omega = None

    Norbs = np.shape(Hk(0,0))[0]

    e1   =  Hk(0,0)[0]
    e2   =  0.1
    lamb =  0.5
    h = np.array([[e1, lamb], [np.conj(lamb), e2]], dtype=complex)

    print 'h\n',h
    evals,R = np.linalg.eig(h)

    beta = Ntau*dtau
    ts   = np.arange(0, Nt*dt, dt)
    taus = np.arange(0, beta, dtau)

    print 'len ts', len(ts)
    print 'len taus', len(taus)

    # compute non-interacting G
    # loop over evals to determine which form to use for f times the exponential based on the sign of each eval

    deltac  = np.tril(np.ones([Ntau,Ntau]), -1) + np.diag(0.5*np.ones(Ntau)) 

    G2x2 = langreth(Nt, Ntau, 2)
    f = 1.0/(np.exp(beta*evals)+1.0)
    G2x2.L  = 1j*np.einsum('ij,mnj,kj->imkn', R, f[None,None,:]*np.exp(-1j*evals[None,None,:]*(ts[:,None,None]-ts[None,:,None])), np.conj(R))[0,:,0,:]
    G2x2.G  = 1j*np.einsum('ij,mnj,kj->imkn', R, (f[None,None,:]-1.0)*np.exp(-1j*evals[None,None,:]*(ts[:,None,None]-ts[None,:,None])), np.conj(R))[0,:,0,:]
    G2x2.RI = 1j*np.einsum('ij,mnj,kj->imkn', R, f[None,None,:]*np.exp(-1j*evals[None,None,:]*(ts[:,None,None]+1j*taus[None,:,None])), np.conj(R))[0,:,0,:]
    G2x2.IR = 1j*np.einsum('ij,mnj,kj->imkn', R, (f[None,None,:]-1.0)*np.exp(-1j*evals[None,None,:]*(-1j*taus[:,None,None]-ts[None,:,None])), np.conj(R))[0,:,0,:]
    G2x2.M  = 1j*np.einsum('ij,mnj,kj->imkn', R, (f[None,None,:]-deltac[:,:,None])*np.exp(-evals[None,None,:]*(taus[:,None,None]-taus[None,:,None])), np.conj(R))[0,:,0,:]
    print 'G2x2\n',G2x2
    G2x2.mysave(savedir+'G2x2dir/G2x2.npy')

    # compute Sigma_embedding
    # Sigma = |lambda|^2 * g22(t,t')
    Sigma = langreth(Nt, Ntau, Norbs)
    f = 1.0/(np.exp(beta*e2)+1.0)
    Sigma.L  = 1j*f*np.exp(-1j*e2*(ts[:,None]-ts[None,:]))
    Sigma.G  = 1j*(f-1.0)*np.exp(-1j*e2*(ts[:,None]-ts[None,:]))
    Sigma.RI = 1j*f*np.exp(-1j*e2*(ts[:,None]+1j*taus[None,:]))
    Sigma.IR = 1j*(f-1.0)*np.exp(-1j*e2*(-1j*taus[:,None]-ts[None,:]))
    Sigma.M  = 1j*(f-deltac)*np.exp(-e2*(taus[:,None]-taus[None,:]))
    Sigma.scale(lamb*np.conj(lamb))
    print 'Sigma\n',Sigma

    #######-----------------------------------------------#########

    G = main(Sigma, Nt, Ntau, dt, dtau, Nkx, Nky, g2, omega, pump)
    
    print 'G\n',G
    G.scale(-1.0)
    G.add(G2x2)

    #def diff(x, xtrue): return np.mean(np.abs(x-xtrue)/np.abs(xtrue))
    def diff(d, xtrue): return np.mean(np.abs(d)/np.abs(xtrue))
     
    diffs = [diff(G.L,  G2x2.L),
             diff(G.G,  G2x2.G),
             diff(G.RI, G2x2.RI),
             diff(G.IR, G2x2.IR),
             diff(G.M,  G2x2.M)]
    print 'diffs', diffs

    MPI.Finalize()
    

    
