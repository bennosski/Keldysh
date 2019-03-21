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
    nprocs = 1
    myrank = 1
    
import pdb
import numpy as np
import time
import sys, os
from functions import *
import shutil
from testing import *
#import plot as plt

savedir   = sys.argv[1]

if myrank==0:
    time0 = time.time()
    
if myrank==0:
    print ' '
    print 'nprocs = ',nprocs
    
mymkdir(savedir)
mymkdir(savedir+'Gdir/')
mymkdir(savedir+'G2x2dir/')

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

    temp = multiply(G0, Sigma, Nt, Ntau, dt, dtau, Norbs)
    temp.scale(-1.0)
    temp.DR = np.ones(Norbs*Nt) / dt
    temp.DM = np.ones(Norbs*Ntau) / (-1j*dtau)
    G = solve(temp, G0, Nt, Ntau, dt, dtau, Norbs)

    if myrank==0:
        print 'finished program'
        print 'total time ',time.time()-time0

    return G0, G


if __name__=='__main__':

    tmax = 100.0
    taumax = 1.0

    diffs_vs_deltat = []

    dts = (0.5,)
    #dts = np.e**np.linspace(np.log(0.1), np.log(0.005), 10)
    for deltat in dts:

        print 'deltat',deltat
    
        Nt = int(round(tmax/deltat))
        dt = deltat
        Ntau = int(round(taumax/deltat))
        dtau = deltat

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
        #ts   = np.arange(0, Nt*dt, dt)
        #taus = np.arange(0, beta, dtau)
        ts = np.linspace(0, (Nt-1)*dt, Nt)
        taus = np.linspace(0, (Ntau-1)*dtau, Ntau)

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

        G0, G = main(Sigma, Nt, Ntau, dt, dtau, Nkx, Nky, g2, omega, pump)

        # plotting
        #plt.myplot(G0, G2x2, G, savedir, Nt, Ntau, dt, dtau)

        print 'G\n',G
        G.scale(-1.0)
        G.add(G2x2)

        def diff(d, xtrue): return np.mean(np.abs(d))

        diffs = [diff(G.L,  G2x2.L),
                 diff(G.G,  G2x2.G),
                 diff(G.RI, G2x2.RI),
                 diff(G.IR, G2x2.IR),
                 diff(G.M,  G2x2.M)]
        print 'diffs', diffs

        diffs.append(np.sum(diffs[:3]))

        diffs_vs_deltat.append(diffs[:])

    np.save(savedir+'diffs', diffs_vs_deltat)
    np.save(savedir+'dts', dts)

    if 'MPI' in sys.modules:
        MPI.Finalize()

    
