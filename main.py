# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 01:44:04 2016

@author: Ben
"""

# this branch is for the embedding self-energy test
# todo:
#  - compute G0 for the 2by2 problem
#  - compute g22 
#  - a version of the code which solves the 1x1 problem with a selfenergy
#
# 1) modularize code by getting rid of separate ARPES specific routines
# 2) improve code efficiency by eliminating for loops over times
# 3) stability of bare G and D on imaginary axis -- rewrite fermi functions and exponentials


import subprocess

import pdb
import numpy as np
import time
import sys, os
from functions import *
from mpi4py import MPI

inputfile = sys.argv[1]
savedir   = sys.argv[2]

comm = MPI.COMM_WORLD
nprocs = comm.size
myrank = comm.rank

if myrank==0:
    time0 = time.time()

def bash_command(cmd):
    subprocess.Popen(['/bin/bash', '-c', cmd])

def mymkdir(mydir):
    if not os.path.exists(mydir):
        print 'making ',mydir
        os.mkdir(mydir)


if myrank==0:
    print ' '
    print 'nprocs = ',nprocs
    
    mymkdir(savedir)
    mymkdir(savedir+'Glocdir/')
    mymkdir(savedir+'Sdir/')

comm.barrier()
        
with open(inputfile,'r') as f:
    Nt    = int(parseline(f.readline()))
    Ntau  = int(parseline(f.readline()))
    dt    = float(parseline(f.readline()))
    dtau  = float(parseline(f.readline()))
    Nkx   = int(parseline(f.readline()))
    Nky   = int(parseline(f.readline()))
    g2    = float(parseline(f.readline()))
    omega = float(parseline(f.readline()))    
    pump  = int(parseline(f.readline()))

try:
    Norbs = np.shape(Hk(0,0))[0]
except:
    Norbs = 1

if myrank==0:
    print '\n','Params'
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
    print '\n'    

if myrank==0:
    startTime = time.time()

#######------------ for embedding test ---------------#########

lamb =  0.2
e1   =  Hk(0,0)
e2   =  0.1
h = np.array([[e1, lamb], [np.conj(lamb), e2]])
evals,R = np.linalg.eig(h)

beta = Ntau*dtau
ts   = np.arange(0, Nt*dt, dt)
taus = np.arange(0, beta, dtau)

print 'len ts', len(ts)
print 'len taus', len(taus)

# compute non-interacting G

f = 1.0/(np.exp(beta*evals)+1.0)
GL = 1j*np.einsum('ij,tj,kj->tik', R, f[None,:]*np.exp(-1j*evals[None,:]*ts[:,None]), np.conj(R))

GG = 1j*np.einsum('ij,tj,kj->tik', R, (f[None,:]-1.0)*np.exp(-1j*evals[None,:]*ts[:,None]), np.conj(R)) 

'''
def f(x): return 1.0/(np.exp(beta*x)+1.0) 
GM = 1j*np.einsum('ij,jt,kj->tik', R, \
          np.vstack(((f(evals[0])-1.0)*np.exp(-evals[0]*taus), \
                     -f(evals[1])*np.exp(evals[1]*(beta-taus)))), \
          np.conj(R))
# fix the tau=0 point
f = 1.0/(np.exp(beta*evals)+1.0)
GM[0] = 1j*np.einsum('ij,j,kj->ik', R, (f-0.5), np.conj(R))
'''

deltac = np.ones(Ntau)
deltac[0] = 0.5
GM = 1j*np.einsum('ij,tj,kj->tik', R, (f[None,:]-deltac[:,None])*np.exp(-evals[None,:]*taus[:,None]), np.conj(R)) 

print 'shape GL', np.shape(GL)
print 'shape GG', np.shape(GG)
print 'shape GM', np.shape(GM)

# compute Sigma_embedding
# Sigma = |lambda|^2 * g22(t,t')

Sigma_phonon = langreth(Nt, Ntau, Norbs)
# compute each of the parts by hand...




#######-----------------------------------------------#########

exit()

## k2p is k indices to processor number
k2p, k2i, i2k = init_k2p_k2i_i2k(Nkx, Nky, nprocs, myrank)

# kpp is the number of k points on this process
kpp = np.count_nonzero(k2p==myrank)

if myrank==0:
    print "kpp =",kpp

UksR, UksI, eks, fks = init_Uks(myrank, Nkx, Nky, kpp, k2p, k2i, Nt, Ntau, dt, dtau, pump, Norbs)

if myrank==0:
    print "done with Uks initialization"

comm.barrier()        
if myrank==0:
    print "Initialization time ", time.time()-startTime,'\n'

volume = Nkx*Nky

########## ---------------- Compute the electron selfenergy due to phonons -------------------- ##############

if myrank==0:
    timeStart = time.time()

D = init_D(omega, Nt, Ntau, dt, dtau, Norbs)
    
if myrank==0:
    print 'max D'
    print D

Gloc_proc = langreth(Nt, Ntau, Norbs)
temp = langreth(Nt, Ntau, Norbs)

# compute local Greens function for each processor
for ik in range(kpp):
    ik1,ik2 = i2k[ik]
    G0k = compute_G0(ik1, ik2, myrank, Nkx, Nky, kpp, k2p, k2i, Nt, Ntau, dt, dtau, fks, UksR, UksI, eks, Norbs)
    Gloc_proc.add(G0k)

if myrank==0:
    print "Initialization of D and G0k time ", time.time()-timeStart,'\n'

iter_selfconsistency = 3
for myiter in range(iter_selfconsistency):
        
    if myrank==0:
        timeStart = time.time()

    Sigma_phonon.zero(Nt, Ntau, Norbs)

    # store complete local Greens function in Sigma_phonon
    comm.Allreduce(Gloc_proc.L,  Sigma_phonon.L,  op=MPI.SUM)
    comm.Allreduce(Gloc_proc.G,  Sigma_phonon.G,  op=MPI.SUM)
    comm.Allreduce(Gloc_proc.IR, Sigma_phonon.IR, op=MPI.SUM)
    comm.Allreduce(Gloc_proc.RI, Sigma_phonon.RI, op=MPI.SUM)
    comm.Allreduce(Gloc_proc.M,  Sigma_phonon.M,  op=MPI.SUM)

    if myrank==0:
        print 'max Gloc'
        print Sigma_phonon
        # save DOS
        Sigma_phonon.mysave(savedir+'Glocdir/Gloc')

    comm.barrier()
    Sigma_phonon.directMultiply(D)
    Sigma_phonon.scale(1j * g2 / volume)

    if myrank==0:
        print 'max Sigma_phonon'
        print Sigma_phonon
    
        print "iteration",myiter
        print "time computing phonon selfenergy ", time.time()-timeStart,'\n'
        timeStart = time.time()

    # unnecessary to compute DOS on the last iteration
    if myiter<iter_selfconsistency-1:

        Gloc_proc.zero(Nt, Ntau, Norbs)

        for ik in range(kpp):
            temp.zero(Nt, Ntau, Norbs)

            ik1, ik2 = i2k[ik]
            G0k = compute_G0(ik1, ik2, myrank, Nkx, Nky, kpp, k2p, k2i, Nt, Ntau, dt, dtau, fks, UksR, UksI, eks, Norbs)

            multiply(G0k, Sigma_phonon, temp, Nt, Ntau, dt, dtau, Norbs)

            temp.scale(-1.0)

            temp.DR = np.ones(Norbs*Nt) / dt
            temp.DM = np.ones(Norbs*Ntau) / (-1j*dtau)

            temp = solve(temp, G0k, Nt, Ntau, dt, dtau, Norbs)

            Gloc_proc.add(temp)

    comm.barrier()       
    if myrank==0:
        print "Done iteration ", time.time()-timeStart,'\n'

# save the selfenergy
if myrank==0:
    Sigma_phonon.mysave(savedir+'Sdir/S')
    
comm.barrier()        
if myrank==0:
    print 'finished program'
    print 'total time ',time.time()-time0

MPI.Finalize()
    

    
