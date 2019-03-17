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
# 2) improve code efficiency by eliminating for loops over times (including in init_Uks function)
# 3) stability of bare G and D on imaginary axis -- rewrite fermi functions and exponentials
# 4) make code look good : create a constants tuple perhaps put all constants at the ends of function calls and use tuple unpacking

import subprocess
import pdb
import numpy as np
import time
import sys, os
from functions import *
from mpi4py import MPI
import shutil
from testing import *

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
    mymkdir(savedir+'Gdir/')
    mymkdir(savedir+'G2x2dir/')
    shutil.copy(inputfile, savedir+'input') 

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

ARPES = False
Norbs = np.shape(Hk(0,0))[0]

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

e1   =  Hk(0,0)[0]
e2   =  0.1
lamb =  0.2
h = np.array([[e1, lamb], [np.conj(lamb), e2]], dtype=complex)
evals,R = np.linalg.eig(h)

beta = Ntau*dtau
ts   = np.arange(0, Nt*dt, dt)
taus = np.arange(0, beta, dtau)

print 'len ts', len(ts)
print 'len taus', len(taus)

# compute non-interacting G
# loop over evals to determine which form to use for f times the exponential based on the sign of each eval

G2x2 = langreth(Nt, Ntau, 2)
f = 1.0/(np.exp(beta*evals)+1.0)
G2x2.L = 1j*np.einsum('ij,mnj,kj->imkn', R, f[None,None,:]*np.exp(-1j*evals[None,None,:]*(ts[:,None,None]-ts[None,:,None])), np.conj(R))[0,:,0,:]
G2x2.G  = 1j*np.einsum('ij,mnj,kj->imkn', R, (f[None,None,:]-1.0)*np.exp(-1j*evals[None,None,:]*(ts[:,None,None]-ts[None,:,None])), np.conj(R))[0,:,0,:]
G2x2.IR = 1j*np.einsum('ij,mnj,kj->imkn', R, f[None,None,:]*np.exp(-1j*evals[None,None,:]*(-1j*taus[:,None,None]-ts[None,:,None])), np.conj(R))[0,:,0,:]
G2x2.RI = 1j*np.einsum('ij,mnj,kj->imkn', R, (f[None,None,:]-1.0)*np.exp(-1j*evals[None,None,:]*(ts[:,None,None]+1j*taus[None,:,None])), np.conj(R))[0,:,0,:]
deltac  = np.tril(np.ones([Ntau,Ntau]), -1) + np.diag(0.5*np.ones(Ntau)) 
G2x2.M  = 1j*np.einsum('ij,mnj,kj->imkn', R, (f[None,None,:]-deltac[:,:,None])*np.exp(-evals[None,None,:]*(taus[:,None,None]-taus[None,:,None])), np.conj(R))[0,:,0,:]
print 'G2x2\n',G2x2

G2x2.mysave(savedir+'G2x2dir/G2x2.npy')

# compute Sigma_embedding
# Sigma = |lambda|^2 * g22(t,t')
Sigma = langreth(Nt, Ntau, Norbs)
f = 1.0/(np.exp(beta*e2)+1.0)
Sigma.L  = 1j*f*np.exp(-1j*e2*(ts[:,None]-ts[None,:]))
Sigma.G  = 1j*(f-1.0)*np.exp(-1j*e2*(ts[:,None]-ts[None,:]))
Sigma.IR = 1j*f*np.exp(-1j*e2*(-1j*taus[:,None]-ts[None,:]))
Sigma.RI = 1j*(f-1.0)*np.exp(-1j*e2*(ts[:,None]+1j*taus[None,:]))
deltac = np.tril(np.ones([Ntau,Ntau]), -1) + np.diag(0.5*np.ones(Ntau)) 
Sigma.M  = 1j*(f-deltac)*np.exp(-e2*(taus[:,None]-taus[None,:]))
Sigma.scale(lamb*np.conj(lamb))
print 'Sigma\n',Sigma

#######-----------------------------------------------#########

## k2p is k indices to processor number
k2p, k2i, i2k = init_k2p_k2i_i2k(Nkx, Nky, nprocs, myrank)

# kpp is the number of k points on this process
kpp = np.count_nonzero(k2p==myrank)

constants = (myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, Nt, Ntau, dt, dtau, pump, Norbs)

if myrank==0:
    print "kpp =",kpp

UksR, UksI, eks, fks = init_Uks(*constants)

if myrank==0:
    print "done with Uks initialization"

comm.barrier()        
if myrank==0:
    print "Initialization time ", time.time()-startTime,'\n'

volume = Nkx*Nky

if myrank==0:
    timeStart = time.time()

time0 = time.time()
D = init_D(omega, Nt, Ntau, dt, dtau, Norbs)
if myrank==0:
    print 'D\n', D
    
G0 = langreth(Nt, Ntau, Norbs)
temp = langreth(Nt, Ntau, Norbs)

for ik in range(kpp):
    ik1,ik2 = i2k[ik]
    G0 = compute_G0(ik1, ik2, fks, UksR, UksI, eks, *constants)

if myrank==0:
    print "Initialization of D and G0k time ", time.time()-timeStart,'\n'

temp.zero(Nt, Ntau, Norbs)

multiply(G0, Sigma, temp, Nt, Ntau, dt, dtau, Norbs)
temp.scale(-1.0)

temp.DR = np.ones(Norbs*Nt) / dt
temp.DM = np.ones(Norbs*Ntau) / (-1j*dtau)
G0 = solve(temp, G0, Nt, Ntau, dt, dtau, Norbs)

if myrank==0:
    print "Done ", time.time()-timeStart,'\n'

# save the Green's function
if myrank==0:
    G0.mysave(savedir+'Gdir/G.npy')


if myrank==0:
    # check the differences

    print 'G0\n', G0
    print 'G2x2\n', G2x2
    G0.scale(-1.0)
    G0.add(G2x2)
    print 'difference\n',G0
    
comm.barrier()        
if myrank==0:
    print 'finished program'
    print 'total time ',time.time()-time0

MPI.Finalize()
    

    
