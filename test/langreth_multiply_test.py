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

def main():

    if myrank==0:
        time0 = time.time()
    
    if myrank==0:
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
    tmax = 10.0
    dt_fine = 0.01
    
    e1 = -0.1
    e2 =  0.1
    lamb = 1.0

    ntau = 400
    
    order = 6
    
    nts = [10, 50, 100, 500]
    
    #nts = [100]
    
    diffs = {}
    diffs['nts'] = nts

    diffs['RxR']   = []
    diffs['MxIR']  = []
    diffs['MxM']   = []
    diffs['MxM_new']   = []
    diffs['RIxIR'] = []
    diffs['IRxA']  = []
    diffs['LxA']   = []
    diffs['RxL']   = []
    diffs['RIxM']  = []
    diffs['RxRI']  = []
    
    for nt in nts:

        ntau = nt
        
        print('nt = %d'%nt)
        
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

        
        norb = 1
        def H(kx, ky, t): return e1*np.ones([1,1])
        constants = (myrank, Nkx, Nky, ARPES, kpp, k2p, k2i, tmax, nt, beta, ntau, norb, pump)
        UksR, UksI, eks, fks, Rs, _ = init_Uks(H, dt_fine, *constants, version='higher order')
        G0M = compute_G0M(0, 0, UksR, UksI, eks, fks, Rs, *constants)
        G0  = compute_G0R(0, 0, G0M, UksR, UksI, eks, fks, Rs, *constants)

        integrator = integration.integrator(order, nt, beta, ntau)
        
        diff_mean = MxM_test(integrator, G0, Sigma, e1, e2, lamb, tmax, nt, beta, ntau)
        diffs['MxM'].append(diff_mean)

        diff_mean = RIxM_test(integrator, G0, Sigma, e1, e2, lamb, tmax, nt, beta, ntau)
        diffs['RIxM'].append(diff_mean)

        #diff_mean = MxM_new_test(integrator, G0, Sigma, e1, e2, lamb, tmax, nt, beta, ntau)
        #diffs['MxM_new'].append(diff_mean)
        
        diff_mean = MxIR_test(integrator, G0, Sigma, e1, e2, lamb, tmax, nt, beta, ntau)
        diffs['MxIR'].append(diff_mean)
        
        diff_mean = RxR_test(integrator, G0, Sigma, e1, e2, lamb, tmax, nt, beta, ntau)
        diffs['RxR'].append(diff_mean)
        
        diff_mean = RIxIR_test(integrator, G0, Sigma, e1, e2, lamb, tmax, nt, beta, ntau)
        diffs['RIxIR'].append(diff_mean)

        diff_mean = IRxA_test(integrator, G0, Sigma, e1, e2, lamb, tmax, nt, beta, ntau)
        diffs['IRxA'].append(diff_mean)

        diff_mean = LxA_test(integrator, G0, Sigma, e1, e2, lamb, tmax, nt, beta, ntau)
        diffs['LxA'].append(diff_mean)

        diff_mean = RxL_test(integrator, G0, Sigma, e1, e2, lamb, tmax, nt, beta, ntau)
        diffs['RxL'].append(diff_mean)

        diff_mean = RxRI_test(integrator, G0, Sigma, e1, e2, lamb, tmax, nt, beta, ntau)
        diffs['RxRI'].append(diff_mean)

        #diff_mean = MxM_test2(integrator, G0, Sigma, e1, e2, lamb, tmax, nt, beta, ntau)
        #diffs['RIxM'].append(diff_mean)

        
    if len(nts)>1:
        #np.save(savedir+'diffs.npy', diffs)
        #np.save(savedir+'nts.npy', nts)
        
        #log_nts = np.log(array(nts))
        #log_diffs = np.log(np.array(diffs))
        #plt(log_nts, [log_diffs], 'diffs')

        #slope = (log_diffs[-1]-log_diffs[0])/(log_nts[-1]-log_nts[0])
        #print('slope = %1.3f'%slope)
        
        plt_diffs(diffs)

def RxRI_test(integrator, G0, Sigma, e1, e2, lamb, tmax, nt, beta, ntau):
    P = integrator.RxRI(G0, Sigma)
    
    ts = linspace(0, tmax, nt)
    taus = linspace(0, beta, ntau)    

    fe1 = 1.0/(np.exp(beta*e1)+1.0)
    fe2 = 1.0/(np.exp(beta*e2)+1.0)
    
    Pexact = fe2*np.exp(-1j*e1*ts[:,None])*np.exp(e2*taus[None,:])/(1j*(e1-e2))*(np.exp(1j*ts[:,None]*(e1-e2))-1.0) 
    Pexact *= lamb*np.conj(lamb)

    y1 = P[:,0,:,0]
    y2 = Pexact[:,:]
    #im([y1.real, y2.real, y1.real-y2.real], [0,beta,0,tmax], 'P and Pexact for RxRI')
    
    #im([y1.imag, y2.imag, y1.imag-y2.imag], [0,beta,0,tmax], 'P and Pexact for RxRI')

    
    diff_mean = np.mean(abs(P[:,0,:,0]-Pexact[:,:]))
    print('diff mean', diff_mean)
        
    return diff_mean    
        
def RIxM_test(integrator, G0, Sigma, e1, e2, lamb, tmax, nt, beta, ntau):
    P = integrator.RIxM(G0, Sigma)
    
    ts = linspace(0, tmax, nt)
    taus = linspace(0, beta, ntau)    

    fe1 = 1.0/(np.exp(beta*e1)+1.0)
    fe2 = 1.0/(np.exp(beta*e2)+1.0)
    
    #Pexact = -fe1*np.exp(-1j*e1*ts[:,None])*np.exp(1j*e2*ts[None,:])/(1j*(e1-e2)) * (np.exp(1j*(e1-e2)*ts[None,:]) - 1.0)
    Pexact = 1j*fe1*fe2*np.exp(-1j*e1*ts[:,None])*np.exp(e2*taus[None,:])/(e1-e2)*(np.exp(taus[None,:]*(e1-e2))-1.0) + 1j*fe1*(fe2-1.0)*np.exp(-1j*e1*ts[:,None])*np.exp(e2*taus[None,:])/(e1-e2)*(np.exp(beta*(e1-e2))-np.exp(taus[None,:]*(e1-e2))) 
    Pexact *= lamb*np.conj(lamb)
    
    y1 = P[:,0,:,0]
    y2 = Pexact[:,:]
    
    diff_mean = np.mean(abs(P[:,0,:,0]-Pexact[:,:]))
    print('diff mean', diff_mean)
        
    return diff_mean    

        
def RxL_test(integrator, G0, Sigma, e1, e2, lamb, tmax, nt, beta, ntau):
    P = integrator.RxL(G0, Sigma)
    
    ts = linspace(0, tmax, nt)
    taus = linspace(0, beta, ntau)    

    fe1 = 1.0/(np.exp(beta*e1)+1.0)
    fe2 = 1.0/(np.exp(beta*e2)+1.0)
    
    Pexact = fe2/(1j*(e1-e2))*np.exp(-1j*e1*ts[:,None])*np.exp(1j*e2*ts[None,:])*(np.exp(1j*(e1-e2)*ts[:,None]) - 1.0)
    Pexact *= lamb*np.conj(lamb)

    y1 = P[:,0,:,0]
    y2 = Pexact[:,:]
    #im([y1.real, y2.real, y1.real-y2.real], [0,beta,0,tmax], 'P and Pexactfor MxIR')
    #im([y1.imag, y2.imag, y1.imag-y2.imag], [0,beta,0,tmax], 'P and Pexactfor MxIR')
    
    diff_mean = np.mean(abs(P[:,0,:,0]-Pexact[:,:]))
    print('diff mean', diff_mean)
        
    return diff_mean    

def LxA_test(integrator, G0, Sigma, e1, e2, lamb, tmax, nt, beta, ntau):
    P = integrator.LxA(G0, Sigma)
    
    ts = linspace(0, tmax, nt)
    taus = linspace(0, beta, ntau)    

    fe1 = 1.0/(np.exp(beta*e1)+1.0)
    fe2 = 1.0/(np.exp(beta*e2)+1.0)
    
    Pexact = -fe1*np.exp(-1j*e1*ts[:,None])*np.exp(1j*e2*ts[None,:])/(1j*(e1-e2)) * (np.exp(1j*(e1-e2)*ts[None,:]) - 1.0)
    Pexact *= lamb*np.conj(lamb)

    y1 = P[:,0,:,0]
    y2 = Pexact[:,:]
    #im([y1.real, y2.real, y1.real-y2.real], [0,beta,0,tmax], 'P and Pexactfor MxIR')

    #im([y1.imag, y2.imag, y1.imag-y2.imag], [0,beta,0,tmax], 'P and Pexactfor MxIR')
    
    diff_mean = np.mean(abs(P[:,0,:,0]-Pexact[:,:]))
    print('diff mean', diff_mean)
        
    return diff_mean    
               
def IRxA_test(integrator, G0, Sigma, e1, e2, lamb, tmax, nt, beta, ntau):
    P = integrator.IRxA(G0, Sigma)
    
    ts = linspace(0, tmax, nt)
    taus = linspace(0, beta, ntau)    

    fe1 = 1.0/(np.exp(beta*e1)+1.0)
    fe2 = 1.0/(np.exp(beta*e2)+1.0)
    
    Pexact = -(fe1-1.0)*np.exp(-e1*taus[:,None])*np.exp(1j*e2*ts[None,:])/(1j*(e1-e2)) * (np.exp(1j*(e1-e2)*ts[None,:]) - 1.0)
    Pexact *= lamb*np.conj(lamb)

    y1 = P[:,0,:,0]
    y2 = Pexact[:,:]
    #im([y1.real, y2.real, y1.real-y2.real], [0,beta,0,tmax], 'P and Pexactfor MxIR')

    #im([y1.imag, y2.imag, y1.imag-y2.imag], [0,beta,0,tmax], 'P and Pexactfor MxIR')
    
    diff_mean = np.mean(abs(P[:,0,:,0]-Pexact[:,:]))
    print('diff mean', diff_mean)
        
    return diff_mean    
        
def RIxIR_test(integrator, G0, Sigma, e1, e2, lamb, tmax, nt, beta, ntau):
    P = integrator.RIxIR(G0, Sigma)
    
    ts = linspace(0, tmax, nt)
    taus = linspace(0, beta, ntau)    

    fe1 = 1.0/(np.exp(beta*e1)+1.0)
    fe2 = 1.0/(np.exp(beta*e2)+1.0)
    
    Pexact  = 1j/(e1-e2)*fe1*(fe2-1)*np.exp(-1j*e1*ts[:,None])*np.exp(-beta*e2)*np.exp(1j*e2*ts[None,:])*(np.exp(beta*e1)-np.exp(beta*e2))
    Pexact *= lamb*np.conj(lamb)

    y1 = P[:,0,:,0]
    y2 = Pexact[:,:]
    #im([y1.real, y2.real, y1.real-y2.real], [0,beta,0,tmax], 'P and Pexactfor MxIR')

    #im([y1.imag, y2.imag, y1.imag-y2.imag], [0,beta,0,tmax], 'P and Pexactfor MxIR')
    
    diff_mean = np.mean(abs(P[:,0,:,0]-Pexact[:,:]))
    print('diff mean', diff_mean)
        
    return diff_mean    
            
def MxM_test(integrator, G0, Sigma, e1, e2, lamb, tmax, nt, beta, ntau):
    norb = 1
    P = integrator.MxM(G0, Sigma)
    
    taus = linspace(0, beta, ntau)
    
    fe1 = 1.0/(np.exp(beta*e1)+1.0)
    fe2 = 1.0/(np.exp(beta*e2)+1.0)

    Pexact = lamb*np.conj(lamb)*1j/(e1-e2)*(fe2-1.0)*np.exp(-e1*taus) * ( -fe1 - np.exp((e1-e2)*taus) + 1.0 + fe1*np.exp((e1-e2)*beta))
                                                       
    y1 = P[:,0,0]
    y2 = Pexact[:]
    #plt(taus, [y1.real, y2.real], 'P and Pexactfor MxM')
    #plt(taus, [y1.imag, y2.imag], 'P and Pexactfor MxM')
    
    diff_mean = np.mean(abs(P[:,0,0]-Pexact[:]))
    print('diff mean', diff_mean)
        
    return diff_mean    

def MxM_new_test(integrator, G0, Sigma, e1, e2, lamb, tmax, nt, beta, ntau):
    norb = 1
    P = integrator.MxM_new(G0, Sigma)
    
    taus = linspace(0, beta, ntau)
    
    fe1 = 1.0/(np.exp(beta*e1)+1.0)
    fe2 = 1.0/(np.exp(beta*e2)+1.0)

    Pexact = lamb*np.conj(lamb)*1j/(e1-e2)*(fe2-1.0)*np.exp(-e1*taus) * ( -fe1 - np.exp((e1-e2)*taus) + 1.0 + fe1*np.exp((e1-e2)*beta))
                                                       
    y1 = P[:,0,0]
    y2 = Pexact[:]
    #plt(taus, [y1.real, y2.real], 'P and Pexactfor MxM')
    #plt(taus, [y1.imag, y2.imag], 'P and Pexactfor MxM')
    
    diff_mean = np.mean(abs(P[:,0,0]-Pexact[:]))
    print('diff mean', diff_mean)
        
    return diff_mean    


def MxM_test2(integrator, G0, Sigma, e1, e2, lamb, tmax, nt, beta, ntau):
    norb = 1
    P1 = integrator.MxM(G0, Sigma)

    P2 = integrator.MxM2(G0, Sigma)
        
    diff_mean = dist(P1, P2)
    print('diff mean', diff_mean)

    taus = np.linspace(0, beta, ntau)
    
    y1 = P1[:,0,0]
    y2 = P2[:,0,0]
    plt(taus, [y1.real, y2.real, y1.real-y2.real], 'real')
    plt(taus, [y1.imag, y2.imag, y1.imag-y2.imag], 'imag')
    
    return diff_mean    


def MxIR_test(integrator, G0, Sigma, e1, e2, lamb, tmax, nt, beta, ntau):
    P = integrator.MxIR(G0, Sigma)
    
    ts = linspace(0, tmax, nt)
    taus = linspace(0, beta, ntau)    

    fe1 = 1.0/(np.exp(beta*e1)+1.0)
    fe2 = 1.0/(np.exp(beta*e2)+1.0)
    
    Pexact  = 1j/(e2-e1)*(fe1-1)*(fe2-1)*np.exp(1j*e2*ts[None,:])*(np.exp(-e1*taus[:,None]) - np.exp(-e2*taus[:,None]))
    Pexact += 1j/(e2-e1)*fe1*(fe2-1)*np.exp(1j*e2*ts[None,:])*(np.exp(-e2*taus[:,None]) - np.exp((beta-taus[:,None])*e1)*np.exp(-beta*e2))
    Pexact *= lamb*np.conj(lamb)

    y1 = P[:,0,:,0]
    y2 = Pexact[:,:]
    #im([y1.real, y2.real, y1.real-y2.real], [0,beta,0,tmax], 'P and Pexactfor MxIR')

    #im([y1.imag, y2.imag, y1.imag-y2.imag], [0,beta,0,tmax], 'P and Pexactfor MxIR')
    
    diff_mean = np.mean(abs(P[:,0,:,0]-Pexact[:,:]))
    print('diff mean', diff_mean)
        
    return diff_mean    
        
def RxR_test(integrator, G0, Sigma, e1, e2, lamb, tmax, nt, beta, ntau):
    P = integrator.RxR(G0, Sigma)

    ts = linspace(0, tmax, nt)
    
    Pexact = lamb*np.conj(lamb)/(1j*(e1-e2)) * (np.exp(-1j*e1*(ts[:,None]-ts[None,:])) - np.exp(-1j*e2*(ts[:,None]-ts[None,:])))

    y1 = P[:,0,:,0]
    y2 = Pexact
    #im([y1.real, y2.real, y1.real-y2.real], [0,tmax,0,tmax], 'P and Pexact')

    #im([y1.imag, y2.imag, y1.imag-y2.imag], [0,tmax,0,tmax], 'P and Pexact')
    
    print('shape P', np.shape(P))
    print('shape Pexact', np.shape(Pexact))
    print('diff max ', np.amax(abs(P[:,0,:,0]-Pexact[:,:])))
    diff_mean = np.mean(abs(P[:,0,:,0]-Pexact[:,:]))
    print('diff mean', diff_mean)
        
    return diff_mean
    
if __name__=='__main__':
    main()
        


