import numpy as np
from scipy import integrate
import volterra 
from matsubara import *
from langreth import *

class integrator:
    def __init__(self, order, nt, beta, ntau, norb):
        self.gregory_matrix_M = self._compute_gregory_matrix(ntau, order)
        self.rcorr = self._compute_rcorr(order)
        self.gregory_matrix_R = self._compute_gregory_matrix(nt, order)
        self.order = order
        self.nt = nt
        self.ntau = ntau
        self.dtau = 1.0*beta/(ntau-1)
        self.norb = norb
    #------------------------------------------------------------                
    def _compute_gregory_matrix(self, nmax, order):
        wstart,omega = volterra.weights(order)

        cff = np.zeros((nmax,nmax))

        for n in range(1,nmax):
            if n < 2*order-2:
                cff[n,0:order] = wstart[n,0:order]
            else:
                cff[n,0:order] = wstart[-1,0:order]

            if n >= order:
                jmax = max(n-order+1,order)
                for j in range(jmax,n+1):
                    cff[n,j] = omega[n-j]

                cff[n,order:n-order+1] = 1.0

        return cff
    #------------------------------------------------------------    
    def _lagrange(self,tpts,j,t):
        Lj = 1.0
        for m in range(0,len(tpts)):
            if not m==j:
                Lj *= (t-tpts[m])/(tpts[j]-tpts[m])
        return Lj
    #------------------------------------------------------------
    def _compute_rcorr(self,p):
        Rcorr = np.zeros((p,p,p))

        tpts = np.linspace(0.0,(p-1),p)

        for i in range(0,p):
            for j in range(0,p):
                for m in range(1,p):
                    def kern(t):
                        return self._lagrange(tpts,i,m-t)*self._lagrange(tpts,j,t)

                    Rcorr[m,i,j],err = integrate.quad(kern, 0.0, m)
        return Rcorr
    #------------------------------------------------------------
    def prep_MxM(self, A):
        '''
        prepare A for higher-order (Gregory and boundary correction) convolution using matrix multiplication

        A : a matsubara or langreth object with a Matsubara matrix of size (ntau x norb, norb)

        output : a matrix of size (ntau x norb x ntau x norb) which can be multiplied by a second vector to produce the convolution
        '''

        ntau  = A.ntau
        dtau  = A.dtau
        norb  = A.norb
        order = self.order
        
        Cmk = np.zeros((ntau,norb,ntau,norb),dtype=np.complex128)        
        
        for iorb in range(0,norb):
            for korb in range(0,norb):

                for m in range(ntau-order,ntau-1):
                    for k in range(0,order):
                        for l in range(0,order):
                            Cmk[m,iorb,ntau-1-k,korb] += -1j*dtau * self.rcorr[ntau-1-m,l,k] * A.sig * A.M[ntau-1-l,iorb,korb]

                for m in range(0,ntau-order):
                    Cmk[m,iorb,m:ntau,korb] += -1j*dtau * self.gregory_matrix_M[ntau-1-m,:ntau-m] * A.sig * A.M[np.arange(ntau-1,m-1,-1),iorb,korb]

                for m in range(1,order):
                    for k in range(0,order):
                        for l in range(0,order):
                            Cmk[m,iorb,k,korb] += -1j*dtau * self.rcorr[m,l,k] * A.M[l,iorb,korb]

                for m in range(order,ntau):
                    Cmk[m,iorb,:m+1,korb] += -1j*dtau * self.gregory_matrix_M[m,:m+1] * A.M[np.arange(m,-1,-1),iorb,korb]

        return np.reshape(Cmk, [ntau*norb, ntau*norb])
    #------------------------------------------------------------
    def MxM(self, A, B, C):
        '''
        A = B*C
        '''
        ntau = A.ntau
        norb = A.norb

        BM = np.reshape(B.M, [ntau*norb, norb])
        C.M = self.prep_MxM(A) @ BM
        C.M = np.reshape(C.M, [ntau,norb,norb])
    #------------------------------------------------------------
    def dyson_matsubara(self, G0, Sigma, G):
        '''
        compute G.M = (I-G0.M*Sigma.M)^(-1)*G0.M
        for the matsubara piece
        G0, Sigma, G are matsubara objects
        G.M will store the final Matsubara solution
        '''
       
        ntau = G0.ntau
        norb = G0.norb

        self.MxM(G0, Sigma, G)

        X = np.diag(np.ones(ntau*norb)) - self.prep_MxM(G)
        G0M = np.reshape(G0.M, [ntau*norb, norb])
        G.M = np.linalg.solve(X, G0M)
        G.M = np.reshape(G.M, [ntau,norb,norb])
    #------------------------------------------------------------
    def prep_MxIR(self, A):
        dtau = A.dtau

        # only needs to be run once!!!!
        # idea : store in the langreth object if component in the langreth object is none otherwise compute the following

        return self.prep_MxM(A)
    #------------------------------------------------------------
    def MxIR(self, A, B):
        ntau = A.ntau
        nt   = B.nt
        norb = A.norb

        BIR = np.reshape(B.IR, [ntau*norb, nt*norb])
        return np.reshape(self.prep_MxIR(A) @ BIR, [ntau,norb,nt,norb])
    #------------------------------------------------------------
    def prep_RIxIR(self, A):
        ntau = A.ntau
        nt   = A.nt
        norb = A.norb
        dtau = A.dtau
        
        ARI = -1j*dtau*np.einsum('i,ibra->raib', self.gregory_matrix_M[-1,:], -A.sig*np.conj(A.IR[::-1]))
        return np.reshape(ARI, [nt*norb, ntau*norb])
    #------------------------------------------------------------
    def RIxIR(self, A, B):
        ntau = A.ntau
        nt   = A.nt
        norb = A.norb

        BIR = np.reshape(B.IR, [ntau*norb, nt*norb])
        return np.reshape(self.prep_RIxIR(A) @ BIR, [nt,norb,nt,norb])
    #------------------------------------------------------------
    def prep_rxA(self, B):
        '''
        prep second matrix for multiplication
        '''
        nt   = B.nt
        norb = B.norb
        dt   = B.dt

        #### CHECK!!!! DO I NEED TO TRANSPOSE ORBITAL INDICES TO GET BA?
        BA = np.einsum('kajb->jbka', np.conj(B.R))
        x = dt * np.einsum('jk,kajb->kajb', self.gregory_matrix_R, BA)
        return np.reshape(x, [nt*norb, nt*norb])
    #------------------------------------------------------------
    def IRxA(self, A, B):
        '''
        prep second matrix for multiplication
        '''
        nt   = B.nt
        ntau = A.ntau
        norb = B.norb
        dt   = B.dt

        AIR = np.reshape(A.IR, [ntau*norb, nt*norb])
        return np.reshape(AIR @ self.prep_rxA(B), [ntau,norb,nt,norb])
    #------------------------------------------------------------
    def LxA(self, A, B):
        '''
        prep second matrix for multiplication
        '''
        nt   = B.nt
        norb = B.norb
        dt   = B.dt

        AL = np.reshape(A.L, [nt*norb, nt*norb])
        return np.reshape(AL @ self.prep_rxA(B), [nt,norb,nt,norb])
    #------------------------------------------------------------
    def prep_Rxr(self, A):
        nt   = A.nt
        norb = A.norb
        dt   = A.dt
        
        x = dt * np.einsum('nk,nakb->nakb', self.gregory_matrix_R, A.R)
        return np.reshape(x, [nt*norb, nt*norb])
    #------------------------------------------------------------
    def RxL(self, A, B):
        nt   = A.nt
        norb = A.norb
        dt   = A.dt

        BL = np.reshape(B.L, [nt*norb, nt*norb])
        return np.reshape(self.prep_Rxr(A) @ BL, [nt,norb,nt,norb])
    #------------------------------------------------------------
    def prep_RxR(self, A, j):
        nt   = A.nt
        norb = A.norb
        dt   = A.dt

        # something weird going on for large j
        # probably need to more carefully handle the boundary
        
        wj = np.zeros((nt,nt))
        wj[j:, j:] = self.gregory_matrix_R[:nt-j, :nt-j]

        #gm = np.tril(np.ones((nt,nt))) - np.diag(0.5*np.ones(nt))
        #wj = np.zeros((nt,nt))
        #wj[j:, j:] = gm[:nt-j, :nt-j]
        
        x = dt * np.einsum('nk,nakb->nakb', wj, A.R)
        return np.reshape(x, [nt*norb, nt*norb])
    #------------------------------------------------------------
    def RxR(self, A, B):
        nt   = A.nt
        norb = A.norb

        out = np.zeros((nt, norb, nt, norb), dtype=np.complex128)

        BR = np.reshape(B.R, [nt*norb, nt, norb])

        for j in range(nt):
            out[:,:,j,:] = np.reshape(self.prep_RxR(A,j) @ BR[:,j,:], [nt,norb,norb])

        #out = out - np.einsum('abcd->cdab', np.conj(out))

        theta = np.tril(np.ones((nt,nt))) - np.diag(0.5*np.ones(nt))     
        out  = np.einsum('mn,manb->manb', theta, out)
        out -= np.einsum('abcd->cdab', np.conj(out))
        
        return out
    #------------------------------------------------------------    
    def multiply_langreth(self, A, B, C):
        C.zero()
        
        C.IR = self.IRxA(A, B) + self.MxIR(A, B)

        C.R = self.RxR(A, B)

        C.L = self.LxA(A, B) + self.RxL(A, B) + self.RIxIR(A, B)
        self.MxM(A, B, C)
        
    #------------------------------------------------------------    
    def dyson_langreth(self, G0, Sigma, G):

        # IS G0*Sigma still fermionic? A.sig=-1?
        # yes I think so because it worked for Matsubara...
        
        M = matsubara(G.beta, G.ntau, G.norb, G.sig)
        A = langreth(G.nt, G.tmax, M)
        self.multiply_langreth(G0, Sigma, A)
        
        nt   = G0.nt
        norb = G0.norb
        ntau = G0.ntau
        
        # solve RxR = R

        # Careful! use extended R or regular R????
        
        G0R = np.reshape(G0.R, [nt*norb, nt, norb])
        theta = np.tril(np.ones((nt,nt))) 
        for j in range(nt):
            sol = np.linalg.solve(np.diag(np.ones(nt*norb)) - self.prep_RxR(A,j), theta[:,j,None]*G0R[:,j,:])
            G.R[:,:,j,:] = np.reshape(sol, [nt,norb,norb])

        theta = np.tril(np.ones((nt,nt))) - np.diag(0.5*np.ones(nt))     
        G.R  = np.einsum('mn,manb->manb', theta, G.R)        
        G.R -= np.einsum('abcd->cdab', np.conj(G.R))
                    
        # solve A_M x B_IR = C_IR - A_IR x B_A
        # note minus sign on A because of I-A
        rhs = np.reshape(G0.IR + self.IRxA(A, G), [ntau*norb, nt*norb])
        sol = np.linalg.solve(np.diag(np.ones(ntau*norb)) - self.prep_MxIR(A), rhs)
        G.IR = np.reshape(sol, [ntau,norb,nt,norb])
        
        # solve A_R x B_L = C_L - A_L x B_A - A_RI x B_IR
        #       A_R x G_L = G0_L - A_L x G_A - A_RI x G_IR
        # note minus sign on A because of I-A
        rhs = np.reshape(G0.L + self.LxA(A, G) + self.RIxIR(A, G), [nt*norb, nt*norb])
        sol = np.linalg.solve(np.diag(np.ones(nt*norb)) - self.prep_Rxr(A), rhs)
        G.L = np.reshape(sol, [nt,norb,nt,norb])
        
        
        

        
