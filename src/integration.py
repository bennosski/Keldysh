import numpy as np
from scipy import integrate
import volterra 
from matsubara import *
from langreth import *

class integrator:
    def __init__(self, order, nt, beta, ntau):
        self.gregory_matrix_M = self._compute_gregory_matrix(ntau, order)
        self.rcorr = self._compute_rcorr(order)
        self.gregory_matrix_R = self._compute_gregory_matrix(nt, order)
        self.order = order
        self.nt = nt
        self.ntau = ntau
        self.dtau = 1.0*beta/(ntau-1)
    #-------------------------------------------------------- 
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
                    for k in range(ntau-order,ntau):
                        for l in range(0,order):                            
                            Cmk[m,iorb,k,korb] += -1j*dtau * self.rcorr[ntau-1-m,l,ntau-1-k] * A.sig * A.M[ntau-1-l,iorb,korb]

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
    def prep_RIxM(self, B):
        '''
        prepare A for higher-order (Gregory and boundary correction) convolution using matrix multiplication

        B : a matsubara or langreth object with a Matsubara matrix of size (ntau x norb, norb)

        output : a matrix of size (ntau x norb x ntau x norb) which can be multiplied by a second vector to produce the convolution
        '''
        
        ntau  = B.ntau
        dtau  = B.dtau
        norb  = B.norb
        order = self.order
        
        Ckm = np.zeros((ntau,norb,ntau,norb),dtype=np.complex128)        
        
        for iorb in range(0,norb):
            for korb in range(0,norb):

                for m in range(ntau-order,ntau-1):
                    for k in range(ntau-order,ntau):
                        for l in range(0,order):                            
                            Ckm[k,iorb,m,korb] += -1j*dtau * self.rcorr[ntau-1-m,ntau-1-k,l] * B.M[l,iorb,korb]

                for m in range(0,ntau-order):
                    Ckm[m:,iorb,m,korb] += -1j*dtau * self.gregory_matrix_M[ntau-1-m,:ntau-m] * B.M[:ntau-m,iorb,korb]

                for m in range(1,order):
                    for k in range(0,order):
                        for l in range(0,order):
                            Ckm[k,iorb,m,korb] += -1j*dtau * self.rcorr[m,k,l] * B.sig * B.M[ntau-1-l,iorb,korb]

                for m in range(order,ntau):
                    Ckm[:m+1,iorb,m,korb] += -1j*dtau * self.gregory_matrix_M[m,:m+1] * B.sig * B.M[ntau-1-m:,iorb,korb]
                    
        return np.reshape(Ckm, [ntau*norb, ntau*norb])
    #------------------------------------------------------------
    def get_IR(self, A):
        return -A.sig * np.einsum('raib->ibra', np.conj(A.RI[:,:,::-1,:]))
    #------------------------------------------------------------
    def get_A(self, A):
        return np.einsum('manb->nbma', np.conj(A.R))    
    #------------------------------------------------------------
    def MxM(self, A, B):
        '''
        A = B*C
        '''
        ntau = A.ntau
        norb = A.norb

        delta = np.einsum('mab,bc->mac', A.M, B.deltaM)
        
        BM = np.reshape(B.M, [ntau*norb, norb])
        out = self.prep_MxM(A) @ BM
        return np.reshape(out, [ntau,norb,norb]) + delta
    #------------------------------------------------------------
    def RIxM(self, A, B):
        ntau = A.ntau
        nt   = A.nt
        norb = A.norb

        delta = np.einsum('manb,bc->manc', A.RI, B.deltaM)
        
        #ARI = -1j*dtau*np.einsum('i,raib->raib', self.gregory_matrix_M[-1,:], A.RI)

        ARI = np.reshape(A.RI, [nt*norb, ntau*norb])
        return np.reshape(ARI @ self.prep_RIxM(B), [nt,norb,ntau,norb]) + delta
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

        G.M = self.MxM(G0, Sigma)
        
        X = np.diag(np.ones(ntau*norb)) - self.prep_MxM(G)
        G0M = np.reshape(G0.M, [ntau*norb, norb])
        G.M = np.linalg.solve(X, G0M)
        G.M = np.reshape(G.M, [ntau,norb,norb])
    #------------------------------------------------------------
    def prep_MxIR(self, A):
        dtau = A.dtau

        # can we run preparation only once?
        # (only prepare SigmaM once with the gregory weights)
        # idea : store in the langreth object if component in the langreth object is none otherwise compute the following

        return self.prep_MxM(A)
    #------------------------------------------------------------
    def MxIR(self, A, B):
        ntau = A.ntau
        nt   = B.nt
        norb = A.norb

        #BIR = np.reshape(B.IR, [ntau*norb, nt*norb])
        BIR = np.reshape(self.get_IR(B), [ntau*norb, nt*norb])
        return np.reshape(self.prep_MxM(A) @ BIR, [ntau,norb,nt,norb])
    #------------------------------------------------------------
    def prep_RIxIR(self, A):
        ntau = A.ntau
        nt   = A.nt
        norb = A.norb
        dtau = A.dtau
        
        #ARI = -1j*dtau*np.einsum('i,ibra->raib', self.gregory_matrix_M[-1,:], -A.sig*np.conj(A.IR[::-1]))

        #ARI = -1j*dtau*np.einsum('i,ibra->raib', self.gregory_matrix_M[-1,:], -A.sig*np.conj(A.IR[::-1]))
        
        ARI = -1j*dtau*np.einsum('i,raib->raib', self.gregory_matrix_M[-1,:], A.RI)
        return np.reshape(ARI, [nt*norb, ntau*norb])
    #------------------------------------------------------------
    def RIxIR(self, A, B):
        ntau = A.ntau
        nt   = A.nt
        norb = A.norb
        
        #BIR = np.reshape(B.IR, [ntau*norb, nt*norb])
        #BIR = -B.sig * np.einsum('raib->ibra', np.conj(A.RI[:,:,::-1,:]))
        BIR = np.reshape(self.get_IR(B), [ntau*norb, nt*norb])
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

        AIR = self.get_IR(A)
        delta = np.einsum('manb,ncb->manc', AIR, np.conj(B.deltaR))

        AIR = np.reshape(self.get_IR(A), [ntau*norb, nt*norb])
        return np.reshape(AIR @ self.prep_rxA(B), [ntau,norb,nt,norb]) + delta
    #------------------------------------------------------------
    def LxA(self, A, B):
        '''
        prep second matrix for multiplication
        '''
        nt   = B.nt
        norb = B.norb
        dt   = B.dt

        delta = np.einsum('manb,ncb->manc', A.L, np.conj(B.deltaR))
        
        AL = np.reshape(A.L, [nt*norb, nt*norb])
        return np.reshape(AL @ self.prep_rxA(B), [nt,norb,nt,norb]) + delta
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

        #delta = np.einsum('manb,nbc->manc', A.R, B.deltaR)
        
        BL = np.reshape(B.L, [nt*norb, nt*norb])
        return np.reshape(self.prep_Rxr(A) @ BL, [nt,norb,nt,norb]) #+ delta
    #------------------------------------------------------------
    def Rxv(self, A, B):
        '''
        B is a vector of size [nt,norb,norb]
        designed to represent the density as a function for time (diagonal of the Green's function)
        this function is used to construct the Hartree selfenergy (integration of DR(t,t') n(t'))
        '''
        nt   = A.nt
        norb = A.norb
        dt   = A.dt

        # any deltaC piece ???? I'm guessing no we only worry
        # about the delta piece on B and B is already the delta piece here (a diagonal of the selfenergy)
        
        B = np.reshape(B, [nt*norb, norb])
        return np.reshape(self.prep_Rxr(A) @ B, [nt,norb,norb])     
    #------------------------------------------------------------
    def prep_RxR(self, As, j):
        nt   = As[0].nt
        norb = As[0].norb
        dt   = As[0].dt

        if j<nt//2:
            wj = np.zeros((nt,nt))
            wj[j:, j:] = self.gregory_matrix_R[:nt-j, :nt-j]
        else:
            wj = np.zeros((nt,nt))
            for n in range(j, nt):
                wj[n,:n+1] = self.gregory_matrix_R[n-j, n::-1]

        xs = [dt * np.einsum('nk,nakb->nakb', wj, A.R) for A in As]
        xs = [np.reshape(x, [nt*norb, nt*norb]) for x in xs]
        return xs
    #------------------------------------------------------------
    def RxR(self, A, B):
        nt   = A.nt
        norb = A.norb

        delta = np.einsum('manb,nbc->manc', A.R, B.deltaR)                
        out = np.zeros((nt, norb, nt, norb), dtype=np.complex128)
        
        AR = np.reshape(A.R, [nt*norb, nt, norb])
        BR = np.reshape(B.R, [nt*norb, nt, norb])
        for j in range(nt):
            ws = self.prep_RxR([A,B], j)
            
            x =  np.reshape(ws[0] @ BR[:,j,:], [nt,norb,norb])[j:]
            out[j:,:,j,:] = x
            
            x =  np.reshape(ws[1] @ AR[:,j,:], [nt,norb,norb])[j:]
            x = -np.einsum('mab->bma', np.conj(x))
            out[j,:,j:,:] = x

        return out + delta
    #------------------------------------------------------------
    def RxRI(self, A, B):
        nt   = B.nt
        ntau = A.ntau
        norb = B.norb
        dt   = B.dt
        
        BRI = np.reshape(B.RI, [nt*norb, ntau*norb])
        return np.reshape(self.prep_Rxr(A) @ BRI, [nt,norb,ntau,norb])
    #------------------------------------------------------------    
    def multiply_langreth(self, A, B, C):
        # not currently used

        exit()
        
        C.zero()
        
        C.IR = self.IRxA(A, B) + self.MxIR(A, B)

        C.R = self.RxR(A, B)

        C.L = self.LxA(A, B) + self.RxL(A, B) + self.RIxIR(A, B)

        C.M = self.MxM(A, B)
    #------------------------------------------------------------    
    def dyson_langreth(self, G0M, SigmaM, GM, G0, Sigma, G):

        # IS G0*Sigma still fermionic? A.sig=-1?
        # yes I think so because it worked for Matsubara...
        
        M = matsubara(GM.beta, GM.ntau, GM.norb, GM.sig)
        A = langreth(G.norb, G.nt, G.tmax, G.ntau, G.beta, G.sig)

        A.RI = self.RxRI(G0, Sigma) + self.RIxM(G0, SigmaM)
        
        A.R = self.RxR(G0, Sigma)

        A.L = self.LxA(G0, Sigma) + self.RxL(G0, Sigma) + self.RIxIR(G0, Sigma)

        #A.M = self.MxM(G0, Sigma)
            
        nt   = G0.nt
        norb = G0.norb
        ntau = G0.ntau
        
        # solve RxR = R
        G0R = np.reshape(G0.R, [nt*norb, nt*norb])
        G.R = np.zeros([nt*norb, nt*norb], dtype=np.complex128)
        for j in range(nt):
            X = np.diag(np.ones(nt*norb)) - self.prep_RxR([A],j)[0]
            for b in range(norb):
                n = j*norb + b
                lhs, rhs = X[n:,n:], G0R[n:,n] - X[n:,:n] @ G.R[:n,n]            
                G.R[n:,n] = np.linalg.solve(lhs, rhs)
                G.R[n,n:] = -np.conj(G.R[n:,n]) 
            
        G.R = np.reshape(G.R, [nt,norb,nt,norb])

        # solve A_M x B_IR = C_IR - A_IR x B_A
        # note minus sign on A because of I-A
        #rhs = np.reshape(G0.IR + self.IRxA(A, G), [ntau*norb, nt*norb])
        #sol = np.linalg.solve(np.diag(np.ones(ntau*norb)) - self.prep_MxIR(A), rhs)
        #G.IR = np.reshape(sol, [ntau,norb,nt,norb])

        # solve A_R x B_RI = C_RI - A_RI x B_M
        # note minus sign on A because of I-A
        rhs = np.reshape(G0.RI + self.RIxM(A, GM), [nt*norb, ntau*norb])
        sol = np.linalg.solve(np.diag(np.ones(nt*norb)) - self.prep_Rxr(A), rhs)
        G.RI = np.reshape(sol, [nt,norb,ntau,norb])
        
        # solve A_R x B_L = C_L - A_L x B_A - A_RI x B_IR
        #       A_R x G_L = G0_L - A_L x G_A - A_RI x G_IR
        # note minus sign on A because of I-A
        rhs = np.reshape(G0.L + self.LxA(A, G) + self.RIxIR(A, G), [nt*norb, nt*norb])
        sol = np.linalg.solve(np.diag(np.ones(nt*norb)) - self.prep_Rxr(A), rhs)
        G.L = np.reshape(sol, [nt,norb,nt,norb])
        
        

        
