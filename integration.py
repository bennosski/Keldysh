import numpy as np
from scipy import integrate
import volterra 

class integrator:
    def __init__(self, nt, ntau, order):
        self.gregory_matrix_M = self._compute_gregory_matrix(ntau, order)
        self.rcorr = self._compute_rcorr(order)
        self.gregory_matrix_R = self._compute_gregory_matrix(nt, order)
        self.order = order
        self.nt = nt
        self.ntau = ntau
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
    def _compute_Cmk(self, A):
        '''
        prepare A for higher-order (Gregory and boundary correction) convolution using matrix multiplication

        A : a matsubara object of with Matsubara matrix of size (ntau x norb, norb)

        output : a matrix of size (ntau x norb x ntau x norb) which can be multiplied by a second vector to produce the convolution
        '''

        ntau  = A.ntau
        dtau   = A.dtau
        norb  = A.norb
        order = self.order
        
        Cmk = np.zeros((ntau,norb,ntau,norb),dtype=np.complex128)        

        AM = np.reshape(A.M, [ntau, norb, norb])
        
        for iorb in range(0,norb):
            for korb in range(0,norb):

                for m in range(ntau-order,ntau-1):
                    for k in range(0,order):
                        for l in range(0,order):
                            Cmk[m,iorb,ntau-1-k,korb] += -1j*dtau * self.rcorr[ntau-1-m,l,k] * A.sig * AM[ntau-1-l,iorb,korb]

                for m in range(0,ntau-order):
                    Cmk[m,iorb,m:ntau,korb] += -1j*dtau * self.gregory_matrix_M[ntau-1-m,:ntau-m] * A.sig * AM[np.arange(ntau-1,m-1,-1),iorb,korb]

                for m in range(1,order):
                    for k in range(0,order):
                        for l in range(0,order):
                            Cmk[m,iorb,k,korb] += -1j*dtau * self.rcorr[m,l,k] * AM[l,iorb,korb]

                for m in range(order,ntau):
                    Cmk[m,iorb,:m+1,korb] += -1j*dtau * self.gregory_matrix_M[m,:m+1] * AM[np.arange(m,-1,-1),iorb,korb]

        return np.reshape(Cmk, [ntau*norb, ntau*norb])
    #------------------------------------------------------------
    def MxM(self, A, B, C):
        '''
        Matsubara convolution
        C = A*B
        A, B and C are langreth objects
        '''
        ntau = A.ntau
        norb = A.norb

        Cmk = self._compute_Cmk(A)

        # Cmk should be of shape (ntau*norb, ntau*norb)
        # B.M should be of shape (ntau*norb, norb)
        
        C.M = Cmk @ B.M
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
        
        X = self._compute_Cmk(G)
        
        X = np.diag(np.ones(ntau*norb)) - X
    
        G.M = np.linalg.solve(X, G0.M)

        
