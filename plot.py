import matplotlib as mpl
mpl.use('Agg')
from matplotlib.pyplot import *
import numpy as np

def myplot(x, y, z, savedir, Nt, Ntau, dt, dtau):

    ts = np.arange(0, Nt*dt, dt) - Nt*dt/2.0
    taus = np.arange(0, Ntau*dtau, dtau) - Ntau*dtau/2.0

    f = figure()
    plot(ts, np.diag(x.G[::-1]).real)
    plot(ts, np.diag(y.G[::-1]).real)
    plot(ts, np.diag(z.G[::-1]).real, '--')
    legend(['bare','exact','test'])
    title('Re Greater')
    xlabel('t')
    savefig(savedir+'Greater')

    figure()
    plot(ts, np.diag(x.L[::-1]).real)
    plot(ts, np.diag(y.L[::-1]).real)
    plot(ts, np.diag(z.L[::-1]).real, '--')
    legend(['bare','exact','test'])
    title('Re Lesser')
    xlabel('t')
    savefig(savedir+'Lesser')
    
    figure()
    plot(taus, np.diag(x.M[::-1]).imag)
    plot(taus, np.diag(y.M[::-1]).imag)
    plot(taus, np.diag(z.M[::-1]).imag, '--')
    legend(['bare','exact','test'])
    title('Im Matsubara')
    xlabel('tau')
    savefig(savedir+'Matsubara')

    figure()
    plot(ts, x.IR[len(taus)//2].real)
    plot(ts, y.IR[len(taus)//2].real)
    plot(ts, z.IR[len(taus)//2].real, '--')
    legend(['bare','exact','test'])
    title('Re IR')
    xlabel('t')
    savefig(savedir+'IR1')

    figure()
    plot(taus, x.IR[:,len(ts)//2].real)
    plot(taus, y.IR[:,len(ts)//2].real)
    plot(taus, z.IR[:,len(ts)//2].real, '--')
    legend(['bare','exact','test'])
    title('Re IR')
    xlabel('tau')
    savefig(savedir+'IR2')

    figure()
    plot(ts, x.RI[:,len(taus)//2].real)
    plot(ts, y.RI[:,len(taus)//2].real)
    plot(ts, z.RI[:,len(taus)//2].real, '--')
    legend(['bare','exact','test'])
    title('Re RI')
    xlabel('t')
    savefig(savedir+'RI1')

    figure()
    plot(taus, x.RI[len(ts)//2].real)
    plot(taus, y.RI[len(ts)//2].real)
    plot(taus, z.RI[len(ts)//2].real, '--')
    legend(['bare','exact','test'])
    title('Re RI')
    xlabel('t')
    savefig(savedir+'RI2')

    

    
    
    
