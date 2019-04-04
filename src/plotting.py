from numpy import *
from matplotlib.pyplot import *


def plt_diffs(diffs):
    figure()
    log_nts = log10(array(diffs['nts']))
    ls = []
    labels = []
    for p in diffs:
        if p!='nts':
            labels.append(p)
            ls.append(plot(log_nts, log10(array(diffs[p])), '.-')[0])
    legend(ls, labels)
    xlabel('$\log(N_t)$', fontsize=12)
    ylabel('$\log(\mathrm{error})$', fontsize=12)
    show()

def plt(x, ys, name):
    figure()
    styles = ['-','--']
    for i,y in enumerate(ys):
        plot(x, y, styles[i])
    title(name)
    show()

def im(ys, extent, name):
    f = figure()
    f.set_size_inches(5*len(ys), 6)
    for i,y in enumerate(ys):
        ax = f.add_axes([0.07 + 0.9/len(ys)*i, 0.15, 0.7/len(ys), 0.8]) 
        image = ax.imshow(y, origin='lower', aspect='auto', extent=extent)
        colorbar(image, ax=ax)
    title(name)
    show()
