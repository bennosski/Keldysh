from numpy import *
from matplotlib.pyplot import *

def plt(x, ys, name):
    figure()
    for y in ys:
        plot(x, y, '.-')
    title(name)
    show()
