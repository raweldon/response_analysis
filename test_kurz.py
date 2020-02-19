import numpy as np
import matplotlib.pyplot as plt

def kurz(E_p, a, b, c, d):
    return a*E_p - b*(1-np.exp(-c*E_p**d))


a = 0.7420
#ds = np.linspace(0.9, 1, 10)
d = 0.9797
bs = np.linspace(2.2335, 3.8951, 10)
b = bs[5]
cs = np.linspace(0.1597, 0.2712, 10)

xvals = np.linspace(0, 10, 1000)
plt.figure()
for c in cs:
    plt.plot(xvals, kurz(xvals, a, b, c, d), label=str(round(c, 4))) 
    plt.legend()
#plt.xlim(0.01, 10)
#plt.xscale('log')
#plt.yscale('log')
plt.show()