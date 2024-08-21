import numpy as np

with np.errstate(all='raise'):
    try:
        a = np.ones(1)/1e-309 # overflow
    except:
        a = 0
# b = np.exp(1000.) # overflow

print(a)
# print(b)
