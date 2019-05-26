import numpy as np
# load时需加上扩展名，save时不用，自动后缀npy
re = np.load('advre.npy')
print(re.shape)
amr = np.argmax(re,axis=1)
print(amr.shape)
print(amr)