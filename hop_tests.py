import numpy as np
import rnn_fxpts as rf

N = 6

data = np.random.random((N,3))*2-1
W = np.matmul(data,data.T)
fxpts, fiber = rf.run_solver(W)
print(fxpts.shape)
