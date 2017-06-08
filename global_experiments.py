import time
import numpy as np
import rnn_fxpts as rfx
import fxpt_experiments as fe

def main():
    N = 3
    test_data = fe.generate_test_data(network_sizes=[N], num_samples=[1], refine_iters = 1)
    W = test_data['N_%d_W_0'%N]
    fxpts = np.empty((N,0))
    fiber = rfx.directional_fiber(W, stop_time=time.clock()+10)
    for status, fxv, VA, c, step_sizes, s_mins, residuals in fiber:
        if status == 'Traversing':
            fxv, converged = rfx.refine_fxpts(W, fxv)
            if converged[0]:
                if not (np.fabs(fxpts - fxv).max(axis=0) < 2**-21).any():
                    fxpts = np.concatenate((fxpts, fxv),axis=1)
    print(status) # status
    print(fxpts)

if __name__=='__main__':
    main()
