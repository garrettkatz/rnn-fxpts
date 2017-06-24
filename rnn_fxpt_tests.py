import numpy as np
import plotter as ptr
import matplotlib.pyplot as plt
import rnn_fxpts as rfx
import fxpt_experiments as fe

def test_fiber_refine():
    N = 3
    test_data = fe.generate_test_data(network_sizes=[N], num_samples=[1], refine_iters = 1)
    W = test_data['N_%d_W_0'%N]
    V = np.empty((N,0))
    plt.ion()
    for iterate in rfx.directional_fiber(W, va=None, c=None, max_nr_iters=2**8, nr_tol=2**-32, max_step_size=None, max_traverse_steps=None, max_refine_steps=2**5, max_fxpts=None, stop_time=None, logfile=None):
        status, fxv, VA, c, step_sizes, s_mins, residuals, refinement = iterate
        if status != 'Traversing':
            print status
            break
        refine_status, refine_fxv, refine_VA, refine_step_sizes, refine_s_mins, refine_residuals = refinement
        assert((refine_fxv==fxv).all())
        V, fx, dup = rfx.process_fxpt(W, V, fxv, tolerance = 2**-21)
        if N == 3: ax = plt.gca(projection='3d')
        else: ax = plt.gca()
        plt.cla()
        ptr.plot(ax,np.concatenate(VA,axis=1)[:N,:],'ko-')
        ptr.plot(ax,np.concatenate(refine_VA,axis=1)[:N,:],'go-')
        ptr.plot(ax,V,'rx')
        ptr.plot(ax,refine_fxv,'r+')
        ptr.set_lims(ax, 3*np.ones((N,1))*np.array([-1,1]))
        plt.show()
        raw_input('refine_status=%s, alpha in=%f, alpha out=%f,fixed=%s,dup=%s'%(refine_status, VA[-1][N], refine_VA[-1][N], fx, dup))
    raw_input('done')

if __name__ == '__main__':
    test_fiber_refine()
