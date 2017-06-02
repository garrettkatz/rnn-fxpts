import pickle as pkl
import numpy as np
import scipy.linalg as spla
import rnn_dynamics as rd
import fxpt_experiments as fe
import matplotlib.pyplot as plt
import plotter as ptr

def main():

    # 2d where some fxpt not on every path.  happens at glancing nullclines
    network_sizes, num_samples, test_data=fe.load_test_data('dl2.npz')
    N, s = 2, 1
    W = test_data['N_%d_W_%d'%(N,s)]
    samp = 10
    ang = np.linspace(0,np.pi,samp)
    C = np.array([np.sin(ang),np.cos(ang)])
    v_ax = plt.subplot(1,2,1)
    a_ax = plt.subplot(1,2,2)
    for a in [0]:#range(samp):
        status, fxV, VA, c, deltas, s_mins = rd.traverse(W, c=C[:,[a]])
        print(a, fxV.shape[1])
        ptr.plot(v_ax, VA[:N,:], '-', c=str(1.0*a/samp))
        ptr.scatter(v_ax, fxV, c='r')
        a_ax.plot(VA[N,:], '-', c=str(1.0*a/samp))
        a_ax.plot(range(VA.shape[1]),np.zeros(VA.shape[1]),'--b')
    # nullclines
    v = np.linspace(-1,1,100)[1:-1]
    v_ax.plot(v, (np.arctanh(v)-W[0,0]*v)/W[0,1], '-b')
    v_ax.plot((np.arctanh(v)-W[1,1]*v)/W[1,0], v, '-b')
    ptr.set_lims(v_ax, np.array([[-2,2],[-2,2]]))
    ptr.set_lims(a_ax, np.array([[0,VA.shape[1]],[-.5,.5]]))
    plt.show()

    # f=open('dbg_64_cloop.pkl','r')
    # # f=open('dbg_64_alpha_plateau4.pkl','r')
    # ground_truth=pkl.load(f)
    # W = ground_truth[0][0]

    # _,VA,c,DVA,NR=rd.fixed_point_path_traversal(W,verbose=1,adapt_tol=2**-16)
    # _,VA,c,DVA,NR,deltas=rd.fixed_point_path_traversal(W,verbose=1,max_iters=100000)
    # VA = np.concatenate(VA,axis=1)
    # DVA = np.concatenate([d[:,:,np.newaxis] for d in DVA],axis=2)
    # deltas = np.array(deltas)

    # network_sizes, num_samples, test_data=fe.load_test_data('dl100.npz')
    # W = test_data['N_%d_W_%d'%(30,0)]
    # _=fe.test_traverse(W, np.zeros(W.shape), c=None, verbose=2)

    # test_data = fe.generate_dense_test_data([32], 1)
    # W = test_data['N_32_W_0']
    # _=fe.test_traverse(W, np.zeros(W.shape), verbose=2)

    # f = open('dbg_nr_step.pkl','r')
    # tup = pkl.load(f)
    # f.close()
    # W, I, c, va, z, delta, max_nr_iters, nr_tol = tup
    # print(W.shape)
    # _=fe.test_traverse(W, np.zeros(W.shape), c=c, verbose=2)    

    # f = open('dbg_nr_step.pkl','r')
    # tup = pkl.load(f)
    # f.close()
    # W, I, c, va, z, delta, max_nr_iters, nr_tol = tup
    # _=rd.take_traverse_step(W, I, c, va, z, delta, max_nr_iters, nr_tol, verbose=1)
    # # Dg,g_root,g,va=rd.take_traverse_step(W, I, c, va, z, delta, max_nr_iters, nr_tol, verbose=1)
    # # rd.mldivide(Dg, g_root-g) # hangs
    # # _=spla.lstsq(Dg, g_root-g)
    # # # return np.linalg.lstsq(A,B)[0]

    # plt.ion()
    # plt.plot(VA.T)
    # plt.show()

    # n = VA.shape[1]-1 #np.fabs(VA).sum(axis=0).argmax()
    # va = VA[:,[n]]

    # N = W.shape[0]
    # I = np.eye(N)
    # Wv = np.dot(W,va[:N,:])
    # J = np.concatenate(((1-np.tanh(Wv)**2)*W - I, -c), axis=1) # Jacobian
    # u, s, v = np.linalg.svd(J) # for null-space
    # dva1 = v[[-1],:].T/np.linalg.norm(v[[-1],:]) # Current tangent vector (unit speed)
    # Jdva1 = -np.concatenate((J,dva1.T),axis=0)
    # Jdva12 = np.append(-2*np.tanh(Wv)*(1-np.tanh(Wv)**2)*Wv**2,[[0]],axis=0)
    # dva2 = rd.mldivide(Jdva1, Jdva12) # Acceleration
    
    # np.set_printoptions(linewidth=200)
    # print('total distance:', deltas.sum())

    # plt.plot(VA.T)
    # plt.show()

    # print('n')
    # print(n)
    # print('va.T')
    # print(va.T)
    # print('W')
    # print(W)
    # print('Wv.T')
    # print(Wv.T)
    # print('J')
    # print(J)
    # print('s')
    # print(s)
    # return W, c, VA, DVA, va, J

if __name__=="__main__":
    # W, c, VA, DVA, va, J = main()
    _ = main()
