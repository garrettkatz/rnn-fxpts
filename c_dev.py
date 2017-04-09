import rnn_fxpts as rfx
import fxpt_experiments as fe
import plotter as pt
import numpy as np
import os

N = 8

trials = 20

works = []

for t in range(trials):

    td = fe.generate_test_data([N], [1], test_data_id=None, refine_iters=2**5,refine_cap=10000)
    
    W = td['N_%d_W_%d'%(N,0)]
    V = td['N_%d_V_%d'%(N,0)]
    
    max_fxpts=None
    
    # use random c
    results, npz = fe.test_traverse(W, V, c=None, logfilename=os.devnull,max_fxpts=max_fxpts)
    for k in results:
        print('%s: %s'%(k,results[k]))
        
    # make c from idea
    A = np.zeros((N,N))
    for i in range(N):
        # check i for hump
        if W[i,i] <= 0 or (W[i,:]**2).sum() < W[i,i]: continue
        for j in range(N):
            if j == i: continue
            A[i,i] += 1 - W[i,i]/(W[i,:]**2).sum()
            A[i,j] += W[i,i]*W[i,j]/(W[i,:]**2).sum()
            A[j,i] += W[i,i]*W[i,j]/(W[i,:]**2).sum()
            A[j,j] += 1
    print('A:')
    print(A)
    eigs,vecs = np.linalg.eig(A)
    idx = np.argsort(eigs)
    # c_inv = vecs[:,[idx[-1]]]
    c_inv = vecs[:,[idx[0]]]
    # c_inv = vecs[:,[idx[int(N/2)]]]
    if (c_inv==0).any():
        c = np.zeros((N,1))
        c[c_inv==0] = 1
    else:
        c = 1/c_inv
    c = c/np.sqrt((c**2).sum())
    print(c)
    results2, npz2 = fe.test_traverse(W, V, c=c, logfilename=os.devnull,max_fxpts=max_fxpts)
    for k in results2:
        print('%s: %s'%(k,results2[k]))
    
    print('cs:')
    print(np.concatenate((npz['c'],c),axis=1))
    
    print('lens:')
    print(results['path_length'],results2['path_length'])
    
    print('steps:')
    print(results['num_steps'],results2['num_steps'])
    
    print('time:')
    print(results['runtime'],results2['runtime'])
    
    print('numfx:')
    print(results['num_fxV'],results2['num_fxV'])
    
    print('work:')
    print(results['runtime']/results['num_fxV'],results2['runtime']/results2['num_fxV'])
    
    works.append([results['runtime']/results['num_fxV'],results2['runtime']/results2['num_fxV']])
    
    # show
    if N < 4:
        ax = pt.plt.gca(projection='3d')
        pt.plot(ax,npz['VA'][:N,:],'-k')
        pt.plot(ax,-npz['VA'][:N,:],'-k')
        pt.plot(ax,npz['fxV'],'ko')
        pt.plot(ax,-npz['fxV'],'ko')
        
        pt.plot(ax,npz2['VA'][:N,:],'-r')
        pt.plot(ax,-npz2['VA'][:N,:],'-r')
        pt.plot(ax,npz2['fxV'],'ko')
        pt.plot(ax,-npz2['fxV'],'ko')
        
        pt.plt.show()


print('work stats:')
print(np.array(works))
print(np.array(works).mean(axis=0))
print(np.array(works).std(axis=0))
