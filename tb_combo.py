import numpy as np
import plotter as pt
import rnn_fxpts as rfx
import fxpt_experiments as fe

# pt.plt.ion()

# simple duplicate test
neighbors = lambda X, y: (np.fabs(X-y) < 2**-21).all(axis=0)

def check_known(fxV, knV):
    # check for ground truth inclusion
    N = fxV.shape[0]
    V_found = np.zeros(N, dtype=bool)
    if fxV.shape[1] > knV.shape[1]:
        for j in range(knV.shape[1]):
            identical = neighbors(fxV, knV[:,[j]])
            V_found[j] = identical.any()
    else:
        for j in range(fxV.shape[1]):
            identical = neighbors(knV, fxV[:,[j]])
            V_found |= identical
    return V_found

def setdiff(fxV1, fxV2):
    fxV = []
    for j in range(fxV1.shape[1]):
        if ~(neighbors(fxV2, fxV1[:,[j]])).any():
            fxV.append(fxV1[:,[j]])
    if len(fxV) > 0:
        fxV = np.concatenate(fxV,axis=1)
    else:
        fxV = np.empty((fxV1.shape[0],0))
    return fxV

def union(fxV1, fxV2):
    fxV = np.concatenate((fxV1, fxV2), axis=1)
    fxV = rfx.get_unique_points_recursively(fxV, neighbors=neighbors)
    return fxV

def add_alpha_mins(W, VA, fxV):
    N = W.shape[0]
    abs_alpha = np.fabs(VA[N,:])
    local_mins = (abs_alpha[:-2] >= abs_alpha[1:-1]) & (abs_alpha[1:-1] <= abs_alpha[2:])
    seed_mask = np.zeros(abs_alpha.shape,dtype=bool)
    seed_mask[:-2] |= local_mins
    seed_mask[1:-1] |= local_mins
    seed_mask[2:] |= local_mins
    slowV = VA[:N, seed_mask]
    fxV = np.concatenate((fxV,slowV),axis=1)
    fxV, _ = rfx.post_process_fxpts(W, fxV,neighbors=neighbors)
    return fxV

def test_tbc(test_data_id, N, s,verbose=0):
    # load data
    npz = {'T': fe.load_npz_file('results/traverse_full_base_N_%d_s_%d.npz'%(N,samp)),
         'B': fe.load_npz_file('results/baseline_full_base_N_%d_s_%d.npz'%(N,samp))}
    knV = npz['T']['V']
    W = npz['T']['W']
    c = npz['T']['c']

    # add alpha mins
    npz['T']['fxV_unique'] = add_alpha_mins(W, npz['T']['VA'], npz['T']['fxV_unique'])

    # Do combo
        
    k = 0 # current component
    
    found = [npz['T']['fxV_unique']] # initial Traverse results
    seeds = [setdiff(npz['B']['fxV_unique'],found[0])] # B - T initial seeds
    seed = [np.zeros((N,1))]
    new = [np.empty((N,0))]
    statuses = ['success']
    known = [check_known(npz['T']['fxV_unique'], knV)]
    
    if verbose > 0:
        print('%d: %s'%(k,statuses[k]))
        print('|found|=%d, |new|=%d, |seeds|=%d, |known|=%d'%(found[k].shape[1], new[k].shape[1], seeds[k].shape[1], known[k].sum()))
        
    while seeds[k].shape[1] > 0:
        k += 1
        seed.append(seeds[k-1][:,[0]])
    
        va = np.concatenate((seed[k],np.array([[0]])),axis=0) # include alpha in seed
        status, fxV, VA, c, _, _, _ = rfx.traverse(W, va=va, c=c, max_traverse_steps = 2**20)
        statuses.append(status)
    
        if status != 'Closed loop detected':
            # need to go both directions if non-cloops
            raw_input('not cloop...')
            pass
    
        fxV = np.concatenate((seed[k],fxV),axis=1) # be sure to include seed(t)
        fxV, _ = rfx.post_process_fxpts(W, fxV, neighbors=neighbors)
    
        new.append(fxV)
        found.append(union(found[k-1], new[k]))
        known.append(check_known(found[k], knV))
        seeds.append(setdiff(seeds[k-1][:,1:], new[k]))
    
        if verbose > 0:
            print('%d: %s'%(k,statuses[k]))
            print('|found|=%d, |new|=%d, |seeds|=%d'%(found[k].shape[1], new[k].shape[1], seeds[k].shape[1]))
            # print(new[k]) # new includes origin every time because of post process
    
    num_cloops = sum([s=='Closed loop detected' for s in statuses[1:]])
    num_comps = len(statuses)
    
    results = {
        '|found|':np.array([f.shape[1] for f in found]),
        '|new|':np.array([n.shape[1] for n in new]),
        '|known|':np.array([k.sum() for k in known]),
        '|seeds|':np.array([s.shape[1] for s in seeds]),
        'num_cloops':num_cloops,
        'num_comps':num_comps,
    }
    return results, (found, new, known, seeds)

        
# N = 3
# N = 6
# N = 8
# N = 10
N = 8

all_cloops = 0.
all_comps = 0.

Ks = []
y_f, y_k, y_s = np.array([]), np.array([]), np.array([])
num_comps = []

for samp in range(50):
    print('samp %d'%samp)
    results, _ = test_tbc('full_base',N,samp,verbose=1)
    K = range(len(results['|found|']))
    Ks += K
    num_comps.append(len(results['|found|']))
    y_f = np.concatenate((y_f,1.*results['|found|']/results['|found|'].max()))
    y_k = np.concatenate((y_k,1.*results['|known|']/N))
    y_s = np.concatenate((y_s,1.*results['|seeds|']/results['|seeds|'].max()))
    # pt.plt.plot(K, 1.*results['|found|']/results['|found|'].max(), 'ko-',label='% found')
    # pt.plt.plot(K, 1.*results['|known|']/N, 'ks-',label='% known')
    # pt.plt.plot(K, 1.*results['|seeds|']/results['|seeds|'].max(), 'kd-',label='% seeds')
    # pt.plt.legend(loc='center right')
    # pt.plt.ylim([0,1])
    # pt.plt.show()
    # print(results)
    # raw_input('.')

handles = []
handles.append(fe.scatter_with_errors(Ks, np.arange(max(Ks)+1), y_f, 'o', 'none'))
handles.append(fe.scatter_with_errors(Ks, np.arange(max(Ks)+1), y_k, 'd', 'none'))
handles.append(fe.scatter_with_errors(Ks, np.arange(max(Ks)+1), y_s, 's', 'none'))
pt.plt.legend(handles,['% found','% known','% seeds'],loc='center right')
pt.plt.ylim([0,1])
pt.plt.ylabel('% of maximum')
pt.plt.xlabel('components traversed')
pt.plt.show()

print(num_comps)
print(min(num_comps),max(num_comps))
pt.plt.figure()
pt.plt.hist(num_comps,bins=range(1,max(num_comps)+1),facecolor='w')
pt.plt.ylabel('# of networks')
pt.plt.xlabel('# of fiber components found')
pt.plt.show()


print('In all, %d cloops of %d comps'%(all_cloops,all_comps))


    # npz = {'T': fe.load_npz_file('results/traverse_full_base_N_%d_s_%d.npz'%(N,samp)),
    #      'B': fe.load_npz_file('results/baseline_full_base_N_%d_s_%d.npz'%(N,samp))}
    # W = npz['T']['W']
    # c = npz['T']['c']

    # # # redo traverse
    # # if samp==3:
    # #     status, fxV, VA, c, _, _, _ = rfx.traverse(W, c=c, max_traverse_steps = 2**20, max_step_size=0.001)
    # #     fxV, _ = rfx.post_process_fxpts(W, fxV)
    # #     npz['T']['fxV_unique'] = fxV
    # #     npz['T']['VA'] = VA
    # #     pt.plt.plot(npz['T']['VA'][N,:])
    # #     pt.plt.plot(np.zeros(npz['T']['VA'][N,:].shape))
    # #     pt.plt.ylim([-1,1])
    # #     raw_input(',;')

    # # add alpha mins
    # npz['T']['fxV_unique'] = add_alpha_mins(W, npz['T']['VA'], npz['T']['fxV_unique'])

    # # if samp==3:
    # #     #N=3:
    # #     ax = pt.plt.gca(projection='3d')
    # #     pt.quiver(ax, npz['T']['VA'][:N,:], c*npz['T']['VA'][[N],:])
    # #     pt.plot(ax, npz['T']['VA'][:N,:],'b-')
    # #     pt.plot(ax, npz['T']['fxV_unique'][:N,:],'ko')

    # #     # # alpha:
    # #     # pt.plt.plot(npz['T']['VA'][N,:])
    # #     # pt.plt.plot(np.zeros(npz['T']['VA'][N,:].shape))
    # #     # pt.plt.ylim([-1,1])
    # #     # print('%g'%np.fabs(np.tanh(W.dot(seed[k]))-seed[k]).max())
    # #     # print('%g'%np.fabs(found[0]-seed[k]).max(axis=0).min())
    # #     raw_input(',;.')
    
    # k = 0 # current component
    
    # found = [npz['T']['fxV_unique']] # initial Traverse results
    # seeds = setdiff(npz['B']['fxV_unique'],found[0]) # B - T initial seeds
    # seed = [np.zeros((N,1))]
    # new = [np.empty((N,0))]
    # statuses = ['success']
    
    # # print('%d: %s'%(k,statuses[k]))
    # # print('|found|=%d, |new|=%d, |seeds|=%d'%(found[k].shape[1], new[k].shape[1], seeds.shape[1]))
    
    # # pt.plotNd(npz['T']['VA'],3*np.ones((N,1))*np.array([[-1,1]]),'r-')
    # # raw_input('...')
    
    # while seeds.shape[1] > 0:
    #     k += 1
    #     seed.append(seeds[:,[0]])
    #     seeds = seeds[:,1:]
    
    #     # need to go both directions! and fxV shouldn't include origin if none found
    #     va = np.concatenate((seed[k],np.array([[0]])),axis=0) # include alpha
    #     status, fxV, VA, c, _, _, _ = rfx.traverse(W, va=va, c=c, max_traverse_steps = 2**20)
    #     statuses.append(status)
    
    #     # if samp == 3:
    #     if status=='Success':
        
    #         if N!=3:
    #             pt.plotNd(npz['T']['VA'],3*np.ones((N,1))*np.array([[-1,1]]),'r-')
    #             pt.plotNd(VA,3*np.ones((N,1))*np.array([[-1,1]]),'b-')

    #         if N==3:
    #             ax = pt.plt.gca(projection='3d')
    #             pt.quiver(ax, npz['T']['VA'][:N,:], c*npz['T']['VA'][[N],:])
    #             pt.plot(ax, npz['T']['VA'][:N,:],'r-')
    #             pt.plot(ax, npz['T']['fxV_unique'][:N,:],'ko')
    #             pt.plot(ax, seed[k],'go')
    #             pt.plot(ax, VA[:N,:],'b-')
    #             # pt.plot(ax, -VA[:N,:],'b-')

    #         # # alpha:
    #         # pt.plt.plot(npz['T']['VA'][N,:])
    #         # pt.plt.plot(np.zeros(npz['T']['VA'][N,:].shape))
    #         # pt.plt.ylim([-1,1])
    #         # print('%g'%np.fabs(np.tanh(W.dot(seed[k]))-seed[k]).max())
    #         # print('%g'%np.fabs(found[0]-seed[k]).max(axis=0).min())
            
    #         raw_input('...')
    #         pass
    
    #     # need rfx.post_process to use simpler neighbor!
    #     fxV = np.concatenate((seed[k],fxV),axis=1) # be sure to include seed(t)
    #     fxV, _ = rfx.post_process_fxpts(W, fxV)
    
    #     new.append(fxV)
    #     found.append(union(found[k-1], new[k]))
    #     seeds = setdiff(seeds, new[k])
    
    #     # print('%d: %s'%(k,statuses[k]))
    #     # print('|found|=%d, |new|=%d, |seeds|=%d'%(found[k].shape[1], new[k].shape[1], seeds.shape[1]))
    #     # # print(new[k]) # new includes origin every time because of post process
    
    # # print(all([s=='Closed loop detected' for s in statuses[1:]]))
    # cloops = sum([s=='Closed loop detected' for s in statuses[1:]])
    # print('%d: %d of %d cloop?'%(samp,cloops, len(statuses)-1))
    # all_cloops += cloops
    # all_comps += len(statuses)-1
    # # Not all cloops all the time! Local minima of |alpha| that don't change sign!!
        
