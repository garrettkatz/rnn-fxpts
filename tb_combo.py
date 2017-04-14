import numpy as np
import rnn_fxpts as rfx
import fxpt_experiments as fe

# simple duplicate test
neighbors = lambda X, y: (np.fabs(X-y) < 2**-21).all(axis=0)

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

N = 8
s = 1

npz = {'T': fe.load_npz_file('results/traverse_full_base_N_%d_s_%d.npz'%(N,s)),
     'B': fe.load_npz_file('results/baseline_full_base_N_%d_s_%d.npz'%(N,s))}
W = npz['T']['W']
c = npz['T']['c']

k = 0 # current component

found = [npz['T']['fxV_unique']] # initial Traverse results
seeds = setdiff(npz['B']['fxV_unique'],found[0]) # B - T initial seeds
seed = [np.zeros((N,1))]
new = [np.empty((N,0))]
statuses = ['success']

print('%d: %s'%(k,statuses[k]))
print('|found|=%d, |new|=%d, |seeds|=%d'%(found[k].shape[1], new[k].shape[1], seeds.shape[1]))

while seeds.shape[1] > 0:
    k += 1
    seed.append(seeds[:,[0]])
    seeds = seeds[:,1:]

    # need to go both directions! and fxV shouldn't include origin if none found
    va = np.concatenate((seed[k],np.array([[0]])),axis=0) # include alpha
    status, fxV, _, c, _, _, _ = rfx.traverse(W, va=va, c=c, max_traverse_steps = 2**20)
    statuses.append(status)


    # need rfx.post_process to use simpler neighbor!
    fxV = np.concatenate((seed[k],fxV),axis=1) # be sure to include seed(t)
    fxV, _ = rfx.post_process_fxpts(W, fxV)

    new.append(fxV)
    found.append(union(found[k-1], new[k]))
    seeds = setdiff(seeds, new[k])

    print('%d: %s'%(k,statuses[k]))
    print('|found|=%d, |new|=%d, |seeds|=%d'%(found[k].shape[1], new[k].shape[1], seeds.shape[1]))
    # print(new[k]) # new includes origin every time because of post process

print('all cloops?')
print(all([s=='Closed loop detected' for s in statuses[1:]]))
# Not all cloops on (N,s)=(8,1)! penultimate has success status?
