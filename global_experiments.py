import time
import numpy as np
import rnn_fxpts as rfx
import fxpt_experiments as fe
import matplotlib.pyplot as plt
import plotter as ptr

def process_fxpt(W, V, v, tolerance = 2**-21):
    """
    Process a new candidate fixed point v against existing set V
    V[:,p] is the p^th fixed point found so far
    tolerance is the maximum infinity norm at which two points are considered duplicates
    Refines v and checks whether v is fixed
    Checks for duplicates of v in V
    Returns V_new, fx, dup where
        V_new is (V union {+v,-v}) with duplicates removed if v is fixed
        V_new is V if v is not fixed
        fx is true iff v is fixed
        dup is true if there were duplicates
    """
    v, fx = rfx.refine_fxpts(W, v)    
    dup = False
    if fx:
        duplicates = (np.fabs(V-v).max(axis=0) < tolerance) | (np.fabs(V+v).max(axis=0) < tolerance)
        dup = duplicates.any()
        if np.fabs(v).max(axis=0) < tolerance: # zero fxpt
            V = np.concatenate((V[:, ~duplicates], v), axis=1)
        else:
            V = np.concatenate((V[:, ~duplicates], v, -v), axis=1)
    return V, fx, dup

def local_trial(W, timeout = .1):
    """
    Run a solver trial using only repeated local search
    W is the rnn weight matrix
    timeout is the number of seconds to continue repeating
    returns V, timestamp, where
        V[i][:,p] is the p^th fixed point found after the i^th iterate
        timestamp[i] is the clock time after the i^th iterate 
    """
    stop_time = time.clock() + timeout
    V = [np.zeros((W.shape[0],1))]
    timestamp = [time.clock()]
    iterates = rfx.local_search(W, stop_time=stop_time)
    for status, v, _ in iterates:
        V_new, _, _ = process_fxpt(W, V[-1], v)
        V.append(V_new)
        timestamp.append(time.clock())
    return V, timestamp

def fiber_trial(W, timeout = .1, repeats = None):
    """
    Run a solver trial using only repeated fiber traversal
    Repeats with different random c until timeout
    W is the rnn weight matrix
    timeout is the number of seconds to continue repeating
    repeats is the number of times to continue repeating
    returns V, timestamp, traversal, c, status, VA, where
        V[i][:,p] is the p^th fixed point found after the i^th iterate
        timestamp[i] is the clock time after the i^th iterate
        traversal[i]: is t, where the t^th traversal returned the i^th iterate
        c[i] is the random c used by the i^th iterate
        status[i] is the traversal status at the i^th iterate
        VA[t][n] is the n^th point along the t^th traversal
    """
    stop_time = time.clock() + timeout
    V = [np.zeros((W.shape[0],1))]
    VA = [[np.zeros((W.shape[0]+1,1))]]
    timestamp = [time.clock()]
    traversal = [0]
    c = [None]
    status = [None]
    t = 0
    while True:
        if stop_time is not None and time.clock() > stop_time: break
        if repeats is not None and t >= repeats: break
        t += 1
        iterates = rfx.directional_fiber(W, stop_time=stop_time)
        for iterate in iterates:
            V_new, _, _ = process_fxpt(W, V[-1], iterate[1])
            V.append(V_new)
            timestamp.append(time.clock())
            traversal.append(t)
            c.append(iterate[3])
            status.append(iterate[0])
        VA.append(iterate[2])
    return V, timestamp, traversal, c, status, VA

def combo_trial(W, c=None, timeout=1, term_ratio=None):
    """
    Run a solver trial using combined local search and fiber traversal
    Repeats traversal with the same c but different initial fixed points until timeout
    W is the rnn weight matrix
    timeout is the number of seconds to continue repeating
    term_ratio, if not None, allows early termination if:
        (the current time elapsed) / (time elapsed at the last new fixed point) > term_ratio
    returns V, timestamp, traversal, c, status, where
        V[i][:,p] is the p^th fixed point found after the i^th iterate
        timestamp[i] is the clock time after the i^th iterate
        traversal[i]: is t, where the t^th traversal returned the i^th iterate
        c is the random c used by all iterates
        status[i] is the traversal status at the i^th iterate, except 
        status[-1] is 'Term ratio satisfied' if method exits successfully
    """
    start_time = time.clock()
    stop_time = start_time + timeout
    # Start with origin component
    V = [np.zeros((W.shape[0],1))]
    timestamp = [time.clock()]
    traversal = [0]
    status = ['Traversing']
    t = 0
    fiber_component = rfx.directional_fiber(W, c=c, stop_time=stop_time)
    for iterate in fiber_component:
        fxpt_time = time.clock()
        V_new, _, _ = process_fxpt(W, V[-1], iterate[1])
        V.append(V_new)
        timestamp.append(time.clock())
        traversal.append(t)
        status.append(iterate[0])
    VA = [iterate[2]]
    c = iterate[3] # Same c for subsequent traversals
    # Do non-origin components with local seeds
    seeds = rfx.local_search(W, stop_time=stop_time)
    for seed_status, fxv, _ in seeds:
        # check if timed out
        if seed_status == 'Timed out': break
        # check if term_ratio exceeded
        current_ratio = (time.clock() - start_time)/(timestamp[-1] - start_time)
        if term_ratio is not None and current_ratio > term_ratio:
            # status[-1] = 'Term ratio satisfied'
            break
        # check if not fixed or already found
        _, fx, dup = process_fxpt(W, V[-1], fxv)
        if dup or not fx: continue
        # traverse component
        va = np.concatenate((fxv, [[0]]), axis=0)
        t += 1
        fiber_component = rfx.directional_fiber(W, va=va, c=c, stop_time=stop_time)
        for iterate in fiber_component:
            V_new, _, _ = process_fxpt(W, V[-1], iterate[1])
            V.append(V_new)
            timestamp.append(time.clock())
            traversal.append(t)
            status.append(iterate[0])
        VA.append(iterate[2])
    return V, timestamp, traversal, c, status, VA

def mini_compare():

    N = 6
    test_data = fe.generate_test_data(network_sizes=[N], num_samples=[1], refine_iters = 1)
    W = test_data['N_%d_W_0'%N]
    V = test_data['N_%d_V_0'%N]

    timeout = 5
    results = {}
    results['local'] = local_trial(W, timeout=timeout)
    results['fiber'] = fiber_trial(W, timeout=timeout)
    results['combo'] = combo_trial(W, timeout=timeout)

    fxpts = (V,) + tuple(results[k][0][-1] for k in results)
    neighbors = lambda X, y: (np.fabs(X-y) < 2**-21).all(axis=0)
    U = rfx.get_unique_points_recursively(np.concatenate(fxpts,axis=1), neighbors=neighbors)

    keys = results.keys()
    for k in keys:
        V_k = results[k][0]
        T_k = results[k][1]
        plt.plot([T_ki - T_k[0] for T_ki in T_k], [V_ki.shape[1] for V_ki in V_k])
    plt.plot([0,timeout], 2*[U.shape[1]])
    plt.legend(keys + ['union'],loc='lower right')
    plt.show()

    # for i in range(len(V)):
    #     if status[i] == 'Traversing': continue
    #     print(timestamp[i],V[i].shape[1], traversal[i], status[i])
        
    # if N == 3: ax = plt.gca(projection='3d')
    # else: ax = plt.gca()
    # for va in VA:
    #     va = np.concatenate(va,axis=1)[:N,:]
    #     print(np.fabs(va[:,1:]-va[:,:-1]).max(axis=0).max())
    #     ptr.plot(ax,va,'ko-')
    # ptr.plot(ax,V[-1],'r.')    
    # plt.show()

def main():

    while True:
        N = 3
        test_data = fe.generate_test_data(network_sizes=[N], num_samples=[1], refine_iters = 1)
        W = test_data['N_%d_W_0'%N]
        V = test_data['N_%d_V_0'%N]
        timeout = 500
        term_ratio = 2
        start = time.clock()
        V, timestamp, traversal, c, status, VA = combo_trial(W, timeout=timeout, term_ratio=term_ratio)
        end_status = []
        bad_status = False
        success_idx = 0
        for i in range(len(traversal)):
            if i == len(traversal)-1 or traversal[i] != traversal[i+1]:
                print(traversal[i],status[i])
                end_status.append(status[i])
                if len(end_status) > 1 and end_status[-1] == 'Success':
                    success_idx = traversal[i]
                    bad_status = True
        print('%d fxpts, %d components, took %f of %f seconds'%(V[-1].shape[1], traversal[-1]+1, time.clock()-start, timeout))
        if bad_status: break

    print(V[-1][:,np.argsort(np.fabs(V[-1]).max(axis=0))[:5]])
    print('\a')
    if N == 3: ax = plt.gca(projection='3d')
    else: ax = plt.gca()
    ptr.plot(ax,np.concatenate(VA[0],axis=1)[:N,:],'ko-')
    ptr.plot(ax,np.concatenate(VA[success_idx],axis=1)[:N,:],'go-')
    ptr.plot(ax,V[-1],'r.')    
    plt.show()
    
if __name__=='__main__':
    main()
