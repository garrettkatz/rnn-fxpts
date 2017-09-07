import time
import numpy as np
import rnn_fxpts as rfx
import fxpt_experiments as fe
import matplotlib.pyplot as plt
import plotter as ptr

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
        V_new, _, _, _ = rfx.process_fxpt(W, V[-1], v)
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
            V_new, _, _, _ = rfx.process_fxpt(W, V[-1], iterate[1])
            V.append(V_new)
            timestamp.append(time.clock())
            traversal.append(t)
            c.append(iterate[3])
            status.append(iterate[0])
        VA.append(iterate[2])
    return V, timestamp, traversal, c, status, VA

def combo_trial(W, c=None, timeout=1, term_ratio=None, max_step_size=None, verbose_prefix = None):
    """
    Run a solver trial using combined local search and fiber traversal
    Repeats traversal with the same c but different initial fixed points until timeout
    W is the rnn weight matrix
    timeout is the number of seconds to continue repeating
    term_ratio, if not None, allows early termination if:
        (the current time elapsed) / (time elapsed at the last new fixed point) > term_ratio
    returns:
        V[i][:,p] is the p^th fixed point found after the i^th iterate
        timestamp[i] is the clock time after the i^th iterate
        traversal[i] is t, where the t^th traversal returned the i^th iterate
        c is the random c used by all iterates
        status[i] is the traversal status at the i^th iterate, except 
        status[-1] is 'Term ratio satisfied' if method exits successfully
        VA[t] is the fiber of the t^th traversal
        seed[t] is the seed used for the t^th traversal
        VA_cp[i] is the candidate point in the i^th iterate
        V_rp[i] is the refined point in the i^th iterate
        step_sizes[t] are the step sizes of the t^th traversal
    """
    start_time = time.clock()
    stop_time = start_time + timeout
    # Start with origin component
    V = [np.zeros((W.shape[0],1))]
    VA_cp = [np.zeros((W.shape[0]+1,1))]
    V_rp = [np.zeros((W.shape[0],1))]
    timestamp = [time.clock()]
    traversal = [0]
    status = ['Traversing']
    t = 0
    fiber_component = rfx.directional_fiber(W, c=c, stop_time=stop_time, max_step_size=max_step_size)
    for iterate in fiber_component:
        V_new, _, _, _ = rfx.process_fxpt(W, V[-1], iterate[1])
        V.append(V_new)
        V_rp.append(iterate[1])
        VA_cp.append(iterate[2][-1])
        timestamp.append(time.clock())
        traversal.append(t)
        status.append(iterate[0])
    VA = [iterate[2]]
    step_sizes = [iterate[4]]
    seed = [np.zeros((W.shape[0],1))]
    c = iterate[3] # Same c for subsequent traversals
    if verbose_prefix is not None: print('%scomponent 1...'%verbose_prefix)
    # Do non-origin components with local seeds
    seeds = rfx.local_search(W, stop_time=stop_time)
    num_components = 1
    for seed_status, fxv, _ in seeds:
        # check if timed out
        if seed_status == 'Timed out': break
        # check if term_ratio exceeded
        current_ratio = (time.clock() - start_time)/(timestamp[-1] - start_time)
        if term_ratio is not None and current_ratio > term_ratio:
            # status[-1] = 'Term ratio satisfied'
            break
        # check if not fixed or already found
        _, fx, dup, fxv = rfx.process_fxpt(W, V[-1], fxv)
        if dup or not fx: continue
        # traverse component
        va = np.concatenate((fxv, [[0]]), axis=0)
        t += 1
        seed.append(va)
        fiber_component = rfx.directional_fiber(W, va=va, c=c, stop_time=stop_time, max_step_size=max_step_size)
        for iterate in fiber_component:
            V_new, _, _, _ = rfx.process_fxpt(W, V[-1], iterate[1])
            V_rp.append(iterate[1])
            VA_cp.append(iterate[2][-1])
            V.append(V_new)
            timestamp.append(time.clock())
            traversal.append(t)
            status.append(iterate[0])
        VA.append(iterate[2])
        step_sizes.append(iterate[4])
        num_components += 1
        if verbose_prefix is not None: print('%scomponent %d...'%(verbose_prefix, num_components))
    return V, timestamp, traversal, c, status, VA, seed, VA_cp, V_rp, step_sizes

def mini_compare():

    N = 32
    test_data = fe.generate_test_data(network_sizes=[N], num_samples=[1], refine_iters = 1)
    W = test_data['N_%d_W_0'%N]
    V = test_data['N_%d_V_0'%N]

    timeout = 60*25
    results = {}
    results['local'] = local_trial(W, timeout=timeout)
    results['fiber'] = fiber_trial(W, timeout=timeout)
    results['combo'] = combo_trial(W, timeout=timeout)

    fxpts = (V,) + tuple(results[k][0][-1] for k in results)
    neighbors = lambda X, y: (np.fabs(X-y) < 2**-21).all(axis=0)
    U = rfx.get_unique_points_recursively(np.concatenate(fxpts,axis=1), neighbors=neighbors)

    keys = results.keys()
    for k in keys:
        V_k = results[k][0] # V[i][:,p] = p^th point found after i^th iterate
        T_k = results[k][1] # timestamp[i] = clock time after i^th iterate
        plt.plot([T_ki - T_k[0] for T_ki in T_k], [V_ki.shape[1] for V_ki in V_k])
    plt.plot([0,timeout], 2*[U.shape[1]])
    plt.xlabel('Time elapsed')
    plt.ylabel('# fixed points found')
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

def combo_checks():

    while True:
        N = 2
        
        test_data = fe.generate_test_data(network_sizes=[N], num_samples=[1], refine_iters = 1)
        c = None
        W = test_data['N_%d_W_0'%N]
        
        # dat = fe.load_npz_file('bad_combo.npz')
        # W, c = dat['W'], dat['c']
        
        timeout = 500
        term_ratio = 2
        start = time.clock()
        V, timestamp, traversal, c, status, VA, seed, VA_cp, V_rp, step_sizes = combo_trial(W, c=c, timeout=timeout, term_ratio=term_ratio, max_step_size=0.05)
        end_status = []
        bad_status = False
        bad_t = 0
        bad_i = 0
        for i in range(len(traversal)):
            if i == len(traversal)-1 or traversal[i] != traversal[i+1]:
                print(traversal[i],status[i])
                end_status.append(status[i])
                if (len(end_status) > 1 and end_status[-1] == 'Success') or (len(end_status) == 1 and end_status[-1] == 'Closed loop detected'): # or (len(end_status) > 2 and N == 3):
                    bad_t = traversal[i]
                    bad_i = i
                    bad_status = True
                    if bad_status: break
        print('%d fxpts, %d components, took %f of %f seconds'%(V[-1].shape[1], traversal[-1]+1, time.clock()-start, timeout))
        if bad_status: break
        break # lork check

    if bad_status: fe.save_npz_file('bad_combo.npz', W=W, c=c)

    # Check alpha dot 0 lork:
    # Get each local maximum of |alpha|
    # At that that (v,alpha), check proximity to each hump:
    # i^th "hump" at W_i v = k, where tanh'(k) = W_ii/|W_i|^2.
    # tanh'(k) = 1 - tanh(k)**2
    # k = arctanh((1 - tanh'(k))**.5)
    # *******conclusions so far: often appear correct or nearly so, but even in 2D, sometimes just doesn't look quite right.  local extrema slightly displaced from humps.
    humps = np.flatnonzero(np.diag(W) > 1)
    k = np.arctanh((1 - np.diag(W)/(W**2).sum(axis=1))**.5)[humps,np.newaxis]
    print('k:')
    print(k.flat[:])

    # if N == 3: ax = plt.gca(projection='3d')
    # else: ax = plt.gca()
    ax=plt.gca(projection='3d')
    
    for t in range(len(VA)):
        VA_t = np.concatenate(VA[t],axis=1)
        A = VA_t[N,:]
        extrema = 1 + np.flatnonzero((A[1:-1]-A[:-2])*(A[1:-1]-A[2:]) > 0)
        K = W[humps,:].dot(VA_t[:N,extrema+0])
        d = (np.fabs(np.fabs(K)-k)/(W[humps,:]**2).sum(axis=1)[:,np.newaxis]).min(axis=0)
        K_m = W[humps,:].dot(VA_t[:N,extrema-1])
        d_m = (np.fabs(np.fabs(K_m)-k)/(W[humps,:]**2).sum(axis=1)[:,np.newaxis]).min(axis=0)
        K_p = W[humps,:].dot(VA_t[:N,extrema+1])
        d_p = (np.fabs(np.fabs(K_p)-k)/(W[humps,:]**2).sum(axis=1)[:,np.newaxis]).min(axis=0)
        print('t=%d:'%t)
        print('||K|-k|/|W_i|^2 (-1,0,1):')
        np.set_printoptions(linewidth=1000)
        print(d_m[:5])
        print(d[:5])
        print(d_p[:5])
        ptr.plot(ax,VA_t[:N,:],'k.-')
        ptr.plot(ax,VA_t[:N,extrema],'ro')
        ptr.plot(ax,VA_t[:,:],'k.-')
        ptr.plot(ax,VA_t[:,extrema],'ro')
        w = W[humps,:]/(W[humps,:]**2).sum(axis=1)[:,np.newaxis]
        for h in range(len(humps)):
            owh = np.array([[-w[h,1],w[h,0]]])
            owh = 2 * owh/((owh**2).sum())**.5
            ptr.plot(ax, (+w[[h],:]*k[h] + np.array([[-1],[1]])*owh).T,'g-')
            ptr.plot(ax, (-w[[h],:]*k[h] + np.array([[-1],[1]])*owh).T,'g-')

    ptr.set_lims(ax, 2*np.ones((3,1))*np.array([-1, 1]))
    ax.set_aspect('equal')
    plt.show()
    raw_input('.')


    return # lork check

    # print(V[-1][:,np.argsort(np.fabs(V[-1]).max(axis=0))[:5]])
    plt.ion()
    print(seed[bad_t])
    print('\a') # beep
    if N == 3: ax = plt.gca(projection='3d')
    else: ax = plt.gca()
    for t in range(len(VA)):
        ptr.plot(ax,np.concatenate(VA[t],axis=1)[:N,:],'ko-') 
    # ptr.plot(ax,np.concatenate(VA[0],axis=1)[:N,:],'ko-')
    # # ptr.plot(ax,np.concatenate(VA[bad_t],axis=1)[:N,:],'go-')
    # # ptr.plot(ax,np.concatenate(VA[1],axis=1)[:N,:],'go-')
    # # ptr.plot(ax,np.concatenate(VA[2],axis=1)[:N,:],'mo-')
    for p in range(len(VA_cp)):
        ptr.plot(ax, np.concatenate((VA_cp[p][:N,:],V_rp[p]),axis=1),'-b')
        ptr.plot(ax,V_rp[p],'b.')
    ptr.plot(ax,V[-1],'r.')
    ptr.set_lims(ax,3*np.ones((N,1))*np.array([-1,1]))
    plt.show()
    plt.figure()
    bad_t = 0
    plt.plot(np.array(step_sizes[bad_t]).cumsum(), np.concatenate(VA[bad_t],axis=1)[N,:],'-b.')
    plt.plot(np.array(step_sizes[bad_t]).cumsum(), np.zeros(len(step_sizes[bad_t])))
    plt.show()
    raw_input('.')

def single_c_search():

    N = 5
    samples = []
    while True:
        
        test_data = fe.generate_test_data(network_sizes=[N], num_samples=[1], refine_iters = 1)
        W = test_data['N_%d_W_0'%N]
        timeout = 60*60
        term_ratio = 2

        num_c = 0
        num_fx = None
        while True:
        
            c = None
            start = time.clock()
            V, timestamp, traversal, c, status, VA, seed, VA_cp, V_rp, step_sizes = combo_trial(W, c=c, timeout=timeout, term_ratio=term_ratio, max_step_size=0.05, verbose_prefix = '  ')
            num_c += 1
            
            
            print(' %d^th c: %d fxpts, %d components, took %f of %f seconds'%(num_c, V[-1].shape[1], traversal[-1]+1, time.clock()-start, timeout))
            
            if num_fx is None: num_fx = V[-1].shape[1]
            if not V[-1].shape[1] == num_fx:
                print('\a')
                raw_input('oh no...')

            if len(VA) == 1: break
            
        print('Found fully connected after %d c\'s'%num_c)
        
if __name__=='__main__':
    single_c_search()
