"""
Large-scale experiments that evaluate rnn-fxpts on many randomly sampled networks
File names provided to methods in this module should follow these naming conventions:
  <test data id>: base name for a set of test networks
  traverse_<test data id>_N_<N>_s_<s>: results for traverse on the s^{th} network of size N
  baseline_<test data id>_N_<N>_s_<s>: results for traverse on the s^{th} network of size N
  TvB_<test data id>_N_<N>_s_<s>: results of traverse-baseline comparison on the s^{th} network of size N

  <test data id>_Wc_N_<N>_s_<s>: results of c-choice comparison on the s^{th} network of size N

  traverse_re_<test data id>_N_<N>_s_<s>: relative errors for round-off in traverse on the s^{th} network of size N
  traverse_rd_<test data id>_N_<N>_s_<s>: relative distances for round-off in traverse on the s^{th} network of size N
  baseline_re_<test data id>_N_<N>_s_<s>: relative errors for round-off in baseline on the s^{th} network of size N
  baseline_rd_<test data id>_N_<N>_s_<s>: relative distances for round-off in baseline on the s^{th} network of size N
"""
import os
import sys
import time
import multiprocessing as mp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.markers as mrk
import plotter as ptr
import rnn_fxpts as rfx
import pickle as pkl

def generate_test_data(network_sizes, num_samples, test_data_id=None, refine_iters=2**5,refine_cap=10000):
    """
    Randomly sample networks for testing with some fixed points by construction
    network_sizes[i] should be the i^{th} network size to include in the test data
    num_samples[i] should be the number of networks to generate with size network_sizes[i]
    test_data_id should be a file name with which to save the test data
      test data is saved as a numpy .npz archive
      if None, no file is saved
    returns test_data, a dictionary with keys
      "network_sizes": the list of network sizes (as a flat numpy.array)
      "num_samples": the list of sample counts at each network size (as a flat numpy.array)
      "N_%d_W_%d"%(N,s): the s^{th} weight matrix sampled at network size N (N by N numpy.array)
      "N_%d_V_%d"%(N,s): the corresponding known fixed points (N by N numpy.array), where
         test_data["N_%d_V_%d"%(N,s)][:,p] is the p^{th} known fixed point
    """
    if test_data_id is not None and os.path.exists(test_data_id):
        regen = raw_input('Test data exists, regenerate? (y/n): ')
        # regen = 'n'
        if regen == 'n':
            return None
    test_data = {"network_sizes": np.array(network_sizes), "num_samples": np.array([num_samples])}
    for (N, S) in zip(network_sizes, num_samples):
        for s in range(S):
            # Random V
            V = 2*np.random.rand(N,N) - 1
            # Construct W
            W = rfx.mrdivide(np.arctanh(V), V)
            # Refine V
            # V, _ = rfx.refine_fxpts(W, V) # too slow on big experiments
            V, _ = rfx.refine_fxpts_capped(W, V, max_iters=refine_iters, cap=refine_cap)
            # Store
            test_data["N_%d_W_%d"%(N,s)] = W
            test_data["N_%d_V_%d"%(N,s)] = V
    # Save test data to file
    if test_data_id is not None:
        np.savez(test_data_id, **test_data)
    return test_data

def generate_scarce_test_data(network_sizes, num_samples, numfx=8, test_data_id=None):
    """
    Randomly sample networks for testing with some fixed points by construction.
    Like generate_test_data, but the networks will tend to have fewer fixed points.
    network_sizes, num_samples, and test_data_id should be as in generate_test_data.
    numfx should be the number of known fixed points to include in the construction.
    The smaller numfx is, the fewer total fixed points the networks are expected to have.
    returns test_data, with same format as in generate_test_data.
    """
    # exposed an issue: W must have no eigenvalues = 1 (else DF low rank at origin)
    test_data = {"network_sizes": np.array(network_sizes), "num_samples": np.array([num_samples])}
    for N in network_sizes:
        for s in range(num_samples):
            # Random V
            V = 2*np.random.rand(N,numfx) - 1
            # Construct W
            A = np.linalg.svd(np.arctanh(V))[0][:,numfx:].dot(np.random.randn(N-numfx,N-numfx)/N)
            B = (np.random.randn(N-numfx,N-numfx)/N).dot(np.linalg.svd(V)[0][:,numfx:].T)
            X = np.concatenate((np.arctanh(V),A),axis=1)
            Y = np.concatenate((rfx.mrdivide(np.eye(numfx),V),B),axis=0)
            W = X.dot(Y)
            # Refine V
            V, _ = rfx.refine_fxpts(W, V)
            # Store
            test_data["N_%d_W_%d"%(N,s)] = W
            test_data["N_%d_V_%d"%(N,s)] = V
    # Save test data to file
    if test_data_id is not None:
        np.savez(test_data_id, **test_data)
    return test_data

def load_test_data(filename):
    """
    Load test data that was generated and saved by generate_test_data.
    filename should be the file name where the data was saved.
    returns test_data, with the same format as in generate_test_data.
    """
    f = open(filename,"r")
    test_data = np.load(f)
    test_data = {k: test_data[k] for k in test_data.files}
    f.close()
    network_sizes = test_data.pop("network_sizes")
    num_samples = test_data.pop("num_samples")[0]
    return network_sizes, num_samples, test_data

def save_pkl_file(filename, data):
    """
    Convenience function for pickling data to a file
    """
    pkl_file = open(filename,'w')
    pkl.dump(data, pkl_file)
    pkl_file.close()
def load_pkl_file(filename):
    """
    Convenience function for loading pickled data from a file
    """
    pkl_file = open(filename,'r')
    data = pkl.load(pkl_file)
    pkl_file.close()
    return data
def save_npz_file(filename, **kwargs):
    """
    Convenience function for saving numpy data to a file
    Each kwarg should have the form
      array_name=array
    """
    npz_file = open(filename,"w")
    np.savez(npz_file, **kwargs)
    npz_file.close()
def load_npz_file(filename):
    """
    Convenience function for loading numpy data from a file
    returns npz, a dictionary with key-value pairs of the form
      array_name: array
    """    
    npz = np.load(filename)
    npz = {k:npz[k] for k in npz.files}
    return npz

def test_traverse(W, V, c=None, result_key=None, logfilename=os.devnull, save_result=False, save_npz=False, max_traverse_steps=2**20,max_fxpts=None):
    """
    Test the traverse algorithm on a single test network.
    W should be the weight matrix (N by N numpy.array)
    V should be the known fixed points (N by K numpy.array)
    c should be the direction vector (N by 1 numpy.array)
      if None, a random c is automatically chosen
    result_key is a unique string identifier for the test
    logfilename is a file name at which to write progress updates
    if save_result == True, results are saved in a file with name based on result_key
    if save_npz == True, traverse numpy outputs are saved in a file with name based on result_key
    max_traverse_steps is number of steps allowed for traverse algorithm
    max_fxpts is number of fxpts after which traverse can terminate
    returns results, npz, where
      results is a dictionary summarizing the test results
      npz is a dictionary with full numpy output from traverse
    """
    N = W.shape[0]
    logfile = open(logfilename,'w')

    # run traversal
    rfx.hardwrite(logfile,'Running traversal: %s...\n'%result_key)
    start = time.clock()
    status, fxV, VA, c, step_sizes, s_mins, residuals = rfx.traverse(W, c=c, max_traverse_steps = max_traverse_steps, max_fxpts=max_fxpts,logfile=logfile)
    runtime = time.clock()-start
    num_steps = VA.shape[1]

    results = {
        "result_key": result_key,
        "status": status,
        "runtime": runtime,
        "N": W.shape[0],
        "num_steps": num_steps,
        "path_length": step_sizes.sum(),
        "min_s_min": s_mins.min(),
        "num_fxV": fxV.shape[1]
    }
    npz = {"W":W, "V":V, "VA":VA, "fxV":fxV, "c":c, "residuals":residuals}
    if save_result: save_pkl_file('results/%s.pkl'%result_key, results)
    if save_npz: save_npz_file('results/%s.npz'%result_key, **npz)

    # Post-process
    # count unique fixed points found
    start = time.clock()
    fxV_unique, fxV_converged = rfx.post_process_fxpts(W, fxV, logfile=logfile)
    post_runtime = time.clock()-start
    results['post_runtime'] = post_runtime
    results['num_fxV_unique'] = fxV_unique.shape[1]
    npz["fxV_unique"] = fxV_unique
    npz["fxV_converged"] = fxV_converged
    if save_result: save_pkl_file('results/%s.pkl'%result_key, results)
    if save_npz: save_npz_file('results/%s.npz'%result_key, **npz)

    # check for ground truth inclusion
    rfx.hardwrite(logfile,'Checking ground truths...\n')
    V_found = np.zeros(N, dtype=bool)
    Winv = np.linalg.inv(W)
    if fxV_converged.shape[1] > V.shape[1]:
        for j in range(V.shape[1]):
            identical, _, _ = rfx.identical_fixed_points(W, fxV_converged, V[:,[j]], Winv)
            V_found[j] = identical.any()
    else:
        for j in range(fxV_converged.shape[1]):
            identical, _, _ = rfx.identical_fixed_points(W, V, fxV_converged[:,[j]], Winv)
            V_found |= identical
    results["num_V_found"] = V_found.sum(),
    npz["V_found"] = V_found
    if save_result: save_pkl_file('results/%s.pkl'%result_key, results)
    if save_npz: save_npz_file('results/%s.npz'%result_key, **npz)

    finish_str = "pid %d: %s, %d fxV (%d unique), %d of %d gt, %d iters (length ~ %f, step_size ~ %f).  restarting..."%(os.getpid(), result_key, fxV.shape[1], fxV_unique.shape[1], V_found.sum(), N, num_steps, step_sizes.sum(), step_sizes.mean())
    rfx.hardwrite(logfile,'%s\n'%finish_str)
    print(finish_str)

    logfile.close()
    return results, npz

def pool_test_traverse(args):
    """
    Wrapper function passed to multiprocessing.Pool
    """
    results, _ = test_traverse(*args)
    return results

def run_traverse_experiments(test_data_id, num_procs, max_traverse_steps=2**20,max_fxpts=None):
    """
    Run test_traverse on every network in the test data
    Uses multi-processing to test on multiple networks in parallel
    test_data_id should be as in generate_test_data (without file extension)
    num_procs is the number of processors to use in parallel
    max_traverse_steps is number of steps allowed for traverse algorithm
    max_fxpts is number of fxpts after which traverse can terminate
    returns pool_results, a list of results with one entry per network
    """

    network_sizes, num_samples, test_data = load_test_data('%s.npz'%test_data_id)

    cpu_count = mp.cpu_count()
    print('%d cpus, using %d'%(cpu_count, num_procs))

    pool_args = []
    for (N, S) in zip(network_sizes, num_samples):
        for s in range(S):
            W = test_data['N_%d_W_%d'%(N,s)]
            V = test_data['N_%d_V_%d'%(N,s)]
            c = None
            result_key = 'traverse_%s_N_%d_s_%d'%(test_data_id, N, s)
            if num_procs > 0:
                logfilename = 'logs/%s.log'%result_key
                save_result=True
                save_npz=True
            else:
                logfilename = 'logs/temp.txt'
                save_result=False
                save_npz=False
            pool_args.append((W,V,c,result_key,logfilename,save_result,save_npz,max_traverse_steps,max_fxpts))
    start_time = time.time()
    test_fun = pool_test_traverse
    if num_procs < 1: # don't multiprocess
        pool_results = [test_fun(args) for args in pool_args]
    else:
        pool = mp.Pool(processes=num_procs)
        pool_results = pool.map(test_fun, pool_args)
        pool.close()
        pool.join()
    print('total time: %f. saving results...'%(time.time()-start_time))

    results_file = open('results/traverse_%s.pkl'%test_data_id, 'w')
    pkl.dump(pool_results, results_file)
    results_file.close()

    return pool_results

def test_Wc(W, V, result_key=None, logfilename=os.devnull, save_result=False):
    """
    Test traverse with different c choices on a single test network.
    One choice is tested for each of the 2^N possible values of numpy.sign(W.dot(c)).
    W should be the weight matrix (N by N numpy.array)
    V should be the known fixed points (N by K numpy.array)
    result_key is a unique string identifier for the test
    logfilename is a file name at which to write progress updates
    if save_result == True, results are saved in a file with name based on result_key
    returns results, a list where
      results[i] is a dictionary summarizing the test results for the i^{th} choice of c
    """
    N = W.shape[0]
    logfile = open(logfilename,'w')
    rfx.hardwrite(logfile,'Running Wc: %s...\n'%result_key)

    signs = ptr.lattice(-np.ones((N,1)),np.ones((N,1)),2)
    C = rfx.solve(W, signs + 0.1*(np.random.rand(*signs.shape)-0.5))
    all_fxV = []
    results = []
    for s in range(signs.shape[1]):

        # run traversal
        rfx.hardwrite(logfile,'Running traversal %d...\n'%s)
        start = time.clock()
        status, fxV, VA, c, step_sizes, s_mins, residuals = rfx.traverse(W, c=C[:,[s]], max_traverse_steps = 2**20, logfile=logfile)
        runtime = time.clock()-start
        num_steps = VA.shape[1]

        # count unique fixed points found
        fxV_unique, fxV_converged = rfx.post_process_fxpts(W, fxV, logfile=logfile)

        all_fxV.append(fxV_unique)
        result = {
            "result_key": result_key,
            "status": status,
            "runtime": runtime,
            "N": W.shape[0],
            "num_steps": num_steps,
            "path_length": step_sizes.sum(),
            "min_s_min": s_mins.min(),
            "num_fxV_unique": fxV_unique.shape[1],
        }

        # check for ground truth inclusion
        rfx.hardwrite(logfile,'Checking ground truths...\n')
        V_found = np.zeros(N, dtype=bool)
        for j in range(V.shape[1]):
            identical, _, _ = rfx.identical_fixed_points(W, fxV_unique, V[:,[j]])
            V_found[j] = identical.any()
        result["num_V_found"] = V_found.sum(),

        results.append(result)

    # union of known fxV
    V = np.concatenate((-V, np.zeros((N,1)), V), axis=1)
    fxV_union = np.concatenate(all_fxV + [V], axis=1)
    rfx.hardwrite(logfile,'post-processing union...\n')
    fxV_union, _ = rfx.post_process_fxpts(W, fxV_union, logfile=logfile)
    for s in range(signs.shape[1]):
        results[s]["num_fxV_union"] = fxV_union.shape[1]

    # return results
    if save_result: save_pkl_file('results/%s.pkl'%result_key, results)
    rfx.hardwrite(logfile,'%s\n'%str([r["num_fxV_unique"] for r in results]))
    best = max([r["num_fxV_unique"] for r in results])
    finish_str = "pid %d: %s, best=%d fxV of %d union.  restarting..."%(os.getpid(), result_key, best, fxV_union.shape[1])
    rfx.hardwrite(logfile,'%s\n'%finish_str)
    print(finish_str)

    return results

def pool_test_Wc(args):
    """
    Wrapper function passed to multiprocessing.Pool
    """
    return test_Wc(*args)

def run_Wc_experiments(test_data_id, num_procs):
    """
    Run test_Wc on every network in the test data
    Uses multi-processing to test on multiple networks in parallel
    test_data_id should be as in generate_test_data (without file extension)
    num_procs is the number of processors to use in parallel
    returns pool_results, a list of results with one entry per network
    """

    network_sizes, num_samples, test_data = load_test_data('%s.npz'%test_data_id)

    cpu_count = mp.cpu_count()
    print('%d cpus, using %d'%(cpu_count, num_procs))

    pool_args = []
    for (N, S) in zip(network_sizes, num_samples):
        for s in range(S):
            W = test_data['N_%d_W_%d'%(N,s)]
            V = test_data['N_%d_V_%d'%(N,s)]
            result_key = '%s_Wc_N_%d_s_%d'%(test_data_id, N, s)
            logfilename =  'logs/%s.log'%result_key
            save_result=True
            pool_args.append((W,V,result_key,logfilename,save_result))
    start_time = time.time()
    test_fun = pool_test_Wc
    if num_procs < 1: # don't multiprocess
        pool_results = [test_fun(args) for args in pool_args]
    else:
        pool = mp.Pool(processes=num_procs)
        pool_results = pool.map(test_fun, pool_args)
        pool.close()
        pool.join()
    print('total time: %f. saving results...'%(time.time()-start_time))

    results_file = open('results/%s_Wc.pkl'%test_data_id, 'w')
    pkl.dump(pool_results, results_file)
    results_file.close()

    return pool_results

def test_baseline(W, V, timeout=60, result_key=None, logfilename=os.devnull, save_result=False, save_npz=False):
    """
    Test the baseline solver on a single test network.
    W should be the weight matrix (N by N numpy.array)
    V should be the known fixed points (N by K numpy.array)
    timeout is the number of seconds before the solver is terminated
    result_key is a unique string identifier for the test
    logfilename is a file name at which to write progress updates
    if save_result == True, results are saved in a file with name based on result_key
    if save_npz == True, solver numpy outputs are saved in a file with name based on result_key
    returns results, npz, where
      results is a dictionary summarizing the test results
      npz is a dictionary with full numpy output from the solver
    """
    N = W.shape[0]
    logfile = open(logfilename,'w')

    # run baseline
    start = time.clock()
    fxV, num_reps = rfx.baseline_solver(W, timeout=timeout, logfile=logfile)
    runtime = time.clock()-start
    results = {
        "result_key": result_key,
        "runtime": runtime,
        "N": W.shape[0],
        "num_reps": num_reps,
        "num_fxV": fxV.shape[1],
    }

    npz = {"W":W, "V":V, "fxV":fxV}
    if save_result: save_pkl_file('results/%s.pkl'%result_key, results)
    if save_npz: save_npz_file('results/%s.npz'%result_key, **npz)

    rfx.hardwrite(logfile,"Post-processing...\n")
    start = time.clock()
    fxV_unique, fxV_converged = rfx.post_process_fxpts(W, fxV, logfile=logfile)
    post_runtime = time.clock()-start
    results["post_runtime"] = post_runtime
    results["num_fxV_unique"] = fxV_unique.shape[1]
    npz["fxV_unique"] = fxV_unique
    npz["fxV_converged"] = fxV_converged
    if save_result: save_pkl_file('results/%s.pkl'%result_key, results)
    if save_npz: save_npz_file('results/%s.npz'%result_key, **npz)

    # check for ground truth inclusion
    rfx.hardwrite(logfile,'checking ground truths...\n')
    V_found = np.zeros(N, dtype=bool)
    for j in range(V.shape[1]):
        identical, _, _ = rfx.identical_fixed_points(W, fxV_converged, V[:,[j]])
        V_found[j] = identical.any()
    results["num_V_found"] = V_found.sum()
    npz["V_found"] = V_found
    if save_result: save_pkl_file('results/%s.pkl'%result_key, results)
    if save_npz: save_npz_file('results/%s.npz'%result_key, **npz)

    finish_str = "pid %d: %s, %d fxV (%d unique), %d of %d gt, %d reps.  restarting..."%(os.getpid(), result_key, fxV_converged.shape[1], fxV_unique.shape[1], V_found.sum(), N, num_reps)
    rfx.hardwrite(logfile,"%s\n"%finish_str)
    print(finish_str)

    logfile.close()
    return results, npz

def pool_test_baseline(args):
    """
    Wrapper function passed to multiprocessing.Pool
    """
    results, _ = test_baseline(*args)
    return results

def run_baseline_experiments(test_data_id, num_procs):
    """
    Run test_baseline on every network in the test data
    Uses multi-processing to test on multiple networks in parallel
    test_data_id should be as in generate_test_data (without file extension)
    num_procs is the number of processors to use in parallel
    returns pool_results, a list of results with one entry per network
    """

    network_sizes, num_samples, test_data = load_test_data('%s.npz'%test_data_id)

    cpu_count = mp.cpu_count()
    print('%d cpus, using %d'%(cpu_count, num_procs))

    pool_args = []
    for (N, S) in zip(network_sizes, num_samples):
        for s in range(S):
            W = test_data['N_%d_W_%d'%(N,s)]
            V = test_data['N_%d_V_%d'%(N,s)]
            traverse_result_key = 'traverse_%s_N_%d_s_%d'%(test_data_id, N, s)
            traverse_results_file = open('results/%s.pkl'%traverse_result_key, 'r')
            traverse_results = pkl.load(traverse_results_file)
            traverse_results_file.close()
            timeout = traverse_results['runtime']
            result_key = 'baseline_%s_N_%d_s_%d'%(test_data_id, N, s)
            logfilename = 'logs/baseline_%s_N_%d_s_%d.log'%(test_data_id, N, s)
            save_result=True
            save_npz=True
            pool_args.append((W,V,timeout,result_key,logfilename,save_result,save_npz))
    start_time = time.time()
    test_fun = pool_test_baseline
    if num_procs < 1: # don't multiprocess
        pool_results = [test_fun(args) for args in pool_args]
    else:
        pool = mp.Pool(processes=num_procs)
        pool_results = pool.map(test_fun, pool_args)
        pool.close()
        pool.join()
    print('total time: %f. saving results...'%(time.time()-start_time))

    results_file = open('results/baseline_%s.pkl'%test_data_id, 'w')
    pkl.dump(pool_results, results_file)
    results_file.close()

    return pool_results

def test_TvB(test_data_id, N, s, logfilename=os.devnull, save_result=False, save_npz=False):
    """
    Compare the traverse and baseline results on a single test network.
    test_data_id should be as in generate_test_data (without file extension)
    Inspects the s^{th} network of size N
    logfilename is a file name at which to write progress updates
    if save_result == True, results are saved in a file with name based on test_data_id
    if save_npz == True, numpy outputs are saved in a file with name based on test_data_id
    returns results, npz, where
      results is a dictionary summarizing the test results
      npz is a dictionary with full numpy output
    """
    logfile = open(logfilename,'w')

    rfx.hardwrite(logfile,'Loading results...\n')
    _,_,test_data = load_test_data('%s.npz'%test_data_id)
    W = test_data['N_%d_W_%d'%(N,s)]
    baseline_npz = np.load('results/baseline_%s_N_%d_s_%d.npz'%(test_data_id, N, s))
    traverse_npz = np.load('results/traverse_%s_N_%d_s_%d.npz'%(test_data_id, N, s))
    fxV_baseline = baseline_npz["fxV_unique"]
    fxV_traverse = traverse_npz["fxV_unique"]
    #### !!! temp simple unique test
    neighbors = lambda X, y: (np.fabs(X-y) < 10**-10).all(axis=0)
    fxV_baseline = rfx.get_unique_points_recursively(fxV_baseline, neighbors=neighbors)
    fxV_traverse = rfx.get_unique_points_recursively(fxV_traverse, neighbors=neighbors)

    T = fxV_traverse.shape[1]
    B = fxV_baseline.shape[1]

    result_key = 'TvB_%s_N_%d_s_%d'%(test_data_id, N, s)
    results = {
        'result_key': result_key,
        'N':N,
        'T':T,
        'B':B,
    }
    npz = {"W":W, "fxV_baseline":fxV_baseline,"fxV_traverse":fxV_traverse}
    if save_result: save_pkl_file('results/%s.pkl'%result_key, results)
    if save_npz: save_npz_file('results/%s.npz'%result_key, **npz)

    # Get union
    rfx.hardwrite(logfile,'unioning %d + %d...\n'%(T, B))
    fxV_union = np.concatenate((fxV_traverse, fxV_baseline), axis=1)
    # neighbors = lambda X, y: rfx.identical_fixed_points(W, X, y)[0]
    fxV_union = rfx.get_unique_points_recursively(fxV_union, neighbors=neighbors)
    TB = fxV_union.shape[1]
    finish_str = 'N:%d,T:%d, B:%d, T|B:%d, T&B:%d, T-B:%d(%f), B-T:%d(%f)'%(N,T,B,TB,T+B-TB,TB-B,1.*(TB-B)/TB,TB-T,1.*(TB-T)/TB)
    rfx.hardwrite(logfile,'%s\n'%finish_str)
    print(finish_str)

    results['T|B']=TB
    results['T&B']=T+B-TB
    results['T-B']=TB-B
    results['B-T']=TB-T
    npz['fxV_union'] = fxV_union
    if save_result: save_pkl_file('results/%s.pkl'%result_key, results)
    if save_npz: save_npz_file('results/%s.npz'%result_key, **npz)

    # distances around means
    baseline_mean = fxV_baseline.mean(axis=1)
    traverse_mean = fxV_traverse.mean(axis=1)
    baseline_dist = np.mean(np.sqrt(((fxV_baseline-baseline_mean[:,np.newaxis])**2).sum(axis=0)))
    traverse_dist = np.mean(np.sqrt(((fxV_traverse-traverse_mean[:,np.newaxis])**2).sum(axis=0)))
    results['baseline_dist'] = baseline_dist
    results['traverse_dist'] = traverse_dist
    if save_result: save_pkl_file('results/%s.pkl'%result_key, results)

    # vertex proximities:
    fxV_baseline_dist_v = np.sqrt(((fxV_baseline - np.sign(fxV_baseline))**2).sum(axis=0))
    fxV_traverse_dist_v = np.sqrt(((fxV_traverse - np.sign(fxV_traverse))**2).sum(axis=0))
    results['baseline_dist_v'] = fxV_baseline_dist_v.mean()
    results['traverse_dist_v'] = fxV_traverse_dist_v.mean()
    if save_result: save_pkl_file('results/%s.pkl'%result_key, results)

    logfile.close()
    return results, npz

def pool_test_TvB(args):
    """
    Wrapper function passed to multiprocessing.Pool
    """
    results, _ = test_TvB(*args)
    return results

def run_TvB_experiments(test_data_id, num_procs):
    """
    Run test_TvB on every network in the test data
    Uses multi-processing to test on multiple networks in parallel
    test_data_id should be as in generate_test_data (without file extension)
    num_procs is the number of processors to use in parallel
    returns pool_results, a list of results with one entry per network
    """

    cpu_count = mp.cpu_count()
    print('%d cpus, using %d'%(cpu_count, num_procs))

    pool_args = []
    network_sizes, num_samples, test_data = load_test_data('%s.npz'%test_data_id)
    for (N,S) in zip(network_sizes, num_samples):
        for s in range(S):
            logfilename = 'logs/tvb_%s_N_%d_s_%d.log'%(test_data_id, N, s)
            save_result=True
            save_npz=True
            pool_args.append((test_data_id, N, s, logfilename, save_result, save_npz))
    start_time = time.time()
    test_fun = pool_test_TvB
    if num_procs < 1: # don't multiprocess
        pool_results = [test_fun(args) for args in pool_args]
    else:
        pool = mp.Pool(processes=num_procs)
        pool_results = pool.map(test_fun, pool_args)
        pool.close()
        pool.join()
    print('total time: %f. saving results...'%(time.time()-start_time))

    results_file = open('results/tvb_%s.pkl'%test_data_id, 'w')
    pkl.dump(pool_results, results_file)
    results_file.close()

    return pool_results

def test_TvB_stability(test_data_id, N, s, logfilename=os.devnull, save_result=False, save_npz=False):
    """
    Compare the stability at traverse and baseline results on a single test network.
    test_data_id should be as in generate_test_data (without file extension)
    Inspects the s^{th} network of size N
    logfilename is a file name at which to write progress updates
    if save_result == True, results are saved in a file with name based on test_data_id
    if save_npz == True, numpy outputs are saved in a file with name based on test_data_id
    returns results, npz, where
      results is a dictionary summarizing the test results
      npz is a dictionary with full numpy output
    """
    logfile = open(logfilename,'w')

    rfx.hardwrite(logfile,'Loading results...\n')
    # _,_,test_data = load_test_data('%s.npz'%test_data_id)
    # W = test_data['N_%d_W_%d'%(N,s)]
    baseline_npz = np.load('results/baseline_%s_N_%d_s_%d.npz'%(test_data_id, N, s))
    traverse_npz = np.load('results/traverse_%s_N_%d_s_%d.npz'%(test_data_id, N, s))
    fxV_baseline = baseline_npz["fxV_unique"]
    fxV_traverse = traverse_npz["fxV_unique"]
    fxVs = {'baseline':fxV_baseline,'traverse':fxV_traverse}
    W = traverse_npz['W']

    result_key = 'TvB_stable_%s_N_%d_s_%d'%(test_data_id, N, s)
    results = {
        'result_key': result_key,
        'N':N,
        'T':fxV_traverse.shape[1],
        'B':fxV_baseline.shape[1],
    }
    npz = {}
    if save_result: save_pkl_file('results/%s.pkl'%result_key, results)
    if save_npz: save_npz_file('results/%s.npz'%result_key, **npz)

    # Stability analysis
    rfx.hardwrite(logfile,'linearizing and checking stability...\n')
    norms, num_big_eigs, max_eigs, num_stable, avg_num_big, min_num_big = {}, {}, {}, {}, {}, {}
    I = np.eye(N)
    for method_key in ['baseline','traverse']:
        fxV = fxVs[method_key]
        norms[method_key] = np.sqrt((fxV**2).sum(axis=0))
        num_big_eigs[method_key] = np.empty(fxV.shape[1])
        max_eigs[method_key] = np.empty(fxV.shape[1])
        for j in range(fxV.shape[1]):
            rfx.hardwrite(logfile,'method %s: j = %d of %d (%d stable so far)\n'%(method_key,j,fxV.shape[1],(num_big_eigs[method_key][:j] == 0).sum()))
            # linearize
            Df = (1-np.tanh(W.dot(fxV[:,[j]]))**2)*W - I
            eigs, _ = np.linalg.eig(Df)
            max_eigs[method_key][j] = np.absolute(eigs).max()
            num_big_eigs[method_key][j] = (np.absolute(eigs) >= 1).sum()
        avg_num_big[method_key] = num_big_eigs[method_key].astype(float).mean()
        min_num_big[method_key] = num_big_eigs[method_key].astype(float).min()
        num_stable[method_key] = (num_big_eigs[method_key] == 0).sum()
        npz['norms_%s'%method_key] = norms[method_key]
        npz['num_big_eigs_%s'%method_key] = num_big_eigs[method_key]
        npz['max_eigs_%s'%method_key] = max_eigs[method_key]
        results['avg_num_big_%s'%method_key] = avg_num_big[method_key]
        results['min_num_big_%s'%method_key] = min_num_big[method_key]
        results['num_stable_%s'%method_key] = num_stable[method_key]
    finish_str = 'N:%d,T:%d of %d stable (~%f big),B:%d of %d stable (~%f big)'%(N,num_stable['traverse'],results['T'],avg_num_big['traverse'],num_stable['baseline'],results['B'],avg_num_big['baseline'])
    rfx.hardwrite(logfile,'%s\n'%finish_str)
    print(finish_str)

    if save_result: save_pkl_file('results/%s.pkl'%result_key, results)
    if save_npz: save_npz_file('results/%s.npz'%result_key, **npz)

    logfile.close()
    return results, npz

def pool_test_TvB_stability(args):
    """
    Wrapper function passed to multiprocessing.Pool
    """
    results, _ = test_TvB_stability(*args)
    return results

def run_TvB_stability_experiments(test_data_id, num_procs):
    """
    Run test_TvB_stability on every network in the test data
    Uses multi-processing to test on multiple networks in parallel
    test_data_id should be as in generate_test_data (without file extension)
    num_procs is the number of processors to use in parallel
    returns pool_results, a list of results with one entry per network
    """

    cpu_count = mp.cpu_count()
    print('%d cpus, using %d'%(cpu_count, num_procs))

    pool_args = []
    network_sizes, num_samples, test_data = load_test_data('%s.npz'%test_data_id)
    for (N,S) in zip(network_sizes, num_samples):
        for s in range(S):
            logfilename = 'logs/tvb_stab_%s_N_%d_s_%d.log'%(test_data_id, N, s)
            save_result=True
            save_npz=True
            pool_args.append((test_data_id, N, s, logfilename, save_result, save_npz))
    start_time = time.time()
    test_fun = pool_test_TvB_stability
    if num_procs < 1: # don't multiprocess
        pool_results = [test_fun(args) for args in pool_args]
    else:
        pool = mp.Pool(processes=num_procs)
        pool_results = pool.map(test_fun, pool_args)
        pool.close()
        pool.join()
    print('total time: %f. saving results...'%(time.time()-start_time))

    results_file = open('results/tvb_stab_%s.pkl'%test_data_id, 'w')
    pkl.dump(pool_results, results_file)
    results_file.close()

def show_tvb_results(test_data_ids=['dl50','dm10','dh5']):
    """
    Plot the results of traverse-baseline performance comparison on one or more testing data sets
    test_data_ids should be the list of ids, each as in generate_test_data (without file extension)
    """
    results = []
    for test_data_id in test_data_ids:
        results += load_pkl_file('results/tvb_%s.pkl'%test_data_id)
        # # temp because accidentally overwrote
        # curr_results = []
        # network_sizes, num_samples, test_data = load_test_data('%s.npz'%test_data_id)
        # for (N,S) in zip(network_sizes, num_samples):
        #     for s in range(S):
        #         print(N,s)
        #         result_key = 'TvB_%s_N_%d_s_%d'%(test_data_id, N, s)
        #         result = load_pkl_file('results/%s.pkl'%result_key)
        #         curr_results.append(result)
        # results += curr_results
        # save_pkl_file('results/tvb_%s.pkl'%test_data_id, curr_results)
    results = [r for r in results if r['N'] in [2,4,7,10,13,16,24,32,48,64,128,256,512,1024]]
    mpl.rcParams['mathtext.default'] = 'regular'
    mpl.rcParams.update({'figure.autolayout': True})
    mpl.rcParams.update({'font.size': 12})
    Ns = np.array([r['N'] for r in results])
    uNs = np.unique(Ns)
    dats = [('T|B','v','k'),('T&B','^','none'),('T-B','s','none'),('B-T','o','k')]
    handles = []
    plt.figure(figsize=(9,4))
    for ym in dats:
        y = [r[ym[0]] if r[ym[0]]>0 else 2**-1 for r in results]
        handles.append(scatter_with_errors(Ns, uNs, y, ym[1],ym[2],log=True,logmin=2**-1))
        # y = [r[ym[0]] for r in results]
        # handles.append(scatter_with_errors(Ns, uNs, y, ym[1],ym[2]))
    handles.append(plt.plot(uNs, np.log2(uNs), 'dk--', ms=9)[0])
    # handles.append(plt.plot(uNs, uNs, 'dk--', ms=9)[0])
    plt.legend(handles, ['$T\cup\,B$', '$T\cap\,B$', '$T-B$', '$B-T$', 'Known'], loc='upper left')
    # plt.ylim([-1,15])
    # plt.gca().set_yscale('log',basey=2)
    plt.xlim([2**.5,1.5*uNs[-1]])
    plt.gca().set_xscale('log',basex=2)
    plt.ylabel('# of fixed points')
    #plt.title('Traverse vs Baseline')
    # plt.draw()
    # ytick_labels = plt.gca().get_yticklabels()
    # plt.gca().set_yticklabels(['2^%s'%(yl.get_text()) for yl in ytick_labels])
    plt.yticks(range(-1,15,2),['0']+['$2^{%d}$'%yl for yl in range(1,15,2)])
    plt.ylim([-2,15])
    plt.tight_layout()
    plt.show()

def show_tvb_dist_results(test_data_ids=['dl50','dm10','dh5']):
    """
    Plot the results of traverse-baseline spatial distribution comparison on one or more testing data sets
    test_data_ids should be the list of ids, each as in generate_test_data (without file extension)
    """
    results = []
    for test_data_id in test_data_ids:
        # results += load_pkl_file('results/tvb_dist_%s.pkl'%test_data_id)
        results += load_pkl_file('results/tvb_%s.pkl'%test_data_id)
    results = [r for r in results if r['N'] in [2,4,7,10,13,16,24,32,48,64,128,256,512,1024]]
    mpl.rcParams['mathtext.default'] = 'regular'
    # mpl.rcParams.update({'figure.autolayout': True})
    mpl.rcParams.update({'font.size': 12})
    Ns = np.array([r['N'] for r in results])
    uNs = np.unique(Ns)
    handles = []
    plt.figure(figsize=(8,3.5))
    # handles.append(scatter_with_errors(Ns, uNs, np.array([r['mean_dist'] for r in results])))
    handles.append(scatter_with_errors(Ns, uNs, np.array([r['traverse_dist'] for r in results]), 'o','k',log=True,logmin=2**-5))
    handles.append(scatter_with_errors(Ns, uNs, np.array([r['baseline_dist'] for r in results]), 'o','none',log=True,logmin=2**-5))
    # handles.append(scatter_with_errors(Ns, uNs, np.array([r['traverse_dist_p'] for r in results]),'^'))
    # handles.append(scatter_with_errors(Ns, uNs, np.array([r['baseline_dist_p'] for r in results]),'v'))
    # handles.append(scatter_with_errors(Ns, uNs, np.array([r['mean_dist_p'] for r in results]), 'd')) # accidental tuple
    # plt.legend(handles, ['$||T - T_{mean}||$','$||B - B_{mean}||$','$||T^{+} - T^{+}_{mean}||$','$||B^{+} - B^{+}_{mean}||$','$||T^{+}_{mean} - B^{+}_{mean}||$'], loc='upper left')
    handles.append(scatter_with_errors(Ns, uNs, np.array([r['traverse_dist_v'] for r in results]),'^','k',log=True,logmin=2**-5))
    handles.append(scatter_with_errors(Ns, uNs, np.array([r['baseline_dist_v'] for r in results]),'^','none',log=True,logmin=2**-5))
    plt.legend(handles, ['$||T - mean(T)||$','$||B - mean(B)||$','$||T - sign(T)||$','$||B - sign(B)||$'], loc='lower center')
    # plt.xlim([uNs[0]-1,uNs[-1]+1])
    plt.xlim([2**.5,2*uNs[-1]])
    plt.gca().set_xscale('log',basex=2)
    plt.ylim([-5,5])
    plt.yticks(range(-5,5,2),['$2^{%d}$'%yl for yl in range(-5,5,2)])
    plt.ylabel('Average distances')
    #plt.title('Traverse vs Baseline')
    # plt.draw()
    # ytick_labels = plt.gca().get_yticklabels()
    # plt.gca().set_yticklabels(['2^%s'%(yl.get_text()) for yl in ytick_labels])
    # plt.yticks(range(-1,15,2),['0']+['$2^{%d}$'%yl for yl in range(1,15,2)])
    # plt.tight_layout()
    plt.show()

def show_tvb_stab_result(test_data_id='full_base', N=24, s=0):
    mpl.rcParams['mathtext.default'] = 'regular'
    # mpl.rcParams.update({'figure.autolayout': True})
    mpl.rcParams.update({'font.size': 16})
    results = load_pkl_file('results/TvB_stable_%s_N_%d_s_%d.pkl'%(test_data_id, N, s))
    npz = load_npz_file('results/TvB_stable_%s_N_%d_s_%d.npz'%(test_data_id, N, s))
    plt.figure(figsize=(11,5))
    plt.subplot(1,2,1)
    ms = 2*(mpl.rcParams['lines.markersize'] ** 2)
    # br = 0.25*np.random.rand(*npz['norms_baseline'].shape)
    # tr = 0.25*np.random.rand(*npz['norms_traverse'].shape)
    # plt.scatter(npz['norms_baseline'],npz['num_big_eigs_baseline']+br,c='',marker='o',s=ms)
    # plt.scatter(npz['norms_traverse'],npz['num_big_eigs_traverse']+.5+tr,c='k',marker='+',s=ms)
    t_idx = np.random.rand(*npz['norms_traverse'].shape) < .05
    b_idx = np.random.rand(*npz['norms_baseline'].shape) < .05
    # npz['max_eigs_%s'%method_key] = max_eigs[method_key]
    plt.scatter(npz['norms_baseline'][b_idx],npz['max_eigs_baseline'][b_idx],c='',marker='o',s=ms)
    plt.scatter(npz['norms_traverse'][t_idx],npz['max_eigs_traverse'][t_idx],c='k',marker='+',s=ms)
    plt.legend(['B','T'],loc='upper left')
    norm_max = max(npz['norms_baseline'].max(),npz['norms_traverse'].max())
    plt.xlim([-.1*norm_max,1.1*norm_max])
    # plt.ylim([-1,15])
    plt.xlabel('Norm')
    # plt.ylabel('# of unstable directions')
    plt.ylabel('Max. eig. mag.')
    # plt.gca().set_yscale('log',basey=2)
    plt.subplot(1,2,2).clear()
    # plt.hist([npz['num_big_eigs_baseline'],npz['num_big_eigs_traverse']],color=['k','w'],bins=range(N+1))
    # big_max = max(npz['num_big_eigs_baseline'].max(),npz['num_big_eigs_traverse'].max())
    # plt.hist([npz['num_big_eigs_baseline'],npz['num_big_eigs_traverse']],color=['k','w'],bins=range(int(big_max+1)))
    # eig_max = max(npz['max_eigs_baseline'].max(),npz['max_eigs_traverse'].max())
    # plt.hist([npz['max_eigs_baseline'],npz['max_eigs_traverse']],color=['k','w'],bins=np.linspace(0,eig_max,10))
    # plt.legend(['B','T'],loc='upper right')
    # plt.xlabel('# of unstable directions')
    # plt.xlabel('Max eigenvalue')
    # plt.ylabel('# of fixed points')
    bs = npz['max_eigs_baseline'] < 1 
    ts = npz['max_eigs_traverse'] < 1 
    plt.hist([npz['norms_baseline'][bs],npz['norms_traverse'][ts],npz['norms_baseline'][~bs],npz['norms_traverse'][~ts]],color=np.array([[0.0,0.33,0.66,1.0]]).T*np.ones((1,3)),bins=range(int(np.ceil(norm_max+1))),align='mid')
    # plt.hist([npz['norms_baseline'][bs],npz['norms_traverse'][ts],npz['norms_baseline'][~bs],npz['norms_traverse'][~ts]],bins=range(int(np.ceil(norm_max+1))),align='mid')
    plt.xlim([0,np.ceil(norm_max)])
    plt.legend(['B st','T st','B un','T un'],loc='upper left')
    plt.xlabel('Norm')
    plt.ylabel('# of fixed points')
    plt.gca().set_yscale('log',basey=2)
    plt.tight_layout()
    # plt.xlim([-30,50])
    # plt.xticks(range(-30,51,10),['']+['$2^{%d}$'%xl for xl in range(-20,51,10)])
    plt.show()
            
def show_tvb_stab_results(test_data_ids):
    """
    Plot the results of traverse-baseline stability analysis on one or more testing data sets
    test_data_ids should be the list of ids, each as in generate_test_data (without file extension)
    """
    results = []
    for test_data_id in test_data_ids:
        # results += load_pkl_file('results/tvb_dist_%s.pkl'%test_data_id)
        # results += load_pkl_file('results/tvb_%s.pkl'%test_data_id)
        next_results = load_pkl_file('results/tvb_stab_%s.pkl'%test_data_id)
        # # temp because left out of code before first run:
        # for r in range(len(next_results)):
        #     npz = load_npz_file('results/%s.npz'%next_results[r]['result_key'])
        #     next_results[r]['avg_big_eigs_traverse'] = npz['num_big_eigs_traverse'].mean()
        #     next_results[r]['avg_big_eigs_baseline'] = npz['num_big_eigs_baseline'].mean()
        #     next_results[r]['min_big_eigs_traverse'] = npz['num_big_eigs_traverse'].min()
        #     next_results[r]['min_big_eigs_baseline'] = npz['num_big_eigs_baseline'].min()
        # # save_pkl_file('results/tvb_stab_%s.pkl'%test_data_id,next_results)
        results += next_results
    results = [r for r in results if r['N'] in [2,4,7,10,13,16,24,32,48,64,128,256,512,1024]]
    mpl.rcParams['mathtext.default'] = 'regular'
    # mpl.rcParams.update({'figure.autolayout': True})
    mpl.rcParams.update({'font.size': 12})
    Ns = np.array([r['N'] for r in results])
    uNs = np.unique(Ns)
    handles = []
    plt.figure(figsize=(8,3.5))
    # handles.append(scatter_with_errors(Ns, uNs, np.array([r['mean_dist'] for r in results])))
    handles.append(scatter_with_errors(Ns, uNs, [max(r['num_stable_traverse'],0.5) for r in results], 'o','none',log=True,logmin=0.5))
    handles.append(scatter_with_errors(Ns, uNs, [max(r['num_stable_baseline'],0.5) for r in results], 'd','none',log=True,logmin=0.5))
    handles.append(scatter_with_errors(Ns, uNs, [max(r['T']-r['num_stable_traverse'],0.5) for r in results], '^','none',log=True,logmin=0.5))
    handles.append(scatter_with_errors(Ns, uNs, [max(r['B']-r['num_stable_baseline'],0.5) for r in results], 's','none',log=True,logmin=0.5))
    # handles.append(scatter_with_errors(Ns, uNs, np.array([r['traverse_dist_p'] for r in results]),'^'))
    # handles.append(scatter_with_errors(Ns, uNs, np.array([r['baseline_dist_p'] for r in results]),'v'))
    # handles.append(scatter_with_errors(Ns, uNs, np.array([r['mean_dist_p'] for r in results]), 'd')) # accidental tuple
    # plt.legend(handles, ['$||T - T_{mean}||$','$||B - B_{mean}||$','$||T^{+} - T^{+}_{mean}||$','$||B^{+} - B^{+}_{mean}||$','$||T^{+}_{mean} - B^{+}_{mean}||$'], loc='upper left')
    # handles.append(scatter_with_errors(Ns, uNs, np.array([r['traverse_dist_v'] for r in results]),'^','k'))
    # handles.append(scatter_with_errors(Ns, uNs, np.array([r['baseline_dist_v'] for r in results]),'^','none'))
    # plt.legend(handles, ['Traverse','Baseline'], loc='upper left')
    plt.legend(handles, ['T st','B st','T un','B un'], loc='upper left')
    plt.xlim([2**.5,2**.5*uNs[-1]])
    # plt.ylim([-1,25])
    plt.ylabel('# of fixed points')
    # plt.xlabel('N')
    #plt.title('Traverse vs Baseline')
    # plt.draw()
    # ytick_labels = plt.gca().get_yticklabels()
    # plt.gca().set_yticklabels(['2^%s'%(yl.get_text()) for yl in ytick_labels])
    plt.yticks(range(-1,13,2),['0']+['$2^{%d}$'%yl for yl in range(1,13,2)])
    plt.ylim([-2,13])
    plt.gca().set_xscale('log',basex=2)
    plt.tight_layout()
    plt.show()

def show_tvb_runtimes(test_data_ids):
    """
    Plot the results of traverse-baseline runtime comparison on one or more testing data sets
    test_data_ids should be the list of ids, each as in generate_test_data (without file extension)
    """
    mpl.rcParams['mathtext.default'] = 'regular'
    # mpl.rcParams.update({'figure.autolayout': True})
    mpl.rcParams.update({'font.size': 12})

    t_res, b_res = [], []
    for test_data_id in test_data_ids:
        t_res += load_pkl_file('results/traverse_%s.pkl'%test_data_id)
        b_res += load_pkl_file('results/baseline_%s.pkl'%test_data_id)
    t_res = [r for r in t_res if r['N'] in [2,4,7,10,13,16,24,32,48,64,128,256,512,1024]]
    b_res = [r for r in b_res if r['N'] in [2,4,7,10,13,16,24,32,48,64,128,256,512,1024]]

    Ns = np.array([r['N'] for r in t_res])
    uNs = np.unique(Ns)
    handles = []
    handles.append(scatter_with_errors(Ns, uNs, (np.array([r['runtime']+r['post_runtime'] for r in b_res]))/60, '^','none'))
    handles.append(scatter_with_errors(Ns, uNs, (np.array([r['runtime']+r['post_runtime'] for r in t_res]))/60, 'o','none'))
    handles.append(scatter_with_errors(Ns, uNs, (np.array([r['runtime'] for r in t_res]))/60, 'x','none'))
    plt.legend(handles, ['With B post-processing','With T post-processing','Runtime'], loc='upper left')
    # plt.xlim([uNs[0]-1,uNs[-1]+1])
    plt.xlim([2**.5,2*uNs[-1]])
    plt.gca().set_xscale('log',basex=2)
    # plt.ylim([-10,200])
    plt.ylabel('Running time (minutes)')
    # plt.gca().set_yscale('log',basey=2)
    #plt.title('Traverse vs Baseline')
    # plt.draw()
    # ytick_labels = plt.gca().get_yticklabels()
    # plt.gca().set_yticklabels(['2^%s'%(yl.get_text()) for yl in ytick_labels])
    # plt.yticks(range(-1,15,2),['0']+['$2^{%d}$'%yl for yl in range(1,15,2)])
    # plt.tight_layout()
    plt.show()

def show_tvb_work(test_data_ids):
    """
    Plot the results of traverse-baseline work complexity comparison on one or more testing data sets
    test_data_ids should be the list of ids, each as in generate_test_data (without file extension)
    """
    mpl.rcParams['mathtext.default'] = 'regular'
    # mpl.rcParams.update({'figure.autolayout': True})
    mpl.rcParams.update({'font.size': 12})

    t_res, b_res = [], []
    for test_data_id in test_data_ids:
        t_res += load_pkl_file('results/traverse_%s.pkl'%test_data_id)
        b_res += load_pkl_file('results/baseline_%s.pkl'%test_data_id)
    t_res = [r for r in t_res if r['N'] in [2,4,7,10,13,16,24,32,48,64,128,256,512,1024]]
    b_res = [r for r in b_res if r['N'] in [2,4,7,10,13,16,24,32,48,64,128,256,512,1024]]

    Ns = np.array([r['N'] for r in t_res])
    uNs = np.unique(Ns)
    handles = []
    plt.figure(figsize=(8,3))
    ylog = True
    if ylog:
        handles.append(scatter_with_errors(Ns, uNs, [(r['runtime']+r['post_runtime'])/r['num_fxV_unique']/60 for r in b_res], 'o','none',log=True,logmin=2**-13))
        handles.append(scatter_with_errors(Ns, uNs, [r['runtime']/r['num_fxV_unique']/60 for r in b_res], 'x','none',log=True,logmin=2**-13))
        handles.append(scatter_with_errors(Ns, uNs, [r['runtime']/r['num_fxV_unique']/60 for r in t_res], '^','none',log=True,logmin=2**-13))
        plt.yticks(range(-13,16,4),['$2^{%d}$'%yl for yl in range(-13,16,4)])
        plt.ylim([-14,15])
    else:
        handles.append(scatter_with_errors(Ns, uNs, [(r['runtime']+r['post_runtime'])/r['num_fxV_unique']/60 for r in b_res], 'o','none'))
        handles.append(scatter_with_errors(Ns, uNs, [r['runtime']/r['num_fxV_unique']/60 for r in b_res], 'x','none'))
        handles.append(scatter_with_errors(Ns, uNs, [r['runtime']/r['num_fxV_unique']/60 for r in t_res], '^','none'))
    plt.legend(handles, ['B with post-processing','B no post-processing','T with post-processing'], loc='upper left')
    # plt.xlim([uNs[0]-1,uNs[-1]+1])
    plt.xlim([2**.5,1.5*uNs[-1]])
    plt.gca().set_xscale('log',basex=2)
    plt.ylabel('Minutes per fixed point found')
    # plt.gca().set_yscale('log',basey=2)
    #plt.title('Traverse vs Baseline')
    # plt.draw()
    # ytick_labels = plt.gca().get_yticklabels()
    # plt.gca().set_yticklabels(['2^%s'%(yl.get_text()) for yl in ytick_labels])
    # plt.yticks(range(-1,15,2),['0']+['$2^{%d}$'%yl for yl in range(1,15,2)])
    # plt.tight_layout()
    plt.show()

def show_tvb_rawcounts(test_data_ids):
    """
    Plot the results of traverse-baseline un-post-processed count comparison on one or more testing data sets
    test_data_ids should be the list of ids, each as in generate_test_data (without file extension)
    """
    mpl.rcParams['mathtext.default'] = 'regular'
    # mpl.rcParams.update({'figure.autolayout': True})
    mpl.rcParams.update({'font.size': 12})
    # mpl.rcParams['lines.linewidth'] = 2

    t_res, b_res = [], []
    for test_data_id in test_data_ids:
        t_res += load_pkl_file('results/traverse_%s.pkl'%test_data_id)
        b_res += load_pkl_file('results/baseline_%s.pkl'%test_data_id)
    t_res = [r for r in t_res if r['N'] in [2,4,7,10,13,16,24,32,48,64,128,256,512,1024]]
    b_res = [r for r in b_res if r['N'] in [2,4,7,10,13,16,24,32,48,64,128,256,512,1024]]

    Ns = np.array([r['N'] for r in t_res])
    uNs = np.unique(Ns)
    handles = []
    handles.append(scatter_with_errors(Ns, uNs, [r['num_fxV'] for r in b_res], 'o', 'k',log=True))
    handles.append(scatter_with_errors(Ns, uNs, [r['num_fxV'] for r in t_res], 'o', 'none',log=True))
    plt.legend(handles, ['Raw B counts','Raw T counts'], loc='lower right')
    plt.xlim([uNs[0]-1,uNs[-1]+1])
    # plt.ylim([-2,90])
    plt.ylabel('Raw point counts')
    plt.yticks(range(2,19,2),['$2^{%d}$'%yl for yl in range(2,19,2)])
    #plt.title('Traverse vs Baseline')
    # plt.draw()
    # ytick_labels = plt.gca().get_yticklabels()
    # plt.gca().set_yticklabels(['2^%s'%(yl.get_text()) for yl in ytick_labels])
    # plt.yticks(range(-1,15,2),['0']+['$2^{%d}$'%yl for yl in range(1,15,2)])
    # plt.tight_layout()
    plt.show()

# def show_Wc_results(results):
# def show_Wc_results(test_data_id='dl15'):
def show_Wc_results(test_data_id='full_choose'):
    """
    Plot the results of c choice comparison
    test_data_id should be as in generate_test_data (without file extension)
    """
    results = load_pkl_file('results/%s_Wc.pkl'%test_data_id)
    mpl.rcParams['mathtext.default'] = 'regular'
    Ns = np.array([r[0]['N'] for r in results])
    uNs = np.unique(Ns)
    handles = []
    plt.figure(figsize=(8,4.2))
    y = [max(r[0]['num_fxV_union'],0.5) for r in results]
    handles.append(scatter_with_errors(Ns, uNs, y, 'o','k',log=True,logmin=.5))
    for (fun, m,fc) in [(np.max,'^','none'),(np.mean,'d','k',),(np.min,'v','none')]:
        y = [max(fun([r['num_fxV_unique'] for r in res]),0.5) for res in results]
        handles.append(scatter_with_errors(Ns, uNs, y,m,fc,log=True,logmin=0.5))
    handles.append(plt.plot(uNs, np.log2(uNs), 'dk--')[0])
    plt.legend(handles, ['Union','Max','Mean','Min','Known'], loc='upper left')
    plt.xlim([uNs[0]-.5,uNs[-1]+.5])
    plt.ylabel('# of fixed points')
    #plt.title('Different Regular Regions')
    # plt.draw()
    # ytick_labels = plt.gca().get_yticklabels()
    # plt.gca().set_yticklabels(['2^%s'%(yl.get_text()) for yl in ytick_labels])
    plt.yticks(range(0,11,1),['$2^{%d}$'%yl for yl in range(0,11,1)])
    plt.ylim([0,9])
    plt.tight_layout()
    plt.show()

def scatter_with_errors(Ns, uNs, y, marker, facecolor, show_scatter=False, log=False,logmin=1):
    """
    Helper function for generating scatter plots with error bars for means and standard deviations.
    Ns[i] should be the size of the i^{th} network being plotted
    y[i] should be the value of the statistic being plotted on the i^{th} network 
    uNs should be a list of the unique network sizes included in the plot
    marker and facecolor should be as in matplotlib.pyplot.scatter
    if show_scatter==False, only means and standard deviations are shown for each network size N
    returns scat, a legend handle for use with matplotlib.pyplot.legend
    """
    y_by_N = np.array([np.array(y)[Ns==N] for N in uNs])
    y_by_N_means = np.array([yy.mean() for yy in y_by_N])
    y_by_N_stds = np.array([yy.std() for yy in y_by_N])
    if log:
        lo = y_by_N_means-y_by_N_stds
        lo[lo <= 0] = logmin
        y_by_N_means[y_by_N_means <= 0] = logmin
        y_err = np.concatenate([
            (np.log2(y_by_N_means)-np.log2(lo))[np.newaxis,:],
            (np.log2(y_by_N_means+y_by_N_stds)-np.log2(y_by_N_means))[np.newaxis,:]],axis=0)
        if show_scatter: y_by_N = np.log2(y_by_N)
        y_by_N_means = np.log2(y_by_N_means)
    else:
        y_err = y_by_N_stds
    if show_scatter:
        scat = plt.scatter(Ns, y, s=30,marker=marker, facecolor=facecolor, edgecolor='0.7')
        plt.errorbar(uNs, y_by_N_means, yerr=y_err, ecolor='k', c='k', marker=marker, ms=9, mfc='none')
    else:
        scat = plt.errorbar(uNs, y_by_N_means, yerr=y_err, ecolor='k', c='k', marker=marker, ms=9, mfc=facecolor)
    plt.xlabel('N')
    return scat

def baseline_comparison_experiments(test_data_id, num_procs, max_traverse_steps=2**20):
    """
    Run traverse, the baseline, and the comparison on every network in the test data
    test_data_id should be as in generate_test_data (without file extension)
    num_procs is the number of processors to use in parallel
    """
    _ = run_traverse_experiments(test_data_id,num_procs,max_traverse_steps)
    _ = run_baseline_experiments(test_data_id,num_procs)
    _ = run_TvB_experiments(test_data_id,num_procs)
