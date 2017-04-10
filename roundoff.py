"""
Methods for assessing treatment of finite-precision issues
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
import fxpt_experiments as fe
import pickle as pkl

def get_relative_errors(test_data_id):
    """
    Compute and save the relative errors of every point found on every network in a testing set.
    Relative error is defined in (Katz and Reggia 2017).
    test_data_id should be as in fxpt_experiments.generate_test_data (without file extension).
    """
    network_sizes, num_samples, _ = fe.load_test_data('%s.npz'%test_data_id)
    for alg in ['traverse','baseline']:
        for (N, S) in zip(network_sizes, num_samples):
            for samp in range(S):
                print('%s, alg %s, N %d,samp %d'%(test_data_id,alg,N,samp))
                npz = np.load('results/%s_%s_N_%d_s_%d.npz'%(alg,test_data_id,N,samp))
                W = npz['W']
                fxV = npz['fxV']
                fxV, converged = rfx.refine_fxpts_capped(W, fxV)
                margin = rfx.estimate_forward_error(W, fxV)
                f = np.tanh(W.dot(fxV))-fxV
                re = np.fabs(f/margin)
                re_fx, re_un = re[:,converged].max(axis=0), re[:,~converged].max(axis=0)
                re_fx = re_fx[re_fx > 0]
                f_fx, f_un = np.fabs(f[:,converged]).max(axis=0), np.fabs(f[:,~converged]).max(axis=0)
                f_fx = f_fx[f_fx > 0]
                re_npz = {}
                re_npz['f_fx'] = f_fx
                re_npz['f_un'] = f_un
                re_npz['re_fx'] = re_fx
                re_npz['re_un'] = re_un
                fe.save_npz_file('results/%s_re_%s_N_%d_s_%d.npz'%(alg,test_data_id,N,samp), **re_npz)

def show_traverse_re_fig(test_data_ids, Ns, samp_range):
    """
    Plot relative errors from points found by fiber traversal.
    test_data_ids and Ns should be length-2 lists.
    Subplots in the first column will show errors networks of size Ns[0] from test_data_ids[0].
    Similarly the second column draws from Ns[1], test_data_ids[1].
    Each network sample within samp_range is shown on a separate row.
    """
    log = True
    mpl.rcParams['mathtext.default'] = 'regular'
    sp = 1
    for samp in samp_range:
        for (test_data_id,N) in zip(test_data_ids, Ns):
            print('samp %d, N %d'%(samp,N))
            npz = np.load('results/traverse_re_%s_N_%d_s_%d.npz'%(test_data_id,N,samp))
            m_fx, m_un = npz['re_fx'], npz['re_un']
            ax = plt.subplot(len(samp_range),len(Ns),sp)
            sp += 1
            if m_un.shape[0] > 0: plt.hist(np.log2(m_un),bins=30,log=log,facecolor='k')
            plt.hist(np.log2(m_fx),bins=10,log=log,facecolor='w')
            lo = 10*(int(np.log2(m_fx).min()/10)-1)
            if m_un.shape[0] > 0: hi = 10*(int(np.log2(m_un).max()/10)+1)
            else: hi = 0
            plt.xticks(range(-10,1,2),['']+['$2^{%d}$'%yl for yl in range(-8,1,2)])
            if N == Ns[0]:
                plt.ylabel('# of points')
            if samp == samp_range[0]:
                ax.set_title('N = %d'%N)
            if samp == samp_range[-1]:
                plt.xlabel('Fiber Relative Error')
    plt.show()

def baseline_re_single_analysis(test_data_id, N, samp, cap=10):
    """
    Analyze edge cases of relative errors on a single network
    Uses the samp^{th} sample network of size N in test data test_data_id.
    Relative errors in the range (0, 2^{cap}) are considered edge cases.
    Returns the number of edge cases divided by the difference |T-B| - |B-T| as a percent.
    T and B are as defined in (Katz and Reggia 2017).
    """
    npz = fe.load_npz_file('results/baseline_re_%s_N_%d_s_%d.npz'%(test_data_id,N,samp))
    res = fe.load_pkl_file('results/TvB_%s_N_%d_s_%d.pkl'%(test_data_id, N, samp))
    re_un = npz['re_un']
    percent = 100.*(re_un < 2**cap).sum()/np.array(res['T-B']-res['B-T'])
    print('N=%d, samp %d: B-T = %d, T-B = %d, %d (%f%%) possibly unique slow RE(B) < 2**%d'%(N, samp, res['B-T'], res['T-B'],(re_un < 2**cap).sum(), percent, cap))
    return percent

def baseline_re_batch_analysis(test_data_id, Ns, cap=10):
    """
    Runs baseline_re_single_analysis on all networks in test_data_id of size N.
    cap is as in baseline_re_single_analysis.
    returns numpy.array percents, where
      percents[i] is as in baseline_re_single_analysis for the i^{th} sample network.
    """
    percents = []
    network_sizes, num_samples, _ = fe.load_test_data('%s.npz'%test_data_id)
    for (N, S) in zip(network_sizes, num_samples):
        if N not in Ns: continue
        for samp in range(S):
            percents.append(baseline_re_single_analysis(test_data_id,N,samp,cap=cap))
    percents = np.array(percents)
    print('mean %%: %f%%'%percents.mean())

def show_baseline_re_fig(test_data_ids, Ns, samp_range):
    """
    Plot relative errors from points found by the baseline solver.
    test_data_ids and Ns should be length-2 lists.
    Subplots in the first column will show errors networks of size Ns[0] from test_data_ids[0].
    Similarly the second column draws from Ns[1], test_data_ids[1].
    Each network sample within samp_range is shown on a separate row.
    """
    log = True
    mpl.rcParams['mathtext.default'] = 'regular'
    sp = 1
    for samp in samp_range:
        for (test_data_id,N) in zip(test_data_ids, Ns):
            print('samp %d, N %d'%(samp,N))
            npz = np.load('results/baseline_re_%s_N_%d_s_%d.npz'%(test_data_id,N,samp))
            m_fx, m_un = npz['re_fx'], npz['re_un']
            ax = plt.subplot(len(samp_range),len(Ns),sp)
            sp += 1
            if m_un.shape[0] > 0: plt.hist(np.log2(m_un),bins=30,log=log,facecolor='k')
            plt.hist(np.log2(m_fx),bins=10,log=log,facecolor='w')
            lo, hi = -20,50
            plt.xticks(range(lo,hi+1,10),[''] + ['$2^{%d}$'%yl for yl in range(lo+10,hi+1,10)])
            if N == Ns[0]:
                plt.ylabel('# of points')
            if samp == samp_range[0]:
                ax.set_title('N = %d'%N)
            if samp == samp_range[-1]:
                plt.xlabel('Baseline Relative Error')
            baseline_re_single_analysis(test_data_id, N, samp)
    plt.show()

def get_baseline_rd(test_data_id,N,samp,cap,logfilename=os.devnull):
    """
    Compute and save relative distances between pairs of points found by the baseline solver.
    Relative distance is defined in (Katz and Reggia 2017).
    Computes for the samp^{th} sample network of size N in test_data_id.
    test_data_id should be as in fxpt_experiments.generate_test_data (without file extension).
    Only pairs within a random subset of points of size cap are inspected.
    logfilename is a file name at which progress updates are written.
    """
    logfile = open(logfilename,'w')
    logfile.write('Running baseline rd (%s,%d,%d)...\n'%(test_data_id,N,samp))
    npz = fe.load_npz_file('results/baseline_%s_N_%d_s_%d.npz'%(test_data_id,N,samp))
    fxV = npz['fxV_converged']
    fxV_unique = npz['fxV_unique']
    W = npz['W']
    if cap is not None and fxV.shape[1] > cap:
        logfile.write('capping...\n')
        perm = np.random.permutation(fxV.shape[1])
        fxV = fxV[:,perm[:cap]]
    in_RR, out_RR = [],[]
    for j in range(fxV_unique.shape[1]):
        logfile.write('duping %d of %d...\n'%(j,fxV_unique.shape[1]))
        dups, RR, R = rfx.identical_fixed_points(W, fxV, fxV_unique[:,[j]])
        in_RR.append(RR[dups])
        out_RR.append(RR[~dups])
    in_RR, out_RR = np.concatenate(in_RR), np.concatenate(out_RR)
    npz["in_RR"], npz["out_RR"] = in_RR, out_RR
    fe.save_npz_file('results/baseline_rd_%s_N_%d_s_%d.npz'%(test_data_id,N,samp), **npz)
    logfile.write('Done.\n')
    logfile.close()
    print('Done %s %d %d'%(test_data_id,N,samp))

def pool_get_baseline_rd(args):
    """
    Wrapper function passed to multiprocessing.Pool
    """
    get_baseline_rd(*args)

def run_baseline_rd(test_data_id, Ns, num_procs):
    """
    Run get_baseline_rd on all networks in test_data_id whose size is in the list Ns.
    Multiprocessing is used to run on multiple networks in parallel.
    num_procs is the number of processors to use.
    """
    cpu_count = mp.cpu_count()
    print('%d cpus, using %d'%(cpu_count, num_procs))

    pool_args = []
    network_sizes, num_samples, _ = fe.load_test_data('%s.npz'%test_data_id)
    for (N, S) in zip(network_sizes, num_samples):
        if N not in Ns: continue
        cap = 20000
        for s in range(S):
            logfilename = 'logs/baseline_rd_%s_N_%d_s_%d.log'%(test_data_id,N,s)
            pool_args.append((test_data_id,N,s,cap,logfilename))
    start_time = time.time()
    test_fun = pool_get_baseline_rd
    if num_procs < 1: # don't multiprocess
        for args in pool_args: test_fun(args)
    else:
        pool = mp.Pool(processes=num_procs)
        pool.map(test_fun, pool_args)
        pool.close()
        pool.join()
    print('total time: %f'%(time.time()-start_time))

def get_traverse_rd(test_data_id,N,samp,cap,logfilename=os.devnull):
    """
    Compute and save relative distances between pairs of points found by the baseline solver.
    Relative distance is defined in (Katz and Reggia 2017).
    Computes for the samp^{th} sample network of size N in test_data_id.
    test_data_id should be as in fxpt_experiments.generate_test_data (without file extension).
    Only pairs within a random subset of points of size cap are inspected.
    logfilename is a file name at which progress updates are written.
    """
    logfile = open(logfilename,'w')
    logfile.write('Running traverse rd (%s,%d,%d)...\n'%(test_data_id,N,samp))
    npz = fe.load_npz_file('results/traverse_%s_N_%d_s_%d.npz'%(test_data_id,N,samp))
    fxV = npz['fxV_converged']
    fxV_unique = npz['fxV_unique']
    W = npz['W']
    if cap is not None and fxV.shape[1] > cap:
        logfile.write('capping...\n')
        perm = np.random.permutation(fxV.shape[1])
        fxV = fxV[:,perm[:cap]]
    in_RR, out_RR = [],[]
    for j in range(fxV_unique.shape[1]):
        logfile.write('duping %d of %d...\n'%(j,fxV_unique.shape[1]))
        dups, RR, R = rfx.identical_fixed_points(W, fxV, fxV_unique[:,[j]])
        in_RR.append(RR[dups])
        out_RR.append(RR[~dups])
    in_RR, out_RR = np.concatenate(in_RR), np.concatenate(out_RR)
    npz["in_RR"], npz["out_RR"] = in_RR, out_RR
    fe.save_npz_file('results/traverse_rd_%s_N_%d_s_%d.npz'%(test_data_id,N,samp), **npz)
    logfile.write('Done.\n')
    logfile.close()
    print('Done %s %d %d'%(test_data_id,N,samp))

def pool_get_traverse_rd(args):
    """
    Wrapper function passed to multiprocessing.Pool
    """
    get_traverse_rd(*args)

def run_traverse_rd(test_data_id, Ns, num_procs):
    """
    Run get_traverse_rd on all networks in test_data_id whose size is in the list Ns.
    Multiprocessing is used to run on multiple networks in parallel.
    num_procs is the number of processors to use.
    """

    cpu_count = mp.cpu_count()
    print('%d cpus, using %d'%(cpu_count, num_procs))

    pool_args = []
    network_sizes, num_samples, _ = fe.load_test_data('%s.npz'%test_data_id)
    for (N,S) in zip(network_sizes, num_samples):
        if N not in Ns: continue
        cap = 20000
        for s in range(S):
            logfilename = 'logs/traverse_rd_%s_N_%d_s_%d.log'%(test_data_id,N,s)
            pool_args.append((test_data_id,N,s,cap,logfilename))
    start_time = time.time()
    test_fun = pool_get_traverse_rd
    if num_procs < 1: # don't multiprocess
        for args in pool_args: test_fun(args)
    else:
        pool = mp.Pool(processes=num_procs)
        pool.map(test_fun, pool_args)
        pool.close()
        pool.join()
    print('total time: %f'%(time.time()-start_time))

def get_simple_rd(test_data_id,N,samp,cap,logfilename=os.devnull):
    """
    Use simple unique test: if max absolute coordinate-wise difference < 2**-32
    Compute and save distances between pairs of points found by both solvers.
    Computes for the samp^{th} sample network of size N in test_data_id.
    test_data_id should be as in fxpt_experiments.generate_test_data (without file extension).
    Only pairs within a random subset of points of size cap are inspected.
    Saves pair-wise distance distribution in histogram with one bucket per integer power of 2
    logfilename is a file name at which progress updates are written.
    """
    logfile = open(logfilename,'w')
    logfile.write('Running simple rd (%s,%d,%d)...\n'%(test_data_id,N,samp))
    buckets = {}
    bins = np.arange(-1025,3)
    for method_key in ['traverse','baseline']:
        npz = fe.load_npz_file('results/%s_%s_N_%d_s_%d.npz'%(method_key,test_data_id,N,samp))
        fxV = npz['fxV_converged']
        buckets[method_key] = np.zeros(len(bins)-1)
        if cap is not None and fxV.shape[1] > cap:
            logfile.write('capping...\n')
            perm = np.random.permutation(fxV.shape[1])
            fxV = fxV[:,perm[:cap]]
        for j in range(fxV.shape[1]):
            logfile.write('disting %d of %d...\n'%(j,fxV.shape[1]))
            dists = np.fabs(fxV-fxV[:,[j]]).max(axis=0)
            dists[dists == 0] = 2.0**bins[0]
            logdists = np.log2(dists)
            logdists[logdists < bins[0]] = bins[0]
            logdists[logdists > bins[-1]] = bins[-1]
            hist,_ = np.histogram(logdists,bins=bins)
            buckets[method_key] += hist
    npz = {'bins':bins,'traverse_buckets':buckets['traverse'],'baseline_buckets':buckets['baseline']}    
    fe.save_npz_file('results/simple_rd_%s_N_%d_s_%d.npz'%(test_data_id,N,samp), **npz)
    logfile.write('Done.\n')
    logfile.close()
    print('Done %s %d %d'%(test_data_id,N,samp))

def pool_get_simple_rd(args):
    """
    Wrapper function passed to multiprocessing.Pool
    """
    get_simple_rd(*args)

def run_simple_rd(test_data_id, Ns, num_procs):
    """
    Run get_traverse_rd on all networks in test_data_id whose size is in the list Ns.
    Multiprocessing is used to run on multiple networks in parallel.
    num_procs is the number of processors to use.
    """

    cpu_count = mp.cpu_count()
    print('%d cpus, using %d'%(cpu_count, num_procs))

    pool_args = []
    network_sizes, num_samples, _ = fe.load_test_data('%s.npz'%test_data_id)
    for (N,S) in zip(network_sizes, num_samples):
        if N not in Ns: continue
        cap = 1000
        for s in range(S):
            logfilename = 'logs/simple_rd_%s_N_%d_s_%d.log'%(test_data_id,N,s)
            pool_args.append((test_data_id,N,s,cap,logfilename))
    start_time = time.time()
    test_fun = pool_get_simple_rd
    if num_procs < 1: # don't multiprocess
        for args in pool_args: test_fun(args)
    else:
        pool = mp.Pool(processes=num_procs)
        pool.map(test_fun, pool_args)
        pool.close()
        pool.join()
    print('total time: %f'%(time.time()-start_time))

def show_traverse_rd_fig(test_data_ids, Ns, samp_range):
    """
    Plot relative distances from points found by fiber traversal.
    test_ids, Ns, and samp_range should be as in show_traverse_re_fig.
    """
    log = True
    mpl.rcParams['mathtext.default'] = 'regular'
    sp = 1
    for samp in samp_range:
        for (test_data_id,N) in zip(test_data_ids, Ns):
            print('samp %d, N %d'%(samp,N))
            npz = np.load('results/traverse_rd_%s_N_%d_s_%d.npz'%(test_data_id,N,samp))
            in_rr, out_rr = npz['in_RR'], npz['out_RR']
            if (in_rr > 0).any(): in_rr[in_rr == 0] = in_rr[in_rr > 0].min()
            else: in_rr[in_rr == 0] = 2**(-30)
            ax = plt.subplot(len(samp_range),len(Ns),sp)
            sp += 1
            if out_rr.shape[0] > 0: plt.hist(np.log2(out_rr),bins=30,log=log,facecolor='k')
            plt.hist(np.log2(in_rr),bins=10,log=log,facecolor='w')
            if N == Ns[0]:
                plt.ylabel('# of pairs')
            if samp == samp_range[0]:
                ax.set_title('N = %d'%N)
            if samp == samp_range[-1]:
                plt.xlabel('Fiber Relative Distance')
            plt.xlim([-30,50])
            plt.xticks(range(-30,51,10),['']+['$2^{%d}$'%xl for xl in range(-20,51,10)])
    plt.show()

def show_baseline_rd_fig(test_data_ids, Ns, samp_range):
    """
    Plot relative distances from points found by the baseline solver.
    test_ids, Ns, and samp_range should be as in show_baseline_re_fig.
    """
    log = True
    mpl.rcParams['mathtext.default'] = 'regular'
    sp = 1
    for samp in samp_range:
        for (test_data_id,N) in zip(test_data_ids, Ns):
            print('samp %d, N %d'%(samp,N))
            npz = np.load('results/baseline_rd_%s_N_%d_s_%d.npz'%(test_data_id,N,samp))
            in_rr, out_rr = npz['in_RR'], npz['out_RR']
            if (in_rr > 0).any(): in_rr[in_rr == 0] = in_rr[in_rr > 0].min()
            else: in_rr[in_rr == 0] = 2**(-30)
            ax = plt.subplot(len(samp_range),len(Ns),sp)
            sp += 1
            if np.isinf(out_rr).any():
                if np.isinf(out_rr).all(): out_rr[:] = 4*in_rr.max()
                else: out_rr[np.isinf(out_rr)] = 4*out_rr[~np.isinf(out_rr)].max()
            print('out_rr:')
            print(out_rr.shape)
            print((out_rr==0).sum())
            print(np.isinf(in_rr).sum())
            print(np.isinf(out_rr).sum())
            print(np.isnan(out_rr).sum())
            if out_rr.shape[0] > 0: plt.hist(np.log2(out_rr),bins=30,log=log,facecolor='k')
            # if out_rr.shape[0] > 0: plt.hist(out_rr,bins=30,facecolor='k')
            plt.hist(np.log2(in_rr),bins=10,log=log,facecolor='w')
            # plt.hist(in_rr,bins=10,facecolor='w')
            if N == Ns[0]:
                plt.ylabel('# of pairs')
            if samp == samp_range[0]:
                ax.set_title('N = %d'%N)
            if samp == samp_range[-1]:
                plt.xlabel('Baseline Relative Distance')
            plt.xlim([-30,50])
            plt.xticks(range(-30,51,10),['']+['$2^{%d}$'%xl for xl in range(-20,51,10)])
    plt.show()

def show_simple_rd_all_fig(test_data_ids, Ns, samp_range):
    """
    Plot relative distances from points found by fiber traversal.
    test_ids, Ns, and samp_range should be as in show_traverse_re_fig.
    """
    log = True
    mpl.rcParams['mathtext.default'] = 'regular'
    buckets = None
    bins = None
    for samp in samp_range:
        for (test_data_id,N) in zip(test_data_ids, Ns):
            print('samp %d, N %d'%(samp,N))
            npz = np.load('results/simple_rd_%s_N_%d_s_%d.npz'%(test_data_id,N,samp))
            if buckets is None:
                buckets = np.zeros(npz['traverse_buckets'].shape)
                bins = npz['bins']
            buckets += npz['traverse_buckets']
            buckets += npz['baseline_buckets']
    plt.figure(figsize=(8,3))
    # plt.hist(buckets,bins=bins,log=log)
    if log:
        buckets[buckets > 0] = np.log2(buckets[buckets > 0])
    plt.bar(left=bins[:-1],height=buckets,width=bins[1:]-bins[:-1])
    plt.ylabel('# of fixed point pairs')
    plt.xlabel('$max_i|v_i^{(1)}-v_i^{(2)}|$') #'Max Coordinate-wise Distance')
    xmin_idx = int(((bins[:-1] > -1000) & (buckets > 0)).argmax())
    xstep = int(np.ceil((bins[-1]-bins[xmin_idx])/10))
    plt.xticks(bins[xmin_idx::xstep],['$2^{%d}$'%xl for xl in bins[xmin_idx::xstep]])
    plt.xlim([bins[xmin_idx]-xstep,bins[-1]+xstep])
    if log:
        ymax = np.ceil(buckets.max())+1
        ystep = np.ceil(ymax/5)
        plt.yticks(np.arange(0,ymax+ystep,ystep),['$2^{%d}$'%yl for yl in np.arange(0,ymax+ystep,ystep)])
        plt.ylim([0,ymax+1])
    plt.tight_layout()
    plt.show()
