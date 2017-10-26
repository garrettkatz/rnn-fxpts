import cProfile
import pstats
import rnn_fxpts as rfx
import fxpt_experiments as fe
import global_experiments as ge

def prof():
    N = 64
    test_data = fe.generate_test_data(network_sizes=[N], num_samples=[1], refine_iters = 1)
    W = test_data['N_%d_W_0'%N]
    # _ = rfx.traverse(W, va=None, c=None, max_nr_iters=2**8, nr_tol=2**-32, max_traverse_steps=None, max_fxpts=None, logfile=None, max_step_size=None)
    for _ in rfx.directional_fiber(W, max_traverse_steps = 2**12, max_nr_iters=2**8, nr_tol=2**-32, max_refine_steps=2**5): pass
    print(_[0]) # status
    # _ = ge.combo_trial(W, timeout=500, term_ratio=2, verbose_prefix = '')
    # print(_[4][-1]) # status

# prof()

cProfile.run('prof()','pstats')
p = pstats.Stats('pstats')
p.strip_dirs().sort_stats('cumulative').print_stats(20)
# p.strip_dirs().sort_stats('cumulative').print_callees(50)
