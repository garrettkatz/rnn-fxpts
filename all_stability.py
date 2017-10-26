import fxpt_experiments as fe

num_procs = 0

test_data_ids = [
    "full_base",
    "big256_base",
    "big512_base",
    "big1024_base",
]

for test_data_id in test_data_ids:
    _ = fe.run_TvB_stability_experiments(test_data_id,num_procs)
