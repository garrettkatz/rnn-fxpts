import fxpt_experiments as fe

num_procs = 10

test_data_ids = [
    "big1024_base",
    "big512_base",
    "big256_base",
    "full_base",
]

for test_data_id in test_data_ids:
    _ = fe.run_TvB_stability_experiments(test_data_id,num_procs)
