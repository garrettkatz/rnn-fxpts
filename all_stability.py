import fxpt_experiments as fe

num_procs = 0

test_data_ids = [
    "full_base",
    "big256",
    "big512",
    "big1024",
]

for test_data_id in test_data_ids
    _ = fe.run_TvB_stability_experiments(test_data_id,num_procs)
