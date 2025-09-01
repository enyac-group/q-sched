!#/bin/bash
device=0

# run Q-Sched and optimize model coeff and sample coeff
CUDA_VISIBLE_DEVICES=$device python src/qsched_optimize.py --checkpoint SDXLTurbo-4Step --quant_type w4a8 --save_folder schedule_ablations/sdxl_turbo_w4a8  --calib_size 25 --max_model_coeff 1.2 --max_sample_coeff 1.2 --ranking_criteria kclip2 --num_points 50

# evaluate the optimized model using learned coeffs from above
CUDA_VISIBLE_DEVICES=$device python src/eval_dataset.py --save_folder raw_data/sdxl_turbo/4step/w4a4_m0.955_m1.099_s1.012_s0.967 --checkpoint SDXLTurbo-4Step --dataset mjhq  --quant_type w4a4 --max_model_coeff 0.955 --start_model_coeff 1.099 --max_sample_coeff 1.012 --start_sample_coeff 0.967
