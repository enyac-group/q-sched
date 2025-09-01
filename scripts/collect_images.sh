#!/bin/bash
device=0
checkpoint=SDXLTurbo-4Step

# evaluate the optimized model using learned coeffs from qsched
CUDA_VISIBLE_DEVICES=$device python src/eval_dataset.py \
        --save_folder raw_data/sdxl_turbo/4step/w4a4_m0.955_m1.099_s1.012_s0.967 \
        --checkpoint $checkpoint \
        --dataset mjhq \
        --quant_type w4a4 \
        --max_model_coeff 0.955 \
        --start_model_coeff 1.099 \
        --max_sample_coeff 1.012 \
        --start_sample_coeff 0.967