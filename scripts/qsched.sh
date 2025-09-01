#!/bin/bash
device=0
checkpoint=SDXLTurbo-4Step
# run Q-Sched and optimize model coeff and sample coeff
CUDA_VISIBLE_DEVICES=$device python src/qsched_optimize.py \
        --checkpoint $checkpoint \
        --quant_type w4a8 \
        --save_folder schedule_ablations/sdxl_turbo_w4a8 \
        --calib_size 25 \
        --max_model_coeff 1.2 \
        --max_sample_coeff 1.2 \
        --ranking_criteria kclip2 \
        --num_points 50 \
        --resume
