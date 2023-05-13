python3 train.py \
    --model 'Highway' \
    --expt_name 'Highway_42_depth0_dim300_lr1e3_stop2_fac30_pat1' \
    --prodfps_prefix 1000000dim_2rad_to_32681_prod_fps \
    --labels_prefix 1000000dim_2rad_to_32681_labels \
    --csv_prefix 1000000dim_2rad_to_32681_csv \
    --bs 300 \
    --bs_eval 300 \
    --random_seed 42 \
    --learning_rate 1e-3 \
    --epochs 30 \
    --early_stop \
    --early_stop_patience 2 \
    --depth 0 \
    --hidden_size 300 \
    --lr_scheduler_factor 0.3 \
    --lr_scheduler_patience 1 \
    --device 0 \
    --checkpoint
