N_CAM=4  # or 3, or 2
SAVE_PRED_PATH=/tmp/preds

python -m metapose.train_metapose \
    --data_root=${DATA_ROOT} \
    --experiment_name=test_github_version \
    --tb_log_dir=/tmp/ \
    --dataset=data/h36m/opt/${N_CAM} \
    --data_splits=train,test \
    --dataset_warmup=false \
    --n_cam=${N_CAM} \
    --train_repeat_k=1 \
    --permute_cams_aug=false \
    --use_equivariant_model=true \
    --debug_show_single_frame_pmpjes=false \
    --debug_enable_check_numerics=false \
    --use_equivariant_model=true \
    --standardize_init_best=false \
    --main_mlp_spec=512,512,ccat,512,512,ccat,512 \
    --load_weights_from=${DATA_ROOT}/ckpt/h36m/cam${N_CAM}/model \
    --load_stages_n=1 \
    --epochs_per_stage=0 \
    --max_stage_attempts=1 \
    --save_preds_to=${SAVE_PRED_PATH}
