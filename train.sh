python tools/train.py configs/motionnet/motionnet_02pillar_stpn_cyclic_nus.py \
     --cfg-options data.train.ann_file='data/nuscenes/motion_mini/nus_5_sweeps_infos_train.pkl' \
     data.train.dataset.ann_file='data/nuscenes/motion_mini/nus_5_sweeps_infos_train.pkl' \
     data.val.ann_file='data/nuscenes/motion_mini/nus_5_sweeps_infos_val.pkl' \
     data.workers_per_gpu=0 data.samples_per_gpu=2
