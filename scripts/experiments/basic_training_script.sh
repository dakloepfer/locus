#!/bin/sh

main_cfg_path="configs/model/standard_feature_extractor_config.py"
data_cfg_path="configs/data/matterport3d_config.py"

python train.py ${main_cfg_path} ${data_cfg_path} --exp_name=basic_training --device_list 0 1 2 3 --batch_size 4 --fast_dev_run True