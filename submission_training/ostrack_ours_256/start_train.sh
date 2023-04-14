cd /apdcephfs/private_qiangqwu/Projects/OSTrack_ours
# /apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python tracking/train.py --script ostrack --config vitb_256_mae_ce_32x4_got10k_ep100 --save_dir /apdcephfs/share_1290939/qiangqwu/ostrack_ours --mode multiple --nproc_per_node 4 --use_lmdb 1 --use_wandb 0
/apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python tracking/test.py ostrack vitb_256_mae_ce_32x4_got10k_ep100 --dataset got10k_test --threads 16 --num_gpus 4
/apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name vitb_256_mae_ce_32x4_got10k_ep100
/apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python tracking/test.py ostrack vitb_256_mae_ce_32x4_got10k_ep99 --dataset got10k_test --threads 16 --num_gpus 4
/apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name vitb_256_mae_ce_32x4_got10k_ep99
/apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python tracking/test.py ostrack vitb_256_mae_ce_32x4_got10k_ep98 --dataset got10k_test --threads 16 --num_gpus 4
/apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name vitb_256_mae_ce_32x4_got10k_ep98
/apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python tracking/test.py ostrack vitb_256_mae_ce_32x4_got10k_ep97 --dataset got10k_test --threads 16 --num_gpus 4
/apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name vitb_256_mae_ce_32x4_got10k_ep97
/apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python tracking/test.py ostrack vitb_256_mae_ce_32x4_got10k_ep96 --dataset got10k_test --threads 16 --num_gpus 4
/apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name vitb_256_mae_ce_32x4_got10k_ep96
/apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python tracking/test.py ostrack vitb_256_mae_ce_32x4_got10k_ep95 --dataset got10k_test --threads 16 --num_gpus 4
/apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name vitb_256_mae_ce_32x4_got10k_ep95
/apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python tracking/test.py ostrack vitb_256_mae_ce_32x4_got10k_ep94 --dataset got10k_test --threads 16 --num_gpus 4
/apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name vitb_256_mae_ce_32x4_got10k_ep94
# /apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python tracking/test.py ostrack vitb_256_mae_ce_32x4_got10k_ep93 --dataset got10k_test --threads 16 --num_gpus 4
# /apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name vitb_256_mae_ce_32x4_got10k_ep93
# /apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python tracking/test.py ostrack vitb_256_mae_ce_32x4_got10k_ep92 --dataset got10k_test --threads 16 --num_gpus 4
# /apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name vitb_256_mae_ce_32x4_got10k_ep92
# /apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python tracking/test.py ostrack vitb_256_mae_ce_32x4_got10k_ep91 --dataset got10k_test --threads 16 --num_gpus 4
# /apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name vitb_256_mae_ce_32x4_got10k_ep91
# /apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python tracking/test.py ostrack vitb_256_mae_ce_32x4_got10k_ep90 --dataset got10k_test --threads 16 --num_gpus 4
# /apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name vitb_256_mae_ce_32x4_got10k_ep90
# /apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python tracking/test.py ostrack vitb_256_mae_ce_32x4_got10k_ep80 --dataset got10k_test --threads 16 --num_gpus 4
# /apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name vitb_256_mae_ce_32x4_got10k_ep80
# /apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python tracking/test.py ostrack vitb_384_mae_ce_32x4_got10k_ep79 --dataset got10k_test --threads 16 --num_gpus 4
# /apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name vitb_384_mae_ce_32x4_got10k_ep79
# /apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python tracking/test.py ostrack vitb_384_mae_ce_32x4_got10k_ep40 --dataset got10k_test --threads 16 --num_gpus 4
# /apdcephfs/private_qiangqwu/anaconda3/envs/ostrack/bin/python lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name vitb_384_mae_ce_32x4_got10k_ep40