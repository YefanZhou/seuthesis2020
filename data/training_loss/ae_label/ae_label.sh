# only do classification no recontruction train 40 epoch
#No.2
''' 
python ../../Protraintest_ae.py --data-basedir '../../../What3D' \
						  --ptcloud-path "ptcloud_0.npz" \
						  --image-size 128 \
						  --view "0" \
						  --alpha 0. \
						  --total-epoch 100 \
						  --train-batch-size 128 \
						  --test-batch-size 200 \
						  --val-batch-size 200 \
						  --verbose_per_n_batch 5 \
						  --lr-G 1e-3 \
						  --lambda-loss-fine 1. \
						  --lambda-loss-classification 1. \
						  --output-dir 'results_adam_1/' \
						  --snapshot-dir 'snapshots_adam_1/' \
						  --log-dir	"logs_adam_1/"	\
						  --momentum 0.9 \
						  --weight-decay 1e-6 \
						  --lr_decay_step 200 \
						  --test \
						  --train 
'''

#No.1 
python ../../Protraintest_ae.py --data-basedir '../../../../What3D' \
						  --ptcloud-path "ptcloud_0.npz" \
						  --image-size 128 \
						  --view "0" \
						  --alpha 0. \
						  --total-epoch 100 \
						  --train-batch-size 128 \
						  --test-batch-size 100 \
						  --val-batch-size 200 \
						  --verbose_per_n_batch 5 \
						  --lr-G 3e-3 \
						  --lambda-loss-fine 1. \
						  --lambda-loss-classification 1. \
						  --output-dir 'results_adam_0/' \
						  --snapshot-dir 'snapshots_adam_0/' \
						  --log-dir	"logs_adam_0/"	\
						  --momentum 0.9 \
						  --weight-decay 1e-6 \
						  --lr_decay_step 30 \
						  --test \


