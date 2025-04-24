TRAIN_AE=train_ae.py
TENSORBOARD=True
LATENT_SPACE=256
MAX_EPOCH=150
WHEAT_GEN=wheat_gen.py
TRAINED_MODEL=pretrained/GEN_WHEAT_PLANT.pt

PHONY: run 


sample: 
	python $(WHEAT_GEN) --ckpt $(TRAINED_MODEL) --save_dir gen_samples --num_images 10  --num_points 2048 --device cuda 

run_nr: 
	python $(TRAIN_AE) --sched_start_epoch 5000 --sched_end_epoch 15000 --max_iters 50000 --max_grad_norm 2 --lr 1e-3  --end_lr 1e-5 --weight_decay 1e-5 --num_steps 5000 --beta_T 5e-2 --val_freq 10000 --train_batch_size 128	--rotate True

run: 
	python $(TRAIN_AE) --resume logs_ae/AE_2025_04_09__18_52_35/ckpt_0.004541_20000.pt --max_iters 10000 --max_grad_norm 2 --lr 1e-4  --end_lr 1e-5 --weight_decay 1e-6 --num_steps 1500 --beta_T 5e-2 --val_freq 5000 --train_batch_size 128	--rotate True


help:
	python $(TRAIN_AE) --help


clean: 
	rm -rf utils/__pycache__
	rm -rf models/__pycache__
	rm -rf models/encoders/__pycache__
