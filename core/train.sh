OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='0','1' python -m torch.distributed.launch --nproc_per_node=2  --master_port=9527 \
main.py \
--log_suffix sift_yfcc \
